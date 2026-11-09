#!/usr/bin/env python3
"""
analyse_reviews_judge.py: the LLM judge's results on the app reviews, and the human check of the judge.

Reviews (items from prepare_reviews_judge.py, results from COMP8240_reviews_judge.ipynb):
  meaning     mean score for the sanitized reviews (SanText, eps = 3) and for identical pairs
  disclosure  the judge as a third annotator: Cohen's kappa with the gold labels and with annotator A on the
              100 original reviews; how often it still finds a disclosure after sanitization
  residual    of the 13 gold-disclosing reviews, how many still reveal the flagged information
Human check (human_check_all.xlsx, filled in by the author without seeing the judge's answers):
  meaning rows: Spearman correlation, quadratic-weighted kappa and agreement within one point between the
  author's 1-5 score and the judge's answer; residual rows: agreement and Cohen's kappa.

Usage
  python analyse_reviews_judge.py --items judge_items.jsonl --results judge_results.jsonl \
      --c_results <notebook C>/judge_results.jsonl --human human_check_all.xlsx --key human_check_all_key.csv \
      [--out reviews_judge_summary.json]
"""
import argparse
import collections
import csv
import json
import statistics


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def kappa(a, b, labels=None, weights=None):
    """Cohen's kappa; weights='quadratic' for ordinal labels."""
    labels = sorted(set(a) | set(b)) if labels is None else labels
    k, n = len(labels), len(a)
    idx = {x: i for i, x in enumerate(labels)}
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    ra = [sum(row) for row in obs]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def w(i, j):
        if weights == "quadratic":
            return ((i - j) ** 2) / ((k - 1) ** 2) if k > 1 else 0.0
        return 0.0 if i == j else 1.0

    po = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k)) / n
    pe = sum(w(i, j) * ra[i] * cb[j] for i in range(k) for j in range(k)) / (n * n)
    return 1.0 - po / pe if pe else float("nan")


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def read_human(path):
    from openpyxl import load_workbook
    ws = load_workbook(path, data_only=True)["Check"]
    out = {}
    for vals in ws.iter_rows(min_row=2, values_only=True):
        if vals[0] is None or vals[5] in (None, ""):
            continue
        out[int(vals[0])] = str(vals[5]).strip().lower()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--c_results")
    ap.add_argument("--human")
    ap.add_argument("--key")
    ap.add_argument("--out")
    args = ap.parse_args()
    items = {it["id"]: it for it in load_jsonl(args.items)}
    res = {r["id"]: r for r in load_jsonl(args.results)}
    out = {"judged": f"{len(res)}/{len(items)}",
           "min_format_mass": round(min(r["format_mass"] for r in res.values()), 4)}

    # meaning
    groups = collections.defaultdict(list)
    for r in res.values():
        if r["task"] == "meaning":
            groups[r["method"]].append(r["expected"])
    out["meaning_mean"] = {m: [round(statistics.mean(v), 2), len(v)] for m, v in groups.items()}

    # disclosure: the judge as a third annotator
    disc = [(items[i], r) for i, r in res.items() if r["task"] == "disclosure"]
    orig = [(it, r) for it, r in disc if it["method"] == "original"]
    if orig:
        j = [r["answer"] for _, r in orig]
        g = [it["gold"] for it, _ in orig]
        a = [it["A"] for it, _ in orig]
        out["disclosure_original"] = {
            "n": len(orig), "judge_yes": j.count("yes"), "gold_yes": g.count("yes"),
            "kappa_gold": round(kappa(j, g, ["no", "yes"]), 3), "agree_gold": round(sum(x == y for x, y in zip(j, g)) / len(j), 3),
            "kappa_A": round(kappa(j, a, ["no", "yes"]), 3), "agree_A": round(sum(x == y for x, y in zip(j, a)) / len(j), 3),
            "judge_yes_gold_no": sum(x == "yes" and y == "no" for x, y in zip(j, g)),
            "judge_no_gold_yes": sum(x == "no" and y == "yes" for x, y in zip(j, g))}
    for method in sorted({it["method"] for it, _ in disc} - {"original"}):
        sub = [(it, r) for it, r in disc if it["method"] == method]
        out[f"disclosure_{method}"] = {
            "n": len(sub), "judge_yes": sum(r["answer"] == "yes" for _, r in sub),
            "yes_among_gold_yes": f"{sum(r['answer'] == 'yes' for it, r in sub if it['gold'] == 'yes')}/{sum(it['gold'] == 'yes' for it, _ in sub)}",
            "yes_among_gold_no": f"{sum(r['answer'] == 'yes' for it, r in sub if it['gold'] == 'no')}/{sum(it['gold'] == 'no' for it, _ in sub)}"}

    # residual
    resid = collections.defaultdict(list)
    for i, r in res.items():
        if r["task"] == "residual":
            resid[r["method"]].append((items[i]["text_key"], items[i]["category"], r["answer"], round(r["p_yes"], 3)))
    out["residual"] = {m: {"still_reveals": f"{sum(x[2] == 'yes' for x in v)}/{len(v)}",
                           "by_category": dict(collections.Counter(f"{x[1]}:{x[2]}" for x in v)),
                           "mean_p_yes": round(statistics.mean(x[3] for x in v), 3), "items": sorted(v)}
                       for m, v in resid.items()}

    # human check
    if args.human and args.key:
        with open(args.key, encoding="utf-8") as f:
            key = {int(r["check_id"]): r["item_id"] for r in csv.DictReader(f)}
        allres = dict(res)
        if args.c_results:
            allres.update({r["id"]: r for r in load_jsonl(args.c_results)})
        human = read_human(args.human)
        m_h, m_j, m_e, r_h, r_j, missing = [], [], [], [], [], []
        for cid, item_id in sorted(key.items()):
            if cid not in human:
                missing.append(cid)
                continue
            if item_id not in allres:
                missing.append(f"{cid}: no judge result for {item_id}")
                continue
            r = allres[item_id]
            if r["task"] == "meaning":
                m_h.append(int(float(human[cid])))
                m_j.append(int(r["answer"]))
                m_e.append(r["expected"])
            else:
                r_h.append("yes" if human[cid].startswith("y") else "no")
                r_j.append(r["answer"])
        hc = {"missing": missing}
        if m_h:
            hc["meaning"] = {"n": len(m_h), "spearman_expected": round(spearman(m_h, m_e), 3),
                             "kappa_quadratic": round(kappa(m_h, m_j, [1, 2, 3, 4, 5], "quadratic"), 3),
                             "exact": round(sum(x == y for x, y in zip(m_h, m_j)) / len(m_h), 3),
                             "within_1": round(sum(abs(x - y) <= 1 for x, y in zip(m_h, m_j)) / len(m_h), 3),
                             "mean_human": round(statistics.mean(m_h), 2), "mean_judge": round(statistics.mean(m_e), 2)}
        if r_h:
            hc["residual"] = {"n": len(r_h), "agree": sum(x == y for x, y in zip(r_h, r_j)),
                              "kappa": round(kappa(r_h, r_j, ["no", "yes"]), 3),
                              "human_yes": r_h.count("yes"), "judge_yes": r_j.count("yes")}
        out["human_check"] = hc
    print(json.dumps(out, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
