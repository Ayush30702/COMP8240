#!/usr/bin/env python3
"""
analyse_c.py: summary tables for notebook C (privacy experiment, sanitizer check, LLM judge).

Reads results_c/ (written by the notebook) and writes results_c/summary_c.md plus CSV tables:
  datasets.json          -> original-dataset statistics (SST-2, QNLI) as measured from the files
  glove_check.csv        -> block-wise sanitizer vs the authors' code (rows; unchanged-token rates)
  bert_santext.csv       -> SanText with BERT embeddings on SST-2: unchanged tokens, utility vs the
                            paper's Table 3, defense rate of the mask token attack (released vs fixed)
  judge_*.csv            -> LLM-judge tables (meaning, disclosure, residual disclosure)
Usage: python analyse_c.py --root results_c --plan_c plan_c.json
"""
import argparse
import collections
import csv
import json
import os
import statistics

TABLE3 = {12.0: 0.6084, 14.0: 0.7548, 16.0: 0.8698}  # Yue et al. (2021), Table 3, SST-2, "Original"
UNSANITIZED_PAPER = 0.9251


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def kappa(a, b):
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fmt(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def unchanged_share(orig_path, san_path, tokenizer):
    with open(orig_path, encoding="utf-8") as f:
        next(f)
        orig = [[t.text for t in tokenizer(line.strip().split("\t")[0])] for line in f]
    with open(san_path, encoding="utf-8") as f:
        next(f)
        san = [line.rstrip("\n").rsplit("\t", 1)[0].split(" ") for line in f]
    same = tot = 0
    for o, s in zip(orig, san):
        if len(o) == len(s):
            tot += len(o)
            same += sum(a == b for a, b in zip(o, s))
    return same / tot if tot else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--plan_c", required=True)
    args = ap.parse_args()
    plan = load_json(args.plan_c)
    R = args.root
    md = ["# Notebook C: summary", ""]
    env = load_json(os.path.join(R, "environment.json"))
    if env:
        md += ["Environment: " + ", ".join(f"{k} {v}" for k, v in env.items()), ""]

    # ---------------------------------------------------------------- original datasets
    ds = load_json(os.path.join(R, "datasets.json"))
    if ds:
        md += ["## Original datasets, measured from the files", ""]
        for name, d in ds.items():
            md.append(f"**{name}**: " + "; ".join(f"{k}: {v}" for k, v in d.items() if k != "examples"))
            for ex in d.get("examples", []):
                md.append(f"  - `{ex}`")
            md.append("")

    # ---------------------------------------------------------------- block-wise sanitizer vs the authors'
    rows = []
    try:
        from spacy.lang.en import English
        tok = English()
    except Exception:
        tok = None
    for e in plan.get("glove_check_eps", []):
        meta = load_json(os.path.join(plan["glove_check_dir"], "eps_%.2f" % e, "meta.json"))
        if not meta:
            continue
        row = {"epsilon": e, "rows_checked": meta.get("check", {}).get("rows_checked"),
               "max_abs_diff": meta.get("check", {}).get("max_abs_diff"),
               "rows_identical": meta.get("check", {}).get("rows_identical"),
               "chunked_unchanged_dev": meta["splits"]["dev"]["share_unchanged"],
               "chunked_peak_gb": meta.get("peak_memory_gb"), "chunked_seconds": meta.get("seconds")}
        if tok:
            for label, folder in plan.get("authors_glove_dirs", {}).items():
                p = os.path.join(folder, "eps_%.2f" % e, "dev.tsv")
                if os.path.exists(p):
                    row[f"authors_unchanged_dev ({label})"] = unchanged_share(plan["sst2_dev"], p, tok)
            row["chunked_unchanged_dev_recount"] = unchanged_share(
                plan["sst2_dev"], os.path.join(plan["glove_check_dir"], "eps_%.2f" % e, "dev.tsv"), tok)
        rows.append(row)
    write_csv(os.path.join(R, "glove_check.csv"), rows)
    if rows:
        md += ["## Block-wise sanitizer vs the authors' code (SST-2, GloVe, SanText, seed 42)", "",
               "| ε | Rows checked | Max abs. difference | Rows bit-identical | Unchanged tokens (dev), block-wise | Unchanged tokens (dev), authors' runs |",
               "|---|---|---|---|---|---|"]
        for r in rows:
            authors = ", ".join(f"{fmt(v, 4)} ({k.split('(')[1][:-1]})" for k, v in r.items() if k.startswith("authors_") and v is not None)
            md.append(f"| {r['epsilon']:g} | {r['rows_checked']} | {r['max_abs_diff']:.1e} | {r['rows_identical']} | "
                      f"{fmt(r.get('chunked_unchanged_dev_recount') or r['chunked_unchanged_dev'], 4)} | {authors or '–'} |")
        md.append("")

    # ---------------------------------------------------------------- BERT-embedding SanText: utility and attack
    rows = []
    base_attack = load_json(os.path.join(R, "attack", "unsanitized.json"))
    unsan = load_json(os.path.join(plan["results_dir"], plan["unsanitized_run"], "metrics.json"))
    for e in [None] + plan.get("bert_eps", []):
        if e is None:
            meta, metrics, attack = None, unsan, base_attack
        else:
            meta = load_json(os.path.join(plan["bert_dir"], "eps_%.2f" % e, "meta.json"))
            metrics = load_json(os.path.join(plan["results_dir"], f"sst2__SanText_bert__eps{e:.2f}__seed42", "metrics.json"))
            attack = load_json(os.path.join(R, "attack", f"eps{e:g}.json"))
        v = (attack or {}).get("variants", {})
        rows.append({"epsilon": "∞ (unsanitized)" if e is None else e,
                     "unchanged_dev": None if not meta else meta["splits"]["dev"]["share_unchanged"],
                     "accuracy": None if not metrics else metrics["final_dev"]["accuracy"],
                     "paper_accuracy": UNSANITIZED_PAPER if e is None else TABLE3.get(e),
                     "defense_released": v.get("released", {}).get("defense_rate"),
                     "defense_fixed": v.get("fixed", {}).get("defense_rate"),
                     "skipped_released": v.get("released", {}).get("skipped_truncated"),
                     "positions": v.get("fixed", {}).get("positions"),
                     "seconds": None if not meta else meta.get("seconds"),
                     "peak_gb": None if not meta else meta.get("peak_memory_gb")})
    write_csv(os.path.join(R, "bert_santext.csv"), rows)
    md += ["## SanText with BERT embeddings on SST-2 (dev): utility and the mask token inference attack", "",
           "| ε | Tokens unchanged | Accuracy (ours) | Accuracy (paper, Table 3) | Defense rate, attack as released | Defense rate, attack fixed | Positions (skipped as released) | Sanitizer s / peak GB |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['epsilon'] if isinstance(r['epsilon'], str) else format(r['epsilon'], 'g')} | "
                  f"{fmt(r['unchanged_dev'])} | {fmt(r['accuracy'], 4)} | {fmt(r['paper_accuracy'], 4)} | "
                  f"{fmt(r['defense_released'])} | {fmt(r['defense_fixed'])} | {r['positions'] or '–'} ({r['skipped_released'] if r['skipped_released'] is not None else '–'}) | "
                  f"{r['seconds'] if r['seconds'] is not None else '–'} / {r['peak_gb'] if r['peak_gb'] is not None else '–'} |")
    md.append("")

    # ---------------------------------------------------------------- LLM judge
    jr = os.path.join(R, "judge", "judge_results.jsonl")
    if os.path.exists(jr):
        with open(jr, encoding="utf-8") as f:
            res = [json.loads(l) for l in f if l.strip()]
        model = res[0]["judge_model"] if res else "?"
        md += [f"## LLM judge ({model}, prompt {res[0]['prompt_version'] if res else '?'})", ""]
        mass = collections.defaultdict(list)
        for r in res:
            mass[r["task"]].append(r["format_mass"])
        md.append("Probability on the allowed answer tokens (format compliance): " +
                  ", ".join(f"{k} {statistics.mean(v):.3f}" for k, v in mass.items()))
        md.append("")
        # meaning
        g = collections.defaultdict(list)
        for r in res:
            if r["task"] == "meaning":
                g[(r["dataset"], r["method"], r["epsilon"])].append(r["expected"])
        mrows = [{"dataset": d, "method": m, "epsilon": e, "n": len(v), "mean": statistics.mean(v),
                  "sd": statistics.stdev(v) if len(v) > 1 else None} for (d, m, e), v in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0))]
        write_csv(os.path.join(R, "judge_meaning.csv"), mrows)
        md += ["### Meaning preserved (1–5, expected value)", "", "| Dataset | Method | ε | n | Mean | SD |", "|---|---|---|---|---|---|"]
        md += [f"| {r['dataset']} | {r['method']} | {'–' if r['epsilon'] is None else format(r['epsilon'], 'g')} | {r['n']} | {r['mean']:.2f} | {fmt(r['sd'], 2)} |" for r in mrows]
        md.append("")
        # disclosure on originals: agreement
        orig = [r for r in res if r["task"] == "disclosure" and r["method"] == "original"]
        if orig:
            judge = [r["answer"] for r in orig]
            gold = [r["gold"] for r in orig]
            a = [r["A"] for r in orig]
            md += ["### Disclosure on the 100 original reviews (the judge as a third annotator)", "",
                   f"- vs gold (adjudicated): agreement {sum(x == y for x, y in zip(judge, gold)) / len(orig):.2f}, κ {fmt(kappa(judge, gold), 2)}",
                   f"- vs annotator A: agreement {sum(x == y for x, y in zip(judge, a)) / len(orig):.2f}, κ {fmt(kappa(judge, a), 2)}",
                   f"- judged as disclosing: {sum(x == 'yes' for x in judge)} of {len(orig)} (gold: {sum(x == 'yes' for x in gold)})", ""]
        # disclosure after sanitization
        g = collections.defaultdict(lambda: [0, 0, 0, 0])
        for r in res:
            if r["task"] == "disclosure":
                k = (r["method"], r["epsilon"])
                if r["gold"] == "yes":
                    g[k][0] += 1
                    g[k][1] += r["answer"] == "yes"
                else:
                    g[k][2] += 1
                    g[k][3] += r["answer"] == "yes"
        drows = [{"method": m, "epsilon": e, "gold_yes": v[0], "still_judged_disclosing": v[1],
                  "share_still_disclosing": v[1] / v[0] if v[0] else None, "gold_no": v[2],
                  "judged_disclosing_spuriously": v[3], "share_spurious": v[3] / v[2] if v[2] else None}
                 for (m, e), v in sorted(g.items(), key=lambda kv: (kv[0][0] != "original", kv[0][0], kv[0][1] or 0))]
        write_csv(os.path.join(R, "judge_disclosure.csv"), drows)
        md += ["### Is a disclosure still judged present after sanitization? (reviews)", "",
               "| Version | ε | Gold-disclosing reviews still judged disclosing | Non-disclosing reviews judged disclosing |", "|---|---|---|---|"]
        md += [f"| {r['method']} | {'–' if r['epsilon'] is None else format(r['epsilon'], 'g')} | {r['still_judged_disclosing']}/{r['gold_yes']} ({fmt(r['share_still_disclosing'], 2)}) | "
               f"{r['judged_disclosing_spuriously']}/{r['gold_no']} ({fmt(r['share_spurious'], 2)}) |" for r in drows]
        md.append("")
        # residual
        g = collections.defaultdict(list)
        for r in res:
            if r["task"] == "residual":
                g[(r["method"], r["epsilon"])].append(r)
        rrows = [{"method": m, "epsilon": e, "n": len(v), "still_reveals": sum(x["answer"] == "yes" for x in v),
                  "share": sum(x["answer"] == "yes" for x in v) / len(v), "mean_p_yes": statistics.mean(x["p_yes"] for x in v)}
                 for (m, e), v in sorted(g.items(), key=lambda kv: (kv[0][0] != "original", kv[0][0], kv[0][1] or 0))]
        write_csv(os.path.join(R, "judge_residual.csv"), rrows)
        md += ["### Targeted check: does the sanitized review still reveal the flagged information?", "",
               "| Version | ε | n | Still reveals | Mean P(yes) |", "|---|---|---|---|---|"]
        md += [f"| {r['method']} | {'–' if r['epsilon'] is None else format(r['epsilon'], 'g')} | {r['n']} | {r['still_reveals']} ({r['share']:.2f}) | {r['mean_p_yes']:.2f} |" for r in rrows]
        md.append("")
    with open(os.path.join(R, "summary_c.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
