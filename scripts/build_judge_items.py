#!/usr/bin/env python3
"""
build_judge_items.py: the items the LLM judge rates, and a blind sheet for the human check.

Items (one JSON object per line in judge_items.jsonl):
  meaning     original vs sanitized dev text, for SST-2, Dreaddit and the app reviews:
              100 texts per dataset (for the reviews, the 100 annotated ones) x SanText and SanText+
              at eps = 1, 2, 3 (seed 42), plus 20 identical pairs per dataset as a sanity check
  disclosure  the 100 annotated reviews, original and all six sanitized versions
  residual    the reviews whose original discloses something (gold label), six sanitized versions,
              plus the originals as a sanity check
human_check.xlsx: 24 meaning and 12 residual items, chosen at random within strata, without the judge's
answers; human_check_key.csv maps each row back to its item.

Usage
  python build_judge_items.py --config judge_inputs.json --out_dir judge
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re

METHODS = (("SanText", "SanText"), ("SanText_plus", "SanText+"))
EPS = (1.0, 2.0, 3.0)


def read_original(path):
    with open(path, encoding="utf-8") as f:
        next(f)
        return [line.strip().split("\t")[0] for line in f]


def read_sanitized(path):
    with open(path, encoding="utf-8") as f:
        next(f)
        return [line.rstrip("\n").rsplit("\t", 1)[0] for line in f]


def norm(text):
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def review_texts(labels, reviews_clean, annotation_sheet):
    """ann_id -> review text: review_key = sha1(review_id)[:12] in reviews_clean.jsonl, else the sheet's text."""
    by_key = {}
    if reviews_clean and os.path.exists(reviews_clean):
        with open(reviews_clean, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                key = hashlib.sha1((r.get("review_id") or r["text"]).encode("utf-8")).hexdigest()[:12]
                by_key[key] = r["text"]
    by_id = {}
    if annotation_sheet and os.path.exists(annotation_sheet):
        with open(annotation_sheet, encoding="utf-8-sig", newline="") as f:
            by_id = {int(r["ann_id"]): r["text"] for r in csv.DictReader(f) if r.get("ann_id")}
    out, how = {}, collections.Counter()
    for r in labels:
        i = int(r["ann_id"])
        if r["review_key"] in by_key:
            out[i] = by_key[r["review_key"]]
            how["review_key"] += 1
        elif i in by_id:
            out[i] = by_id[i]
            how["annotation_sheet"] += 1
    return out, dict(how)


def align(texts, dev_texts):
    """ann_id -> row in the dev (test) split: exact text, then normalised text."""
    exact = {t: k for k, t in reversed(list(enumerate(dev_texts)))}
    normed = {norm(t): k for k, t in reversed(list(enumerate(dev_texts)))}
    out, how = {}, collections.Counter()
    for i, t in texts.items():
        if t in exact:
            out[i] = exact[t]
            how["exact"] += 1
        elif norm(t) in normed:
            out[i] = normed[norm(t)]
            how["normalised"] += 1
        else:
            how["not_found"] += 1
    return out, dict(how)


def sanitized_path(san_root, ds, method, eps, old_sst2=None):
    """Where notebook A/B wrote the seed-42 output; for SST-2 SanText, fall back to the 9 Oct run."""
    sub = os.path.join(san_root, ds, method, "seed_42", "eps_%.2f" % eps)
    if method == "SanText_plus":
        sub = os.path.join(sub, "sword_0.90_p_0.30")
    p = os.path.join(sub, "dev.tsv")
    if os.path.exists(p):
        return p, "notebook A/B"
    if ds == "sst2" and method == "SanText" and old_sst2:
        q = os.path.join(old_sst2, "eps_%.2f" % eps, "dev.tsv")
        if os.path.exists(q):
            return q, "9 Oct run"
    return None, None


def info_string(words, cats):
    w = [x.strip() for x in words.split(";") if x.strip()]
    c = [x.strip().lower() for x in cats.split(";") if x.strip()]
    return "; ".join(f'"{x}" ({c[k] if k < len(c) else c[0]})' for k, x in enumerate(w))


def write_human_sheet(rows, path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    how = wb.active
    how.title = "How to"
    lines = [
        "Human check of the LLM judge (about 20 minutes)",
        "",
        "Each row on the 'Check' sheet shows an original text (A) and a version made from it by SanText (B).",
        "Fill only the 'your answer' column. Don't look anything up; go with your first reading.",
        "",
        "Rows marked MEANING: how well does B preserve the meaning of A?",
        "  1 = unrelated or unintelligible",
        "  2 = mostly lost; only fragments survive",
        "  3 = the topic survives, but important details are lost or changed",
        "  4 = mostly the same meaning; minor details are lost",
        "  5 = the same meaning",
        "",
        "Rows marked STILL REVEALS: the 'information' column says what the original review disclosed.",
        "  Answer yes if B still reveals that information, or enough of it that a reader could infer it; otherwise no.",
        "",
        "Save the file and upload it in the chat when you're done.",
    ]
    for k, line in enumerate(lines, start=1):
        how.cell(row=k, column=1, value=line).font = Font(bold=(k == 1), size=12 if k == 1 else 11)
    how.column_dimensions["A"].width = 120
    ws = wb.create_sheet("Check")
    header = ["check_id", "task", "A: original", "B: sanitized", "information (STILL REVEALS rows)", "your answer"]
    ws.append(header)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDE7F0")
    for r in rows:
        ws.append([r["check_id"], r["task_label"], r["a"], r["b"], r.get("info", ""), None])
    for col, width in zip("ABCDEF", (10, 15, 60, 60, 35, 14)):
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    n = len(rows) + 1
    meaning_rows = [k + 2 for k, r in enumerate(rows) if r["task"] == "meaning"]
    residual_rows = [k + 2 for k, r in enumerate(rows) if r["task"] == "residual"]
    if meaning_rows:
        dv = DataValidation(type="whole", operator="between", formula1="1", formula2="5", allow_blank=True)
        dv.error = "Enter a whole number from 1 to 5"
        ws.add_data_validation(dv)
        for k in meaning_rows:
            dv.add(f"F{k}")
    if residual_rows:
        dv2 = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
        ws.add_data_validation(dv2)
        for k in residual_rows:
            dv2.add(f"F{k}")
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:F{n}"
    wb.active = 1
    wb.save(path)


def main():
    ap = argparse.ArgumentParser(description="Build the LLM-judge items and the human-check sheet.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--per_dataset", type=int, default=100)
    ap.add_argument("--sanity", type=int, default=20)
    ap.add_argument("--human_meaning_per_dataset", type=int, default=8)
    ap.add_argument("--human_residual", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    report = {"sources": {}, "counts": {}}

    with open(cfg["labels"], encoding="utf-8") as f:
        labels = list(csv.DictReader(f))
    texts, how_text = review_texts(labels, cfg.get("reviews_clean"), cfg.get("annotation_sheet"))
    report["review_texts_found"] = how_text

    originals = {ds: read_original(p) for ds, p in cfg["original_dev"].items() if p and os.path.exists(p)}
    versions = {}  # (ds, method, eps) -> sanitized dev texts
    for ds in originals:
        for method, _ in METHODS:
            for e in EPS:
                p, src = sanitized_path(cfg["san_root"], ds, method, e, cfg.get("old_sst2"))
                if p:
                    san = read_sanitized(p)
                    if len(san) != len(originals[ds]):
                        raise ValueError(f"{p}: {len(san)} rows, original has {len(originals[ds])}")
                    versions[(ds, method, e)] = san
                    report["sources"][f"{ds}/{method}/eps{e:g}"] = src
    items = []

    # which dev rows each dataset uses
    picks = {}
    if "reviews" in originals:
        rowmap, how_align = align(texts, originals["reviews"])
        report["review_alignment"] = how_align
        picks["reviews"] = {f"r{i:03d}": k for i, k in sorted(rowmap.items())}
    for ds in originals:
        if ds != "reviews":
            idx = sorted(random.Random(f"{args.seed}-{ds}").sample(range(len(originals[ds])),
                                                                    min(args.per_dataset, len(originals[ds]))))
            picks[ds] = {f"{ds[0]}{k:04d}": k for k in idx}

    # meaning
    for ds, chosen in picks.items():
        for key, row in chosen.items():
            for method, mlabel in METHODS:
                for e in EPS:
                    if (ds, method, e) in versions:
                        items.append({"id": f"meaning|{ds}|{key}|{method}|{e:g}", "task": "meaning", "dataset": ds,
                                      "text_key": key, "row": row, "method": mlabel, "epsilon": e,
                                      "a": originals[ds][row], "b": versions[(ds, method, e)][row]})
        for key in list(chosen)[: args.sanity]:
            row = chosen[key]
            items.append({"id": f"meaning|{ds}|{key}|identical|0", "task": "meaning", "dataset": ds, "text_key": key,
                          "row": row, "method": "identical", "epsilon": None,
                          "a": originals[ds][row], "b": originals[ds][row]})

    # disclosure and residual (annotated reviews)
    lab = {f"r{int(r['ann_id']):03d}": r for r in labels}
    for key, row in picks.get("reviews", {}).items():
        r = lab[key]
        base = {"dataset": "reviews", "text_key": key, "row": row, "gold": r["gold_disclosure"],
                "A": r["A_disclosure"], "category": r["category"]}
        items.append({"id": f"disclosure|{key}|original|0", "task": "disclosure", "method": "original", "epsilon": None,
                      "a": originals["reviews"][row], **base})
        for method, mlabel in METHODS:
            for e in EPS:
                if ("reviews", method, e) in versions:
                    items.append({"id": f"disclosure|{key}|{method}|{e:g}", "task": "disclosure", "method": mlabel,
                                  "epsilon": e, "a": versions[("reviews", method, e)][row], **base})
        if r["gold_disclosure"] == "yes":
            info = info_string(r["gold_words"], r["gold_categories"])
            items.append({"id": f"residual|{key}|original|0", "task": "residual", "method": "original", "epsilon": None,
                          "info": info, "b": originals["reviews"][row], **base})
            for method, mlabel in METHODS:
                for e in EPS:
                    if ("reviews", method, e) in versions:
                        items.append({"id": f"residual|{key}|{method}|{e:g}", "task": "residual", "method": mlabel,
                                      "epsilon": e, "info": info, "a": originals["reviews"][row],
                                      "b": versions[("reviews", method, e)][row], **base})

    with open(os.path.join(args.out_dir, "judge_items.jsonl"), "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    report["counts"] = dict(collections.Counter(f"{it['task']}/{it['dataset']}" for it in items))

    # blind human-check sheet
    human = []
    for ds in picks:
        pool = [it for it in items if it["task"] == "meaning" and it["dataset"] == ds and it["method"] != "identical"]
        by_combo = collections.defaultdict(list)
        for it in pool:
            by_combo[(it["method"], it["epsilon"])].append(it)
        combos = sorted(by_combo)
        used_texts, chosen = set(), []
        k = 0
        while len(chosen) < args.human_meaning_per_dataset and any(by_combo.values()):
            combo = combos[k % len(combos)]
            k += 1
            cands = [it for it in by_combo[combo] if it["text_key"] not in used_texts]
            if not cands:
                if k > 10 * len(combos):
                    break
                continue
            it = rng.choice(cands)
            used_texts.add(it["text_key"])
            chosen.append(it)
        human += chosen
    res_pool = [it for it in items if it["task"] == "residual" and it["method"] != "original"]
    for mlabel in ("SanText", "SanText+"):
        sub = [it for it in res_pool if it["method"] == mlabel]
        human += rng.sample(sub, min(args.human_residual // 2, len(sub)))
    rng.shuffle(human)
    rows, key_rows = [], []
    for n, it in enumerate(human, start=1):
        rows.append({"check_id": n, "task": it["task"], "task_label": "MEANING" if it["task"] == "meaning" else "STILL REVEALS",
                     # STILL REVEALS rows show what the judge sees: the information and B, not the original
                     "a": it["a"] if it["task"] == "meaning" else "(not shown: judge from the information column and B)",
                     "b": it["b"], "info": it.get("info", "")})
        key_rows.append({"check_id": n, "item_id": it["id"]})
    if rows:
        write_human_sheet(rows, os.path.join(args.out_dir, "human_check.xlsx"))
        with open(os.path.join(args.out_dir, "human_check_key.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["check_id", "item_id"])
            w.writeheader()
            w.writerows(key_rows)
    report["human_check_rows"] = dict(collections.Counter(r["task"] for r in rows))
    with open(os.path.join(args.out_dir, "items_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
