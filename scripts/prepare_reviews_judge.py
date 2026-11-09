#!/usr/bin/env python3
"""
prepare_reviews_judge.py: LLM-judge items for the app reviews, rebuilt from the 9 October outputs.

The reviews' raw training split is no longer available, so notebooks A/B and C could not rerun the
reviews. The 9 October outputs (comp8240_update_outputs.zip) still hold everything the judge needs for
the test split:
  results/reviews_unsanitized/dev_predictions.tsv              original test reviews and labels
  SanText/output_SanText_glove/app_reviews/eps_3.00/dev.tsv    the same reviews after SanText, eps = 3
  new_data/app_reviews/annotation/annotation_sheet.csv         the texts of the 100 annotated reviews
This script lays them out the way build_judge_items.py expects, builds the items (meaning, disclosure and
residual, as for the other datasets), and merges the reviews' rows of the human check into notebook C's
sheet, so that one person rates all rows in one blind, shuffled sheet.

Usage
  python prepare_reviews_judge.py --a2 <unzipped 9 Oct outputs> --labels annotation_labels.csv \
      --c_judge <notebook C's results_c/judge> --out <folder>
Outputs in <folder>/judge: judge_items.jsonl (reviews only), items_report.json,
  human_check_all.xlsx and human_check_all_key.csv (notebook C's rows plus the reviews' rows).
"""
import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_judge_items import write_human_sheet  # noqa: E402


def read_sheet(path):
    """Rows of a human-check sheet written by build_judge_items.py."""
    from openpyxl import load_workbook
    ws = load_workbook(path)["Check"]
    rows = []
    for vals in ws.iter_rows(min_row=2, values_only=True):
        if vals[0] is None:
            continue
        rows.append({"check_id": int(vals[0]), "task": "meaning" if vals[1] == "MEANING" else "residual",
                     "task_label": vals[1], "a": vals[2], "b": vals[3], "info": vals[4] or ""})
    return rows


def read_key(path):
    with open(path, encoding="utf-8") as f:
        return {int(r["check_id"]): r["item_id"] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--a2", required=True, help="unzipped comp8240_update_outputs.zip")
    ap.add_argument("--labels", required=True, help="annotation_labels.csv (with the unredacted flagged words)")
    ap.add_argument("--c_judge", required=True, help="notebook C's results_c/judge folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    orig_dir = os.path.join(args.out, "original")
    san_dir = os.path.join(args.out, "san", "reviews", "SanText", "seed_42", "eps_3.00")
    judge_dir = os.path.join(args.out, "judge")
    for d in (orig_dir, san_dir, judge_dir):
        os.makedirs(d, exist_ok=True)

    # original test split, from the unsanitized run's predictions (same rows, same order as dev.tsv)
    with open(os.path.join(args.a2, "results", "reviews_unsanitized", "dev_predictions.tsv"), encoding="utf-8") as f:
        next(f)
        rows = [line.rstrip("\n").split("\t") for line in f]
    with open(os.path.join(orig_dir, "dev.tsv"), "w", encoding="utf-8") as f:
        f.write("sentence\tlabel\n")
        for r in rows:
            f.write(f"{r[0]}\t{r[2]}\n")
    shutil.copy2(os.path.join(args.a2, "SanText", "output_SanText_glove", "app_reviews", "eps_3.00", "dev.tsv"),
                 os.path.join(san_dir, "dev.tsv"))

    cfg = {"labels": args.labels,
           "annotation_sheet": os.path.join(args.a2, "new_data", "app_reviews", "annotation", "annotation_sheet.csv"),
           "original_dev": {"reviews": os.path.join(orig_dir, "dev.tsv")},
           "san_root": os.path.join(args.out, "san")}
    with open(os.path.join(args.out, "judge_inputs.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    # 8 meaning rows and 12 of the 13 residual rows (only SanText at eps = 3 exists for the reviews)
    subprocess.run([sys.executable, os.path.join(HERE, "build_judge_items.py"), "--config",
                    os.path.join(args.out, "judge_inputs.json"), "--out_dir", judge_dir, "--human_residual", "24",
                    "--seed", str(args.seed)], check=True)
    with open(os.path.join(judge_dir, "items_report.json"), encoding="utf-8") as f:
        print(json.dumps(json.load(f), indent=1))

    # one blind sheet: notebook C's rows and the reviews' rows, shuffled together
    old_rows, old_key = read_sheet(os.path.join(args.c_judge, "human_check.xlsx")), read_key(
        os.path.join(args.c_judge, "human_check_key.csv"))
    new_rows, new_key = read_sheet(os.path.join(judge_dir, "human_check.xlsx")), read_key(
        os.path.join(judge_dir, "human_check_key.csv"))
    pool = [(r, old_key[r["check_id"]]) for r in old_rows] + [(r, new_key[r["check_id"]]) for r in new_rows]
    random.Random(f"{args.seed}-merge").shuffle(pool)
    rows, key = [], []
    for n, (r, item_id) in enumerate(pool, start=1):
        rows.append({**r, "check_id": n})
        key.append({"check_id": n, "item_id": item_id})
    write_human_sheet(rows, os.path.join(judge_dir, "human_check_all.xlsx"))
    with open(os.path.join(judge_dir, "human_check_all_key.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["check_id", "item_id"])
        w.writeheader()
        w.writerows(key)
    print(f"human_check_all.xlsx: {len(rows)} rows ({len(old_rows)} from notebook C, {len(new_rows)} reviews)")


if __name__ == "__main__":
    main()
