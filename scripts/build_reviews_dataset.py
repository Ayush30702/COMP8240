#!/usr/bin/env python3
"""
build_reviews_dataset.py: builds the project's newly constructed dataset,
"AU health & banking app reviews", for the SanText replication (COMP8240).

Why this data: SanText was motivated by health and finance text. App-store
reviews of health and banking apps often disclose exactly that kind of
information (conditions, medications, banks, money problems, family), and each
review's star rating gives a sentiment label for free, as in IMDb (Maas et al., 2011).

Stages (run in order, or all at once with `all`):
  collect     Newest Google Play reviews for each app in APPS (country=au,
              lang=en) via the google-play-scraper package. Reviewer names,
              avatars and developer replies are never stored.
  preprocess  clean -> drop duplicates -> keep >= 5 words -> keep English ->
              star label (1-2 stars = 0 negative, 4-5 = 1 positive, 3 dropped) ->
              stratified 80/20 train/dev split -> SST-2-format TSVs that
              run_SanText.py --task SST-2 and finetune_sanitized.py read unchanged.
              Writes stats.json with the stage-by-stage funnel.
  sample      Stratified sample of reviews for manual annotation (CSV to fill in).

Usage (as run on 9 October 2026)
  pip install google-play-scraper          # only needed for `collect`
  python build_reviews_dataset.py collect --per_app 500
  python build_reviews_dataset.py preprocess
  python build_reviews_dataset.py sample --n 100

Outputs (under --out_dir, default new_data/app_reviews/)
  raw_reviews.jsonl, collection_log.json     collect
  reviews_clean.jsonl, train.tsv, dev.tsv, stats.json   preprocess
  annotation/annotation_sheet.csv            sample
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

# (Google Play package id, category, display name)
APPS = [
    ("com.mysugr.android.companion", "health", "mySugr (diabetes)"),
    ("com.medisafe.android.client", "health", "Medisafe (medications)"),
    ("eu.smartpatient.mytherapy", "health", "MyTherapy (medications)"),
    ("com.getsomeheadspace.android", "health", "Headspace (mental health)"),
    ("com.commbank.netbank", "finance", "CommBank"),
    ("org.westpac.bank", "finance", "Westpac"),
    ("au.com.nab.mobile", "finance", "NAB"),
    ("com.anz.android.gomoney", "finance", "ANZ"),
]

# Common English function words; a review needs at least two to count as English.
EN_HINTS = set(
    "the and to is it i this my of for a in not you with on have but was be can "
    "so me that when are just very all no an at or its it's app".split()
)
WS = re.compile(r"\s+")  # also matches tabs and newlines, which would break the TSV format
WORD = re.compile(r"[a-z']+")


# ----------------------------------------------------------------------------- collect
def collect(args):
    try:
        from google_play_scraper import Sort, reviews
    except ImportError:
        raise SystemExit("google-play-scraper is not installed: run  pip install google-play-scraper")
    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, "raw_reviews.jsonl")
    log = {"collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "country": args.country, "lang": args.lang, "per_app_target": args.per_app, "apps": []}
    seen = set()
    with open(raw_path, "w", encoding="utf-8") as out:
        for app_id, category, name in APPS:
            got, token, error = 0, None, None
            try:
                while got < args.per_app:
                    batch, token = reviews(app_id, lang=args.lang, country=args.country, sort=Sort.NEWEST,
                                           count=min(200, args.per_app - got), continuation_token=token)
                    batch = [r for r in batch if r.get("reviewId") not in seen][: args.per_app - got]
                    if not batch:
                        break
                    for r in batch:
                        seen.add(r.get("reviewId"))
                        at = r.get("at")
                        record = {
                            "review_id": r.get("reviewId"),
                            "app_id": app_id,
                            "app_name": name,
                            "category": category,
                            "stars": r.get("score"),
                            "text": r.get("content") or "",
                            "date": at.isoformat() if hasattr(at, "isoformat") else at,
                            "app_version": r.get("appVersion") or r.get("reviewCreatedVersion"),
                            "thumbs_up": r.get("thumbsUpCount"),
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    got += len(batch)
                    if getattr(token, "token", None) is None:
                        break
                    time.sleep(args.sleep)
            except Exception as e:  # unknown package id, network problem, layout change
                error = f"{type(e).__name__}: {e}"
            log["apps"].append({"app_id": app_id, "app_name": name, "category": category,
                                "collected": got, "error": error})
            print(f"  {name:<30} {got:>5} reviews" + (f"   ! {error}" if error else ""))
    with open(os.path.join(args.out_dir, "collection_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"collected {sum(a['collected'] for a in log['apps'])} reviews -> {raw_path}")


# ----------------------------------------------------------------------------- preprocess
def clean_text(text):
    return WS.sub(" ", (text or "").replace("​", " ")).strip()


def looks_english(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ascii_share = sum(c.isascii() for c in letters) / len(letters)
    hits = sum(w in EN_HINTS for w in WORD.findall(text.lower()))
    return ascii_share >= 0.9 and hits >= 2


def star_label(stars):
    if stars in (1, 2):
        return 0
    if stars in (4, 5):
        return 1
    return None  # 3 stars (neutral) or missing


def preprocess(args):
    raw_path = os.path.join(args.out_dir, "raw_reviews.jsonl")
    with open(raw_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    funnel = []

    def record(stage, kept):
        per_cat = Counter(r["category"] for r in kept)
        funnel.append({"stage": stage, "total": len(kept), **{c: per_cat.get(c, 0) for c in ("health", "finance")}})
        return kept

    rows = record("collected", rows)
    for r in rows:
        r["text"] = clean_text(r.get("text"))
    rows = record("non-empty text", [r for r in rows if r["text"]])
    seen, unique = set(), []
    for r in rows:
        key = r["text"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    rows = record("duplicates removed", unique)
    rows = record(f">= {args.min_words} words", [r for r in rows if len(r["text"].split()) >= args.min_words])
    rows = record("English only", [r for r in rows if looks_english(r["text"])])
    for r in rows:
        r["label"] = star_label(r.get("stars"))
    rows = record("3-star/unrated dropped", [r for r in rows if r["label"] is not None])

    # stratified split by (category, label) so both splits keep the same mix
    rng = random.Random(args.seed)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["category"], r["label"])].append(r)
    for key in sorted(groups):
        members = groups[key]
        rng.shuffle(members)
        n_dev = round(len(members) * args.dev_share)
        for i, r in enumerate(members):
            r["split"] = "dev" if i < n_dev else "train"

    for split in ("train", "dev"):
        with open(os.path.join(args.out_dir, f"{split}.tsv"), "w", encoding="utf-8") as f:
            f.write("sentence\tlabel\n")
            for r in rows:
                if r["split"] == split:
                    f.write(f"{r['text']}\t{r['label']}\n")
    with open(os.path.join(args.out_dir, "reviews_clean.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lengths = [len(r["text"].split()) for r in rows]
    vocab = Counter(w for r in rows for w in WORD.findall(r["text"].lower()))
    stats = {
        "funnel": funnel,
        "final": {
            "reviews": len(rows),
            "train": sum(r["split"] == "train" for r in rows),
            "dev": sum(r["split"] == "dev" for r in rows),
            "share_positive": round(sum(r["label"] for r in rows) / max(1, len(rows)), 3),
            "by_category_and_label": {f"{c}_{'pos' if l else 'neg'}": len(g) for (c, l), g in sorted(groups.items())},
            "by_app": dict(Counter(r["app_name"] for r in rows)),
            "words_per_review": {
                "mean": round(statistics.mean(lengths), 1) if lengths else 0,
                "median": statistics.median(lengths) if lengths else 0,
                "max": max(lengths, default=0),
            },
            "word_types": len(vocab),
            "santext_peak_memory_gb_upper_bound": round(5 * len(vocab) ** 2 * 4 / 1e9, 2),
        },
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"{'stage':<26}{'total':>7}{'health':>8}{'finance':>9}")
    for s in funnel:
        print(f"{s['stage']:<26}{s['total']:>7}{s['health']:>8}{s['finance']:>9}")


# ----------------------------------------------------------------------------- sample
SHEET_COLUMNS = ["ann_id", "review_key", "category", "app_name", "stars", "star_label", "text",
                 "text_sentiment", "discloses_personal_info", "sensitive_words", "sensitive_categories", "notes"]


def sample(args):
    with open(os.path.join(args.out_dir, "reviews_clean.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["category"], r["label"])].append(r)
    keys = sorted(groups)
    rng = random.Random(args.seed)
    for k in keys:
        rng.shuffle(groups[k])
    # equal share per (category, label) cell; leftover quota goes to cells that still have rows
    quota = {k: 0 for k in keys}
    remaining = args.n
    while remaining > 0 and any(quota[k] < len(groups[k]) for k in keys):
        for k in keys:
            if remaining and quota[k] < len(groups[k]):
                quota[k] += 1
                remaining -= 1
    picked = [r for k in keys for r in groups[k][: quota[k]]]
    rng.shuffle(picked)  # mix categories so the annotator can't infer the stars from position
    os.makedirs(os.path.join(args.out_dir, "annotation"), exist_ok=True)
    path = os.path.join(args.out_dir, "annotation", "annotation_sheet.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        for i, r in enumerate(picked, start=1):
            key = hashlib.sha1((r.get("review_id") or r["text"]).encode("utf-8")).hexdigest()[:12]
            w.writerow({"ann_id": i, "review_key": key, "category": r["category"], "app_name": r["app_name"],
                        "stars": r["stars"], "star_label": "positive" if r["label"] else "negative", "text": r["text"]})
    print(f"annotation sheet with {len(picked)} reviews -> {path}")


def main():
    p = argparse.ArgumentParser(description="Build the AU health & banking app-review dataset.")
    p.add_argument("stage", choices=["collect", "preprocess", "sample", "all"])
    p.add_argument("--out_dir", default="new_data/app_reviews")
    p.add_argument("--per_app", type=int, default=500, help="reviews to collect per app")
    p.add_argument("--country", default="au")
    p.add_argument("--lang", default="en")
    p.add_argument("--sleep", type=float, default=1.0, help="pause between requests, in seconds")
    p.add_argument("--min_words", type=int, default=5)
    p.add_argument("--dev_share", type=float, default=0.2)
    p.add_argument("--n", type=int, default=100, help="reviews in the annotation sample")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.stage in ("collect", "all"):
        collect(args)
    if args.stage in ("preprocess", "all"):
        preprocess(args)
    if args.stage in ("sample", "all"):
        sample(args)


if __name__ == "__main__":
    main()
