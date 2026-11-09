#!/usr/bin/env python3
"""
survival_gold.py: do the words annotators flagged as sensitive survive sanitization?

Uses notebook A/B's analysis/aligned/reviews__<method>__eps<e>__seed42.tsv (original and sanitized token at
every position of the 542 test reviews) and analysis/vocab/reviews.tsv (whether each word is in SanText+'s
sensitive set and has a GloVe vector). The 13 gold-disclosing reviews are located by a distinctive phrase of
their text; their flagged words come from annotation_labels.csv.
Usage: python survival_gold.py <unzipped A/B outputs> <annotation_labels.csv>
"""
import collections
import csv
import glob
import os
import re
import sys

TOK = re.compile(r"\w+|[^\w\s]", re.UNICODE)
STOP = set("the a an and or of to in on for with my me i you your it is was be are this that at by from as not "
           "no but so our we they their its has have had can just all any get got".split())
LOCATE = {1: "antidepressants", 3: "onto payid transactions", 11: "Shellhabour", 13: "accu chek device",
          14: "Ombudsman", 21: "Diagnosed type 2", 25: "Westpac is still secure", 29: "connect the meter to my phone",
          35: "19% interest", 40: "locked my account on Friday night", 80: "open the mortgage manager",
          94: "provider I've seen virtually", 99: "total knee replacement"}


def content(tok):
    return (tok.isdigit() or (tok.isalpha() and len(tok) >= 3)) and tok.lower() not in STOP


def main():
    root, labels_path = sys.argv[1], sys.argv[2]
    labels = {int(r["ann_id"]): r for r in csv.DictReader(open(labels_path, encoding="utf-8"))}
    vocab = {}
    vp = os.path.join(root, "analysis", "vocab", "reviews.tsv")
    if os.path.exists(vp):
        for r in csv.DictReader(open(vp, encoding="utf-8"), delimiter="\t"):
            vocab[r["token"]] = "no vector" if r["has_glove_vector"] == "0" else (
                "rare 90%" if r["sensitive_w0.9"] == "1" else "frequent 10%")
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "analysis", "aligned", "reviews__*__seed42.tsv"))):
        name = os.path.basename(path)[:-4]
        _, method, eps, _ = name.split("__")
        docs = collections.defaultdict(list)
        for r in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL):
            docs[int(r["doc"])].append((r["orig"], r["san"]))
        text = {d: " ".join(o for o, _ in toks) for d, toks in docs.items()}
        allc = [(o, s) for toks in docs.values() for o, s in toks if content(o)]
        flagged, by_cat, by_class, missing = [], collections.Counter(), collections.Counter(), []
        kept_cat, kept_class = collections.Counter(), collections.Counter()
        for i, phrase in LOCATE.items():
            key = " ".join(TOK.findall(phrase))
            hits = [d for d, t in text.items() if key in t]
            if len(hits) != 1:
                missing.append((i, len(hits)))
                continue
            toks = docs[hits[0]]
            used = set()
            words = [w.strip() for w in labels[i]["gold_words"].split(";") if w.strip()]
            cats = [c.strip() for c in labels[i]["gold_categories"].split(";") if c.strip()]
            for k, w in enumerate(words):
                cat = cats[k] if k < len(cats) else cats[0]
                for t in TOK.findall(w):
                    if not content(t):
                        continue
                    pos = next((p for p, (o, _) in enumerate(toks) if p not in used and o.lower() == t.lower()), None)
                    if pos is None:
                        continue
                    used.add(pos)
                    o, s = toks[pos]
                    keep = o == s
                    cls = vocab.get(o, "?")
                    flagged.append(keep)
                    by_cat[cat] += 1
                    kept_cat[cat] += keep
                    by_class[cls] += 1
                    kept_class[cls] += keep
        rows.append({"method": method.replace("_plus", "+"), "eps": float(eps[3:]),
                     "flagged": f"{sum(flagged)}/{len(flagged)} ({sum(flagged) / max(1, len(flagged)):.0%})",
                     "all content": f"{sum(o == s for o, s in allc) / max(1, len(allc)):.0%}",
                     "by category": ", ".join(f"{c} {kept_cat[c]}/{n}" for c, n in sorted(by_cat.items())),
                     "by word class": ", ".join(f"{c} {kept_class[c]}/{n}" for c, n in sorted(by_class.items())),
                     "not located": missing})
    for r in sorted(rows, key=lambda r: (r["method"], r["eps"])):
        print(r)


if __name__ == "__main__":
    main()
