#!/usr/bin/env python3
"""
filter_glove.py: write the part of GloVe that run_SanText.py can use for the given datasets.

run_SanText.py (Yue et al., 2021) reads all ~2.2 million lines of glove.840B.300d.txt on every
run. It keeps a line only if the line's word, after NFD normalisation, is in the dataset's
vocabulary, and the first matching line wins. This script copies all those candidate lines,
unchanged and in their original order, to a small file. Given the small file, run_SanText.py
therefore selects the same words, the same vectors and the same word order as from the full
file, so its output is unchanged; each run just loads in seconds instead of minutes.

It also keeps lines whose word matches a vocabulary word only after both are NFD-normalised
(e.g. a precomposed "café" in the data). run_SanText.py never selects these, because it compares
the normalised GloVe word with the raw token, so they do not change its output; they let the
analysis count the words that miss their vector for this reason.

The vocabulary is built with the authors' own get_vocab_SST2 and spaCy's blank English
tokenizer, exactly as run_SanText.py builds it for --task SST-2 (the single-sentence format
that SST-2, Dreaddit and the app-review dataset all use).

Usage
  python filter_glove.py --glove data/glove.840B.300d.txt --out data/glove.filtered.txt \
      --santext_dir SanText DATA_DIR [DATA_DIR ...]
"""
import argparse
import json
import os
import sys
import time
import unicodedata


def main():
    ap = argparse.ArgumentParser(description="Keep only the GloVe lines run_SanText.py can use for these datasets.")
    ap.add_argument("--glove", required=True, help="full GloVe text file")
    ap.add_argument("--out", required=True, help="filtered GloVe file to write")
    ap.add_argument("--santext_dir", required=True, help="the authors' repository (provides utils.get_vocab_SST2)")
    ap.add_argument("data_dirs", nargs="+", help="folders holding train.tsv and dev.tsv in SST-2 format")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.santext_dir))
    from utils import get_vocab_SST2  # the authors' function, unchanged
    from spacy.lang.en import English

    tokenizer = English()
    vocab, sizes = set(), {}
    for d in args.data_dirs:
        v = get_vocab_SST2(d, tokenizer, tokenizer_type="word")
        sizes[d] = len(v)
        vocab.update(v)
        print(f"vocabulary of {d}: {len(v)} word types", flush=True)
    keep = vocab | {unicodedata.normalize("NFD", t) for t in vocab}

    t0 = time.time()
    n_lines = n_kept = 0
    tmp = args.out + ".part"
    # Same reading as run_SanText.py: default text encoding, words split on single spaces.
    with open(args.glove) as f, open(tmp, "w") as out:
        for row in f:
            n_lines += 1
            word = row.rstrip().split(" ")[0]
            if unicodedata.normalize("NFD", word) in keep:
                out.write(row)
                n_kept += 1
    os.replace(tmp, args.out)

    info = {"glove": os.path.abspath(args.glove), "lines_read": n_lines, "lines_kept": n_kept,
            "union_vocabulary": len(vocab), "vocabulary_per_dataset": sizes,
            "seconds": round(time.time() - t0, 1)}
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"kept {n_kept:,} of {n_lines:,} GloVe lines for {len(vocab):,} word types -> {args.out} "
          f"({info['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
