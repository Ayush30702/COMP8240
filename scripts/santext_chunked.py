#!/usr/bin/env python3
"""
santext_chunked.py: SanText (Yue et al., 2021) computed a block of probability rows at a time.

The authors' run_SanText.py builds the whole |V| x |V| probability matrix
    P[w, w'] = softmax over w' of ( -eps/2 * ||phi(w) - phi(w')|| )
in memory and holds about five copies of it at its peak: about 5.3 GB for SST-2 with GloVe
(14,283 words with vectors), about 19 GB with BERT's 30,522-token vocabulary and about 160 GB
for QNLI with GloVe (88,159 words). Colab has 12.7 GB, so the paper's BERT-embedding experiments
and QNLI cannot run as released. This script samples from exactly the same rows, computed with
the authors' own cal_probability function, but only for words that occur in the data and only a
block of rows at a time.

Differences from run_SanText.py, all deliberate:
  * One seeded NumPy generator in a single process draws every replacement, so --seed fixes the
    output. (In the released code, forked worker processes reuse copies of one random state.)
  * SanText only (every word is sensitive), which is all the paper's privacy experiments use.
Everything else follows run_SanText.py: the vocabulary (the authors' get_vocab_SST2 /
get_vocab_QNLI for GloVe; every BERT token for BERT), the tokenizers (spaCy's blank English for
GloVe, bert-base-uncased WordPiece for BERT), the order of the embedding rows, the uniform
replacement of words that have no GloVe vector, and the output format (tokens joined by single
spaces, then the label; QNLI as index, question, sentence, label).

Usage
  python santext_chunked.py --task SST-2 --embedding bert --epsilon 14 --data_dir data/SST-2 \
      --output_dir san_bert/SST-2 --santext_dir SanText
  python santext_chunked.py --task SST-2 --embedding glove --glove data/glove.filtered.txt \
      --epsilon 3 --data_dir data/SST-2 --output_dir san_glove/SST-2 --santext_dir SanText --check_rows 2000
Output: <output_dir>/eps_<eps>/{train.tsv, dev.tsv, meta.json}
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="SanText with the authors' probabilities, computed in blocks of rows.")
    p.add_argument("--task", choices=["SST-2", "QNLI"], default="SST-2",
                   help="input format: SST-2 (sentence, label) or QNLI (index, question, sentence, label)")
    p.add_argument("--embedding", choices=["glove", "bert"], required=True)
    p.add_argument("--glove", default=None, help="GloVe text file (full or filtered); needed for --embedding glove")
    p.add_argument("--bert_model", default="bert-base-uncased")
    p.add_argument("--epsilon", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", required=True, help="folder with train.tsv and dev.tsv")
    p.add_argument("--output_dir", required=True, help="results go to <output_dir>/eps_<epsilon>/")
    p.add_argument("--santext_dir", required=True, help="the authors' repository (utils.py, SanText.py)")
    p.add_argument("--block", type=int, default=1024, help="probability rows computed at once")
    p.add_argument("--splits", default="train,dev", help="which files to sanitize")
    p.add_argument("--check_rows", type=int, default=0,
                   help="compare this many rows with the authors' full matrix (GloVe vocabularies up to 20,000 words)")
    return p.parse_args()


# ----------------------------------------------------------------------------- vocabulary and embeddings
def load_glove(data_dir, task, glove_path, santext_dir):
    """Exactly run_SanText.py's construction for --embedding_type glove (SanText: all words sensitive)."""
    from utils import get_vocab_SST2, get_vocab_QNLI, word_normalize  # the authors' functions
    from spacy.lang.en import English
    tokenizer = English()
    get_vocab = {"SST-2": get_vocab_SST2, "QNLI": get_vocab_QNLI}[task]
    vocab = get_vocab(data_dir, tokenizer, tokenizer_type="word")
    all_words = [k for k, _ in vocab.most_common()]
    word2id, embs = {}, []
    with open(glove_path) as f:
        line = f.readline().rstrip().split(" ")
        if len(line) != 2:
            f.seek(0)
        for row in f:
            content = row.rstrip().split(" ")
            cur_word = word_normalize(content[0])
            if cur_word in vocab and cur_word not in word2id:
                word2id[cur_word] = len(embs)
                embs.append([float(x) for x in content[1:]])
    emb = np.array(embs, dtype="f")
    id2word = [None] * len(word2id)
    for w, i in word2id.items():
        id2word[i] = w

    def tokenize(text):
        return [t.text for t in tokenizer(text)]

    return tokenize, all_words, word2id, id2word, emb, len(vocab)


def load_bert_embeddings(model_name):
    """BERT's input embedding matrix (one row per token id) and its tokenizer."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    emb = model.get_input_embeddings().weight.detach().cpu().numpy().astype("f")
    del model
    return tokenizer, emb


def load_bert(model_name):
    """run_SanText.py with --embedding_type bert: every token of BERT's vocabulary is in V (and sensitive),
    rows in token-id order, docs tokenized into word pieces."""
    tokenizer, emb_matrix = load_bert_embeddings(model_name)
    items = sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])
    word2id = {tok: k for k, (tok, _) in enumerate(items)}
    id2word = [tok for tok, _ in items]
    emb = emb_matrix[[i for _, i in items]].astype("f")
    return tokenizer.tokenize, id2word, word2id, id2word, emb, len(id2word)


# ----------------------------------------------------------------------------- data
def read_split(path, task):
    """As run_SanText.py reads it: header kept; SST-2 -> (text, label); QNLI -> (q, s, label) per row."""
    with open(path, encoding="utf-8") as f:
        header = next(f)
        rows = []
        for line in f:
            content = line.strip().split("\t")
            if task == "SST-2":
                rows.append((content[0], int(content[1])))
            else:
                rows.append((content[1], content[2], content[-1]))
    return header, rows


def write_split(path, header, rows, san_docs, task):
    with open(path, "w", encoding="utf-8") as out:
        out.write(header)
        if task == "SST-2":
            for (_, label), doc in zip(rows, san_docs):
                out.write(" ".join(doc) + "\t" + str(label) + "\n")
        else:
            for i, (_, _, label) in enumerate(rows):
                out.write(str(i) + "\t" + " ".join(san_docs[2 * i]) + "\t" + " ".join(san_docs[2 * i + 1])
                          + "\t" + str(label) + "\n")


# ----------------------------------------------------------------------------- sampling
def sanitize_docs(docs, word2id, id2word, all_words, emb, eps, rng, block, cal_probability):
    """Replace every token: in-vocabulary tokens from their row of the authors' probability matrix,
    tokens without a vector uniformly from the whole vocabulary (as run_SanText.py does)."""
    lengths = [len(d) for d in docs]
    flat = [t for d in docs for t in d]
    rows = np.fromiter((word2id.get(t, -1) for t in flat), dtype=np.int64, count=len(flat))
    out = np.empty(len(flat), dtype=object)
    oov = np.flatnonzero(rows < 0)
    if len(oov):
        out[oov] = np.asarray(all_words, dtype=object)[rng.integers(0, len(all_words), size=len(oov))]
    inv = np.flatnonzero(rows >= 0)
    order = inv[np.argsort(rows[inv], kind="stable")]
    uniq, starts, counts = np.unique(rows[order], return_index=True, return_counts=True)
    id2word_arr = np.asarray(id2word, dtype=object)
    for b in range(0, len(uniq), block):
        ids = uniq[b:b + block]
        prob = cal_probability(emb[ids], emb, eps)  # the authors' function, on a block of rows
        for k in range(len(ids)):
            s, c = starts[b + k], counts[b + k]
            cdf = np.cumsum(prob[k], dtype=np.float64)
            idx = np.searchsorted(cdf, rng.random(c) * cdf[-1], side="right")
            out[order[s:s + c]] = id2word_arr[np.minimum(idx, len(cdf) - 1)]
        del prob
    san, pos = [], 0
    for n in lengths:
        san.append(list(out[pos:pos + n]))
        pos += n
    unchanged = float(np.mean([a == b for a, b in zip(flat, out)])) if flat else None
    return san, {"tokens": len(flat), "without_vector": int(len(oov)), "rows_computed": int(len(uniq)),
                 "share_unchanged": unchanged}


def check_rows(emb, eps, n, block, rng, cal_probability):
    """Rows computed in blocks vs the authors' full matrix (small vocabularies), or block-size invariance."""
    sample = np.sort(rng.choice(len(emb), size=min(n, len(emb)), replace=False))
    if len(emb) <= 20000:
        reference = cal_probability(emb, emb, eps)[sample]
        how = "full |V| x |V| matrix, as run_SanText.py computes it"
    else:
        reference = np.vstack([cal_probability(emb[sample[i:i + 97]], emb, eps) for i in range(0, len(sample), 97)])
        how = "blocks of 97 rows (block-size invariance; the full matrix does not fit in memory)"
    mine = np.vstack([cal_probability(emb[sample[i:i + block]], emb, eps) for i in range(0, len(sample), block)])
    diff = np.abs(reference.astype(np.float64) - mine.astype(np.float64))
    return {"rows_checked": int(len(sample)), "reference": how, "max_abs_diff": float(diff.max()),
            "rows_identical": int(np.sum(np.all(reference == mine, axis=1))),
            "argmax_identical": int(np.sum(reference.argmax(1) == mine.argmax(1)))}


def main():
    args = parse_args()
    sys.path.insert(0, os.path.abspath(args.santext_dir))
    from SanText import cal_probability  # the authors' distance + softmax, unchanged

    t0 = time.time()
    if args.embedding == "glove":
        if not args.glove:
            raise SystemExit("--glove is required with --embedding glove")
        tokenize, all_words, word2id, id2word, emb, vocab_size = load_glove(args.data_dir, args.task, args.glove,
                                                                            args.santext_dir)
    else:
        tokenize, all_words, word2id, id2word, emb, vocab_size = load_bert(args.bert_model)
    print(f"vocabulary {vocab_size:,} | rows with vectors {len(emb):,} x {emb.shape[1]} | "
          f"one full matrix would be {len(emb) ** 2 * 4 / 1e9:.2f} GB", flush=True)

    rng = np.random.default_rng(args.seed)
    out_dir = os.path.join(args.output_dir, "eps_%.2f" % args.epsilon)
    os.makedirs(out_dir, exist_ok=True)
    meta = {"task": args.task, "embedding": args.embedding, "epsilon": args.epsilon, "seed": args.seed,
            "block": args.block, "vocabulary": vocab_size, "rows_with_vectors": int(len(emb)),
            "full_matrix_gb": round(len(emb) ** 2 * 4 / 1e9, 2), "splits": {}}
    for split in args.splits.split(","):
        header, rows = read_split(os.path.join(args.data_dir, f"{split}.tsv"), args.task)
        texts = [r[0] for r in rows] if args.task == "SST-2" else [t for r in rows for t in (r[0], r[1])]
        docs = [tokenize(t) for t in texts]
        san, info = sanitize_docs(docs, word2id, id2word, all_words, emb, args.epsilon, rng, args.block, cal_probability)
        write_split(os.path.join(out_dir, f"{split}.tsv"), header, rows, san, args.task)
        meta["splits"][split] = {"rows": len(rows), **info}
        print(f"{split}: {len(rows):,} rows | {info['tokens']:,} tokens | unchanged {info['share_unchanged']:.4f} | "
              f"without vector {info['without_vector']:,}", flush=True)
    if args.check_rows:
        meta["check"] = check_rows(emb, args.epsilon, args.check_rows, args.block, np.random.default_rng(0), cal_probability)
        print("row check:", json.dumps(meta["check"]), flush=True)
    meta["seconds"] = round(time.time() - t0, 1)
    meta["peak_memory_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9, 2)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"done in {meta['seconds']} s, peak memory {meta['peak_memory_gb']} GB -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
