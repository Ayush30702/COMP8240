#!/usr/bin/env python3
"""
analyse_runs.py: summarise the final runs for the report.

Reads plan.json (written by the notebook) and writes, under --out:
  runs.csv           one row per fine-tuning run: dataset, method, epsilon, seed, accuracy, macro-F1,
                     F1 of label 1, plus the 9 Oct runs if their metrics were copied in
  survival.csv       one row per sanitized output (dev split): share of tokens left unchanged, overall
                     and by word class (content words, words without a GloVe vector, the rarest 90%
                     that SanText+ treats as sensitive, the frequent 10% it treats as non-sensitive)
  resources.csv      vocabulary, matrix size, wall time and peak memory of every sanitization
  aligned/<id>.tsv   review test split only: original and sanitized token at every position, for
                     the analysis against the human-flagged sensitive words
  vocab/<name>.tsv   the authors' vocabulary for each dataset: frequency rank, count, whether GloVe
                     has a vector, whether SanText+ (w = 0.9) treats it as sensitive, and whether the
                     authors' NFD comparison misses a vector GloVe does have
  repro.json         reproducibility checks (same seed with 1 vs 2 worker processes; one review
                     repeated 1,000 times; 9 Oct output vs today's with the same seed)
  summary.md         all of the above as readable tables, with the paper's numbers alongside

Word classes and alignment follow run_SanText.py: texts are tokenized with spaCy's blank English
tokenizer (the authors' tokenizer), the sanitized text is the replacement tokens joined by single
spaces, so position i of the sanitized text replaces token i of the original.
"""
import argparse
import collections
import csv
import json
import os
import re
import statistics
import sys
import unicodedata

STOP = set("the a an and or of to in on for with my me i you your it is was be are this that at by from as not "
           "no but so our we they their its has have had can just all any get got".split())

# Yue et al. (2021), Table 1, GloVe embeddings, SST-2 dev accuracy
PAPER_SST2 = {("unsanitized", None): 0.9251,
              ("SanText", 1.0): 0.5101, ("SanText", 2.0): 0.5838, ("SanText", 3.0): 0.8374,
              ("SanText_plus", 1.0): 0.7796, ("SanText_plus", 2.0): 0.7943, ("SanText_plus", 3.0): 0.8516}
METHOD_NAME = {"unsanitized": "Unsanitized", "SanText": "SanText", "SanText_plus": "SanText+"}


def read_original_texts(path):
    """As run_SanText.py reads them: strip the line, take the first tab-separated field."""
    with open(path, encoding="utf-8") as f:
        next(f)
        return [line.strip().split("\t")[0] for line in f]


def read_sanitized_texts(path):
    """run_SanText.py writes '<tokens joined by spaces>\\t<label>' after a header row."""
    with open(path, encoding="utf-8") as f:
        next(f)
        return [line.rstrip("\n").rsplit("\t", 1)[0] for line in f]


def load_glove_words(path):
    words = set()
    with open(path) as f:
        for row in f:
            words.add(unicodedata.normalize("NFD", row.rstrip().split(" ")[0]))
    return words


def authors_vocab(data_dir, tokenizer, w=0.9):
    from utils import get_vocab_SST2  # the authors' function
    vocab = get_vocab_SST2(data_dir, tokenizer, tokenizer_type="word")
    words = [k for k, _ in vocab.most_common()]
    n_sensitive = int(w * len(vocab))
    sensitive = set(words[-n_sensitive - 1:])  # the authors' slice, including its extra word
    return vocab, words, sensitive


def is_content(tok):
    return tok.isalpha() and len(tok) >= 3 and tok.lower() not in STOP


def mean_sd(values):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], None
    return statistics.mean(values), statistics.stdev(values)


def fmt(x, nd=4):
    return "–" if x is None else f"{x:.{nd}f}"


def compare_files(a_dir, b_dir):
    """Share of identical lines (and of identical tokens where lengths match) between two outputs."""
    out = {}
    for split in ("train", "dev"):
        pa, pb = os.path.join(a_dir, f"{split}.tsv"), os.path.join(b_dir, f"{split}.tsv")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        a, b = read_sanitized_texts(pa), read_sanitized_texts(pb)
        n = min(len(a), len(b))
        same_lines = sum(x == y for x, y in zip(a[:n], b[:n]))
        tok_total = tok_same = 0
        for x, y in zip(a[:n], b[:n]):
            xs, ys = x.split(" "), y.split(" ")
            if len(xs) == len(ys):
                tok_total += len(xs)
                tok_same += sum(p == q for p, q in zip(xs, ys))
        out[split] = {"lines": n, "identical_lines": same_lines, "share_identical_lines": round(same_lines / max(1, n), 4),
                      "share_identical_tokens": round(tok_same / max(1, tok_total), 4)}
    return out


def main():
    ap = argparse.ArgumentParser(description="Summarise the final runs.")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.plan, encoding="utf-8") as f:
        plan = json.load(f)
    cfg = plan["config"]
    sys.path.insert(0, cfg["santext_dir"])
    from spacy.lang.en import English
    tokenizer = English()
    os.makedirs(os.path.join(args.out, "aligned"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "vocab"), exist_ok=True)
    md = ["# Final runs: summary", ""]

    env_path = os.path.join(args.out, "environment.json")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            env = json.load(f)
        md += ["Environment: " + ", ".join(f"{k} {v}" for k, v in env.items()), ""]

    # ---------------------------------------------------------------- vocabularies
    glove = load_glove_words(cfg["glove_filtered"]) if os.path.exists(cfg["glove_filtered"]) else None
    vinfo, vsummary = {}, []
    for name, data_dir in plan["datasets"].items():
        if not os.path.exists(os.path.join(data_dir, "dev.tsv")):
            continue
        vocab, words, sensitive = authors_vocab(data_dir, tokenizer)
        has_vec = {t: (glove is not None and t in glove) for t in words}
        nfd_miss = {t: (glove is not None and t != unicodedata.normalize("NFD", t)
                        and unicodedata.normalize("NFD", t) in glove) for t in words}
        vinfo[name] = {"vocab": vocab, "sensitive": sensitive, "has_vec": has_vec}
        with open(os.path.join(args.out, "vocab", f"{name}.tsv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["rank", "token", "count", "has_glove_vector", "sensitive_w0.9", "nfd_mismatch"])
            for r, t in enumerate(words, start=1):
                w.writerow([r, t, vocab[t], int(has_vec[t]), int(t in sensitive), int(nfd_miss[t])])
        vsummary.append((name, len(words), sum(has_vec.values()), len(words) - sum(has_vec.values()),
                         sum(nfd_miss.values()), len(sensitive), sum(1 for t in sensitive if has_vec[t])))
    if vsummary:
        md += ["## Vocabularies (the authors' construction: train + dev, spaCy blank English tokenizer)", "",
               "| Dataset | Word types | With GloVe vector | Without (replaced uniformly at random) | of which NFD mismatch | Sensitive, w = 0.9 | Sensitive with vector |",
               "|---|---|---|---|---|---|---|"]
        md += [f"| {n} | {a:,} | {b:,} | {c:,} | {d:,} | {e:,} | {g:,} |" for n, a, b, c, d, e, g in vsummary]
        md.append("")

    # ---------------------------------------------------------------- sanitizations: resources + survival
    jobs = {j["id"]: j for j in plan["sanitize"]}
    res_rows, surv_rows = [], []
    for jid, job in jobs.items():
        out = job["out_path"]
        if not os.path.exists(os.path.join(out, ".done")):
            continue
        log_path = os.path.join(cfg["logs"], f"san__{jid}.log")
        log = open(log_path, errors="ignore").read() if os.path.exists(log_path) else ""
        tw = re.search(r"#Total Words: (\d+), #Sensitive Words: (\d+)", log)
        mat = re.search(r"All Word Embedding Matrix: \((\d+), (\d+)\)", log)
        smat = re.search(r"Sensitive Word Embedding Matrix: \((\d+), (\d+)\)", log)
        meas = re.search(r"\[measure\] exit=(\d+) wall=(\d+)s peak_memory=([\d.]+) GB", log)
        res_rows.append({"job": jid, "dataset": job["dataset"], "method": job["method"], "epsilon": job["epsilon"],
                         "seed": job["seed"], "threads": job["threads"],
                         "total_words": tw.group(1) if tw else "", "sensitive_words": tw.group(2) if tw else "",
                         "rows_with_vector": mat.group(1) if mat else "", "columns_sensitive_with_vector": smat.group(1) if smat else "",
                         "wall_s": meas.group(2) if meas else "", "peak_gb": meas.group(3) if meas else ""})

        info = vinfo.get(job["dataset"])
        if info is None or job.get("diagnostic"):
            continue
        orig = read_original_texts(os.path.join(job["data_dir"], "dev.tsv"))
        san = read_sanitized_texts(os.path.join(out, "dev.tsv"))
        c = collections.Counter()
        aligned_rows = []
        for d, (o_text, s_text) in enumerate(zip(orig, san)):
            o = [t.text for t in tokenizer(o_text)]
            s = s_text.split(" ") if s_text else []
            c["docs"] += 1
            if len(o) != len(s):
                continue
            c["docs_aligned"] += 1
            for pos, (ot, st) in enumerate(zip(o, s)):
                same = ot == st
                classes = ["all"]
                if is_content(ot):
                    classes.append("content")
                if not info["has_vec"].get(ot, False):
                    classes.append("no_vector")
                elif ot in info["sensitive"]:
                    classes.append("rare90")
                else:
                    classes.append("frequent10")
                for k in classes:
                    c[f"n_{k}"] += 1
                    c[f"kept_{k}"] += same
                if job["dataset"] == "reviews":
                    aligned_rows.append((d, pos, ot, st))
        row = {"job": jid, "dataset": job["dataset"], "method": job["method"], "epsilon": job["epsilon"],
               "seed": job["seed"], "threads": job["threads"], "docs": c["docs"], "docs_aligned": c["docs_aligned"]}
        for k in ("all", "content", "rare90", "frequent10", "no_vector"):
            row[f"tokens_{k}"] = c[f"n_{k}"]
            row[f"unchanged_{k}"] = c[f"kept_{k}"]
            row[f"share_{k}"] = round(c[f"kept_{k}"] / c[f"n_{k}"], 4) if c[f"n_{k}"] else ""
        surv_rows.append(row)
        if aligned_rows:
            with open(os.path.join(args.out, "aligned", f"{jid}.tsv"), "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
                w.writerow(["doc", "pos", "orig", "san"])
                w.writerows(aligned_rows)

    for name, rows in (("resources.csv", res_rows), ("survival.csv", surv_rows)):
        if rows:
            with open(os.path.join(args.out, name), "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    if surv_rows:
        md += ["## Tokens left unchanged by sanitization (dev split)", "",
               "rare 90% = the words SanText+ treats as sensitive (w = 0.9); frequent 10% = the rest; "
               "no vector = words GloVe lacks, replaced uniformly at random.", "",
               "| Dataset | Method | ε | Seed | Docs aligned | All | Content words | Rare 90% | Frequent 10% | No vector |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(surv_rows, key=lambda r: (r["dataset"], r["method"], r["epsilon"], r["seed"])):
            pct = lambda k: "–" if r[f"share_{k}"] == "" else f"{100 * r[f'share_{k}']:.1f}%"
            md.append(f"| {r['dataset']} | {METHOD_NAME[r['method']]} | {r['epsilon']:g} | {r['seed']} | "
                      f"{r['docs_aligned']}/{r['docs']} | {pct('all')} | {pct('content')} | {pct('rare90')} | "
                      f"{pct('frequent10')} | {pct('no_vector')} |")
        md.append("")
    if res_rows:
        md += ["## Sanitization resources", "",
               "| Job | Words | Rows with vector | Sensitive columns | Wall (s) | Peak memory (GB) |", "|---|---|---|---|---|---|"]
        md += [f"| {r['job']} | {r['total_words']} | {r['rows_with_vector']} | {r['columns_sensitive_with_vector']} | "
               f"{r['wall_s']} | {r['peak_gb']} |" for r in res_rows]
        md.append("")

    # ---------------------------------------------------------------- fine-tuning results
    runs = []
    for job in plan["finetune"]:
        p = os.path.join(cfg["base"], "results", job["run_name"], "metrics.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        d = m["final_dev"]
        runs.append({"run": job["run_name"], "source": "this notebook", "dataset": job["dataset"], "method": job["method"],
                     "epsilon": job["epsilon"], "seed": job["seed"], "n_train": m["n_train"], "n_dev": m["n_dev"],
                     "accuracy": d["accuracy"], "macro_f1": d["macro_f1"], "f1_label1": d.get("f1_label_index_1"),
                     "minutes": m["runtime_min"], "transformers": m["environment"]["transformers"],
                     "torch": m["environment"]["torch"]})
    for old in plan.get("old_runs", []):
        p = os.path.join(old["path"], "metrics.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
            d = m["final_dev"]
            runs.append({"run": old["run_name"], "source": "9 Oct", "dataset": old["dataset"], "method": old["method"],
                         "epsilon": old["epsilon"], "seed": old["seed"], "n_train": m["n_train"], "n_dev": m["n_dev"],
                         "accuracy": d["accuracy"], "macro_f1": d["macro_f1"], "f1_label1": d.get("f1_label_index_1"),
                         "minutes": m["runtime_min"], "transformers": m["environment"]["transformers"],
                         "torch": m["environment"]["torch"]})
    if runs:
        with open(os.path.join(args.out, "runs.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(runs[0]))
            w.writeheader()
            w.writerows(runs)

        cur = [r for r in runs if r["source"] == "this notebook"]
        sst = [r for r in cur if r["dataset"] == "sst2"]
        if sst:
            md += ["## SST-2 dev accuracy against the paper (Yue et al., 2021, Table 1, GloVe)", "",
                   "| Method | ε | Paper | Ours, per seed | Mean | SD | Mean − paper |", "|---|---|---|---|---|---|---|"]
            for (method, eps), paper in PAPER_SST2.items():
                vals = sorted((r["seed"], r["accuracy"]) for r in sst if r["method"] == method and r["epsilon"] == eps)
                if not vals:
                    continue
                mu, sd = mean_sd([v for _, v in vals])
                per_seed = ", ".join(f"{v:.4f} (s{s})" for s, v in vals)
                eps_txt = "–" if eps is None else f"{eps:g}"
                md.append(f"| {METHOD_NAME[method]} | {eps_txt} | {paper:.4f} | {per_seed} | "
                          f"{fmt(mu)} | {fmt(sd)} | {mu - paper:+.4f} |")
            md.append("")
        new = [r for r in cur if r["dataset"] != "sst2"]
        if new:
            md += ["## New datasets (dev = held-out test split)", "",
                   "| Dataset | Method | ε | Seed | Accuracy | Macro-F1 | F1 (label 1) |", "|---|---|---|---|---|---|---|"]
            for r in sorted(new, key=lambda r: (r["dataset"], r["method"] != "unsanitized", r["method"], r["epsilon"] or 0, r["seed"])):
                eps = "–" if r["epsilon"] is None else f"{r['epsilon']:g}"
                md.append(f"| {r['dataset']} | {METHOD_NAME[r['method']]} | {eps} | "
                          f"{r['seed']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {fmt(r['f1_label1'])} |")
            md.append("")
        old = [r for r in runs if r["source"] == "9 Oct"]
        if old:
            md += ["## 9 Oct runs, for comparison", "", "| Run | Accuracy | Macro-F1 | transformers |", "|---|---|---|---|"]
            md += [f"| {r['run']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['transformers']} |" for r in old]
            md.append("")

    # ---------------------------------------------------------------- reproducibility
    repro = {}
    for pair in plan.get("repro", {}).get("pairs", []):
        a, b = jobs[pair["a"]], jobs[pair["b"]]
        if all(os.path.exists(os.path.join(j["out_path"], ".done")) for j in (a, b)):
            repro[pair["name"]] = {"description": pair["description"], **compare_files(a["out_path"], b["out_path"])}
    for item in plan.get("repro", {}).get("dupdoc", []):
        j = jobs[item["job"]]
        if os.path.exists(os.path.join(j["out_path"], ".done")):
            dev = read_sanitized_texts(os.path.join(j["out_path"], "dev.tsv"))
            cnt = collections.Counter(dev)
            repro[item["name"]] = {"description": item["description"], "copies": len(dev), "distinct_outputs": len(cnt),
                                   "copies_sharing_their_output": sum(n for n in cnt.values() if n > 1),
                                   "largest_group": max(cnt.values()) if cnt else 0}
    for item in plan.get("repro", {}).get("old_vs_new", []):
        j = jobs.get(item["job"])
        if j and os.path.isdir(item["old_dir"]) and os.path.exists(os.path.join(j["out_path"], ".done")):
            repro[item["name"]] = {"description": item["description"], **compare_files(item["old_dir"], j["out_path"])}
    if repro:
        with open(os.path.join(args.out, "repro.json"), "w", encoding="utf-8") as f:
            json.dump(repro, f, indent=2)
        md += ["## Reproducibility checks", ""]
        for name, r in repro.items():
            md.append(f"- **{name}**: {r['description']}")
            for k, v in r.items():
                if k != "description":
                    md.append(f"  - {k}: {v}")
        md.append("")

    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
