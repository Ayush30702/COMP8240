#!/usr/bin/env python3
"""
make_plan.py: the run plan for the final report, in priority order.

Tier 1 (essentials):  new datasets at eps = 1, 2, 3 with SanText and SanText+; SanText+ on SST-2;
                      reproducibility checks (cheap: they use the small review dataset).
Tier 2 (should):      SST-2 SanText at eps = 1, 2, 3 and the unsanitized baseline with seeds 42-44,
                      so the replication reports run-to-run spread instead of single runs.
Tier 3 (could):       SanText+ on SST-2 with seeds 43-44; new datasets at eps = 3 with seeds 43-44.

Every sanitization uses the authors' run_SanText.py with --threads 8 (as on 9 Oct; Colab has two
cores, so this means two worker processes), except where a reproducibility check sets it.
SanText+ uses the paper's p = 0.3 and w = 0.9 (Yue et al., 2021, Sec. 5.1).
Fine-tuning settings match 9 Oct: SST-2 with the paper's settings; Dreaddit with 256-token inputs
and batch 16; reviews with batch 16 (small datasets, so more optimizer steps).
"""
import json
import os

P, W = 0.3, 0.9
EPS = (1.0, 2.0, 3.0)
FT_ARGS = {"sst2": ["--fp16", "--log_every", "300"],
           "dreaddit": ["--max_seq_length", "256", "--batch_size", "16", "--fp16"],
           "reviews": ["--batch_size", "16", "--fp16"]}


def out_path(output_dir, method, eps):
    path = os.path.join(output_dir, "eps_%.2f" % eps)
    return os.path.join(path, "sword_%.2f_p_%.2f" % (W, P)) if method == "SanText_plus" else path


def build_plan(base, santext_dir, data_dirs, max_tier=3, backup_root=None, old_root=None, seeds=(42, 43, 44)):
    """data_dirs: {"sst2": ..., "dreaddit": ..., "reviews": ..., "dupdoc": ...}; missing datasets are left out."""
    have = {k: v for k, v in data_dirs.items() if v and os.path.exists(os.path.join(v, "dev.tsv"))}
    san, ft = [], []

    def S(ds, method, eps, seed, tier, threads=8, tag="", diagnostic=False):
        if ds not in have:
            return None
        jid = f"{ds}__{method}__eps{eps:.2f}__seed{seed}{tag}"
        output_dir = os.path.join(base, "san", ds, method, f"seed_{seed}{tag}")
        job = {"id": jid, "tier": tier, "dataset": ds, "method": method, "epsilon": eps, "seed": seed,
               "threads": threads, "p": P, "w": W, "data_dir": have[ds], "output_dir": output_dir,
               "out_path": out_path(output_dir, method, eps), "diagnostic": diagnostic}
        san.append(job)
        return job

    def F(ds, method, eps, seed, tier, needs=None):
        if ds not in have or (method != "unsanitized" and needs is None):
            return
        name = f"{ds}__unsanitized__seed{seed}" if method == "unsanitized" else f"{ds}__{method}__eps{eps:.2f}__seed{seed}"
        ft.append({"run_name": name, "tier": tier, "dataset": ds, "method": method,
                   "epsilon": None if method == "unsanitized" else eps, "seed": seed,
                   "data_dir": have[ds] if method == "unsanitized" else needs["out_path"],
                   "needs": None if needs is None else needs["id"], "args": FT_ARGS[ds] + ["--seed", str(seed)]})

    # ---------------- sanitizations, in the order the background queue runs them
    s42 = seeds[0]
    new_jobs, repro_jobs = {}, {}
    for ds in ("reviews", "dreaddit"):
        for method in ("SanText", "SanText_plus"):
            for e in EPS:
                new_jobs[(ds, method, e, s42)] = S(ds, method, e, s42, tier=1)
        if ds == "reviews":   # reproducibility checks right after the first (fast) review jobs
            repro_jobs = {
                "t1a": S("reviews", "SanText", 1.0, s42, 1, threads=1, tag="_rep_t1a", diagnostic=True),
                "t1b": S("reviews", "SanText", 1.0, s42, 1, threads=1, tag="_rep_t1b", diagnostic=True),
                "t2a": S("reviews", "SanText", 1.0, s42, 1, threads=8, tag="_rep_t2a", diagnostic=True),
                "t2b": S("reviews", "SanText", 1.0, s42, 1, threads=8, tag="_rep_t2b", diagnostic=True),
                "p1a": S("reviews", "SanText_plus", 1.0, s42, 1, threads=1, tag="_rep_p1a", diagnostic=True),
                "p1b": S("reviews", "SanText_plus", 1.0, s42, 1, threads=1, tag="_rep_p1b", diagnostic=True),
                "dup1": S("dupdoc", "SanText", 1.0, s42, 1, threads=1, tag="_rep_t1", diagnostic=True),
                "dup2": S("dupdoc", "SanText", 1.0, s42, 1, threads=8, tag="_rep_t2", diagnostic=True),
            }
    sst_plus = {(e, s42): S("sst2", "SanText_plus", e, s42, tier=1) for e in EPS}
    sst_san = {(e, s): S("sst2", "SanText", e, s, tier=2) for s in seeds for e in EPS}
    for s in seeds[1:]:
        for e in EPS:
            sst_plus[(e, s)] = S("sst2", "SanText_plus", e, s, tier=3)
    for ds in ("reviews", "dreaddit"):
        for method in ("SanText", "SanText_plus"):
            for s in seeds[1:]:
                new_jobs[(ds, method, 3.0, s)] = S(ds, method, 3.0, s, tier=3)

    # ---------------- fine-tuning, in the order the GPU loop runs them
    for ds in ("reviews", "dreaddit", "sst2"):
        F(ds, "unsanitized", None, s42, tier=1)
    for ds in ("reviews", "dreaddit"):
        for method in ("SanText", "SanText_plus"):
            for e in EPS:
                F(ds, method, e, s42, 1, new_jobs.get((ds, method, e, s42)))
    for e in EPS:
        F("sst2", "SanText_plus", e, s42, 1, sst_plus.get((e, s42)))
    for e in EPS:
        F("sst2", "SanText", e, s42, 2, sst_san.get((e, s42)))
    for s in seeds[1:]:
        F("sst2", "unsanitized", None, s, tier=2)
        for e in EPS:
            F("sst2", "SanText", e, s, 2, sst_san.get((e, s)))
    for s in seeds[1:]:
        for e in EPS:
            F("sst2", "SanText_plus", e, s, 3, sst_plus.get((e, s)))
    for ds in ("reviews", "dreaddit"):
        for method in ("SanText", "SanText_plus"):
            for s in seeds[1:]:
                F(ds, method, 3.0, s, 3, new_jobs.get((ds, method, 3.0, s)))

    # ---------------- reproducibility comparisons
    repro = {"pairs": [], "dupdoc": [], "old_vs_new": []}
    if "reviews" in have and repro_jobs:
        repro["pairs"] = [
            {"name": "SanText, same seed, 1 worker, two runs", "a": repro_jobs["t1a"]["id"], "b": repro_jobs["t1b"]["id"],
             "description": "Reviews, SanText eps = 1, seed 42, --threads 1, run twice. Identical files mean the seed controls the output."},
            {"name": "SanText, same seed, 2 workers, two runs", "a": repro_jobs["t2a"]["id"], "b": repro_jobs["t2b"]["id"],
             "description": "Reviews, SanText eps = 1, seed 42, --threads 8 (two workers on Colab), run twice."},
            {"name": "SanText+, same seed, 1 worker, two runs", "a": repro_jobs["p1a"]["id"], "b": repro_jobs["p1b"]["id"],
             "description": "Reviews, SanText+ eps = 1, seed 42, --threads 1, run twice. The keep/replace coin uses Python's "
                            "random module, which reseeds itself in every worker process, so the seed cannot fix it."}]
    if "dupdoc" in have and repro_jobs.get("dup1"):
        repro["dupdoc"] = [
            {"name": "One review x 1,000, 1 worker", "job": repro_jobs["dup1"]["id"],
             "description": "Dev file = the first test review repeated 1,000 times; SanText eps = 1, --threads 1. "
                            "Independent sampling gives (almost) 1,000 distinct outputs."},
            {"name": "One review x 1,000, 2 workers", "job": repro_jobs["dup2"]["id"],
             "description": "Same input with two worker processes. Each worker inherits the same NumPy random state, "
                            "so copies processed at the same point by different workers get identical outputs."}]
    if old_root:
        for e in EPS:
            j = sst_san.get((e, s42))
            if j:
                repro["old_vs_new"].append({"name": f"SST-2 SanText eps = {e:g}: 9 Oct vs today (seed 42)", "job": j["id"],
                                            "old_dir": os.path.join(old_root, "SST-2", "eps_%.2f" % e),
                                            "description": "Same code, data and seed (42), --threads 8, on two days."})
    return {"config": {"base": base, "santext_dir": santext_dir,
                       "measure": os.path.join(base, "measure.py"), "filter": os.path.join(base, "filter_glove.py"),
                       "glove_full": os.path.join(base, "data", "glove.840B.300d.txt"),
                       "glove_ready": os.path.join(base, "data", "glove.ready"),
                       "glove_filtered": os.path.join(base, "data", "glove.filtered.txt"),
                       "filter_data_dirs": [v for k, v in have.items()],
                       "logs": os.path.join(base, "logs"), "status": os.path.join(base, "logs", "queue_status.json"),
                       "backup_root": backup_root, "max_tier": max_tier},
            "datasets": {k: v for k, v in have.items() if k != "dupdoc"},
            "sanitize": san, "finetune": ft, "repro": repro, "old_runs": []}


if __name__ == "__main__":
    import sys
    plan = build_plan(*sys.argv[1:3], json.loads(sys.argv[3]))
    print(json.dumps({"sanitize": len(plan["sanitize"]), "finetune": len(plan["finetune"])}))
