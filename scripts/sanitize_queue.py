#!/usr/bin/env python3
"""
sanitize_queue.py: run the planned SanText sanitizations one after another, in the background.

The notebook writes plan.json and starts this script, then fine-tunes on the GPU while this
script keeps the CPU busy. Every job calls the authors' unchanged run_SanText.py (through
measure.py, which records wall time and peak memory). The queue:
  1. waits for the full GloVe download, then builds the filtered GloVe file (filter_glove.py),
     which gives run_SanText.py the same words and vectors but loads in seconds;
  2. runs each job whose tier is within config.max_tier, in plan order, skipping jobs that a
     previous session already finished;
  3. marks each output folder with .done (or .failed), so the fine-tuning loop knows when its
     input is ready, and copies the output and its log to the Drive backup folder straight away.

Usage: python sanitize_queue.py plan.json
"""
import json
import os
import shutil
import subprocess
import sys
import time


def backup(path, cfg):
    root = cfg.get("backup_root")
    if not root or not os.path.exists(path):
        return
    dest = os.path.join(root, os.path.relpath(path, cfg["base"]))
    if os.path.isdir(path):
        shutil.copytree(path, dest, dirs_exist_ok=True)
    else:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(path, dest)


def write_status(cfg, status):
    status["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = cfg["status"] + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp, cfg["status"])


def last_measure_line(log_path):
    """The '[measure] exit=.. wall=..s peak_memory=.. GB' line that measure.py prints last."""
    try:
        with open(log_path, errors="ignore") as f:
            lines = [l.strip() for l in f if l.startswith("[measure]")]
        return lines[-1] if lines else None
    except OSError:
        return None


def build_filtered_glove(cfg, status):
    if os.path.exists(cfg["glove_filtered"]):
        try:
            with open(cfg["glove_filtered"] + ".json", encoding="utf-8") as f:
                covered = set(json.load(f)["vocabulary_per_dataset"])
        except (OSError, ValueError, KeyError):
            covered = set()
        if set(cfg["filter_data_dirs"]) <= covered:
            print("filtered GloVe already present:", cfg["glove_filtered"], flush=True)
            return True
        print("filtered GloVe misses a dataset; rebuilding it", flush=True)
        os.remove(cfg["glove_filtered"])
    t0 = time.time()
    while not os.path.exists(cfg["glove_ready"]):
        status["current"] = f"waiting for the GloVe download ({(time.time() - t0) / 60:.0f} min)"
        write_status(cfg, status)
        if time.time() - t0 > cfg.get("glove_timeout_min", 90) * 60:
            print("GloVe download did not finish; see logs/glove_download.log", flush=True)
            return False
        time.sleep(15)
    status["current"] = "filtering GloVe"
    write_status(cfg, status)
    log = os.path.join(cfg["logs"], "filter_glove.log")
    cmd = [sys.executable, "-u", cfg["filter"], "--glove", cfg["glove_full"], "--out", cfg["glove_filtered"],
           "--santext_dir", cfg["santext_dir"], *cfg["filter_data_dirs"]]
    with open(log, "w") as lf:
        code = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    print(f"filter_glove.py exit={code} ({time.time() - t0:.0f} s); log: {log}", flush=True)
    if code == 0:
        backup(cfg["glove_filtered"], cfg)
        backup(cfg["glove_filtered"] + ".json", cfg)
        backup(log, cfg)
    return code == 0


def run_job(job, cfg):
    out = job["out_path"]
    log = os.path.join(cfg["logs"], f"san__{job['id']}.log")
    cmd = [sys.executable, "-u", cfg["measure"], sys.executable, "-u", "run_SanText.py",
           "--task", "SST-2", "--method", job["method"], "--epsilon", str(job["epsilon"]),
           "--seed", str(job["seed"]), "--word_embedding_path", cfg["glove_filtered"],
           "--word_embedding_size", "300", "--data_dir", job["data_dir"] + "/",
           "--output_dir", job["output_dir"] + "/", "--threads", str(job["threads"])]
    if job["method"] == "SanText_plus":
        cmd += ["--p", str(job["p"]), "--sensitive_word_percentage", str(job["w"])]
    t0 = time.time()
    with open(log, "w") as lf:
        lf.write("command: " + " ".join(cmd) + "\n")
        lf.flush()
        code = subprocess.run(cmd, cwd=cfg["santext_dir"], stdout=lf, stderr=subprocess.STDOUT).returncode
    ok = code == 0 and all(os.path.exists(os.path.join(out, f)) for f in ("train.tsv", "dev.tsv"))
    os.makedirs(out, exist_ok=True)
    marker = {"id": job["id"], "exit": code, "seconds": round(time.time() - t0), "measure": last_measure_line(log),
              "log": os.path.relpath(log, cfg["base"])}
    with open(os.path.join(out, ".done" if ok else ".failed"), "w", encoding="utf-8") as f:
        json.dump(marker, f)
    backup(out, cfg)
    backup(log, cfg)
    return ok, marker


def main():
    with open(sys.argv[1], encoding="utf-8") as f:
        plan = json.load(f)
    cfg = plan["config"]
    os.makedirs(cfg["logs"], exist_ok=True)
    status = {"current": None, "done": [], "failed": [], "skipped_already_done": [], "not_in_tier": []}
    write_status(cfg, status)

    if not build_filtered_glove(cfg, status):
        status["current"] = "stopped: no GloVe"
        status["failed"] = [j["id"] for j in plan["sanitize"]]
        for j in plan["sanitize"]:
            os.makedirs(j["out_path"], exist_ok=True)
            with open(os.path.join(j["out_path"], ".failed"), "w") as f:
                f.write('{"reason": "GloVe unavailable"}')
        write_status(cfg, status)
        return

    for job in plan["sanitize"]:
        if job["tier"] > cfg["max_tier"]:
            status["not_in_tier"].append(job["id"])
            continue
        if os.path.exists(os.path.join(job["out_path"], ".done")):
            status["skipped_already_done"].append(job["id"])
            continue
        failed_marker = os.path.join(job["out_path"], ".failed")
        if os.path.exists(failed_marker):
            os.remove(failed_marker)  # a new session retries failed jobs once
        status["current"] = job["id"]
        write_status(cfg, status)
        print(f"sanitizing {job['id']} ...", flush=True)
        ok, marker = run_job(job, cfg)
        (status["done"] if ok else status["failed"]).append(job["id"])
        print(f"  {'done' if ok else 'FAILED'}: {job['id']} | {marker['measure']}", flush=True)
        write_status(cfg, status)

    status["current"] = "finished"
    write_status(cfg, status)
    backup(cfg["status"], cfg)
    print("queue finished", flush=True)


if __name__ == "__main__":
    main()
