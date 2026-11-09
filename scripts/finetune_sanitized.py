#!/usr/bin/env python3
"""
finetune_sanitized.py: fine-tune BERT on (sanitized) text and report dev-set utility.

Why this script exists
  SanText's released fine-tuning code (run_glue.py + dataset_glue.py) imports
  transformers 3.3.1 module paths such as `transformers.tokenization_bart`,
  which were removed in transformers 4.0, so it no longer runs on a current
  install. This script re-implements the same fine-tuning step with the paper's
  settings (Yue et al., 2021, Sec. 5.1, SST-2/QNLI):
      bert-base-uncased, max_seq_length 128, batch size 64, learning rate 2e-5,
      3 epochs, AdamW without weight decay, linear decay without warmup,
      gradient clipping 1.0, seed 42.
  One deliberate difference: each batch is padded to its longest example rather
  than to 128 tokens. Padding is masked out of attention, so the model computes
  the same thing, but short-sentence data such as SST-2 trains several times faster.

Input
  --data_dir must contain train.tsv and dev.tsv with a header row, as written by
  run_SanText.py (or the original GLUE files):
    single sentence (SST-2 style):  sentence <TAB> label
    sentence pair  (QNLI style):    index <TAB> text_a <TAB> text_b <TAB> label
  The format is detected from the header (2 columns = single, 4 or more = pair).

Output
  results/<run_name>/metrics.json         settings, environment, per-epoch and final dev metrics
  results/<run_name>/dev_predictions.tsv  gold and predicted label for every dev row

Examples
  # 1-minute smoke test (run this first on a new machine)
  python finetune_sanitized.py --data_dir data/SST-2 --max_train_samples 512 --epochs 1 --run_name smoke_test
  # unsanitized baseline (paper: 0.9251 accuracy)
  python finetune_sanitized.py --data_dir data/SST-2 --run_name SST-2_unsanitized --fp16
  # sanitized data written by run_SanText.py
  python finetune_sanitized.py --data_dir output_SanText_glove/SST-2/eps_3.00 --fp16
  # long texts (e.g. Dreaddit): longer inputs, same effective batch of 64 in two halves
  python finetune_sanitized.py --data_dir new_data/dreaddit --max_seq_length 256 --grad_accum 2 --fp16
"""
import argparse
import json
import math
import os
import platform
import random
import time
from contextlib import nullcontext

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.utils.data import DataLoader

import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune BERT on (sanitized) TSV data and report dev-set utility.")
    p.add_argument("--data_dir", required=True, help="folder containing train.tsv and dev.tsv")
    p.add_argument("--run_name", default=None, help="results folder name (default: built from data_dir)")
    p.add_argument("--output_root", default="results")
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64, help="effective batch size per optimizer step")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="split each batch into this many micro-batches to save GPU memory (effective batch unchanged)")
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", help="mixed precision on GPU (faster on T4/P100-class GPUs)")
    p.add_argument("--format", choices=["auto", "single", "pair"], default="auto")
    p.add_argument("--max_train_samples", type=int, default=None, help="use only the first N training rows (smoke tests)")
    p.add_argument("--log_every", type=int, default=200, help="print training loss every N optimizer steps")
    p.add_argument("--save_model", action="store_true", help="also save the fine-tuned model (about 440 MB)")
    args = p.parse_args()
    if args.batch_size % args.grad_accum != 0:
        p.error("--batch_size must be divisible by --grad_accum")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_tsv(path, fmt):
    """Return (format, texts_a, texts_b or None, labels, skipped_rows)."""
    texts_a, texts_b, labels, skipped = [], [], [], 0
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if fmt == "auto":
            fmt = "single" if len(header) == 2 else "pair"
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if fmt == "single":
                text, label = "\t".join(cols[:-1]).strip(), cols[-1].strip()
                if len(cols) < 2 or not text or not label:
                    skipped += 1
                    continue
                texts_a.append(text)
            else:
                if len(cols) < 4 or not cols[-1].strip():
                    skipped += 1
                    continue
                texts_a.append(cols[1])
                texts_b.append(cols[2])
                label = cols[-1].strip()
            labels.append(label)
    return fmt, texts_a, (texts_b if fmt == "pair" else None), labels, skipped


def build_label_map(labels):
    def key(x):
        return (0, int(x), "") if x.lstrip("-").isdigit() else (1, 0, x)
    return {lab: i for i, lab in enumerate(sorted(set(labels), key=key))}


def make_examples(texts_a, texts_b, labels, label_map, split_name):
    unknown = sorted(set(labels) - set(label_map))
    if unknown:
        raise ValueError(f"{split_name}.tsv has labels not seen in train.tsv: {unknown[:5]}")
    b = texts_b if texts_b is not None else [None] * len(texts_a)
    return [(a, bb, label_map[lab]) for a, bb, lab in zip(texts_a, b, labels)]


def make_collate(tokenizer, max_len):
    def collate(batch):
        a = [ex[0] for ex in batch]
        b = [ex[1] for ex in batch] if batch[0][1] is not None else None
        enc = tokenizer(a, b, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([ex[2] for ex in batch], dtype=torch.long)
        return enc
    return collate


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def amp(enabled):
    """Mixed-precision context on GPU; a no-op otherwise."""
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else nullcontext()


def compute_metrics(gold, pred, n_labels):
    gold, pred = np.asarray(gold), np.asarray(pred)
    f1s = []
    for c in range(n_labels):
        tp = int(((pred == c) & (gold == c)).sum())
        fp = int(((pred == c) & (gold != c)).sum())
        fn = int(((pred != c) & (gold == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    out = {"accuracy": float((gold == pred).mean()), "macro_f1": float(np.mean(f1s))}
    if n_labels == 2:
        out["f1_label_index_1"] = float(f1s[1])  # positive class for SST-2, "stress" for Dreaddit
    return out


@torch.no_grad()
def evaluate(model, loader, device, use_fp16, n_labels):
    model.eval()
    preds, golds = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with amp(use_fp16):
            logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
        preds.extend(logits.argmax(dim=-1).tolist())
        golds.extend(batch["labels"].tolist())
    model.train()
    return compute_metrics(golds, preds, n_labels), preds


def main():
    args = parse_args()
    set_seed(args.seed)
    t_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.fp16 and device.type == "cuda"
    if args.fp16 and not use_fp16:
        print("note: --fp16 ignored because no GPU is available")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None

    # ---- data -------------------------------------------------------------
    fmt, tr_a, tr_b, tr_y, tr_skip = read_tsv(os.path.join(args.data_dir, "train.tsv"), args.format)
    _, dv_a, dv_b, dv_y, dv_skip = read_tsv(os.path.join(args.data_dir, "dev.tsv"), fmt)
    if args.max_train_samples:
        tr_a, tr_y = tr_a[: args.max_train_samples], tr_y[: args.max_train_samples]
        tr_b = tr_b[: args.max_train_samples] if tr_b is not None else None
    label_map = build_label_map(tr_y)
    n_labels = len(label_map)
    train_ex = make_examples(tr_a, tr_b, tr_y, label_map, "train")
    dev_ex = make_examples(dv_a, dv_b, dv_y, label_map, "dev")
    print(f"data: {args.data_dir} | format={fmt} | train={len(train_ex)} dev={len(dev_ex)} "
          f"| skipped rows train={tr_skip} dev={dv_skip} | labels={label_map}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate = make_collate(tokenizer, args.max_seq_length)
    micro_bs = args.batch_size // args.grad_accum
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ex, batch_size=micro_bs, shuffle=True, collate_fn=collate, generator=gen)
    dev_loader = DataLoader(dev_ex, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)

    # ---- model, optimizer, schedule ------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=n_labels).to(device)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate, eps=1e-8)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = make_scaler(use_fp16)

    # ---- training -------------------------------------------------------------
    print(f"training on {gpu_name or 'CPU'} | {args.epochs} epochs x {steps_per_epoch} steps | "
          f"batch {args.batch_size} (micro-batch {micro_bs}) | lr {args.learning_rate} | fp16={use_fp16}")
    history, global_step = [], 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss, n_loss = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with amp(use_fp16):
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            running_loss += loss.item() * args.grad_accum
            n_loss += 1
            if i % args.grad_accum == 0 or i == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every == 0:
                    print(f"  epoch {epoch} step {global_step}/{total_steps} | loss {running_loss / n_loss:.4f} "
                          f"| lr {scheduler.get_last_lr()[0]:.2e} | {(time.time() - t_start) / 60:.1f} min")
        dev_metrics, preds = evaluate(model, dev_loader, device, use_fp16, n_labels)
        history.append({"epoch": epoch, "train_loss": running_loss / max(1, n_loss), "dev": dev_metrics,
                        "elapsed_min": round((time.time() - t_start) / 60, 2)})
        print(f"epoch {epoch} done | train loss {running_loss / max(1, n_loss):.4f} | dev {dev_metrics}")

    # ---- results (final = after the last epoch, as the original run_glue.py reports) ----
    final_metrics = history[-1]["dev"]
    run_name = args.run_name or "_".join(
        [p for p in os.path.normpath(os.path.abspath(args.data_dir)).split(os.sep) if p][-3:])
    out_dir = os.path.join(args.output_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    id2label = {v: k for k, v in label_map.items()}
    with open(os.path.join(out_dir, "dev_predictions.tsv"), "w", encoding="utf-8") as f:
        f.write("text_a\ttext_b\tgold\tpred\n")
        for (a, b, gold), pred in zip(dev_ex, preds):
            f.write(f"{a}\t{b or ''}\t{id2label[gold]}\t{id2label[pred]}\n")
    runtime_min = round((time.time() - t_start) / 60, 2)
    report = {
        "run_name": run_name,
        "data_dir": os.path.abspath(args.data_dir),
        "format": fmt,
        "label_map": label_map,
        "n_train": len(train_ex),
        "n_dev": len(dev_ex),
        "skipped_rows": {"train": tr_skip, "dev": dv_skip},
        "settings": vars(args),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": gpu_name or "CPU",
            "platform": platform.platform(),
        },
        "epochs": history,
        "final_dev": final_metrics,
        "runtime_min": runtime_min,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if args.save_model:
        model.save_pretrained(os.path.join(out_dir, "model"))
        tokenizer.save_pretrained(os.path.join(out_dir, "model"))
    summary = " ".join(f"{k}={v:.4f}" for k, v in final_metrics.items())
    print(f"RESULT {run_name} | {summary} | n_dev={len(dev_ex)} | {runtime_min} min on {gpu_name or 'CPU'}")
    print(f"saved to {out_dir}/")


if __name__ == "__main__":
    main()
