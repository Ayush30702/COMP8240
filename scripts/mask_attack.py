#!/usr/bin/env python3
"""
mask_attack.py: the mask token inference attack of Yue et al. (2021, Sec. 5.3), as released and as fixed.

For every word piece j of a sanitized document, replace piece j with [MASK], let a pretrained BERT
masked language model predict it, and count a success when the top-1 prediction is the ORIGINAL
piece at j. Defense rate = 1 - success rate (higher = more private).

Two variants:
  released  What the authors' bert_inference_token.py computes in its own environment
            (transformers 3.3.1). Its TensorDataset holds (input_ids, token_type_ids,
            attention_mask, labels), but the model call passes attention_mask=batch[1] (the
            token-type ids: all 0 for one sentence) and token_type_ids=batch[2] (the attention
            mask). In 3.3.1 an all-zero mask adds the same -10000 to every attention score, which
            changes nothing, so the model attends to the padding as well; and every real token gets
            the second-segment (B) embedding. Inputs are padded to 64 word pieces as in the script
            (--max_seq_length 64); positions beyond the 62 that fit lose their [MASK] and are
            skipped and counted here (the released script would stop with a shape error).
            We reproduce this by passing an all-ones attention mask and the real attention mask as
            token_type_ids, which is the same computation.
  fixed     The same attack with the inputs in their intended places and no truncation.

Input: the original dev.tsv (SST-2 format) and a sanitized dev.tsv whose text is word pieces joined
by single spaces (as run_SanText.py and santext_chunked.py write it with --embedding bert), or
--sanitized none to attack the unsanitized text (epsilon = infinity).

Usage
  python mask_attack.py --original data/SST-2/dev.tsv --sanitized san_bert/SST-2/eps_14.00/dev.tsv \
      --name eps14 --out_dir results_c/attack [--variants released,fixed]
"""
import argparse
import json
import os
import time


def read_texts(path, sanitized):
    with open(path, encoding="utf-8") as f:
        next(f)
        if sanitized:
            return [line.rstrip("\n").rsplit("\t", 1)[0] for line in f]
        return [line.strip().split("\t")[0] for line in f]  # as bert_inference_token.py reads it


def build_instances(orig_docs, san_docs, variant, to_ids, cls_id, sep_id, mask_id, max_len=64):
    """One instance per word piece: (input ids, real length, mask position, label id, kept, doc, pos).
    Returns the instances and the number of positions skipped because truncation removed the mask."""
    instances, skipped = [], 0
    for d, (o, s) in enumerate(zip(orig_docs, san_docs)):
        if len(o) != len(s):
            raise ValueError(f"document {d}: {len(o)} original pieces but {len(s)} sanitized pieces")
        san_ids, orig_ids = to_ids(s), to_ids(o)
        for j in range(len(s)):
            ids = list(san_ids)
            ids[j] = mask_id
            if variant == "released":
                if j >= max_len - 2:
                    skipped += 1
                    continue
                ids = ids[: max_len - 2]
            else:
                ids = ids[:510]
                if j >= 510:
                    skipped += 1
                    continue
            seq = [cls_id] + ids + [sep_id]
            instances.append((seq, len(seq), j + 1, orig_ids[j], s[j] == o[j], d, j))
    return instances, skipped


def make_batch(batch, variant, pad_id, max_len, torch, device):
    """Tensors for one batch. released: pad to max_len, attention over everything, segment ids = real mask."""
    width = max_len if variant == "released" else max(b[1] for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    real = torch.zeros((len(batch), width), dtype=torch.long)
    for k, b in enumerate(batch):
        ids[k, : b[1]] = torch.tensor(b[0], dtype=torch.long)
        real[k, : b[1]] = 1
    if variant == "released":
        attention, segments = torch.ones_like(real), real
    else:
        attention, segments = real, torch.zeros_like(real)
    return ids.to(device), attention.to(device), segments.to(device)


def predict_at(model, ids, attention, segments, pos, torch, device):
    """Top-1 prediction at each row's [MASK] position. Runs BERT once and applies the masked-LM head
    only at the masked positions (same numbers as the full head, far less work)."""
    rows = torch.arange(ids.shape[0], device=device)
    body, head = getattr(model, "bert", None), getattr(model, "cls", None)
    if body is not None and head is not None:
        hidden = body(input_ids=ids, attention_mask=attention, token_type_ids=segments).last_hidden_state
        return head(hidden[rows, pos]).argmax(-1).tolist()
    logits = model(input_ids=ids, attention_mask=attention, token_type_ids=segments).logits
    return logits[rows, pos].argmax(-1).tolist()


def run_attack(instances, variant, model, pad_id, max_len, batch_size, torch, device, autocast=None):
    correct = []
    order = sorted(range(len(instances)), key=lambda i: instances[i][1]) if variant == "fixed" else range(len(instances))
    order = list(order)
    with torch.no_grad():
        for b in range(0, len(order), batch_size):
            batch = [instances[i] for i in order[b:b + batch_size]]
            ids, attention, segments = make_batch(batch, variant, pad_id, max_len, torch, device)
            pos = torch.tensor([x[2] for x in batch], device=device)
            if autocast is not None:
                with autocast():
                    pred = predict_at(model, ids, attention, segments, pos, torch, device)
            else:
                pred = predict_at(model, ids, attention, segments, pos, torch, device)
            correct.extend((order[b + k], int(p == x[3])) for k, (p, x) in enumerate(zip(pred, batch)))
    hits = [0] * len(instances)
    for i, c in correct:
        hits[i] = c
    return hits


def summarise(instances, hits, skipped):
    n = len(instances)
    kept = [h for inst, h in zip(instances, hits) if inst[4]]
    changed = [h for inst, h in zip(instances, hits) if not inst[4]]
    rate = lambda xs: round(1 - sum(xs) / len(xs), 4) if xs else None
    return {"positions": n, "skipped_truncated": skipped, "attack_success": round(sum(hits) / n, 4) if n else None,
            "defense_rate": rate(hits), "positions_kept_by_sanitizer": len(kept), "defense_rate_kept": rate(kept),
            "positions_changed_by_sanitizer": len(changed), "defense_rate_changed": rate(changed)}


def main():
    ap = argparse.ArgumentParser(description="Mask token inference attack, as released and fixed.")
    ap.add_argument("--original", required=True, help="original dev.tsv (SST-2 format)")
    ap.add_argument("--sanitized", required=True, help="sanitized dev.tsv, or 'none' for unsanitized text")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--variants", default="released,fixed")
    ap.add_argument("--max_seq_length", type=int, default=64, help="released script's default")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--fp32", action="store_true", help="disable mixed precision on the GPU")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(device).eval()
    autocast = None
    if device.type == "cuda" and not args.fp32:
        autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)

    orig_docs = [tok.tokenize(t) for t in read_texts(args.original, sanitized=False)]
    if args.sanitized == "none":
        san_docs = [list(d) for d in orig_docs]
    else:
        san_docs = [t.split(" ") if t else [] for t in read_texts(args.sanitized, sanitized=True)]
    to_ids = tok.convert_tokens_to_ids
    report = {"name": args.name, "sanitized": args.sanitized, "model": args.model, "documents": len(orig_docs),
              "max_seq_length_released": args.max_seq_length, "mixed_precision": autocast is not None, "variants": {}}
    for variant in args.variants.split(","):
        t0 = time.time()
        inst, skipped = build_instances(orig_docs, san_docs, variant, to_ids, tok.cls_token_id, tok.sep_token_id,
                                        tok.mask_token_id, args.max_seq_length)
        hits = run_attack(inst, variant, model, tok.pad_token_id, args.max_seq_length, args.batch_size, torch, device,
                          autocast)
        report["variants"][variant] = {**summarise(inst, hits, skipped), "seconds": round(time.time() - t0, 1)}
        print(f"{args.name} [{variant}]: " + json.dumps(report["variants"][variant]), flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{args.name}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
