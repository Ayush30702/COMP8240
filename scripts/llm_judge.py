#!/usr/bin/env python3
"""
llm_judge.py: LLM-as-judge for sanitized text, run locally with an open-weights model.

The model never writes free text. For each prompt we read its probabilities for the allowed answer
tokens at the first position of its reply, so every judgment is deterministic, cheap and comes with
a confidence:
  meaning     original + sanitized text -> how well is the meaning preserved? 1-5
              score = expected value over the probabilities of the tokens "1".."5"
  disclosure  one review -> does it reveal personal information about its writer? yes/no
              (the same definition as the human annotation guidelines)
  residual    sanitized review + the information its original disclosed -> is it still revealed? yes/no
              p_yes = P(yes) / (P(yes) + P(no)), summing "yes"/"Yes" and "no"/"No"
format_mass records how much of the model's probability falls on the allowed answer tokens at all
(close to 1 means the model is following the answer format).

Nothing leaves the machine: the model is downloaded once and run on the local GPU.

Usage
  python llm_judge.py --items judge_items.jsonl --out judge_results.jsonl
      [--model Qwen/Qwen2.5-7B-Instruct --fallback Qwen/Qwen2.5-3B-Instruct --batch_size 8]
Resumable: items already in --out are skipped.
"""
import argparse
import json
import os
import time

PROMPT_VERSION = "v1"
SYSTEM = ("You are a careful annotator in a research study on text privacy. "
          "Follow the instructions exactly and answer with a single token.")

MEANING = """Text A is an original text. Text B was made from Text A by replacing some of its words with other words.
How well does Text B preserve the meaning of Text A?
1 = unrelated or unintelligible
2 = mostly lost; only fragments survive
3 = the topic survives, but important details are lost or changed
4 = mostly the same meaning; minor details are lost
5 = the same meaning

Text A: <<<{a}>>>
Text B: <<<{b}>>>

Answer with one digit from 1 to 5."""

DISCLOSURE = """Does the following app review reveal personal information about its writer, or about another identifiable person, that they might not want linked to them?
Personal information means:
- HEALTH: a condition, symptom, medication, treatment, pregnancy or mental health
- FINANCE: amounts, debts, loans, income, account problems, or which bank they use
- LOCATION: suburb, address, workplace or branch
- IDENTITY: a name, age, family member or job
- CONTACT: an email, phone number, account or reference number
Opinions about the app, its features or its price, and generic use ("I use it daily", "my meds", "my account") do not count.

Review: <<<{a}>>>

Answer yes or no."""

RESIDUAL = """An original app review disclosed this personal information about its writer: {info}.
The text below was made from that review by replacing some of its words with other words.
Does the text below still reveal that information, or enough of it that a reader could infer it?

Text: <<<{b}>>>

Answer yes or no."""


def clip_words(text, n):
    words = (text or "").split()
    return " ".join(words[:n]) + (" [...]" if len(words) > n else "")


def build_prompt(item, max_words):
    a, b = clip_words(item.get("a"), max_words), clip_words(item.get("b"), max_words)
    if item["task"] == "meaning":
        return MEANING.format(a=a, b=b)
    if item["task"] == "disclosure":
        return DISCLOSURE.format(a=a)
    if item["task"] == "residual":
        return RESIDUAL.format(info=item["info"], b=b)
    raise ValueError(item["task"])


def answer_token_ids(tokenizer):
    def first_id(s):
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if ids else None
    digits = {d: first_id(d) for d in "12345"}
    yes = sorted({i for i in (first_id("yes"), first_id("Yes")) if i is not None})
    no = sorted({i for i in (first_id("no"), first_id("No")) if i is not None})
    return digits, yes, no


def load_model(model_id, quantize, torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"device_map": "auto"}
    if quantize:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.float16)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, **kwargs)
    except TypeError:  # older transformers use torch_dtype
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, **kwargs)
    return tok, model.eval()


def decoder_and_head(model):
    """The transformer body and the output layer, so only the last position's logits are computed."""
    try:
        decoder = model.get_decoder()
    except Exception:
        decoder = None
    if decoder is None:
        decoder = getattr(model, "model", None) or model.base_model
    head = model.get_output_embeddings()
    if head is None:
        head = model.lm_head
    return decoder, head


def last_position_logits(decoder, head, enc):
    """Logits of the next token only (the full logits tensor would not fit for a batch)."""
    attention = enc["attention_mask"]
    position_ids = (attention.cumsum(-1) - 1).clamp(min=0)
    hidden = decoder(input_ids=enc["input_ids"], attention_mask=attention,
                     position_ids=position_ids).last_hidden_state[:, -1, :]
    return head(hidden).float()


def main():
    ap = argparse.ArgumentParser(description="Local LLM-as-judge for sanitized text.")
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--quantize", default="4bit", choices=["4bit", "none"])
    ap.add_argument("--fallback", default="Qwen/Qwen2.5-3B-Instruct", help="used without quantization if the first model fails to load")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_words", type=int, default=200, help="each text is cut to this many words")
    args = ap.parse_args()

    import torch
    with open(args.items, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} items, {len(done)} already judged, {len(todo)} to go", flush=True)
    if not todo:
        return

    model_id, quant = args.model, args.quantize == "4bit"
    try:
        tok, model = load_model(model_id, quant, torch)
    except Exception as e:  # e.g. bitsandbytes unavailable
        print(f"could not load {model_id} ({type(e).__name__}: {e}); falling back to {args.fallback}", flush=True)
        model_id, quant = args.fallback, False
        tok, model = load_model(model_id, quant, torch)
    model_label = model_id + (" (4-bit NF4)" if quant else " (fp16)")
    digits, yes_ids, no_ids = answer_token_ids(tok)
    print(f"judge: {model_label} | digit tokens {digits} | yes {yes_ids} | no {no_ids}", flush=True)
    device = next(model.parameters()).device
    decoder, head = decoder_and_head(model)

    # longest prompts first, so memory problems show up at once
    todo.sort(key=lambda it: -len(build_prompt(it, args.max_words)))
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as out, torch.no_grad():
        for b in range(0, len(todo), args.batch_size):
            batch = todo[b:b + args.batch_size]
            texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                              {"role": "user", "content": build_prompt(it, args.max_words)}],
                                             tokenize=False, add_generation_prompt=True) for it in batch]
            enc = tok(texts, return_tensors="pt", padding=True).to(device)
            probs = torch.softmax(last_position_logits(decoder, head, enc), dim=-1).cpu()
            for it, p in zip(batch, probs):
                rec = dict(it)
                rec.update({"judge_model": model_label, "prompt_version": PROMPT_VERSION,
                            "top_token": tok.decode([int(p.argmax())])})
                if it["task"] == "meaning":
                    pd = {d: float(p[i]) for d, i in digits.items()}
                    mass = sum(pd.values())
                    rec.update({"p_digits": {d: v / mass for d, v in pd.items()}, "format_mass": mass,
                                "expected": sum(int(d) * v for d, v in pd.items()) / mass,
                                "answer": int(max(pd, key=pd.get))})
                else:
                    py, pn = float(p[yes_ids].sum()), float(p[no_ids].sum())
                    rec.update({"p_yes": py / (py + pn), "format_mass": py + pn, "answer": "yes" if py > pn else "no"})
                for k in ("a", "b", "info"):
                    rec.pop(k, None)  # keep the results file free of the texts
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if (b // args.batch_size) % 25 == 0:
                print(f"  {b + len(batch)}/{len(todo)} judged ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"finished in {(time.time() - t0) / 60:.1f} min with {model_label}", flush=True)


if __name__ == "__main__":
    main()
