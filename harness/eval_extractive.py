"""
eval_extractive.py — v2 eval: run the QA head over a split, score officially.

Forward pass -> start/end logits -> postprocess to one best span per qid ->
harness/scoring.py (the SAME official CUAD scorer used for every frontier
baseline). Output n-best + AUPR land in results/pilot_v2/, directly comparable
to v1 and the frontier rows. Features come from the disk-cached arrow Dataset;
offsets are read lazily per feature so full splits never load into RAM.

    python -m harness.eval_extractive --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --adapter outputs/ext_0p5b/final_adapter --split-json data/cuad/dev.json \
        --out results/pilot_v2/dev_0p5b.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from harness import scoring
from harness.extractive import (CACHE_ROOT, ROOT, load_examples, load_qa_model,
                                make_dataset, postprocess)


def _infer_logits(model, ds, batch_size, device):
    """Return (start_logits, end_logits) as per-feature python lists, in order."""
    starts, ends = [], []
    model.eval()
    n = len(ds)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            sl = ds[i:i + batch_size]
            ids = torch.tensor(sl["input_ids"], device=device)
            mask = torch.tensor(sl["attention_mask"], device=device)
            out = model(input_ids=ids, attention_mask=mask)
            starts.extend(out.start_logits.float().cpu().tolist())
            ends.extend(out.end_logits.float().cpu().tolist())
            if (i // batch_size) % 20 == 0:
                print(f"  eval {min(i+batch_size,n)}/{n} windows", flush=True)
    return starts, ends


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit = raw head)")
    ap.add_argument("--split-json", default=str(ROOT / "data" / "cuad" / "dev.json"))
    ap.add_argument("--limit", type=int, default=None, help="cap #contracts (smoke)")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cache-tag", default=None,
                    help="reuse a prebuilt eval feature cache under outputs/ext_cache/")
    ap.add_argument("--out", default="results/pilot_v2/eval.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # tokenizer comes from the base model (a LoRA adapter never changes it, and
    # intermediate Trainer checkpoints don't save tokenizer files).
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = load_qa_model(args.base_model, dtype=torch.bfloat16)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device)

    cache_dir = (CACHE_ROOT / args.cache_tag) if args.cache_tag else None
    examples = load_examples(args.split_json, limit=args.limit)
    ds = make_dataset(examples, tok, args.max_seq_len, args.stride,
                      is_training=False, cache_dir=cache_dir)
    print(f"[eval] {len(examples)} qas -> {len(ds)} windows on {device}", flush=True)

    starts, ends = _infer_logits(model, ds, args.batch_size, device)
    example_ids = ds["example_id"]              # small: one string per feature
    nbest = postprocess(examples, example_ids,
                        lambda fi: ds[fi]["offset_mapping"],  # lazy per-feature
                        starts, ends, max_answer_len=args.max_seq_len)

    gt_full = scoring.load_gt(args.split_json)
    gt = {k: gt_full[k] for k in nbest}
    overall = scoring.score_overall(nbest, gt)
    per = scoring.score_per_category(nbest, gt)
    print(f"\n[AUPR] overall={overall['aupr']:.4f}  "
          f"p@80r={overall['prec_at_80_recall']:.3f}  "
          f"p@90r={overall['prec_at_90_recall']:.3f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "base_model": args.base_model, "adapter": args.adapter,
        "split": args.split_json, "limit": args.limit,
        "n_qas": len(examples), "overall": overall, "per_category": per,
    }, indent=2), encoding="utf-8")
    (out.parent / (out.stem + "_predictions.json")).write_text(
        json.dumps(nbest, indent=2), encoding="utf-8")
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
