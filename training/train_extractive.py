"""
train_extractive.py — v2: LoRA fine-tune an extractive QA head on CUAD.

Same integrity rules as v1: trains ONLY on the train split, contamination gate
runs first, checkpoints are dev-selected (test touched once). The QA head is
randomly initialised, so it is added to `modules_to_save` (LoRA alone would
leave it random). Backbone-agnostic: --base-model swaps Qwen 0.5B (local smoke)
for Qwen3-14B (cloud) with no code change.

    python -m training.train_extractive --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --limit 5 --max-steps 10 --output-dir outputs/ext_smoke     # GPU smoke
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
from transformers import (AutoTokenizer, Trainer, TrainingArguments,
                          default_data_collator)
from peft import LoraConfig, TaskType, get_peft_model

from harness.extractive import (CACHE_ROOT, ROOT, _titles, load_examples,
                                load_qa_model, make_dataset)

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def _training_args(**want) -> TrainingArguments:
    """Filter kwargs to what THIS transformers version's TrainingArguments
    accepts (v5 renamed/dropped several) — same robustness pattern as v1."""
    ok = set(inspect.signature(TrainingArguments.__init__).parameters)
    return TrainingArguments(**{k: v for k, v in want.items() if k in ok})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train-json", default=str(ROOT / "data" / "cuad" / "train.json"))
    ap.add_argument("--output-dir", default="outputs/ext")
    ap.add_argument("--limit", type=int, default=None, help="cap #contracts (smoke)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--save-steps", type=int, default=250)
    ap.add_argument("--neg-per-pos", type=int, default=2,
                    help="null windows kept per positive window (imbalance)")
    ap.add_argument("--exclude-json", default=str(ROOT / "data" / "cuad" / "dev.json"),
                    help="drop these contracts from training (dev is carved from train)")
    ap.add_argument("--cache-tag", default=None,
                    help="reuse a prebuilt feature cache under outputs/ext_cache/")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cache_dir = (CACHE_ROOT / args.cache_tag) if args.cache_tag else None
    print(f"[data] loading train (limit={args.limit}, exclude={args.exclude_json}) ...", flush=True)
    examples = load_examples(args.train_json, limit=args.limit,
                             exclude_json=args.exclude_json)
    # rule #1, in code: training titles must not touch dev or test
    train_titles = {e["title"] for e in examples}
    for guard in ("dev.json", "test.json"):
        gp = ROOT / "data" / "cuad" / guard
        if gp.exists():
            overlap = train_titles & _titles(gp)
            assert not overlap, f"CONTAMINATION: {len(overlap)} train titles in {guard}"
    print(f"[gate] {len(train_titles)} train contracts, 0 overlap with dev/test", flush=True)
    feats = make_dataset(examples, tok, args.max_seq_len, args.stride,
                         is_training=True, neg_per_pos=args.neg_per_pos,
                         cache_dir=cache_dir)
    pos = sum(1 for s in feats["start_positions"] if s != 0)
    print(f"[data] {len(examples)} qas -> {len(feats)} windows "
          f"({pos} positive, {len(feats)-pos} null @ neg_per_pos={args.neg_per_pos})",
          flush=True)

    model = load_qa_model(args.base_model, dtype=torch.bfloat16)
    peft_cfg = LoraConfig(task_type=TaskType.QUESTION_ANS, r=args.lora_r,
                          lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                          target_modules=LORA_TARGETS, modules_to_save=["qa_outputs"])
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    targs = _training_args(
        output_dir=args.output_dir, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        warmup_ratio=0.03, lr_scheduler_type="cosine", bf16=True,
        logging_steps=20, save_steps=args.save_steps, save_total_limit=8,
        report_to=[], dataloader_pin_memory=False,
        remove_unused_columns=False,  # peft forward sig hides start/end_positions
    )
    keep_cols = ["input_ids", "attention_mask", "start_positions", "end_positions"]
    feats = feats.remove_columns([c for c in feats.column_names if c not in keep_cols])
    trainer = Trainer(model=model, args=targs, train_dataset=feats,
                      data_collator=default_data_collator)
    result = trainer.train()
    loss = result.training_loss
    print(f"[done] final training_loss={loss:.4f}", flush=True)
    assert loss == loss and loss < 20, f"loss broken: {loss}"

    out = Path(args.output_dir) / "final_adapter"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
