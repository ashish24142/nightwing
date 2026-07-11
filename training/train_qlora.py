"""
train_qlora.py — P2.3: QLoRA fine-tune of Qwen-14B on CUAD train (Unsloth).

Runs on the LINUX SPOT GPU box (A100 80GB / H100), NOT the Windows harness box.
Heavy deps (unsloth/torch/trl) are imported lazily so this file is importable
anywhere for inspection.

Pipeline:
  1. HARD GATE: run data/verify_split.py — refuse to train on contamination (rule #1).
  2. Load Qwen-14B 4-bit (Unsloth) + attach QLoRA adapters from config/training.yaml.
  3. Load data/cuad/train_instruct.jsonl, render with the model's chat template.
  4. SFT with frequent checkpoints (spot preemption — rule #6); resume if present.
  5. Save the adapter to output_dir.

Run (on GPU box):  python -m training.train_qlora
Config:            config/training.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "config" / "training.yaml"


def load_cfg() -> dict:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def gate_contamination() -> None:
    """Rule #1: never train if train/test overlap can't be proven zero."""
    sys.path.insert(0, str(ROOT))
    from data.verify_split import main as verify
    code = verify()
    if code != 0:
        raise SystemExit("ABORT: verify_split failed — refusing to train (rule #1).")


def _load_model_and_tokenizer(cfg: dict, base_model: str, prefer_unsloth: bool):
    """Portable model+adapter loader.
      - cloud (Linux): Unsloth FastLanguageModel (fast, 4-bit) when available.
      - local smoke (Windows/Blackwell): transformers + peft fallback. 4-bit only
        if load_in_4bit and bitsandbytes are present, else plain bf16.
    Returns (model, tokenizer, backend_name)."""
    lora = cfg["lora"]
    use_4bit = bool(cfg.get("load_in_4bit", True))

    if prefer_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tok = FastLanguageModel.from_pretrained(
                model_name=base_model, max_seq_length=cfg["max_seq_length"],
                load_in_4bit=use_4bit, dtype=None)
            model = FastLanguageModel.get_peft_model(
                model, r=lora["r"], lora_alpha=lora["alpha"],
                lora_dropout=lora["dropout"], target_modules=lora["target_modules"],
                bias="none", use_gradient_checkpointing="unsloth",
                random_state=cfg["seed"])
            return model, tok, "unsloth"
        except Exception as e:  # Unsloth missing or unsupported (e.g. Windows) -> fallback
            print(f"   Unsloth unavailable ({type(e).__name__}: {str(e)[:80]}); "
                  "using transformers+peft fallback")

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quant = None
    if use_4bit:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        except ImportError:
            print("   load_in_4bit set but bitsandbytes missing -> bf16 (ok for small models)")
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:  # SDPA attention is much faster than eager for long sequences
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=quant, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation="sdpa")
    except (ValueError, ImportError):
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=quant, torch_dtype=torch.bfloat16,
            device_map="auto")
    if quant is not None:
        model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM"))
    return model, tok, "transformers+peft"


def build_trainer(cfg: dict, max_steps: int | None = None,
                  base_model: str | None = None, prefer_unsloth: bool = True):
    """Construct the model + TRL SFTTrainer. GPU-only imports inside.
    `base_model`/`prefer_unsloth` let the local smoke use a small bf16 model."""
    try:
        from datasets import load_dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        raise SystemExit(
            f"ABORT: training deps not available ({e}). Install: "
            "pip install -r training/requirements-train.txt")

    base_model = base_model or cfg["base_model"]
    model, tokenizer, backend = _load_model_and_tokenizer(cfg, base_model, prefer_unsloth)
    print(f"   model backend: {backend}  (base={base_model})")

    # adamw_8bit needs bitsandbytes; fall back to adamw_torch if unavailable
    optim = cfg["optim"]
    if "8bit" in optim:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            optim = "adamw_torch"
            print(f"   {cfg['optim']} needs bitsandbytes -> using adamw_torch")

    train_file = ROOT / cfg["train_file"]
    if not train_file.exists():
        raise SystemExit(f"ABORT: {train_file} not found — run "
                         "`python -m data.prepare_train` first.")
    ds = load_dataset("json", data_files=str(train_file), split="train")
    if len(ds) == 0:
        raise SystemExit(f"ABORT: {train_file} is empty — nothing to train on.")

    def render(batch):
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in batch["messages"]
        ]
        return {"text": texts}

    ds = ds.map(render, batched=True, remove_columns=ds.column_names)

    import inspect
    sft_kwargs = dict(
        output_dir=str(ROOT / cfg["output_dir"]),
        per_device_train_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_ratio=cfg["warmup_ratio"],
        num_train_epochs=cfg["epochs"] if max_steps is None else 1,
        max_steps=max_steps if max_steps is not None else -1,
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler"],
        weight_decay=cfg["weight_decay"],
        optim=optim,
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        seed=cfg["seed"],
        dataset_text_field="text",
        report_to=cfg.get("report_to", "none"),
        run_name=cfg.get("run_name", "qwen14b-cuad-qlora"),
    )
    # version-robust: trl renamed max_seq_length -> max_length in 1.x; and drop
    # any kwargs this installed trl/transformers version doesn't accept.
    accepted = set(inspect.signature(SFTConfig.__init__).parameters)
    if "max_length" in accepted:
        sft_kwargs["max_length"] = cfg["max_seq_length"]
    elif "max_seq_length" in accepted:
        sft_kwargs["max_seq_length"] = cfg["max_seq_length"]
    dropped = [k for k in list(sft_kwargs) if k not in accepted]
    for k in dropped:
        sft_kwargs.pop(k)
    if dropped:
        print(f"   (SFTConfig: this trl version ignores {dropped})")
    sft = SFTConfig(**sft_kwargs)
    try:  # trl >=0.13 renamed tokenizer -> processing_class
        trainer = SFTTrainer(model=model, processing_class=tokenizer,
                             train_dataset=ds, args=sft)
    except TypeError:
        trainer = SFTTrainer(model=model, tokenizer=tokenizer,
                             train_dataset=ds, args=sft)
    return trainer, model, tokenizer


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=None, help="override base_model")
    ap.add_argument("--max-steps", type=int, default=None, help="cap optimizer steps")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--output-dir", default=None, help="override output_dir")
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--no-unsloth", action="store_true",
                    help="force transformers+peft (portable / local)")
    ap.add_argument("--train-file", default=None, help="override train_file")
    ap.add_argument("--save-steps", type=int, default=None,
                    help="checkpoint interval (lower = safer on unstable boxes)")
    ap.add_argument("--save-total-limit", type=int, default=None,
                    help="how many checkpoints to keep (high = keep all for a curve)")
    args = ap.parse_args()

    cfg = dict(load_cfg())
    if args.save_steps:        cfg["save_steps"] = args.save_steps
    if args.save_total_limit:  cfg["save_total_limit"] = args.save_total_limit
    if args.train_file:    cfg["train_file"] = args.train_file
    if args.output_dir:    cfg["output_dir"] = args.output_dir
    if args.max_seq_length: cfg["max_seq_length"] = args.max_seq_length
    if args.batch_size:    cfg["per_device_batch_size"] = args.batch_size
    if args.grad_accum:    cfg["gradient_accumulation_steps"] = args.grad_accum
    if args.epochs:        cfg["epochs"] = args.epochs

    print("== train_qlora ==")
    base = args.base_model or cfg["base_model"]
    print(f"   base={base}  seq={cfg['max_seq_length']}  "
          f"epochs={cfg['epochs']}  max_steps={args.max_steps}  lr={cfg['learning_rate']}")
    gate_contamination()  # rule #1 — must pass

    trainer, model, tokenizer = build_trainer(
        cfg, max_steps=args.max_steps, base_model=args.base_model,
        prefer_unsloth=not args.no_unsloth)
    out = ROOT / cfg["output_dir"]
    ckpts = list(out.glob("checkpoint-*")) if out.exists() else []
    resume = bool(ckpts) and cfg.get("resume_from_checkpoint", True)
    print(f"   resume_from_checkpoint={resume} ({len(ckpts)} checkpoints found)")

    trainer.train(resume_from_checkpoint=resume)

    adapter_dir = out / "final_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nDONE. Adapter saved -> {adapter_dir}")
    print("   set backends.local.adapter_path to this dir in config/models.yaml (P2.4)")


if __name__ == "__main__":
    main()
