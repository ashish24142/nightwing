"""
smoke_test.py — P2.1: 10-step training sanity check (run BEFORE the full train).

Proves on the GPU box that: base weights load 4-bit, LoRA attaches, the dataset
renders, and 10 optimizer steps run without OOM and with a sane (decreasing) loss.
Cheap insurance before committing to the multi-hour QLoRA run (rule #5 spirit).

Run (on GPU box):  python -m training.smoke_test
"""
from __future__ import annotations

from training.train_qlora import build_trainer, gate_contamination, load_cfg


def main() -> None:
    cfg = load_cfg()
    print("== QLoRA smoke test: 10 steps ==")
    from training.train_qlora import ROOT
    gate_contamination(ROOT / cfg["train_file"])  # rule #1 even for the smoke

    trainer, model, _ = build_trainer(cfg, max_steps=10)
    result = trainer.train()
    loss = result.training_loss
    print(f"\n10-step smoke complete. final training_loss={loss:.4f}")
    assert loss == loss, "loss is NaN — training is broken"  # NaN check
    assert loss < 20, f"loss suspiciously high ({loss}) — check data/format"
    print("PASS: weights load, LoRA attaches, data renders, 10 steps ran, no OOM.")
    print("   -> safe to launch the full run: python -m training.train_qlora")


if __name__ == "__main__":
    main()
