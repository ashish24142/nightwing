"""
local_smoke.py — full-pipeline smoke on a LOCAL GPU with a tiny proxy model.

NOT the real pilot (that needs the cloud A100/H100 + Qwen-14B). This validates
the CODE end-to-end on a 12 GB laptop GPU using Qwen2.5-0.5B in bf16:
    contamination gate -> QLoRA train (few steps) -> save adapter
        -> LocalBackend windowed inference -> official scoring
If this passes, the pipeline is correct; only scale changes on the cloud box.

Run:  python -m training.local_smoke
"""
from __future__ import annotations

import gc

from harness import scoring
from harness.backends.local_model import LocalBackend
from harness.run_eval import _load_contracts
from training.train_qlora import ROOT, build_trainer, gate_contamination, load_cfg

SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # ~1 GB bf16 — fits 12 GB easily
ADAPTER = ROOT / "outputs" / "smoke_adapter"
STEPS = 20
EVAL_CONTRACTS = 1


def main() -> None:
    cfg = dict(load_cfg())
    cfg["max_seq_length"] = 2048          # smaller for speed on the laptop
    cfg["output_dir"] = "outputs/smoke_train"
    cfg["per_device_batch_size"] = 1
    cfg["gradient_accumulation_steps"] = 4

    print("== LOCAL PIPELINE SMOKE ==")
    gate_contamination()  # rule #1

    # subset the data so tokenization is fast (smoke only needs a few hundred)
    full = ROOT / cfg["train_file"]
    subset = full.parent / "train_instruct_smoke.jsonl"
    with open(full, encoding="utf-8") as fin:
        lines = [next(fin) for _ in range(800)]
    subset.write_text("".join(lines), encoding="utf-8")
    cfg["train_file"] = str(subset.relative_to(ROOT))
    print(f"   using {len(lines)}-example subset for the smoke")

    print(f"\n[1/3] train {STEPS} steps of {SMOKE_MODEL} (bf16 LoRA)...")
    trainer, model, tok = build_trainer(
        cfg, max_steps=STEPS, base_model=SMOKE_MODEL, prefer_unsloth=False)
    out = trainer.train()
    loss = float(out.training_loss)
    print(f"   train loss={loss:.4f}")
    assert loss == loss and loss < 50, f"bad loss {loss}"
    model.save_pretrained(str(ADAPTER))
    tok.save_pretrained(str(ADAPTER))
    print(f"   adapter saved -> {ADAPTER}")

    # free training memory before loading the eval copy
    del trainer, model
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n[2/3] windowed inference on {EVAL_CONTRACTS} test contract(s)...")
    backend = LocalBackend({
        "base_model": SMOKE_MODEL, "adapter_path": str(ADAPTER),
        "max_new_tokens": 128, "max_seq_length": 2048,
        "win_chars": 8000, "overlap_chars": 2000,
    })
    nbest: dict = {}
    for title, text, qs in _load_contracts("test", EVAL_CONTRACTS):
        res = backend.predict_contract(text, qs)
        nbest.update(res.nbest)
        preds = sum(1 for q in qs if res.nbest.get(q.qid))
        print(f"   {title[:40]:40s} qerrors={res.errors} spans-predicted={preds}")

    print("\n[3/3] score via official metric...")
    gt_full = scoring.load_gt(ROOT / "data" / "cuad" / "test.json")
    gt = {qid: gt_full[qid] for qid in nbest}
    overall = scoring.score_overall(nbest, gt)
    print(f"   overall (1 contract, tiny model — value is meaningless): {overall}")
    print("\nPASS: train -> adapter -> windowed eval -> official score ran end-to-end.")
    print("   The pipeline is correct; the cloud run only changes model + scale.")


if __name__ == "__main__":
    main()
