"""
eval_checkpoints.py — AUPR-vs-training-steps curve for the local experiment.

Each saved checkpoint is a snapshot of the LoRA adapter at that step. We
evaluate each on the SAME set of DEV contracts (never test — checkpoint
selection on test would leak it into model selection), producing the curve
that picks the best checkpoint. Test is then touched ONCE via run_eval.

NOTE: run AFTER training finishes — a single 12 GB GPU can't train and eval at
once. Evaluating snapshots post-hoc gives the identical curve to live eval.

Run:  python -m training.eval_checkpoints
Out:  results/pilot/<ckpt-dir-name>_curve.json
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

from harness import scoring
from harness.backends.local_model import LocalBackend
from harness.run_eval import _load_contracts
from training.train_qlora import ROOT

def _checkpoints(CKPT_DIR: Path) -> list[tuple[int, Path]]:
    found = []
    for d in CKPT_DIR.glob("checkpoint-*"):
        m = re.search(r"checkpoint-(\d+)$", d.name)
        if m and (d / "adapter_config.json").exists():
            found.append((int(m.group(1)), d))
    fa = CKPT_DIR / "final_adapter"
    if fa.exists() and (fa / "adapter_config.json").exists():
        # treat final as the highest step (use a big number so it sorts last)
        found.append((10**9, fa))
    return sorted(found)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contracts", type=int, default=20)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--every", type=int, default=4,
                    help="evaluate every Nth checkpoint (always keeps the last "
                         "numbered one + final_adapter); full sweep of 30+ "
                         "checkpoints would take ~10 GPU-hours for no extra signal")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--base", default="Qwen/Qwen3-14B")
    ap.add_argument("--ckpt-dir", default="outputs/qwen14b-cuad-lora")
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--win-chars", type=int, default=8000,
                    help="MUST match prepare_train --win-chars")
    ap.add_argument("--overlap-chars", type=int, default=2000)
    args = ap.parse_args()
    BASE = args.base
    CKPT_DIR = ROOT / args.ckpt_dir
    OUT = ROOT / "results" / "pilot" / (Path(args.ckpt_dir).name + "_curve.json")

    ckpts = _checkpoints(CKPT_DIR)
    if not ckpts:
        raise SystemExit(f"no checkpoints in {CKPT_DIR} — train first.")
    if args.every > 1:
        numbered = [c for c in ckpts if c[0] < 10**9]
        finals = [c for c in ckpts if c[0] >= 10**9]
        kept = numbered[::args.every]
        if numbered and numbered[-1] not in kept:
            kept.append(numbered[-1])
        ckpts = sorted(kept) + finals
    print(f"sweeping {len(ckpts)} checkpoints (every {args.every}th): "
          f"{[s if s < 10**9 else 'final' for s, _ in ckpts]}")

    # DEV split ONLY (carved from train by prepare_train). Checkpoint selection
    # on test would leak the test set into model selection — test is touched
    # ONCE, by run_eval --split test, after the checkpoint is chosen here.
    dev_json = ROOT / "data" / "cuad" / "dev.json"
    if not dev_json.exists():
        raise SystemExit("ABORT: data/cuad/dev.json missing — run "
                         "`python -m data.prepare_train` (it carves the dev split).")
    contracts = _load_contracts("dev", args.contracts)
    gt_full = scoring.load_gt(dev_json)
    import torch

    curve = []
    backend = None
    for steps, path in ckpts:
        label = "final" if steps >= 10**9 else str(steps)
        print(f"\n== checkpoint {label} ({path.name}) ==")
        if backend is None:
            backend = LocalBackend({
                "base_model": BASE, "adapter_path": str(path),
                "max_new_tokens": args.max_new_tokens,
                "max_seq_length": args.max_seq_length,
                "win_chars": args.win_chars, "overlap_chars": args.overlap_chars,
                "gen_batch_size": args.batch, "load_in_4bit": False,
            })
        else:
            # swap ONLY the adapter — reloading the 14B base per checkpoint
            # would waste ~an hour of billed GPU time across a full sweep
            backend.model.load_adapter(str(path), adapter_name=label)
            backend.model.set_adapter(label)
        nbest: dict = {}
        for title, text, qs in contracts:
            r = backend.predict_contract(text, qs)
            nbest.update(r.nbest)
            preds = sum(1 for q in qs if r.nbest.get(q.qid))
            print(f"   {title[:38]:38s} spans={preds:2d} qerr={r.errors}")
        gt = {qid: gt_full[qid] for qid in nbest}
        overall = scoring.score_overall(nbest, gt)
        curve.append({"steps": label, "aupr": overall["aupr"],
                      "prec_at_80": overall["prec_at_80_recall"]})
        print(f"   -> AUPR={overall['aupr']:.4f}")
        gc.collect()
        torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"base": BASE, "n_contracts": args.contracts,
                               "curve": curve}, indent=2), encoding="utf-8")
    print("\n=== AUPR vs STEPS (DEV split - checkpoint selection) ===")
    for pt in curve:
        bar = "#" * int(pt["aupr"] * 50)
        print(f"  steps {pt['steps']:>5}:  AUPR {pt['aupr']:.4f}  {bar}")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
