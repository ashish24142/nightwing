"""
eval_checkpoints.py — AUPR-vs-training-steps curve for the local experiment.

Each saved checkpoint (outputs/local3b/checkpoint-50, -100, ...) is a snapshot of
the LoRA adapter at that step. We evaluate each on the SAME small set of test
contracts (windowed inference + official AUPR), producing a tuning curve that
shows whether more steps -> higher AUPR.

NOTE: run AFTER training finishes — a single 12 GB GPU can't train and eval at
once. Evaluating snapshots post-hoc gives the identical curve to live eval.

Run:  python -m training.eval_checkpoints --contracts 6
Out:  results/pilot/local3b_curve.json
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
    ap.add_argument("--contracts", type=int, default=6)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ckpt-dir", default="outputs/local15b")
    args = ap.parse_args()
    BASE = args.base
    CKPT_DIR = ROOT / args.ckpt_dir
    OUT = ROOT / "results" / "pilot" / (Path(args.ckpt_dir).name + "_curve.json")

    ckpts = _checkpoints(CKPT_DIR)
    if not ckpts:
        raise SystemExit(f"no checkpoints in {CKPT_DIR} — train first.")
    print(f"found {len(ckpts)} checkpoints: "
          f"{[s if s < 10**9 else 'final' for s, _ in ckpts]}")

    contracts = _load_contracts("test", args.contracts)
    gt_full = scoring.load_gt(ROOT / "data" / "cuad" / "test.json")
    import torch

    curve = []
    for steps, path in ckpts:
        label = "final" if steps >= 10**9 else str(steps)
        print(f"\n== checkpoint {label} ({path.name}) ==")
        backend = LocalBackend({
            "base_model": BASE, "adapter_path": str(path),
            "max_new_tokens": args.max_new_tokens, "max_seq_length": 2048,
            "win_chars": 4000, "overlap_chars": 800, "gen_batch_size": args.batch,
            "load_in_4bit": False,
        })
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
        del backend
        gc.collect()
        torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"base": BASE, "n_contracts": args.contracts,
                               "curve": curve}, indent=2), encoding="utf-8")
    print("\n=== AUPR vs STEPS (local 3B, secondary experiment) ===")
    for pt in curve:
        bar = "#" * int(pt["aupr"] * 50)
        print(f"  steps {pt['steps']:>5}:  AUPR {pt['aupr']:.4f}  {bar}")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
