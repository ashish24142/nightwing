"""
calibrate.py — v3: per-category Platt calibration of span confidences.

Raw extractive confidence sigmoid(span - null) is comparable within a
category but not across categories, and overall AUPR ranks predictions
across ALL categories on one confidence axis. Fix: fit a per-category
logistic map on DEV predictions (label = official Jaccard >= 0.5 match
against gold, via the unmodified official helper), freeze it, apply to test.
Per-category AUPR is invariant to this (monotone within category); only the
cross-category ranking, and therefore overall AUPR, changes.

  python -m analysis.calibrate fit   --preds DEVPRED --gold data/cuad/dev.json  --out params.json
  python -m analysis.calibrate apply --preds TESTPRED --params params.json     --out CALPRED
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import scoring
from harness.scoring import _OFF, qid_category   # official jaccard, unmodified

MIN_POINTS = 10        # per class (pos and neg) to fit a category; else identity
CLIP = 1e-6


def _logit(p: float) -> float:
    p = min(1 - CLIP, max(CLIP, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def fit(preds_path: str, gold_json: str, out_path: str) -> None:
    preds = json.loads(Path(preds_path).read_text(encoding="utf-8"))
    gt = scoring.load_gt(gold_json)
    from sklearn.linear_model import LogisticRegression

    by_cat: dict[str, list[tuple[float, int]]] = {}
    for qid, entries in preds.items():
        golds = gt.get(qid, [])
        for e in entries:
            if not e["text"]:
                continue
            correct = int(any(_OFF.get_jaccard(g, e["text"]) >= 0.5 for g in golds))
            by_cat.setdefault(qid_category(qid), []).append(
                (_logit(e["probability"]), correct))

    params: dict[str, list[float]] = {}
    for cat, pts in sorted(by_cat.items()):
        pos = sum(c for _, c in pts)
        neg = len(pts) - pos
        if pos < MIN_POINTS or neg < MIN_POINTS:
            continue  # identity fallback
        X = [[x] for x, _ in pts]
        y = [c for _, c in pts]
        lr = LogisticRegression(C=1.0).fit(X, y)
        params[cat] = [float(lr.coef_[0][0]), float(lr.intercept_[0])]

    Path(out_path).write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"[fit] {len(params)} categories calibrated "
          f"({len(by_cat) - len(params)} identity fallback) -> {out_path}")


def apply(preds_path: str, params_path: str, out_path: str) -> None:
    preds = json.loads(Path(preds_path).read_text(encoding="utf-8"))
    params = json.loads(Path(params_path).read_text(encoding="utf-8"))
    n = 0
    for qid, entries in preds.items():
        ab = params.get(qid_category(qid))
        if not ab:
            continue
        a, b = ab
        for e in entries:
            if e["text"]:
                e["probability"] = _sigmoid(a * _logit(e["probability"]) + b)
                n += 1
    Path(out_path).write_text(json.dumps(preds, indent=2), encoding="utf-8")
    print(f"[apply] recalibrated {n} predictions -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--preds", required=True)
    f.add_argument("--gold", required=True)
    f.add_argument("--out", required=True)
    a = sub.add_parser("apply")
    a.add_argument("--preds", required=True)
    a.add_argument("--params", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "fit":
        fit(args.preds, args.gold, args.out)
    else:
        apply(args.preds, args.params, args.out)


if __name__ == "__main__":
    main()
