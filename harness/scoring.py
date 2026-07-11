"""
scoring.py — P1.2: thin wrapper over the OFFICIAL CUAD metric.

We do NOT reimplement the metric (operating rule #2). We import the unmodified
official `cuad_official_evaluate.py` (fetched from TheAtticusProject/cuad) and
call its functions: AUPR + precision@80%/90% recall, Jaccard>=0.5 matching.

Prediction format (the official "nbest") that every backend must produce:
    { qid: [ {"text": <span>, "probability": <float 0..1>}, ... ] }
The metric sweeps a confidence threshold over `probability` to trace the PR curve.
For "clause absent", emit [{"text": "", "probability": ...}] (empty text is
ignored by the metric) or an empty list.

Usage:
    from harness import scoring
    gt = scoring.load_gt("data/cuad/test.json")
    res = scoring.score_overall(pred_nbest, gt)            # {aupr, prec_at_80_recall, prec_at_90_recall}
    per = scoring.score_per_category(pred_nbest, gt)       # {category: {...}}

Self-check (DoD): score gold-as-prediction -> perfect.
    python -m harness.scoring --selfcheck
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OFFICIAL_PATH = _HERE / "cuad_official_evaluate.py"
CATEGORY_RE = re.compile(r'related to "(.*?)"')


@contextlib.contextmanager
def _cwd(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _load_official():
    """Import the official module. It reads category_descriptions.csv at import
    time via a relative path, so import from within harness/."""
    if not _OFFICIAL_PATH.exists():
        raise FileNotFoundError(
            f"official scorer missing: {_OFFICIAL_PATH.name}. Re-fetch from "
            "TheAtticusProject/cuad (see README P1.2).")
    if not (_HERE / "category_descriptions.csv").exists():
        raise FileNotFoundError(
            "category_descriptions.csv missing next to the official scorer "
            "(it is read at import time). Re-fetch from TheAtticusProject/cuad.")
    spec = importlib.util.spec_from_file_location("cuad_official_evaluate", _OFFICIAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    with _cwd(_HERE):
        spec.loader.exec_module(mod)
    return mod


_OFF = _load_official()


# ---------------------------------------------------------------------------
# Ground truth + categories
# ---------------------------------------------------------------------------
def _load_test_json(test_json_path: str | Path) -> dict:
    path = Path(test_json_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m data.download_cuad` first.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "data" not in data:
        raise ValueError(f"{path} is not CUAD SQuAD-nested JSON (no 'data' key).")
    return data


def load_gt(test_json_path: str | Path) -> dict:
    """{qid: [answer_text, ...]} via the official get_answers()."""
    return _OFF.get_answers(_load_test_json(test_json_path))


def categories(test_json_path: str | Path) -> list[str]:
    """The 41 category names, taken from the question text (canonical)."""
    data = _load_test_json(test_json_path)["data"]
    cats = set()
    for contract in data:
        for para in contract["paragraphs"]:
            for qa in para["qas"]:
                m = CATEGORY_RE.search(qa["question"])
                if m:
                    cats.add(m.group(1))
    return sorted(cats)


# ---------------------------------------------------------------------------
# Scoring (delegates entirely to official functions)
# ---------------------------------------------------------------------------
def _score(pred_nbest: dict, gt_dict: dict, category: str | None) -> dict:
    precisions, recalls, confs = _OFF.get_precisions_recalls(
        pred_nbest, gt_dict, category=category
    )
    aupr = _OFF.get_aupr(precisions, recalls)
    p80, _ = _OFF.get_prec_at_recall(precisions, recalls, confs, recall_thresh=0.8)
    p90, _ = _OFF.get_prec_at_recall(precisions, recalls, confs, recall_thresh=0.9)
    return {
        "aupr": float(aupr),
        "prec_at_80_recall": float(p80),
        "prec_at_90_recall": float(p90),
    }


def score_overall(pred_nbest: dict, gt_dict: dict) -> dict:
    # official get_results asserts identical key sets; mirror that safety check
    assert sorted(pred_nbest.keys()) == sorted(gt_dict.keys()), (
        "prediction qids must exactly match ground-truth qids"
    )
    return _score(pred_nbest, gt_dict, category=None)


def score_per_category(pred_nbest: dict, gt_dict: dict,
                       category_names: list[str] | None = None) -> dict:
    """Per-category AUPR/prec@recall. The official compute_precision_recall
    filters qids by `category in qid`, so we pass each category name through."""
    if category_names is None:
        # derive from qids: '<title>__<Category>_<idx>'
        cats = sorted({re.sub(r"_\d+$", "", k.split("__", 1)[1]) for k in gt_dict})
    else:
        cats = category_names
    return {c: _score(pred_nbest, gt_dict, category=c) for c in cats}


def selfcheck_nbest_from_gt(test_json_path: str | Path) -> dict:
    """Build a perfect prediction file: each gold answer at probability 1.0."""
    gt = load_gt(test_json_path)
    nbest = {}
    for qid, answers in gt.items():
        if answers:
            nbest[qid] = [{"text": a, "probability": 1.0} for a in answers]
        else:
            nbest[qid] = [{"text": "", "probability": 1.0}]
    return nbest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _selfcheck(test_json_path: str) -> None:
    print(f"Self-check: scoring ground truth against itself ({test_json_path})")
    gt = load_gt(test_json_path)
    pred = selfcheck_nbest_from_gt(test_json_path)
    overall = score_overall(pred, gt)
    print(f"  overall: {overall}")
    assert overall["aupr"] > 0.999, f"AUPR not ~1.0: {overall['aupr']}"
    assert overall["prec_at_80_recall"] > 0.999
    assert overall["prec_at_90_recall"] > 0.999
    # Per-category: a category with ZERO positive gold answers in test has an
    # undefined PR curve (recall never defined) -> AUPR is degenerate, NOT 1.0.
    # Assert perfection only for categories that actually have positives.
    cats = categories(test_json_path)
    per = score_per_category(pred, gt, cats)
    positives = {c: 0 for c in cats}
    for qid, ans in gt.items():
        c = re.sub(r"_\d+$", "", qid.split("__", 1)[1])
        if ans and c in positives:
            positives[c] += 1
    no_positives = sorted(c for c in cats if positives[c] == 0)
    scorable = [c for c in cats if positives[c] > 0]
    bad = {c: per[c] for c in scorable if per[c]["aupr"] < 0.999}
    print(f"  categories: {len(cats)} | scorable (>=1 positive): {len(scorable)} | "
          f"no positives in test: {no_positives}")
    assert not bad, f"scorable categories that did NOT score perfectly: {list(bad)[:5]}"
    print("\nPASS: official metric wired; GT-vs-GT = perfect (AUPR=1.0) overall and "
          "for all 40 scorable categories. ('Price Restrictions' has 0 positives in "
          "test -> excluded from per-category comparison.)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true",
                    help="score gold-as-prediction; assert perfect score")
    ap.add_argument("--test-json", default=str(_HERE.parent / "data" / "cuad" / "test.json"))
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck(args.test_json)
    else:
        ap.print_help()
