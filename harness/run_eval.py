"""
run_eval.py — model-agnostic CUAD evaluation harness entry point (P1.3).

Pipeline:  load test contracts + 41 questions each
        -> call the configured backend (config/models.yaml `active`, or --backend)
        -> collect official nbest predictions
        -> score with the official CUAD metric (harness/scoring.py)
        -> track tokens + $ and extrapolate per-contract cost (rule #5)
        -> save predictions + scores to results/baselines/

Examples:
    python -m harness.run_eval --backend claude --limit 5     # 5-contract smoke
    python -m harness.run_eval --backend claude --split test  # full baseline
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import scoring
from .backends.base import Question, build_backend, load_config
from .cost_tracker import CostTracker

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "cuad"
RESULTS_DIR = ROOT / "results" / "baselines"
CATEGORY_RE = re.compile(r'related to "(.*?)"')
FULL_TEST_CONTRACTS = 102


def _category(question: str) -> str:
    m = CATEGORY_RE.search(question)
    return m.group(1) if m else "UNKNOWN"


def _load_contracts(split: str, limit: int | None):
    path = DATA_DIR / f"{split}.json"
    if not path.exists():
        raise SystemExit(f"ABORT: {path} not found — run `python -m data.download_cuad` first.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f).get("data")
    if not data:
        raise SystemExit(f"ABORT: {path} has no contracts (empty/corrupt).")
    if limit:
        data = data[:limit]
    contracts = []
    for entry in data:
        para = entry["paragraphs"][0]
        questions = [
            Question(qid=qa["id"], question=qa["question"],
                     category=_category(qa["question"]))
            for qa in para["qas"]
        ]
        contracts.append((entry["title"], para["context"], questions))
    return contracts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None,
                    help="backend name in models.yaml (default: its `active`)")
    ap.add_argument("--split", default="test", choices=["test", "train", "dev"])
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only first N contracts (smoke test)")
    ap.add_argument("--workers", type=int, default=1,
                    help="process N contracts concurrently (each keeps its 41 "
                         "questions sequential, preserving prompt caching)")
    ap.add_argument("--out", default=None, help="output JSON path override")
    args = ap.parse_args()

    args.workers = max(1, min(args.workers, 32))  # clamp to a sane range
    config = load_config()
    backend_name = args.backend or config["active"]
    is_smoke = args.limit is not None
    label = f"{backend_name}_{args.split}" + (f"_smoke{args.limit}" if is_smoke else "_full")
    print(f"== run_eval: backend={backend_name} split={args.split} "
          f"limit={args.limit or 'ALL'} ==")

    backend = build_backend(backend_name, config)
    print(f"   model_id={backend.model_id}")

    contracts = _load_contracts(args.split, args.limit)
    n = len(contracts)
    n_q = sum(len(q) for _, _, q in contracts)
    print(f"   {n} contracts, {n_q} questions")

    pricing_id = getattr(backend, "pricing_model_id", backend.model_id)
    ct = CostTracker(pricing_id, config.get("pricing", {}))
    nbest: dict = {}

    # --- crash-safe checkpoint: one JSONL line per completed contract.
    # Resume skips done contracts so a killed run never loses paid work (rule #6).
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = RESULTS_DIR / f"{label}.ckpt.jsonl"
    done_titles: set = set()
    if ckpt_path.exists():
        bad = 0
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:  # tolerate a truncated/corrupt last line from a crash
                    rec = json.loads(line)
                    title, nb, u = rec["title"], rec["nbest"], rec["usage"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    bad += 1
                    continue
                done_titles.add(title)
                nbest.update(nb)
                ct.add(u.get("input_tokens", 0), u.get("cache_write_tokens", 0),
                       u.get("cache_read_tokens", 0), u.get("output_tokens", 0))
        print(f"   RESUME: {len(done_titles)} contracts from checkpoint "
              f"(${ct.cost_usd:.2f} already spent"
              + (f"; skipped {bad} corrupt line(s)" if bad else "") + ")")

    todo = [(t, txt, qs) for (t, txt, qs) in contracts if t not in done_titles]
    lock = threading.Lock()
    abort = threading.Event()
    state = {"done": len(done_titles), "ok": 0, "qerr": 0, "failed": []}
    t0 = time.time()

    def process(item):
        title, text, questions = item
        if abort.is_set():
            return
        try:
            r = backend.predict_contract(text, questions)  # 41 seq -> caching intact
        except Exception as e:  # contract-level failure (setup, not per-question)
            with lock:
                state["failed"].append((title, str(e)[:140]))
                if len(state["failed"]) >= 5 and state["ok"] == 0:
                    abort.set()
            return
        full_fail = r.errors >= len(questions)
        preds = sum(1 for q in questions if r.nbest.get(q.qid))
        with lock:
            if full_fail:
                # whole contract failed -> do NOT checkpoint (so resume retries it)
                state["failed"].append((title, f"all {r.errors} questions failed"))
                if len(state["failed"]) >= 5 and state["ok"] == 0:
                    abort.set()  # systemic (bad key/endpoint/quota) -> stop the run
                return
            nbest.update(r.nbest)
            ct.add(r.usage.input_tokens, r.usage.cache_write_tokens,
                   r.usage.cache_read_tokens, r.usage.output_tokens)
            with open(ckpt_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"title": title, "nbest": r.nbest, "usage": {
                    "input_tokens": r.usage.input_tokens,
                    "cache_write_tokens": r.usage.cache_write_tokens,
                    "cache_read_tokens": r.usage.cache_read_tokens,
                    "output_tokens": r.usage.output_tokens}}) + "\n")
                f.flush()
            state["done"] += 1
            state["ok"] += 1
            state["qerr"] += r.errors
            tag = f"  (!{r.errors} q-err)" if r.errors else ""
            print(f"   [{state['done']}/{n}] {title[:45]:45s}  spans={preds:2d}{tag}  "
                  f"${ct.cost_usd:.3f}  {time.time()-t0:.0f}s")

    if args.workers > 1:
        print(f"   concurrency: {args.workers} contracts in parallel")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process, it) for it in todo]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:  # never let one worker kill the whole run
                    with lock:
                        state["failed"].append(("<worker>", str(e)[:140]))
    else:
        for it in todo:
            process(it)
            if abort.is_set():
                break
    elapsed = time.time() - t0

    # systemic failure -> refuse to score/commit a garbage baseline (rule #7)
    if todo and state["ok"] == 0:
        first = state["failed"][0][1] if state["failed"] else "unknown"
        raise SystemExit(
            f"ABORT: 0/{len(todo)} contracts succeeded (likely systemic: bad "
            f"key/endpoint/quota). Nothing scored or committed. First error: {first}")
    if not nbest:
        raise SystemExit("ABORT: no predictions to score.")
    if state["failed"]:
        print(f"   WARNING: {len(state['failed'])} contract(s) failed and were "
              f"NOT checkpointed (re-run to retry): "
              f"{[t for t, _ in state['failed']][:5]}")

    # -- score (restrict GT to evaluated qids) --
    gt_full = scoring.load_gt(DATA_DIR / f"{args.split}.json")
    unknown = [qid for qid in nbest if qid not in gt_full]
    if unknown:  # defensive: drop any prediction whose qid isn't in ground truth
        print(f"   WARNING: {len(unknown)} prediction qids not in ground truth "
              f"(dropped): {unknown[:3]}")
        for qid in unknown:
            nbest.pop(qid, None)
    gt = {qid: gt_full[qid] for qid in nbest}
    overall = scoring.score_overall(nbest, gt)
    cats = sorted({_category(q.question) for _, _, qs in contracts for q in qs})
    per_cat = scoring.score_per_category(nbest, gt, cats)

    # -- cost summary + extrapolation --
    cost = ct.summary()
    per_contract = ct.per_unit(n)
    extrapolated_full = round(per_contract * FULL_TEST_CONTRACTS, 2)
    ledger = ct.commit_to_ledger(label)

    result = {
        "label": label,
        "backend": backend_name,
        "model_id": backend.model_id,
        "split": args.split,
        "n_contracts": n,
        "n_contracts_scored": len(nbest) // 41 if n_q else len(nbest),
        "n_questions": n_q,
        "question_errors": state["qerr"],
        "failed_contracts": [t for t, _ in state["failed"]],
        "elapsed_sec": round(elapsed, 1),
        "overall": overall,
        "per_category": per_cat,
        "cost": cost,
        "cost_per_contract_usd": round(per_contract, 4),
        "extrapolated_full_test_usd": extrapolated_full,
        "cumulative_spend_usd": ledger["cumulative_usd"],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    # save raw predictions alongside (gitignored if large)
    with open(RESULTS_DIR / f"{label}_predictions.json", "w", encoding="utf-8") as f:
        json.dump(nbest, f)
    # final results are durable now -> checkpoint no longer needed
    ckpt_path.unlink(missing_ok=True)

    # -- report --
    print("\n=== RESULT ===")
    print(f"  overall AUPR={overall['aupr']:.3f}  "
          f"P@80={overall['prec_at_80_recall']:.3f}  "
          f"P@90={overall['prec_at_90_recall']:.3f}")
    print(f"  cost this run: ${cost['cost_usd']:.4f}  "
          f"({cost['calls']} calls, in={cost['input_tokens']} "
          f"cache_w={cost['cache_write_tokens']} cache_r={cost['cache_read_tokens']} "
          f"out={cost['output_tokens']}, cache_hit={cost['cache_hit_rate']:.0%})")
    print(f"  per-contract: ${per_contract:.4f}")
    if not cost["pricing_known"]:
        print("  WARNING: pricing for this model_id not in models.yaml -> $ is 0; add it.")
    if is_smoke:
        print(f"  >>> EXTRAPOLATED full {FULL_TEST_CONTRACTS}-contract cost: "
              f"${extrapolated_full}  (review before full run — rule #5)")
    print(f"  cumulative project spend: ${ledger['cumulative_usd']}")
    if ledger["alert"]:
        print(f"  *** ALERT: cumulative spend >= ${ledger['threshold']} (rule #4) ***")
    print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
