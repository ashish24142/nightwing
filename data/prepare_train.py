"""
prepare_train.py — P2.2: CUAD TRAIN split -> windowed instruction dataset.

For each (train contract, category question), slide windows over the contract
(harness/windowing.py) and emit one instruction example per window:
  - target JSON  {"present": true,  "spans": [<gold spans in window>], "confidence": 1.0}
                 {"present": false, "spans": [],                        "confidence": 0.0}
The prompt format MIRRORS the eval harness (harness/prompt.py) so the fine-tune
learns exactly the response contract the harness scores.

ONLY the train split is touched (rule #1). Each record carries its source
`title` so data/verify_split.py can prove no test contract leaked in.

Negatives (windows with no gold span) vastly outnumber positives, so they are
subsampled to `--neg-per-pos` per contract-question; total capped by --max-examples.

Run:  python -m data.prepare_train
Out:  data/cuad/train_instruct.jsonl   (gitignored)
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from harness import prompt as P
from harness.windowing import iter_windows, spans_in_window

DATA_DIR = Path(__file__).resolve().parent / "cuad"
CATEGORY_RE = re.compile(r'related to "(.*?)"')


def _category(q: str) -> str:
    m = CATEGORY_RE.search(q)
    return m.group(1) if m else "UNKNOWN"


def _example(window_text: str, question: str, category: str,
             spans: list[str], title: str) -> dict:
    """One chat-format training example mirroring the eval prompt."""
    user = (P.build_contract_block(window_text) + "\n\n"
            + P.build_question_text(question, category))
    target = {
        "present": bool(spans),
        "spans": spans,
        "confidence": 1.0 if spans else 0.0,
    }
    return {
        "title": title,                 # provenance for verify_split.py
        "category": category,
        "messages": [
            {"role": "system", "content": P.SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-per-pos", type=float, default=1.0,
                    help="max negative windows kept per positive (per contract-question)")
    ap.add_argument("--max-examples", type=int, default=40000,
                    help="hard cap on total examples (pilot-sized)")
    ap.add_argument("--win-chars", type=int, default=8000)
    ap.add_argument("--overlap-chars", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(DATA_DIR / "train_instruct.jsonl"))
    ap.add_argument("--dev-contracts", type=int, default=40,
                    help="contracts held out of training as a DEV split for "
                         "checkpoint selection (carved from TRAIN, never test). "
                         "0 disables.")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    train_json = DATA_DIR / "train.json"
    if not train_json.exists():
        raise SystemExit(f"ABORT: {train_json} not found — run "
                         "`python -m data.download_cuad` first.")
    with open(train_json, "r", encoding="utf-8") as f:
        data = json.load(f).get("data")
    if not data:
        raise SystemExit(f"ABORT: {train_json} has no contracts (empty/corrupt).")

    # ---- dev carve-out: whole contracts (prevents window leakage), seeded ----
    if args.dev_contracts > 0:
        titles = sorted(c["title"] for c in data)
        dev_titles = set(rng.sample(titles, args.dev_contracts))
        dev_data = [c for c in data if c["title"] in dev_titles]
        data = [c for c in data if c["title"] not in dev_titles]
        dev_path = DATA_DIR / "dev.json"
        with open(dev_path, "w", encoding="utf-8") as f:
            json.dump({"version": "cuad-dev-from-train", "data": dev_data}, f,
                      ensure_ascii=False)
        print(f"dev split: {len(dev_data)} contracts -> {dev_path} "
              f"(training uses remaining {len(data)})")

    pos, neg = [], []
    for contract in data:
        title = contract["title"]
        for para in contract["paragraphs"]:
            context = para["context"]
            windows = iter_windows(context, args.win_chars, args.overlap_chars)
            for qa in para["qas"]:
                cat = _category(qa["question"])
                answers = qa.get("answers", [])
                contract_pos, contract_neg = [], []
                for w in windows:
                    spans = spans_in_window(answers, w)
                    ex = _example(w.text, qa["question"], cat, spans, title)
                    (contract_pos if spans else contract_neg).append(ex)
                pos.extend(contract_pos)
                # keep at most neg_per_pos * (#positives) negatives for this Q;
                # if no positives, keep a single negative so 'absent' is learned
                keep = max(1, int(round(args.neg_per_pos * len(contract_pos))))
                rng.shuffle(contract_neg)
                neg.extend(contract_neg[:keep])

    rng.shuffle(neg)
    examples = pos + neg
    rng.shuffle(examples)
    if len(examples) > args.max_examples:
        examples = examples[:args.max_examples]

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    n_pos = sum(1 for e in examples if json.loads(e["messages"][2]["content"])["present"])
    print(f"wrote {len(examples)} examples -> {out_path}")
    print(f"  positives={n_pos}  negatives={len(examples)-n_pos}  "
          f"(pos rate {n_pos/len(examples):.1%})")
    print(f"  source contracts={len(data)} (TRAIN split only)")
    print(f"  window={args.win_chars} chars, overlap={args.overlap_chars}")
    print("  next: python -m data.verify_split")


if __name__ == "__main__":
    main()
