"""
verify_split.py — P2.2: PROVE zero train/test contamination (operating rule #1).

Three checks, all must pass or the script exits non-zero (so it can gate a
training run in CI / a launch script):

  1. Raw split disjointness: train.json contract titles vs test.json titles.
  2. Prepared-data provenance: every record in train_instruct.jsonl carries a
     `title`; assert NONE of those titles is a test contract.
  3. (defensive) assert the prepared data's titles are a subset of the train split.

If you cannot prove zero overlap, STOP and report (rule #1).

Run:  python -m data.verify_split
Exit: 0 = clean, 1 = contamination / missing files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "cuad"
TRAIN_JSON = DATA_DIR / "train.json"
TEST_JSON = DATA_DIR / "test.json"
TRAIN_INSTRUCT = DATA_DIR / "train_instruct.jsonl"


def _titles(squad_json: Path) -> set[str]:
    with open(squad_json, "r", encoding="utf-8") as f:
        return {c["title"] for c in json.load(f)["data"]}


def main(instruct_path: str | Path | None = None) -> int:
    """instruct_path: the prepared training file to audit — MUST be the file
    training will actually consume (train_qlora passes its cfg['train_file'])."""
    instruct = Path(instruct_path) if instruct_path else TRAIN_INSTRUCT
    if not (TRAIN_JSON.exists() and TEST_JSON.exists()):
        print("FAIL: run data/download_cuad.py first (train/test json missing)")
        return 1

    train_titles = _titles(TRAIN_JSON)
    test_titles = _titles(TEST_JSON)

    # --- check 1: raw split disjoint ---
    overlap = train_titles & test_titles
    print(f"[1] raw split: train={len(train_titles)} test={len(test_titles)} "
          f"overlap={len(overlap)}")
    if overlap:
        print(f"    CONTAMINATION: {sorted(overlap)[:5]} ...")
        return 1

    # --- check 2 & 3: prepared training data provenance ---
    if not instruct.exists():
        print(f"[2] {instruct.name} not found — run data/prepare_train.py "
              "before training. (split check above passed.)")
        return 0  # nothing prepared yet; split itself is clean

    prepared_titles, n = set(), 0
    with open(instruct, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            prepared_titles.add(json.loads(line)["title"])

    leaked = prepared_titles & test_titles
    print(f"[2] prepared data: {n} examples from {len(prepared_titles)} contracts; "
          f"test-leak={len(leaked)}")
    if leaked:
        print(f"    CONTAMINATION: test contracts in training data: {sorted(leaked)[:5]}")
        return 1

    not_in_train = prepared_titles - train_titles
    print(f"[3] provenance: prepared titles subset-of train split? "
          f"{'YES' if not not_in_train else 'NO'}")
    if not_in_train:
        print(f"    UNEXPECTED: titles not from train split: {sorted(not_in_train)[:5]}")
        return 1

    # --- check 4: dev split (checkpoint selection) must be disjoint from test
    #     AND from the prepared training data (we select on dev, so training on
    #     it would make the selection meaningless) ---
    dev_json = DATA_DIR / "dev.json"
    if dev_json.exists():
        dev_titles = _titles(dev_json)
        dev_test = dev_titles & test_titles
        dev_trained = dev_titles & prepared_titles
        print(f"[4] dev split: {len(dev_titles)} contracts; "
              f"dev&test={len(dev_test)} dev&prepared-train={len(dev_trained)}")
        if dev_test or dev_trained:
            print("    CONTAMINATION: dev overlaps "
                  + ("test " if dev_test else "") + ("training-data" if dev_trained else ""))
            return 1

    print("\nPASS: zero train/test contamination. Safe to train (rule #1).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
