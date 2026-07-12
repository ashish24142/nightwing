"""
download_cuad.py — P1.1: fetch CUAD and write it to data/cuad/ in the OFFICIAL
SQuAD-nested format (the format the official CUAD evaluate.py consumes).

Source: HuggingFace `theatticusproject/cuad-qa` — the canonical CUAD QA release.
Its train/test split IS the official CUAD split (408 / 102 contracts, 510 total).

Outputs (all gitignored):
    data/cuad/train.json        SQuAD-nested, 408 contracts
    data/cuad/test.json         SQuAD-nested, 102 contracts
    data/cuad/manifest.json     verification numbers (contracts, overlap, 41 cats)

Run:  python -m data.download_cuad
"""
from __future__ import annotations
import json
import os
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

HF_DATASET = "theatticusproject/cuad-qa"
OUT_DIR = Path(__file__).resolve().parent / "cuad"
CATEGORY_RE = re.compile(r'related to "(.*?)"')


def category_of(question: str) -> str:
    """The 41 CUAD categories are the quoted clause name in each question."""
    m = CATEGORY_RE.search(question)
    return m.group(1) if m else "UNKNOWN"


def to_squad_nested(split) -> dict:
    """Flat HF rows -> SQuAD-nested: one entry per contract, context stored once."""
    by_contract: dict[str, dict] = {}
    order: list[str] = []
    for ex in split:
        title = ex["title"]
        if title not in by_contract:
            by_contract[title] = {"context": ex["context"], "qas": []}
            order.append(title)
        ans = ex["answers"]
        by_contract[title]["qas"].append(
            {
                "id": ex["id"],
                "question": ex["question"],
                "answers": [
                    {"text": t, "answer_start": s}
                    for t, s in zip(ans["text"], ans["answer_start"])
                ],
                "is_impossible": len(ans["text"]) == 0,
            }
        )
    data = [
        {"title": t, "paragraphs": [{"context": by_contract[t]["context"],
                                     "qas": by_contract[t]["qas"]}]}
        for t in order
    ]
    return {"version": "cuad-v1-from-hf", "data": data}


def stats(split) -> dict:
    titles = list(dict.fromkeys(split["title"]))
    cats = sorted({category_of(q) for q in split["question"]})
    qpc = defaultdict(int)
    for t in split["title"]:
        qpc[t] += 1
    empty = sum(1 for a in split["answers"] if len(a["text"]) == 0)
    return {
        "qa_pairs": len(split),
        "contracts": len(titles),
        "categories": len(cats),
        "q_per_contract_min": min(qpc.values()),
        "q_per_contract_max": max(qpc.values()),
        "empty_answer_frac": round(empty / len(split), 4),
        "category_names": cats,
        "contract_titles": titles,
    }


def main() -> None:
    from datasets import load_dataset

    print(f"Loading {HF_DATASET} ...")
    try:
        ds = load_dataset(HF_DATASET, trust_remote_code=True)
    except (RuntimeError, TypeError, ValueError):
        # datasets>=4 dropped script-based datasets; use HF's auto-converted
        # parquet branch instead (same splits/fields, no script needed)
        print("  (script dataset unsupported on this datasets version -> parquet branch)")
        ds = load_dataset(HF_DATASET, revision="refs/convert/parquet")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tr, te = stats(ds["train"]), stats(ds["test"])
    overlap = sorted(set(tr["contract_titles"]) & set(te["contract_titles"]))

    for name, split in (("train", ds["train"]), ("test", ds["test"])):
        path = OUT_DIR / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_squad_nested(split), f, ensure_ascii=False)
        print(f"  wrote {path}  ({(path.stat().st_size/1e6):.1f} MB)")

    manifest = {
        "source": HF_DATASET,
        "split_is_official": True,
        "train": {k: v for k, v in tr.items() if k != "contract_titles"},
        "test": {k: v for k, v in te.items() if k != "contract_titles"},
        "contract_overlap_count": len(overlap),
        "contract_overlap": overlap,
        "total_contracts": tr["contracts"] + te["contracts"],
    }
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n=== VERIFICATION ===")
    print(f"  contracts: train={tr['contracts']} test={te['contracts']} "
          f"total={manifest['total_contracts']}  (expect 408/102/510)")
    print(f"  contract overlap: {len(overlap)}  (MUST be 0)")
    print(f"  categories: train={tr['categories']} test={te['categories']}  (expect 41)")
    print(f"  q/contract test: {te['q_per_contract_min']}..{te['q_per_contract_max']}")
    print(f"  test empty-answer frac: {te['empty_answer_frac']}")

    assert len(overlap) == 0, "TRAIN/TEST CONTRACT OVERLAP DETECTED — STOP."
    assert tr["categories"] == 41 and te["categories"] == 41, "Expected 41 categories."
    print("\nOK: split clean, 41 categories. Manifest -> data/cuad/manifest.json")


if __name__ == "__main__":
    main()
