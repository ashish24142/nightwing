"""
category_breakdown.py — P1.6: per-category frontier weakness analysis.

Builds the 41-category x frontier-model AUPR table from the saved baselines and
ranks categories by how WEAK the best frontier model is on each — those are the
categories with the most room for a 14B specialist to win in Phase 2.

Excludes 'Price Restrictions' (0 positive gold answers in the test split -> AUPR
undefined; see harness/scoring.py).

Run:  python -m analysis.category_breakdown
Outputs:
    results/baselines/category_breakdown.json   (committed)
    results/category_breakdown.md               (committed, human-readable)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "baselines"
DEGENERATE = {"Price Restrictions"}  # 0 positives in test -> not scorable

# the three frontier baselines (label -> file)
BASELINES = {
    "claude-opus-4-8": BASE / "claude48_test_full.json",
    "claude-opus-4-6": BASE / "claude46_test_full.json",
    "gpt-5.2": BASE / "gpt_test_full.json",
}


def _load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    results = {}
    for name, p in BASELINES.items():
        if p.exists():
            results[name] = _load(p)
        else:
            print(f"  (skipping {name}: {p.name} not found)")
    if not results:
        raise SystemExit("ABORT: no baseline result files found in results/baselines/ "
                         "— run the frontier baselines first.")
    models = list(results)

    # union of categories across baselines, minus degenerate
    cats = set()
    for r in results.values():
        cats.update(r["per_category"].keys())
    cats = sorted(c for c in cats if c not in DEGENERATE)

    rows = []
    for c in cats:
        auprs = {m: results[m]["per_category"].get(c, {}).get("aupr", 0.0)
                 for m in models}
        best = max(auprs.values())
        best_model = max(auprs, key=auprs.get)
        rows.append({
            "category": c,
            "aupr": auprs,
            "frontier_best": round(best, 3),
            "frontier_best_model": best_model,
            "frontier_mean": round(sum(auprs.values()) / len(models), 3),
        })

    # winnable = weakest frontier-best AUPR (most headroom for a specialist)
    rows.sort(key=lambda x: x["frontier_best"])
    top10 = rows[:10]

    out = {
        "models": models,
        "overall": {m: results[m]["overall"]["aupr"] for m in models},
        "n_categories_scored": len(cats),
        "excluded": sorted(DEGENERATE),
        "categories": rows,
        "top10_winnable": [r["category"] for r in top10],
    }
    (BASE / "category_breakdown.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    # ---- markdown ----
    md = ["# CUAD per-category frontier weakness (P1.6)\n",
          f"Overall AUPR — " +
          " · ".join(f"**{m}** {results[m]['overall']['aupr']:.3f}" for m in models)
          + "\n",
          f"{len(cats)} categories scored ( 'Price Restrictions' excluded: "
          "0 positives in test ).\n",
          "## Top-10 winnable categories (weakest frontier — most room for the pilot)\n",
          "| # | Category | best frontier AUPR | (by) | mean |",
          "|---|---|---|---|---|"]
    for i, r in enumerate(top10, 1):
        md.append(f"| {i} | {r['category']} | {r['frontier_best']:.3f} | "
                  f"{r['frontier_best_model']} | {r['frontier_mean']:.3f} |")
    md.append("\n## Full 41-category table (sorted weakest-frontier first)\n")
    md.append("| Category | " + " | ".join(models) + " | best | mean |")
    md.append("|---|" + "---|" * (len(models) + 2))
    for r in rows:
        md.append(f"| {r['category']} | "
                  + " | ".join(f"{r['aupr'][m]:.3f}" for m in models)
                  + f" | **{r['frontier_best']:.3f}** | {r['frontier_mean']:.3f} |")
    (ROOT / "results" / "category_breakdown.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")

    # ---- console ----
    print("Overall AUPR:",
          "  ".join(f"{m}={results[m]['overall']['aupr']:.3f}" for m in models))
    print(f"\nTop-10 WINNABLE categories (weakest best-frontier AUPR):")
    for i, r in enumerate(top10, 1):
        print(f"  {i:2d}. {r['category']:32s} best={r['frontier_best']:.3f} "
              f"({r['frontier_best_model']})  mean={r['frontier_mean']:.3f}")
    n_zero = sum(1 for r in rows if r["frontier_best"] == 0.0)
    print(f"\n  ({n_zero} categories where ALL frontier models score AUPR=0)")
    print("\nsaved -> results/baselines/category_breakdown.json, "
          "results/category_breakdown.md")


if __name__ == "__main__":
    main()
