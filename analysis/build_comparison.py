"""
build_comparison.py — the full grid: every model, every category, one table.

Reads every result JSON in results/baselines/ (frontier), results/pilot/
(v1 generative fine-tune) and results/pilot_v2/ (v2 extractive fine-tune),
and builds THE comparison: overall ranking + a wide per-category table with
the best score per category bolded, head-to-head win counts, and the FIXED
pre-committed Green/Yellow/Red band for the v1 pilot decision (historical,
never auto-acted on — surfaced for a HUMAN).

Run:  python -m analysis.build_comparison
Out:  results/comparison.md  +  results/comparison.json
"""
from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.scoring import DEGENERATE_CATEGORIES as DEGENERATE

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / "results" / "baselines"
PILOT_DIR = ROOT / "results" / "pilot"
PILOT_V2_DIR = ROOT / "results" / "pilot_v2"
MATCH_MARGIN = 0.02                   # "matches" = within 2 AUPR pts

# Fixed pre-committed bands (v1 pilot decision; chosen BEFORE results existed).
GREEN_GAP_PTS = -4.0
GREEN_WINS = 5
YELLOW_GAP_PTS = -8.0
YELLOW_MATCHES = 3

DISPLAY = {  # normalise raw model_ids to the published names
    "local:Qwen3-14B+checkpoint-1250": "nightwing-v1-14b (generative)",
}


def _load_dir(d: Path, pattern: str) -> dict[str, dict]:
    out = {}
    for p in sorted(d.glob(pattern)):
        if p.stem.endswith("_predictions"):
            continue
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        name = r.get("model_id", p.stem)
        out[DISPLAY.get(name, name)] = r
    return out


def main() -> None:
    frontier = _load_dir(BASE_DIR, "*_test_full.json")
    v1 = _load_dir(PILOT_DIR, "*_test_full.json")
    v2 = _load_dir(PILOT_V2_DIR, "test_*.json")
    v2 = {f"nightwing-v2-{k.replace('test_', '')} (extractive)": r
          for k, r in v2.items()} if any("model_id" not in r for r in v2.values()) else v2
    if not frontier:
        raise SystemExit("ABORT: no frontier baselines in results/baselines/.")

    specialists = {**v1, **v2}
    models = {**frontier, **specialists}
    best_frontier_overall = max(r["overall"]["aupr"] for r in frontier.values())
    cats = sorted({c for r in models.values() for c in r["per_category"]}
                  - DEGENERATE)

    md = ["# nightwing — every model, every category "
          "(official CUAD AUPR, full 102-contract test split)\n"]

    # ---- overall ranking -------------------------------------------------
    md.append("## Overall\n")
    md.append("| Model | AUPR | P@80R | P@90R | gap to best frontier |")
    md.append("|---|---|---|---|---|")
    ranked = sorted(models.items(), key=lambda kv: -kv[1]["overall"]["aupr"])
    for name, r in ranked:
        o = r["overall"]
        gap = o["aupr"] - best_frontier_overall
        md.append(f"| {name} | **{o['aupr']:.3f}** | "
                  f"{o['prec_at_80_recall']:.3f} | {o['prec_at_90_recall']:.3f} | "
                  f"{gap:+.3f} |")

    # ---- the full grid ---------------------------------------------------
    order = [n for n, _ in ranked]      # columns in overall-rank order
    md.append(f"\n## Per-category — the full grid ({len(cats)} scorable; "
              "'Price Restrictions' excluded — 0 positives in test)\n")
    md.append("Best score per category in **bold**.\n")
    md.append("| Category | " + " | ".join(order) + " |")
    md.append("|---|" + "---|" * len(order))

    def cell(name: str, c: str) -> float:
        return models[name]["per_category"].get(c, {}).get("aupr", 0.0)

    for c in cats:
        vals = {n: cell(n, c) for n in order}
        best = max(vals.values())
        row = [f"**{v:.3f}**" if v == best and best > 0 else f"{v:.3f}"
               for v in (vals[n] for n in order)]
        md.append(f"| {c} | " + " | ".join(row) + " |")

    # ---- head-to-head win counts ----------------------------------------
    md.append("\n## Head-to-head: categories won by each specialist\n")
    md.append("| Specialist | " + " | ".join(f"vs {n}" for n in frontier)
              + " | beats ALL frontier |")
    md.append("|---|" + "---|" * (len(frontier) + 1))
    result_h2h = {}
    for sname, sr in specialists.items():
        wins = {fn: sum(1 for c in cats
                        if cell(sname, c) > cell(fn, c))
                for fn in frontier}
        beats_all = [c for c in cats
                     if cell(sname, c) > max(cell(fn, c) for fn in frontier)]
        md.append(f"| {sname} | "
                  + " | ".join(f"{wins[fn]}/{len(cats)}" for fn in frontier)
                  + f" | {len(beats_all)} |")
        result_h2h[sname] = {"wins": wins, "beats_all": beats_all}

    # ---- pre-committed band (v1 pilot decision — historical) -------------
    result = {"overall": {n: r["overall"] for n, r in models.items()},
              "best_frontier_aupr": best_frontier_overall,
              "head_to_head": result_h2h}
    if v1:
        pilot_name, pilot = next(iter(v1.items()))
        wins, matches = [], []
        for c in cats:
            delta = cell(pilot_name, c) - max(cell(fn, c) for fn in frontier)
            if delta > MATCH_MARGIN:
                wins.append(c)
            elif delta >= -MATCH_MARGIN:
                matches.append(c)
        gap_pts = (pilot["overall"]["aupr"] - best_frontier_overall) * 100
        if gap_pts >= GREEN_GAP_PTS or len(wins) >= GREEN_WINS:
            band = "GREEN"
        elif gap_pts >= YELLOW_GAP_PTS and (wins or len(matches) >= YELLOW_MATCHES):
            band = "YELLOW"
        else:
            band = "RED"
        md.append(f"\n## Signal band (pre-committed thresholds, v1 pilot "
                  f"decision): **{band}**\n")
        md.append(f"*v1 gap {gap_pts:+.1f} pts, {len(wins)} outright wins. "
                  "Presented for the human funding decision — not auto-acted on. "
                  "A RED result is still a successful pilot outcome. "
                  "v2 (extractive framing, same budget) closed the gap to "
                  f"{(next(iter(v2.values()))['overall']['aupr'] - best_frontier_overall) * 100:+.1f} pts — "
                  "see docs/RUN_JOURNAL_V2.md.*" if v2 else "*…*")
        result.update({"pilot": pilot_name, "wins": wins, "matches": matches,
                       "gap_pts": round(gap_pts, 2), "band": band})

    (ROOT / "results" / "comparison.md").write_text("\n".join(md) + "\n",
                                                    encoding="utf-8")
    (ROOT / "results" / "comparison.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print("\n".join(md[:12]).encode("ascii", "replace").decode("ascii"))
    print("\nsaved -> results/comparison.md, results/comparison.json")


if __name__ == "__main__":
    main()
