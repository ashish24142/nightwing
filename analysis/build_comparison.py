"""
build_comparison.py — P2.5 (nightwing): pilot vs frontier, overall + per-category.

Reads every result JSON in results/baselines/ (frontier) and results/pilot/
(fine-tuned models), builds the comparison table, flags categories where the
pilot wins/matches (within 2 AUPR pts), computes gap-to-frontier, and classifies
the headline result against the FIXED, pre-committed Green/Yellow/Red bands.
The classification is surfaced for a HUMAN decision — never auto-acted on.

Run:  python -m analysis.build_comparison
Out:  results/comparison.md  +  results/comparison.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / "results" / "baselines"
PILOT_DIR = ROOT / "results" / "pilot"
DEGENERATE = {"Price Restrictions"}   # 0 positives in test -> AUPR undefined
MATCH_MARGIN = 0.02                   # "matches frontier" = within 2 AUPR pts


def _load_results(d: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(d.glob("*_test_full.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        out[r.get("model_id", p.stem)] = r
    return out


def main() -> None:
    frontier = _load_results(BASE_DIR)
    pilots = _load_results(PILOT_DIR)
    if not frontier:
        raise SystemExit("ABORT: no frontier baselines in results/baselines/.")

    best_frontier_overall = max(r["overall"]["aupr"] for r in frontier.values())
    models = {**frontier, **pilots}
    cats = sorted({c for r in models.values() for c in r["per_category"]}
                  - DEGENERATE)

    md = ["# nightwing — pilot vs frontier (official CUAD AUPR, full test split)\n"]
    md.append("## Overall\n")
    md.append("| Model | AUPR | P@80R | P@90R | gap to best frontier |")
    md.append("|---|---|---|---|---|")
    ranked = sorted(models.items(), key=lambda kv: -kv[1]["overall"]["aupr"])
    for name, r in ranked:
        o = r["overall"]
        gap = o["aupr"] - best_frontier_overall
        tag = " *(pilot)*" if name in pilots else ""
        md.append(f"| {name}{tag} | **{o['aupr']:.3f}** | "
                  f"{o['prec_at_80_recall']:.3f} | {o['prec_at_90_recall']:.3f} | "
                  f"{gap:+.3f} |")

    result = {"overall": {n: r["overall"] for n, r in models.items()},
              "best_frontier_aupr": best_frontier_overall}

    if pilots:
        # per-category: pilot vs best frontier, flag wins/matches
        md.append(f"\n## Per-category ({len(cats)} scorable; 'Price Restrictions' "
                  "excluded — 0 positives in test)\n")
        pilot_name, pilot = next(iter(pilots.items()))  # headline pilot
        wins, matches = [], []
        md.append(f"| Category | {pilot_name} | best frontier | (model) | Δ |")
        md.append("|---|---|---|---|---|")
        rows = []
        for c in cats:
            p_aupr = pilot["per_category"].get(c, {}).get("aupr", 0.0)
            f_scores = {n: r["per_category"].get(c, {}).get("aupr", 0.0)
                        for n, r in frontier.items()}
            f_best = max(f_scores.values())
            f_name = max(f_scores, key=f_scores.get)
            delta = p_aupr - f_best
            if delta > MATCH_MARGIN:
                wins.append(c)
            elif delta >= -MATCH_MARGIN:
                matches.append(c)
            rows.append((delta, c, p_aupr, f_best, f_name))
        for delta, c, p_aupr, f_best, f_name in sorted(rows, reverse=True):
            flag = " 🏆" if delta > MATCH_MARGIN else (" ≈" if delta >= -MATCH_MARGIN else "")
            md.append(f"| {c}{flag} | {p_aupr:.3f} | {f_best:.3f} | {f_name} | {delta:+.3f} |")

        gap_pts = (pilot["overall"]["aupr"] - best_frontier_overall) * 100
        md.append(f"\n**Pilot wins {len(wins)} categories, matches {len(matches)} "
                  f"(±{MATCH_MARGIN*100:.0f} pts). Overall gap: {gap_pts:+.1f} AUPR pts.**\n")

        # fixed §7 bands (presented for the HUMAN decision)
        if gap_pts >= -4 or len(wins) >= 5:
            band = "GREEN"
        elif gap_pts >= -8 and wins:
            band = "YELLOW"
        else:
            band = "RED"
        md.append(f"## Signal band (pre-committed thresholds): **{band}**\n")
        md.append("*Presented for the human funding decision — not auto-acted on. "
                  "A RED result is still a successful pilot outcome.*")
        result.update({"pilot": pilot_name, "wins": wins, "matches": matches,
                       "gap_pts": round(gap_pts, 2), "band": band})
    else:
        md.append("\n*(No pilot results in results/pilot/ yet — frontier baselines "
                  "only. Re-run after P2.4.)*")

    (ROOT / "results" / "comparison.md").write_text("\n".join(md) + "\n",
                                                    encoding="utf-8")
    (ROOT / "results" / "comparison.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print("\n".join(md[:14]))
    print(f"\nsaved -> results/comparison.md, results/comparison.json")


if __name__ == "__main__":
    main()
