# nightwing, Project State

Snapshot date: 2026-07-20. One page: what exists, what it cost, what is open.

## Headline results (official CUAD AUPR, full 102-contract test split)

| Model | AUPR | Cost/run |
|---|---|---|
| claude-opus-4-8 | 0.561 | ~$165 |
| claude-opus-4-6 | 0.498 | ~$154 |
| gpt-5.2 | 0.423 | ~$13 |
| gpt-4o | 0.421 | ~$58 |
| nightwing-v2-14b (extractive) | 0.389 | ~$0 after ~$32 training |
| nightwing-v1-14b (generative) | 0.291 | ~$0 after ~$41 training |

Key findings, in one line each:
1. The frontier gap is category-shaped: v2 beats gpt-4o in 25/40 categories
   and gpt-5.2 in 22/40 while trailing overall.
2. Framing beats scale: generation -> extraction gained +9.8 pts at fixed
   budget; a 28x parameter sweep gained +1.9 dev pts; gpt-4o and gpt-5.2 tie
   a generation apart.
3. Recipes do not transfer naively: an LR swept on a 0.5B proxy (won by
   0.0004) cost the 14B seventeen dev points at epoch 1 (v3 finding).
4. Both pre-registered 0.50 predictions missed and are published as misses;
   the test split was touched exactly once per shipped model and never by v3.

## What is published where

- GitHub (github.com/ashish24142/nightwing): harness, training pipelines,
  all raw predictions for every model, run journals v1/v2/v3, full 6x40
  comparison grid (results/comparison.md), this file.
- Website (migrationassurance.com): two essays (the v1 loss, the v2 sequel),
  cross-linked homepage.
- Hugging Face (ashish24142/nightwing-14b-cuad-extractive): v2 adapter +
  honest model card with the required backbone-loading recipe.
- arXiv: paper source READY in paper/ (compiles clean, 6 pp), NOT yet
  submitted; needs the author's arXiv account (see paper/SUBMIT_ARXIV.md).

## Money

| Phase | Spend |
|---|---|
| v1 (frontier baselines + generative pilot) | ~$545 |
| v2 (extractive + size curve + test) | ~$32 |
| v3 pod session (LR sweep, failed recipe, idle, v3.1 stub) | ~$43 |
| **Total** | **~$620 of the $1,000 cap** |

RunPod: NO pods exist (terminated 2026-07-20); balance ~$5 remains on the
account. Azure API keys live in .env (gitignored), unused since the gpt-4o
baseline.

## Open items

1. **arXiv upload** (author action, $0): create account at arxiv.org, follow
   paper/SUBMIT_ARXIV.md. Endorsement for cs.CL can take 1-2 days.
2. **v3.1, postponed by decision**: 3 epochs + neg 4:1 at lr 2e-4, the clean
   epochs-and-negatives test. Resume: fresh A100 SXM 80GB pod, clone repo,
   scp data/cuad/*.json, run scripts/v3_1_cloud.sh (~19 h, ~$40; account
   needs a top-up first). Full steps: docs/RUN_JOURNAL_V3.md.
3. **NLLP @ EMNLP 2026 submission** (deadline Aug 11, OpenReview): port
   paper/ to the ACL template + add the v3/v3.1 recipe section. Needs v3.1
   resumed by ~Aug 1 for comfortable margin, or submit without v3.1 using
   the v3 LR-transfer finding as the closing section.
4. Optional: Hugging Face demo Space (ZeroGPU); Google Search Console
   registration for the site.

## Local assets (this laptop, not in git)

- runpod_artifacts/: v1 checkpoints + adapters, v2 final adapters (3 sizes),
  v3 run logs. The v2 14B adapter is also on Hugging Face (its backup).
- Obsidian vault (Downloads/claude_notes): running project notes + activity
  log, current through this snapshot.
- data/cuad/: verified CUAD splits (md5s in data manifest).

## How to resume work in a new session

Read this file, docs/RUN_JOURNAL_V3.md, and the Obsidian Nightwing note.
The eval harness is config-only swappable (config/models.yaml); training any
new model is three commands (README, "Fine-tune a different model").
