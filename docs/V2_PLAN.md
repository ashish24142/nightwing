# nightwing v2 — Experiment Plan (pre-committed before spend)

**Question:** v1 showed a 14B *generative* fine-tune scores 0.291 on CUAD while
frontier tops out at 0.561. The interpretation was "task framing, not scale, is
the constraint." v2 tests that directly: same backbone family, same data, same
metric, but **extractive framing** (a QA span head instead of generation).

**Pre-registered prediction (written before the cloud run, already public in
the v1 essay): the 14B extractive model clears 0.50 test AUPR.**

## Evidence so far ($0, local RTX 5070 Ti 12GB)

| Model | Framing | Params | AUPR | Split | Epochs |
|---|---|---|---|---|---|
| nightwing-14b (v1) | generative | 14B | 0.291 | test | ~0.6 (dev-selected) |
| Qwen2.5-0.5B extractive | extractive | 0.5B | 0.271 | dev | 0.48 |
| Qwen2.5-0.5B extractive | extractive | 0.5B | **0.284** | dev | 0.96 |

A 0.5B extractive model matches the 14B generative pilot at 1/28th the
parameters, curve still rising at the step cap. Reference point: published
DeBERTa-xlarge (0.9B, extractive) scores ~0.47.

## Design (fixed)

- **Pipeline:** `training/train_extractive.py` + `harness/eval_extractive.py`
  (committed, validated end-to-end locally; caught two silent bugs before any
  cloud spend). Reuses transformers' QA head; scored by the unmodified official
  CUAD evaluator via `harness/scoring.py` — identical to every frontier row.
- **Data:** CUAD train split minus the 40 dev contracts (368 clean), sliding
  512-token windows (stride 128), negatives subsampled 2:1 per positive.
  Contamination gate asserted in code at every training launch (rule #1).
- **Selection:** checkpoint sweep on dev; best-dev checkpoint per size.
  **Test split touched exactly once**, by the final selected model only.
- **Backbones:** Qwen2.5-1.5B, Qwen2.5-7B, Qwen3-14B (matches v1's backbone).
  With the local 0.5B this gives a 4-point size-vs-AUPR curve.

## Run schedule (one A100 80GB spot pod, RunPod ~$1.52/hr)

| Stage | Est. GPU time |
|---|---|
| Setup + stage repo/data + contamination gate | ~0.5 h |
| Feature caches (per-tokenizer) | ~0.5 h |
| 1.5B train (2000 steps) + dev eval | ~2 h |
| 7B train + dev eval | ~4 h |
| 14B train + dev eval | ~6 h |
| Dev sweep + ONE test eval (best model; extractive eval is a single forward pass, not 11.5 h of generation) | ~1.5 h |
| **Total** | **~14–15 h ≈ $22–25** |

Realistic all-in **~$40–60** (setup, retries, idle); hard ceiling **$300** —
abort and reassess if crossed. Project total stays inside the original $1,000
cap (~$545 spent to date).

## Decision criteria (fixed)

- **Prediction resolves TRUE:** 14B extractive ≥ 0.50 test AUPR → framing
  thesis confirmed at scale; publish v2 results (journal, comparison, essay
  follow-up, HF release of the best adapter).
- **Prediction resolves FALSE:** publish that too, with the measured curve.
  A miss with a clean curve is still the honest, useful result.
- No threshold-moving after launch. Test is scored once.

## Out of scope (v3+ candidates, not this run)

Data augmentation, hard-negative mining beyond the 2:1 subsample, per-category
calibration, ensembles, multi-answer spans (current head predicts the single
best span per window; some CUAD questions have multiple gold spans — noted as
a known ceiling on recall-heavy categories).
