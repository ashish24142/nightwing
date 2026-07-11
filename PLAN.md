# nightwing — Publication Plan (open-source + HuggingFace release)

*Drafted 2026-07-11. Research-backed (HF competition scan, licensing verification,
signoff-repo style analysis, HF release best practices).*

## The opening (why this is worth publishing)

**Nobody has done this.** Verified on HuggingFace:
- Largest existing CUAD fine-tune is **8B** (Qwen3/Llama-3.1). A search for "cuad 14b" returns **zero results**.
- **None** of the modern LLM CUAD fine-tunes (1B–8B Llama/Qwen/Mistral adapters, 2023–2026) report the **official CUAD AUPR metric** — they report Macro-F1, win-rates, or nothing. Most-used CUAD models are still 2021-era RoBERTa.
- Nobody publishes frontier-model comparisons (Claude/GPT) on the official metric, and nobody ships auditable predictions JSON.

**Our claim:** *first 14B contract-clause specialist scored with the official CUAD
metric, head-to-head against frontier models (Claude Opus 4.8/4.6, GPT-5.2), fully
reproducible, at ~$0 marginal inference cost vs ~$165/run for frontier.*

The cost axis is the hook: frontier eval on 102 contracts cost us $165 (Claude 4.8);
a local specialist costs ~$0 after training (~$10 of spot GPU).

## Licensing — all green (verified)

| Item | License | Obligation |
|---|---|---|
| CUAD dataset | CC BY 4.0 | Attribute Atticus Project + cite Hendrycks et al. 2021 |
| Qwen2.5-**14B**-Instruct | Apache 2.0 (14B specifically; 3B/72B differ!) | Include license, note modification |
| Benchmarks vs Claude/GPT/Azure | Permitted | Publish methodology (we do anyway); **never train on frontier outputs** (we don't — CUAD ground truth only) |

## Name

**nightwing** — the sidekick who trained up to match the mentor: a fine-tuned open
model going head-to-head with frontier. (PyPI free; HF namespace clear of real
projects; unrelated-industry neighbor: Raytheon's cyber spinoff.)

## The three artifacts

### 1. GitHub repo (the signoff-style open-source asset)
Position like `signoff`: lean, hype-free, problem-first, "audit-ready" evidence
framing. The repo IS the eval harness + training pipeline — reusable by anyone
benchmarking legal LLMs.

Have: model-agnostic harness (official metric, caching, checkpointing, parallel),
contamination gate, portable QLoRA trainer, 3 committed frontier baselines,
44-test hardening suite, 41-category weakness analysis.

Need: Apache-2.0 LICENSE · README rewrite (signoff style: one-line value prop →
Install → Use → results table → Why → Roadmap) · GitHub Actions CI running the test
suite · sanitize check (creds never committed; .env gitignored — verify before push)
· `git init` history is clean (nothing committed yet — good) · publish predictions
JSONs (auditability differentiator).

### 2. HuggingFace release
- `nightwing-14b-cuad-lora` — **adapter repo** (primary artifact, peft)
- `nightwing-14b-cuad` — **merged bf16 repo** (needed for vLLM/GGUF users)
- optional `-GGUF` repo (Q4_K_M + Q8_0) — signals seriousness
- Model card: TL;DR → **Evaluation table** (us vs Claude 4.8 0.561 / Claude 4.6
  0.498 / GPT-5.2 0.423 vs published DeBERTa anchor, same metric/split/harness,
  commit-pinned) → usage code (peft + merged) → training details (QLoRA, 1×A100
  spot, wall-clock, contamination check) → intended use ("research/benchmark; NOT
  legal advice") → limitations → citations (ours + CUAD + Qwen)
- `model-index` YAML metadata (renders eval widget), `base_model_relation: adapter`
  (builds the Model Tree), `datasets: theatticusproject/cuad-qa`
- **Gradio Space on ZeroGPU** (free H200 slices): paste clause + 41-category
  dropdown + preloaded examples; `models:` metadata cross-links Space ↔ model
- **HF Collection** "nightwing" grouping all artifacts

### 3. Writeup
- migrationassurance.com article (companion to the signoff one): *"Can a $10
  fine-tune match a frontier model on contract review?"*
- Stretch: NLLP workshop @ EMNLP or arXiv preprint — needs the scaling curve (below)

## Phased execution

### Phase A — free, no GPU (Claude Code does now)
1. **Fix confidence granularity** — P@80/90 = 0 everywhere is the credibility hole:
   single per-answer confidence → coarse PR curve. Move to graded confidence so the
   official prec-at-recall sweep is meaningful. Re-score existing frontier
   predictions (free — predictions are saved; re-run scoring only if format allows,
   else note as limitation).
2. **Build `analysis/build_comparison.py` (P2.5)** — pilot vs frontier table generator.
3. **Dev split** — carve dev set from *train* (never test) for checkpoint selection.
4. **Repo release prep** — LICENSE, README rewrite, CI workflow, sanitize audit, first commit.

### Phase B — cloud GPU (~$40–80, A100 80GB spot) — "maximize win probability" config
5. Train **Qwen3-14B** (fallback Qwen2.5-14B) with **bf16 LoRA, not QLoRA** — 80GB
   fits it; removes quantization noise (~1–2 AUPR pts). 4–5 epochs, checkpoint
   every ~250 steps, **select best checkpoint on dev** (carved from train), then
   eval on test ONCE. ~4–5 hr + eval.
6. **Scaling story (recommended, +~$15):** also train 1.5B → 7B; one figure:
   *AUPR vs model size vs cost* with frontier lines overlaid. This figure is the paper.
7. 2 seeds on the 14B if budget allows (variance bars).

**Why we have a real shot at beating 0.561 (not hype):** CUAD scores by Jaccard≥0.5
against expert span conventions. Zero-shot frontier models can't know the annotation
style; a fine-tune learns it (a 125M RoBERTa scores ~45 AUPR — near frontier's
0.42–0.56 under this harness). Plus a structural advantage frontier APIs can't match:
**logprob-based confidence** from the local model → smooth calibrated PR curve
(fixes our P@80/90=0 hole for the pilot, which frontier predictions inherently have).
**Integrity line (fixed):** tune only on dev; test touched once per model; results
published win or lose.

### Phase C — release (free)
8. Merge adapter → upload adapter + merged (+ GGUF) with full model card.
9. Gradio ZeroGPU Space + Collection.
10. GitHub repo public; predictions JSONs committed.

### Phase D — writeup
11. Blog post. 12. Optional workshop paper (needs step 6's curve).

## Budget

Spent $493.89 of $1,000. Phase B needs ~$15–40 GPU (scaling story +$15).
Everything else is free. Total stays well under cap.

## Honest-result policy (unchanged from CLAUDE.md)

If the 14B lands RED (>8–10 pts below frontier), we publish anyway — as an honest
negative result with the cost analysis ("what $10 of fine-tuning does NOT get you").
Negative results with clean methodology are still credible content; goalposts do not move.
