# nightwing v1 — Run Journal

**Experiment:** Can a fine-tuned open 14B match frontier LLMs on CUAD contract-clause
extraction, on the official metric, at ~$0 marginal inference cost?
**Dates:** 2026-07-12 → 2026-07-13 (single RunPod A100 SXM 80GB pod, ~30 h wall-clock)
**Verdict:** RED band (−27.0 AUPR pts overall vs best frontier), **2 outright category
wins** held against every frontier model. Honest negative result, published as
pre-committed.

---

## 1. Design (fixed before the run)

- **Model:** Qwen3-14B (Apache-2.0) + bf16 LoRA (r=32, α=64, dropout 0.05, all
  attention+MLP projections). *Not* QLoRA — 80 GB removed the need to quantize.
- **Data:** CUAD train split only (408 contracts → 368 train + 40-contract dev
  carve-out), 8000-char sliding windows (2000 overlap), instruction format identical
  to the eval prompt. 33,244 examples (35.4% positive); 47 over-length examples
  (0.1%) dropped rather than silently truncating their answer targets.
- **Selection protocol:** checkpoint every 250 steps; sweep on the dev split;
  best-dev checkpoint goes to test. **Test split touched exactly once.**
- **Metric:** official CUAD evaluator (AUPR / precision-at-recall), unmodified.
- **Contamination:** `verify_split.py` gates every training launch — train/test
  overlap 0, dev∩test 0, dev∩train-data 0, asserted in code each time.
- **Decision bands** (pre-committed): GREEN within 3–4 pts of frontier, YELLOW
  5–8 pts below, RED >8–10 pts below.

Frontier baselines (measured earlier through the *identical* harness, full
102-contract test split): claude-opus-4-8 **0.561**, claude-opus-4-6 **0.498**,
gpt-5.2 **0.423**. Frontier eval cost: ~$165 / ~$154 / ~$13 per run respectively.

## 2. Environment

| | |
|---|---|
| Pod | RunPod Secure Cloud, 1× A100 SXM 80 GB, 24 vCPU, 117 GB RAM, 210 GB disk, $1.52/hr all-in |
| Image | runpod/pytorch 2.4 template (py3.11, cuda 12.4); torch 2.10.0+cu128 at run time |
| Stack | transformers + peft + trl (bf16 LoRA path); **no Unsloth** (see §3) |
| Orchestration | `scripts/cloud_run.sh` (stages 1–4) + `scripts/stage5plus.sh` (stages 5–7), nohup + SSH monitoring, crash-resumable at every stage |

## 3. What broke and what fixed it (engineering log)

Four failures, all in the dependency ecosystem, none in the experiment code:

1. **datasets 4.x dropped script-datasets** → `theatticusproject/cuad-qa` unloadable
   on the pod (training stack forced datasets ≥4; re-pin lost the resolver fight).
   *Fix:* uploaded the locally-verified data files directly (350 MB SCP) + added a
   parquet-branch fallback to `download_cuad.py` for public users.
2. **trl `<EOS_TOKEN>` sentinel crash** at trainer construction. Passing the real
   `tokenizer.eos_token` explicitly didn't clear it —
3. — because the sentinel came from **Unsloth's tokenizer patching** clashing with
   this trl version. *Fix:* removed Unsloth entirely; the portable
   transformers+peft path (validated end-to-end locally before the run) took over.
   Cost: ~1.5× slower steps. Two runs were lost to this pair of failures (~45 min).
4. **Throughput reality check:** measured 24 s/step (vs ~16 hoped) → the planned
   8,300 steps (4 epochs) would have taken 34+ hr. *Fix:* capped at **2,000 steps
   (~0.96 epoch)** and let the dev sweep decide empirically whether to extend.
   The dev curve later *vindicated the cap* — it peaked at step 1250 (0.6 epoch)
   and declined after.

Operational lessons that made it survivable: per-contract checkpointing with
resume everywhere (a killed 13-hr process costs ≤100 min); progress must be
unbuffered (`python -u`) or a silent pipeline is indistinguishable from a dead
one; `pkill -f` patterns must bracket-escape or they kill their own SSH session;
SCP of changed files beat `git pull` on an unreliable pod network.

## 4. Training

| | |
|---|---|
| Steps | 2,000 (effective batch 16 → ~32k examples ≈ 0.96 epoch) |
| Wall-clock | 13.1 h @ ~23.9 s/step, GPU ~100%, 51–73 GB VRAM |
| Loss | 1.77 (step 5) → 0.98 (205) → 0.62 (605) → 0.31 (905) → ~0.25–0.34 plateau (1005–1205) |
| Checkpoints | every 250 steps, all kept (752 MB each incl. optimizer) |

## 5. Dev sweep (checkpoint selection — 8 dev contracts, never test)

| Checkpoint | ~epoch | Dev AUPR |
|---|---|---|
| 250 | 0.12 | 0.2308 |
| **1250** | **0.60** | **0.2547** ← selected |
| 2000 | 0.96 | 0.2367 |
| final_adapter | 0.96 | 0.2367 |

Clean overfitting peak at ~0.6 epoch. More steps on this data would not have
helped; the binding constraint is data/framing, not capacity or duration.

## 6. Final test result (102 contracts, 4,182 questions, touched once)

| Model | AUPR | Gap |
|---|---|---|
| claude-opus-4-8 | **0.561** | — |
| claude-opus-4-6 | 0.498 | −6.3 |
| gpt-5.2 | 0.423 | −13.8 |
| **nightwing-14b (ckpt-1250)** | **0.291** | **−27.0** |

- **Band: RED** (pre-committed thresholds). Eval integrity: **0 question errors,
  0 failed contracts** across all 4,182 questions; 11.5 h generative eval.
- **Category wins (beats ALL three frontier models):**
  - **Agreement Date: 0.687** vs best-frontier 0.168 (+0.52; GPT-5.2 scores 0.054)
  - **Effective Date: 0.369** vs best-frontier 0.105 (+0.26)
- **Head-to-head win counts:** vs gpt-5.2 **11**/40 categories; vs claude-opus-4-6
  **3**/40; vs claude-opus-4-8 **3**/40.
- Near-misses (within 0.1 of best frontier): Volume Restriction −0.03,
  Post-Termination Services −0.05, Non-Transferable License −0.06, Irrevocable Or
  Perpetual License −0.07, Expiration Date −0.08, Minimum Commitment −0.09.
- Worst collapses: Parties 0.247 vs 0.954; Source Code Escrow 0.100 vs 0.800;
  Covenant Not To Sue 0.107 vs 0.642.
- Full table: [`results/comparison.md`](../results/comparison.md). Raw predictions
  committed (`results/pilot/local_test_full_predictions.json`) — every number is
  re-scorable by anyone.

## 7. Interpretation

The wins and losses split on one axis: **whether the category is decided by
learned annotation convention or by semantic reasoning.** Date/identity fields are
convention: the fine-tune learned exactly which span CUAD annotators mark, which
zero-shot frontier models cannot infer (they answer "correctly" with different
boundaries and Jaccard≥0.5 punishes them). Reasoning-heavy or rare categories
(Parties, Source Code Escrow, Covenant Not To Sue) exposed the recipe's limits:
generative span-copying + one epoch of naive data.

Context that reframes the RED band: DeBERTa-xlarge (0.9B, extractive head) scores
~0.47 published on CUAD. Parameter count is not the constraint — task framing is.
A 14B backbone with an extractive span head is the credible path past 0.561
(planned as experiment v2).

## 8. Cost accounting

| Item | Spend |
|---|---|
| Frontier baselines + smokes (API, incl. one $136 crash loss) | $493.94 |
| GPU pod (~33 h × $1.52: train 13.1 h, sweep 2.1 h, test eval 11.5 h, setup/babysitting/idle rest) | ~$50 |
| **Project total** | **~$545 of the $1,000 cap** |
| This training run alone (train+sweep+eval share) | ~$41 |

## 9. Artifacts

- `runpod_artifacts/` (local, gitignored): full run logs, all 8 checkpoints +
  final adapter (~6.4 GB, byte-verified against the pod), results.
- Repo (public): pilot scores + predictions, dev curve, comparison, this journal.
- Adapter for release: `checkpoint-1250` (752 MB; adapter weights ~514 MB).

## 10. What v2 should be (decided ahead of spend)

1. **Extractive span head on the 14B backbone** — the framing fix; ~$300 lean /
   ~$800 full including a 0.9B/7B/14B scaling curve. ~90–95% of spend is GPU;
   data prep is code-only (CUAD ships char offsets).
2. Cheap adjuncts regardless of path: per-category confidence calibration,
   window-aggregation tuning, rare-category oversampling, hard-negative mining
   (dev-driven, test touched once at the end, published as v2 alongside v1).
