# nightwing v2, Run Journal

**Experiment:** v1 showed a generative 14B fine-tune scores 0.291 on CUAD vs frontier's
0.561. The diagnosis was "task framing, not scale, is the constraint." v2 tested that
directly: identical backbone family, identical data, identical metric, but an
**extractive span head** instead of generation.
**Dates:** 2026-07-16 -> 2026-07-18 (local RTX 5070 Ti smoke + curve start; one RunPod
A100 SXM 80GB pod for the 1.5B/7B/14B curve and the single test eval)
**Pre-registered prediction (public before spend): 14B extractive clears 0.50 test
AUPR. Result: MISS, 0.3885.** Published as committed.
**But the framing thesis held emphatically: +9.8 pts over v1 (0.291 -> 0.389) at the
same size, same data, same lean budget.**

---

## 1. Design (fixed before the run, see V2_PLAN.md)

- **Framing:** CUAD is natively SQuAD-format extractive QA. v2 uses transformers'
  QA head (`Qwen2/3ForQuestionAnswering`) + sliding-window tokenization (512 tokens,
  stride 128). No custom head, no reimplemented metric, postprocessed n-best feeds
  the same official CUAD scorer as every frontier baseline.
- **Training:** LoRA r16 (alpha 32) on all attention+MLP projections + the randomly-init
  `qa_outputs` head in `modules_to_save`. bf16, effective batch 16, lr 2e-4, 2000
  steps (~1 epoch), negatives subsampled 2:1 per positive window.
- **Contamination:** dev (40 contracts) is carved from train, so training excludes it:
  368 clean contracts, title-overlap asserted in code at every launch. Test touched
  exactly once, by the dev-selected winner only.
- **Scaling curve:** 0.5B (local, free) -> 1.5B -> 7B -> 14B (one A100 session).

## 2. Results

### The scaling curve (dev split, identical recipe)

| Size | 0.5 epoch | 1 epoch |
|---|---|---|
| 0.5B | 0.2710 | 0.2841 |
| 1.5B | 0.2768 | 0.2893 |
| 7B | 0.2815 | 0.2999 |
| **14B** | 0.2952 | **0.3034** <- dev winner |

**28x parameters bought +0.019 dev AUPR.** The curve is monotonic but nearly flat:
under this lean recipe, scale is almost decorative. Framing did the work, the
*0.5B* extractive (0.284 dev) already matches the *14B generative* v1 (0.291 test).

### The test result (102 contracts, 4,182 questions, touched once)

| Model | Test AUPR |
|---|---|
| claude-opus-4-8 | 0.561 |
| claude-opus-4-6 | 0.498 |
| gpt-5.2 | 0.423 |
| gpt-4o | 0.421 |
| **nightwing-v2-14b (extractive)** | **0.3885** |
| nightwing-v1-14b (generative) | 0.291 |

- Dev->test moved 0.303->0.389, same direction and similar magnitude as v1
  (dev 0.255->test 0.291): the 40-contract dev carve-out is the harder split.
- **Head-to-head category wins:** vs gpt-4o **25/40**, vs gpt-5.2 **22/40**,
  vs claude-opus-4-6 **11/40**, vs claude-opus-4-8 **8/40**. The specialist now
  beats both GPT frontier models in a **majority** of categories (v1: 11/40).
- **Beats ALL four frontier models in 6 categories** (v1: 2): Agreement Date
  0.829 (+0.66 over best frontier), Effective Date 0.395 (+0.24), Expiration Date
  0.853 (+0.14), Volume Restriction, Document Name, Post-Termination Services.
- **The framing fix reached reasoning categories too**, not just dates, v1's worst
  collapses recovered hard: Covenant Not To Sue 0.107->0.539, Joint IP Ownership
  0.184->0.500, Anti-Assignment 0.224->0.497, Uncapped Liability 0.150->0.415.
  Median category AUPR 0.219->0.340.

## 3. Prediction resolution

**0.3885 < 0.50 -> the pre-registered prediction is wrong.** What the miss teaches:
the published ~0.47 for a 0.9B extractive model (DeBERTa-xlarge) comes from the
full CUAD training recipe, multiple epochs, full negative sampling, tuned
hyperparameters. v2's deliberately lean recipe (1 epoch, 2:1 negative subsample,
one LR, zero tuning) costs roughly 8-10 points against that reference. The gap to
frontier is now recipe, not architecture and not scale, which is the most
actionable place it could be.

## 4. What broke and what fixed it (engineering log)

Local GPU testing before the cloud run caught two silent correctness bugs for $0:

1. **Random-backbone load:** `Qwen2ForQuestionAnswering` names its backbone
   `transformer.*` but checkpoints store `model.*`, HF silently loads a RANDOM
   backbone (only the QA head is expected-random). Every number would have been
   garbage with zero errors. Fix: `load_qa_model()` injects the real weights and
   asserts. Caught because a "successful" local smoke was inspected, not trusted.
2. **Dev contamination trap:** dev is carved *from* train.json, so v2 training on
   train.json would have trained on all 40 dev-selection contracts. Fix:
   dev-exclusion + in-code overlap assertion (368 clean contracts).

Cloud-side, the smoke-first gate (rule #5, run with the *biggest* model) converted
four would-be run killers into ~$0.15 of failed minutes: torch 2.4 template vs
transformers 5.x (pinned the locally-validated stack), stale torchvision poisoning
imports (removed), 14B OOM at 80GB without gradient checkpointing (enabled,
validated locally first), missing sklearn for the official scorer (added). One
selection-stage bug (glob matched `*_predictions.json`) crashed *before* test
contact, `set -u` halted the script and the touched-once protocol survived.

## 5. Cost accounting

| Item | Spend |
|---|---|
| Local 0.5B curve point + all pipeline validation | $0 (own GPU) |
| A100 pod (~21 h x $1.52: 3 trainings 4.5 h, 8 dev evals ~9 h, test eval 3.5 h, setup/idle rest) | ~$31.60 |
| **v2 total** | **~$32** |
| **Project total (v1 + v2)** | **~$577 of the $1,000 cap** |

## 6. Artifacts

- Repo (public): all dev/test scores + raw predictions (`results/pilot_v2/`),
  every number re-scorable; this journal; `scripts/v2_cloud.sh` (the exact
  runbook that produced the result).
- Local (gitignored): all three final adapters (`runpod_artifacts/v2/`), full
  run log. Adapter for release: `ext_14b_final_adapter` (257 MB).

## 7. What v3 should be (decided ahead of spend)

The lean-recipe deficit is now the measured bottleneck, so v3 is recipe, not
architecture: 3 epochs, full negative sampling (drop the 2:1 subsample), modest
LR/warmup sweep on dev, and per-category confidence calibration. The reference
point says ~0.47+ is reachable at 0.9B; at 14B with the convention categories
already saturated (Agreement Date 0.83, Expiration Date 0.85, Document Name 0.89),
clearing 0.50 on the second attempt is the same prediction made with better
information, and this time the recipe variable is controlled.
