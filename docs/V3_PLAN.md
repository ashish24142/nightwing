# nightwing v3, Experiment Plan (pre-committed before spend)

**Question:** v2 localized the gap to frontier in the training recipe: its lean
settings (1 epoch, 2:1 negative subsample, untuned LR) trail the published
recipe-complete 0.9B extractive baseline (~0.47) by ~8 points. v3 changes ONLY
the recipe, on the same backbone, data, splits, metric and harness.

**Pre-registered prediction (carried forward from v2, written before this
run's spend): the recipe-complete 14B extractive model clears 0.50 test AUPR.**

## Recipe changes (everything else identical to v2)

| Variable | v2 (lean) | v3 |
|---|---|---|
| Epochs | ~1 | 3 |
| Negative windows per positive | 2:1 | 4:1 (see note) |
| Learning rate | 2e-4 (untuned) | swept {1e-4, 2e-4, 4e-4} on the 0.5B proxy, winner transferred |

**Note on "full negatives":** the literal full negative set is ~700k windows
per epoch; 3 epochs at the measured 4.7 s/step is ~a week of A100 (~$250),
which the published 0.9B baseline could afford only because its model is 15x
smaller. 4:1 doubles v2's negative exposure at ~55k windows/epoch and is the
budget-shaped approximation; this is declared here, before results exist.

## Protocol (unchanged)

- Train on the 368 clean contracts (dev excluded), contamination gate asserted
  in code at launch. Checkpoints ~each epoch; selection on the 40-contract dev
  split. **Test touched once**, by the dev-selected checkpoint.
- Official CUAD evaluator, unmodified. All predictions committed.
- Calibration note (decided before spend): per-category Platt calibration was
  prototyped and validated on v2's committed dev predictions; it DEGRADED
  overall dev AUPR in-sample (0.303 -> 0.243, analysis/calibrate.py), so it is
  excluded from v3. The primary metric is the uncalibrated overall AUPR,
  identical to every other model in the comparison.

## Run schedule (one A100 80GB, $1.52/hr, from measured v2 rates)

| Stage | Est. |
|---|---|
| Setup + caches (neg 4:1 rebuild) | ~1 h |
| LR sweep: 3x 0.5B (1 epoch + dev eval each) | ~3 h |
| 14B train: 3 epochs (~10.3k steps @ 4.7 s) | ~13.5 h |
| Dev evals: 3 checkpoints x ~2.5 h | ~7.5 h |
| Single test eval | ~3.5 h |
| **Total** | **~29 h ~ $44** |

Realistic all-in ~$45-55; hard ceiling **$100**: abort and reassess if crossed.

## Decision criteria (fixed)

- Prediction TRUE (>= 0.50 test AUPR): framing + recipe thesis
  complete; publish and submit to NLLP @ EMNLP 2026 (deadline Aug 11).
- Prediction FALSE: publish the miss and the measured recipe curve; the NLLP
  paper reports the honest arc either way.
- No threshold-moving after launch.

## Out of scope

Hard-negative mining, ensembles, multi-span heads, data augmentation, backbone
changes. One backbone (Qwen3-14B); the size curve is done.
