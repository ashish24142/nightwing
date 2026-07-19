# nightwing v3, Run Journal (in progress)

**Experiment:** recipe-complete the v2 extractive 14B (3 epochs, negatives 4:1,
LR swept on a 0.5B proxy). Pre-registered prediction: clears 0.50 test AUPR.
Plan: `V3_PLAN.md`, committed before spend.

**Status: v3 recipe FAILED via LR transfer; test never touched (guard worked).
v3.1 (same recipe at the v2-proven lr 2e-4) approved and queued.**

## The LR sweep (0.5B proxy, 1 epoch, neg 4:1, dev split)

| LR | Dev AUPR |
|---|---|
| 1e-4 | 0.2717 |
| 2e-4 | 0.2873 |
| 4e-4 | **0.2877** (winner, by 0.0004) |

## The recipe comparison the run actually produced (14B, dev split)

| Run | Recipe | ep 0.5 | ep 1 | ep 2 | ep 3 | final |
|---|---|---|---|---|---|---|
| v2 | lr 2e-4, neg 2:1, 1 epoch | 0.2952 | **0.3034** | - | - | - |
| v3 | lr 4e-4, neg 4:1, 3 epochs | - | 0.1287 | 0.2411 | 0.2451 | 0.2445 |
| v3.1 | lr 2e-4, neg 4:1, 3 epochs | - | queued | queued | queued | queued |

## Finding: proxy-swept learning rates do not transfer 0.5B -> 14B

lr 4e-4 won the 0.5B sweep by four ten-thousandths of a point and then cost
the 14B seventeen points at epoch 1 (0.129 vs v2's 0.303 at the same lr-2e-4
budget point). Epochs 2 and 3 spent their compute *recovering* from the early
damage (0.129 -> 0.245), not compounding gains, and still ended six points
below v2's single epoch. The standard folk wisdom (larger models want cooler
learning rates) beat the measured-but-underpowered proxy sweep. A 0.0004
margin on a 28x smaller model is noise, and we treated it as signal; the
pre-registered protocol said "sweep winner transfers", so it was followed and
is reported as designed.

## Protocol outcome

- Best v3 dev = 0.2451 < the 0.25 sanity guard -> selection auto-aborted;
  **the test split was never touched by v3.** v2 remains the selected model.
- Money: ~$31 (sweep ~$4, 14B train ~$21, dev evals ~$6). The run answered a
  real question (LR transfer), just not the one it was aimed at (epochs and
  negatives), so v3.1 re-asks that question with the LR confound removed.

## v3.1 (queued at abort time)

Same 3 epochs + neg 4:1 at lr 2e-4, reusing the pod's feature caches.
Stricter test gate than v3: the single test eval fires ONLY if v3.1's best
dev beats v2's 0.3034; otherwise v2 stays the model and test stays pristine.
Runbook: `scripts/v3_1_cloud.sh`. This section will be completed with results.
