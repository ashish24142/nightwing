#!/usr/bin/env bash
# v2 local scaling run: extractive QA head on Qwen2.5 0.5B + 1.5B, full CUAD
# train (dev excluded), eval on full dev. Free (this GPU). Each stage is
# independent so a late-stage OOM never loses an earlier result.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/pilot_v2 outputs
LOG=outputs/v2_local.log
: > "$LOG"

run() {
  echo "" >>"$LOG"
  echo "=== $(date +%H:%M:%S) START: $* ===" >>"$LOG"
  "$@" >>"$LOG" 2>&1
  echo "=== $(date +%H:%M:%S) EXIT $?: $1 ... ===" >>"$LOG"
}

TR="--epochs 3 --max-steps 2000 --neg-per-pos 2 --lr 2e-4 --save-steps 1000 --cache-tag v2_train_full"
EV="--split-json data/cuad/dev.json --cache-tag v2_dev_full"

# ---- 0.5B (builds both caches) ----
run python -u -m training.train_extractive --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir outputs/ext_0p5b --batch-size 8 --grad-accum 2 $TR
run python -u -m harness.eval_extractive --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter outputs/ext_0p5b/final_adapter $EV --out results/pilot_v2/dev_0p5b.json

# ---- 1.5B (reuses caches; smaller batch for 12GB) ----
run python -u -m training.train_extractive --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir outputs/ext_1p5b --batch-size 4 --grad-accum 4 $TR
run python -u -m harness.eval_extractive --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/ext_1p5b/final_adapter $EV --out results/pilot_v2/dev_1p5b.json

echo "" >>"$LOG"
echo "=== ALL DONE $(date +%H:%M:%S) ===" >>"$LOG"
