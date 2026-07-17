#!/usr/bin/env bash
# Resume 0.5B extractive from checkpoint-1000 -> step 2000, then eval full dev.
cd "$(dirname "$0")/.." || exit 1
LOG=outputs/v2_resume_0p5b.log
: > "$LOG"
run() {
  echo "" >>"$LOG"; echo "=== $(date +%H:%M:%S) START: $* ===" >>"$LOG"
  "$@" >>"$LOG" 2>&1
  echo "=== $(date +%H:%M:%S) EXIT $?: $1 ===" >>"$LOG"
}

run python -u -m training.train_extractive --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir outputs/ext_0p5b --cache-tag v2_train_full \
  --max-steps 2000 --neg-per-pos 2 --lr 2e-4 --save-steps 1000 \
  --resume-from outputs/ext_0p5b/checkpoint-1000

run python -u -m harness.eval_extractive --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter outputs/ext_0p5b/final_adapter --split-json data/cuad/dev.json \
  --cache-tag v2_dev_full --out results/pilot_v2/dev_0p5b_ckpt2000.json

echo "" >>"$LOG"; echo "=== ALL DONE $(date +%H:%M:%S) ===" >>"$LOG"
