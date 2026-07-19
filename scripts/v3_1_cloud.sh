#!/usr/bin/env bash
# nightwing v3.1 — the v3 recipe WITHOUT the LR confound.
# v3 showed lr 4e-4 (picked by a 0.0004 margin on a 0.5B proxy) is toxic at
# 14B: ep1 dev 0.129 vs v2's 0.303. v3.1 reruns 3 epochs + neg 4:1 at the
# v2-proven lr 2e-4. Reuses the pod's existing v3 feature caches.
# EXTRA GUARD: test is burned ONLY if best v3.1 dev BEATS v2's dev (0.3034);
# otherwise v2 remains the selected model and test stays untouched.
set -uo pipefail
cd "$(dirname "$0")/.."

export HF_HOME=/workspace/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/pilot_v3
STAMP() { date -u +"%H:%M:%S"; }
V2_DEV=0.3034

echo "== [1/3] $(STAMP) 14B: 3 epochs, neg 4:1, lr 2e-4 =="
python - <<'PY'
from harness.extractive import load_examples, _titles
ex = load_examples("data/cuad/train.json", exclude_json="data/cuad/dev.json")
t = {e["title"] for e in ex}
assert len(t) == 368 and not (t & _titles("data/cuad/dev.json")) and not (t & _titles("data/cuad/test.json"))
print("GATE OK")
PY
if [ ! -f outputs/ext14b_v31/final_adapter/adapter_model.safetensors ]; then
  RESUME=""
  ls outputs/ext14b_v31/checkpoint-*/trainer_state.json >/dev/null 2>&1 && RESUME="--resume-from auto"
  python -u -m training.train_extractive --base-model Qwen/Qwen3-14B \
    --output-dir outputs/ext14b_v31 --epochs 3 --max-steps -1 --lr 2e-4 \
    --neg-per-pos 4 --batch-size 8 --grad-accum 2 --save-steps 3450 \
    --cache-tag v3_q3_train_n4 $RESUME || { echo "TRAIN v31 FAILED"; exit 1; }
fi

echo "== [2/3] $(STAMP) dev evals =="
for ck in $(ls -d outputs/ext14b_v31/checkpoint-* 2>/dev/null) outputs/ext14b_v31/final_adapter; do
  tag=$(basename $ck)
  outj=results/pilot_v3/dev_v31_${tag}.json
  if [ ! -f "$outj" ]; then
    echo "-- $(STAMP) dev-eval $tag --"
    python -u -m harness.eval_extractive --split-json data/cuad/dev.json \
      --base-model Qwen/Qwen3-14B --adapter $ck --batch-size 32 \
      --cache-tag v3_q3_dev --out "$outj" || { echo "EVAL $tag FAILED"; exit 1; }
  fi
done

echo "== [3/3] $(STAMP) select; test ONLY if dev beats v2 ($V2_DEV) =="
WINCK=$(python - <<PY
import json, sys
from pathlib import Path
cands = [p for p in Path("results/pilot_v3").glob("dev_v31_*.json")
         if not p.stem.endswith("_predictions")]
best = max(cands, key=lambda p: json.loads(p.read_text())["overall"]["aupr"])
a = json.loads(best.read_text())["overall"]["aupr"]
print(f"# BEST v3.1 dev={a:.4f} (v2 dev=$V2_DEV)", file=sys.stderr)
if a <= $V2_DEV:
    sys.exit("NO-TEST: v3.1 dev does not beat v2 dev; test stays untouched, v2 remains the model.")
print(best.stem.replace("dev_v31_", ""))
PY
) || { echo "V31 DECIDED: NO TEST EVAL"; echo "== DONE $(STAMP) =="; exit 0; }
echo "WINNER-CKPT: $WINCK"
if [ ! -f results/pilot_v3/test_14b_v31.json ]; then
  python -u -m harness.eval_extractive --split-json data/cuad/test.json \
    --base-model Qwen/Qwen3-14B --adapter outputs/ext14b_v31/$WINCK \
    --batch-size 32 --cache-tag v3_q3_test --out results/pilot_v3/test_14b_v31.json
fi
echo "== DONE $(STAMP) =="