#!/usr/bin/env bash
# nightwing v3 cloud runbook — recipe-complete 14B extractive on one A100 80GB.
# Recipe (pre-registered in docs/V3_PLAN.md): 3 epochs, negatives 4:1,
# LR swept {1e-4, 2e-4, 4e-4} on a 0.5B proxy and transferred.
# Usage (from /workspace):
#   git clone https://github.com/ashish24142/nightwing && cd nightwing
#   # scp the verified data/cuad/*.json in
#   nohup bash scripts/v3_cloud.sh > /workspace/v3.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

export HF_HOME=/workspace/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" results/pilot_v3
STAMP() { date -u +"%H:%M:%S"; }

SMALL=Qwen/Qwen2.5-0.5B-Instruct
BIG=Qwen/Qwen3-14B
EV="python -u -m harness.eval_extractive --split-json data/cuad/dev.json"

echo "== [1/6] $(STAMP) deps =="
python -c "import torch; v=torch.__version__; exit(0 if v.startswith('2.1') and int(v.split('.')[1])>=8 else 1)" 2>/dev/null || \
  pip install -q -U torch --index-url https://download.pytorch.org/whl/cu128
pip uninstall -q -y torchvision torchaudio 2>/dev/null || true
pip install -q "transformers==5.5.0" "peft==0.19.1" "datasets==3.2.0" accelerate sentencepiece pyyaml tenacity python-dotenv tqdm pandas scikit-learn

echo "== [2/6] $(STAMP) data + contamination gate (rule #1) =="
[ -f data/cuad/test.json ] || python -m data.download_cuad
python - <<'PY'
from harness.extractive import load_examples, _titles
ex = load_examples("data/cuad/train.json", exclude_json="data/cuad/dev.json")
t = {e["title"] for e in ex}
assert len(t) == 368 and not (t & _titles("data/cuad/dev.json")) and not (t & _titles("data/cuad/test.json"))
print(f"GATE OK: {len(t)} train contracts, zero dev/test overlap")
PY

echo "== [3/6] $(STAMP) 14B 6-step smoke at the v3 recipe (rule #5) =="
if [ ! -f outputs/smoke14b_v3.ok ]; then
  python -u -m training.train_extractive --base-model $BIG --output-dir outputs/smk3 \
    --limit 3 --max-steps 6 --neg-per-pos 4 --batch-size 8 --grad-accum 2 --save-steps 999 \
    && touch outputs/smoke14b_v3.ok && rm -rf outputs/smk3
fi
[ -f outputs/smoke14b_v3.ok ] || { echo "SMOKE FAILED"; exit 1; }

echo "== [4/6] $(STAMP) LR sweep on the 0.5B proxy (1 epoch, neg 4:1) =="
for lr in 1e-4 2e-4 4e-4; do
  outj=results/pilot_v3/lrsweep_${lr}.json
  if [ ! -f "$outj" ]; then
    python -u -m training.train_extractive --base-model $SMALL \
      --output-dir outputs/lr_${lr} --epochs 1 --max-steps -1 --lr $lr \
      --neg-per-pos 4 --batch-size 16 --grad-accum 1 --save-steps 99999 \
      --cache-tag v3_q25_train_n4 || { echo "SWEEP TRAIN $lr FAILED"; exit 1; }
    $EV --base-model $SMALL --adapter outputs/lr_${lr}/final_adapter \
      --batch-size 96 --cache-tag v3_q25_dev --out "$outj" \
      || { echo "SWEEP EVAL $lr FAILED"; exit 1; }
  fi
done
BESTLR=$(python - <<'PY'
import json
from pathlib import Path
cands = [p for p in Path("results/pilot_v3").glob("lrsweep_*.json")
         if not p.stem.endswith("_predictions")]
best = max(cands, key=lambda p: json.loads(p.read_text())["overall"]["aupr"])
print(best.stem.replace("lrsweep_", ""))
PY
)
[ -n "$BESTLR" ] || { echo "LR SELECTION FAILED"; exit 1; }
echo "WINNER-LR: $BESTLR"

echo "== [5/6] $(STAMP) 14B: 3 epochs, neg 4:1, lr $BESTLR =="
if [ ! -f outputs/ext14b_v3/final_adapter/adapter_model.safetensors ]; then
  RESUME=""
  ls outputs/ext14b_v3/checkpoint-*/trainer_state.json >/dev/null 2>&1 && RESUME="--resume-from auto"
  python -u -m training.train_extractive --base-model $BIG \
    --output-dir outputs/ext14b_v3 --epochs 3 --max-steps -1 --lr $BESTLR \
    --neg-per-pos 4 --batch-size 8 --grad-accum 2 --save-steps 3450 \
    --cache-tag v3_q3_train_n4 $RESUME || { echo "TRAIN 14B FAILED"; exit 1; }
fi
for ck in $(ls -d outputs/ext14b_v3/checkpoint-* 2>/dev/null) outputs/ext14b_v3/final_adapter; do
  tag=$(basename $ck)
  outj=results/pilot_v3/dev_14b_${tag}.json
  if [ ! -f "$outj" ]; then
    echo "-- $(STAMP) dev-eval $tag --"
    $EV --base-model $BIG --adapter $ck --batch-size 32 \
      --cache-tag v3_q3_dev --out "$outj" || { echo "EVAL $tag FAILED"; exit 1; }
  fi
done

echo "== [6/6] $(STAMP) select best dev ckpt -> single TEST eval =="
WINCK=$(python - <<'PY'
import json, sys
from pathlib import Path
cands = [p for p in Path("results/pilot_v3").glob("dev_14b_*.json")
         if not p.stem.endswith("_predictions")]
best = max(cands, key=lambda p: json.loads(p.read_text())["overall"]["aupr"])
a = json.loads(best.read_text())["overall"]["aupr"]
if a < 0.25:
    sys.exit(f"ABORT: best dev {a} < 0.25 — broken; NOT burning test.")
print(best.stem.replace("dev_14b_", ""), file=sys.stdout)
print(f"# WINNER dev={a:.4f}", file=sys.stderr)
PY
) || { echo "SELECTION ABORTED"; exit 1; }
echo "WINNER-CKPT: $WINCK"
if [ ! -f results/pilot_v3/test_14b_v3.json ]; then
  python -u -m harness.eval_extractive --split-json data/cuad/test.json \
    --base-model $BIG --adapter outputs/ext14b_v3/$WINCK \
    --batch-size 32 --cache-tag v3_q3_test --out results/pilot_v3/test_14b_v3.json
fi

echo "== DONE $(STAMP) =="
for f in results/pilot_v3/*.json; do
  case $f in *predictions*) continue;; esac
  python -c "import json;d=json.load(open('$f'));print('$f'.split('/')[-1], round(d['overall']['aupr'],4))"
done