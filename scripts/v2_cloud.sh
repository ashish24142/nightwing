#!/usr/bin/env bash
# nightwing v2 cloud runbook — extractive scaling curve on one A100 80GB.
# Usage (from /workspace):
#   git clone https://github.com/ashish24142/nightwing && cd nightwing
#   # scp the verified data/cuad/*.json in (or let download_cuad try)
#   nohup bash scripts/v2_cloud.sh > /workspace/v2.log 2>&1 &
# Idempotent: finished stages are skipped via their output files; training
# resumes from checkpoints (--resume-from auto picks up after preemption).
set -uo pipefail
cd "$(dirname "$0")/.."

export HF_HOME=/workspace/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" results/pilot_v2
STAMP() { date -u +"%H:%M:%S"; }

# effective batch 16 for every size (matches the local 0.5B point: 8x2)
TRAIN="python -u -m training.train_extractive --max-steps 2000 --neg-per-pos 2 --lr 2e-4 --save-steps 500"
EVAL="python -u -m harness.eval_extractive --split-json data/cuad/dev.json"

echo "== [1/6] $(STAMP) deps (pin to the locally-validated stack) =="
# template ships torch 2.4; transformers 5.x needs newer torch (DTensor import).
# Match the stack the pipeline was validated on locally: torch 2.11 + tf 5.5.
python -c "import torch; v=torch.__version__; exit(0 if v.startswith('2.1') and int(v.split('.')[1])>=8 else 1)" 2>/dev/null || \
  pip install -q -U torch --index-url https://download.pytorch.org/whl/cu128
# template's torchvision/torchaudio are built for torch 2.4 and poison
# transformers' imports after the torch upgrade; text pipeline needs neither.
pip uninstall -q -y torchvision torchaudio 2>/dev/null || true
pip install -q "transformers==5.5.0" "peft==0.19.1" "datasets==3.2.0" accelerate sentencepiece pyyaml tenacity python-dotenv tqdm pandas scikit-learn

echo "== [2/6] $(STAMP) data + contamination gate (rule #1) =="
[ -f data/cuad/test.json ] || python -m data.download_cuad
python - <<'PY'
from harness.extractive import load_examples, _titles
ex = load_examples("data/cuad/train.json", exclude_json="data/cuad/dev.json")
t = {e["title"] for e in ex}
assert len(t) == 368, f"expected 368 clean train contracts, got {len(t)}"
assert not (t & _titles("data/cuad/dev.json")), "dev leaked into train"
assert not (t & _titles("data/cuad/test.json")), "test leaked into train"
print(f"GATE OK: {len(t)} train contracts, zero dev/test overlap")
PY

echo "== [3/6] $(STAMP) 14B 6-step smoke (rule #5: biggest model proves fit first) =="
if [ ! -f outputs/smoke14b.ok ]; then
  $TRAIN --base-model Qwen/Qwen3-14B --output-dir outputs/ext14b_smoke \
    --limit 3 --max-steps 6 --batch-size 8 --grad-accum 2 --save-steps 999 \
    && touch outputs/smoke14b.ok && rm -rf outputs/ext14b_smoke
fi
[ -f outputs/smoke14b.ok ] || { echo "SMOKE FAILED — aborting"; exit 1; }

echo "== [4/6] $(STAMP) scaling curve: train + dev-eval each size =="
# Qwen2.5 1.5B/7B share a tokenizer (one cache); Qwen3-14B gets its own.
declare -A BASE=( [1p5b]=Qwen/Qwen2.5-1.5B-Instruct [7b]=Qwen/Qwen2.5-7B-Instruct [14b]=Qwen/Qwen3-14B )
declare -A TCACHE=( [1p5b]=v2c_q25_train [7b]=v2c_q25_train [14b]=v2c_q3_train )
declare -A DCACHE=( [1p5b]=v2c_q25_dev  [7b]=v2c_q25_dev  [14b]=v2c_q3_dev )
declare -A BS=( [1p5b]=16 [7b]=16 [14b]=8 )
declare -A GA=( [1p5b]=1 [7b]=1 [14b]=2 )
declare -A EBS=( [1p5b]=64 [7b]=48 [14b]=32 )

for size in 1p5b 7b 14b; do
  b=${BASE[$size]}
  if [ ! -f outputs/ext_${size}/final_adapter/adapter_model.safetensors ]; then
    echo "-- $(STAMP) train $size ($b) --"
    RESUME=""
    ls outputs/ext_${size}/checkpoint-*/trainer_state.json >/dev/null 2>&1 && RESUME="--resume-from auto"
    $TRAIN --base-model "$b" --output-dir outputs/ext_${size} \
      --batch-size ${BS[$size]} --grad-accum ${GA[$size]} \
      --cache-tag ${TCACHE[$size]} $RESUME || { echo "TRAIN $size FAILED"; exit 1; }
  fi
  for ck in checkpoint-1000 final_adapter; do
    tag=$( [ "$ck" = "final_adapter" ] && echo 2000 || echo 1000 )
    outj=results/pilot_v2/dev_${size}_ckpt${tag}.json
    if [ ! -f "$outj" ]; then
      echo "-- $(STAMP) dev-eval $size $ck --"
      $EVAL --base-model "$b" --adapter outputs/ext_${size}/$ck \
        --batch-size ${EBS[$size]} --cache-tag ${DCACHE[$size]} --out "$outj" \
        || { echo "EVAL $size $ck FAILED"; exit 1; }
    fi
  done
done

echo "== [5/6] $(STAMP) select best dev model (abort if broken) =="
python - <<'PY' > outputs/v2_winner.env
import json, sys
from pathlib import Path
BASE = {"0p5b": "Qwen/Qwen2.5-0.5B-Instruct", "1p5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "7b": "Qwen/Qwen2.5-7B-Instruct", "14b": "Qwen/Qwen3-14B"}
best = None
for p in sorted(Path("results/pilot_v2").glob("dev_*_ckpt*.json")):
    d = json.loads(p.read_text())
    size = p.stem.split("_")[1]
    ck = "checkpoint-1000" if p.stem.endswith("1000") else "final_adapter"
    a = d["overall"]["aupr"]
    print(f"# {p.name}: {a:.4f}", file=sys.stderr)
    if size in ("1p5b", "7b", "14b") and (best is None or a > best[0]):
        best = (a, size, ck)
if best is None or best[0] < 0.25:
    sys.exit(f"ABORT: best cloud dev AUPR {best} < 0.25 — broken; NOT burning test.")
a, size, ck = best
print(f"# WINNER dev={a:.4f}", file=sys.stderr)
print(f"WIN_BASE={BASE[size]}")
print(f"WIN_ADAPTER=outputs/ext_{size}/{ck}")
print(f"WIN_SIZE={size}")
print(f"WIN_EBS={dict(zip(['1p5b','7b','14b'],[64,48,32]))[size]}")
PY
cat outputs/v2_winner.env
source outputs/v2_winner.env

echo "== [6/6] $(STAMP) TEST split — touched exactly once =="
TESTCACHE=v2c_q25_test; [ "$WIN_SIZE" = "14b" ] && TESTCACHE=v2c_q3_test
if [ ! -f results/pilot_v2/test_${WIN_SIZE}.json ]; then
  python -u -m harness.eval_extractive --split-json data/cuad/test.json \
    --base-model "$WIN_BASE" --adapter "$WIN_ADAPTER" \
    --batch-size "$WIN_EBS" --cache-tag "$TESTCACHE" \
    --out results/pilot_v2/test_${WIN_SIZE}.json
fi

echo "== DONE $(STAMP) =="
python - <<'PY'
import json
from pathlib import Path
for p in sorted(Path("results/pilot_v2").glob("*.json")):
    if p.stem.endswith("predictions"): continue
    d = json.loads(p.read_text())
    print(f"{p.name}: AUPR={d['overall']['aupr']:.4f}")
PY
