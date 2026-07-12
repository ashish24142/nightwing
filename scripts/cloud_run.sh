#!/usr/bin/env bash
# nightwing cloud runbook — single A100 80GB (RunPod PyTorch template).
# Usage (from /workspace):
#   git clone https://github.com/ashish24142/nightwing && cd nightwing
#   nohup bash scripts/cloud_run.sh > /workspace/run.log 2>&1 &
# Idempotent: training resumes from checkpoints; re-running skips finished stages.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME=/workspace/hf          # weights on the big persistent volume
mkdir -p "$HF_HOME"
STAMP() { date -u +"%H:%M:%S"; }

echo "== [1/7] $(STAMP) deps =="
pip install -q -r requirements.txt
# unsloth is optional (train_qlora falls back to transformers+peft); don't let
# its failure kill the run
pip install -q transformers peft trl accelerate bitsandbytes sentencepiece
# NO unsloth: its tokenizer patching injects an '<EOS_TOKEN>' sentinel that this
# trl version never resolves (killed two runs). The transformers+peft path is
# validated end-to-end; ~1.5x slower training is the price of it working.
pip uninstall -q -y unsloth unsloth_zoo 2>/dev/null || true
pip install -q "datasets==3.2.0"

echo "== [2/7] $(STAMP) data + contamination gate =="
[ -f data/cuad/test.json ] || python -m data.download_cuad
[ -f data/cuad/train_instruct.jsonl ] || python -m data.prepare_train
python -m data.verify_split   # rule #1 — exits non-zero on any overlap

echo "== [3/7] $(STAMP) 10-step training smoke (rule #5) =="
python -m training.smoke_test

echo "== [4/7] $(STAMP) full LoRA train (resumes from checkpoints) =="
python -m training.train_qlora

echo "== [5/7] $(STAMP) dev-split checkpoint sweep =="
python -m training.eval_checkpoints

echo "== [6/7] $(STAMP) select best dev checkpoint =="
python - <<'PY'
import json, re, sys
from pathlib import Path
curve = json.loads(Path("results/pilot/qwen14b-cuad-lora_curve.json").read_text())["curve"]
best = max(curve, key=lambda p: p["aupr"])
print(f"dev curve: {[(p['steps'], round(p['aupr'],4)) for p in curve]}")
print(f"best: steps={best['steps']} AUPR={best['aupr']:.4f}")
if best["aupr"] < 0.05:
    sys.exit("ABORT: best dev AUPR < 0.05 — something is broken; NOT burning the test run.")
name = "final_adapter" if best["steps"] == "final" else f"checkpoint-{best['steps']}"
path = f"outputs/qwen14b-cuad-lora/{name}"
cfg = Path("config/models.yaml")
s = cfg.read_text(encoding="utf-8")
s2 = re.sub(r'adapter_path: ""[^\n]*', f'adapter_path: {path}', s, count=1)
assert s2 != s, "adapter_path placeholder not found in config/models.yaml"
cfg.write_text(s2, encoding="utf-8")
print(f"config/models.yaml -> adapter_path: {path}")
PY

echo "== [7/7] $(STAMP) TEST split — touched exactly once — then comparison =="
python -m harness.run_eval --backend local --split test
python -m analysis.build_comparison

echo "== DONE $(STAMP) — final artifacts =="
cat results/pilot/local_test_full.json | head -40
echo "----------------------------------------"
cat results/comparison.md
