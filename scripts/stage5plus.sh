#!/usr/bin/env bash
# Remaining nightwing stages (post-training): dev sweep -> select -> ONE test
# eval -> comparison. Separate from cloud_run.sh so a restart never re-enters
# the smoke/train stages against an already-finished trainer state.
set -euo pipefail
cd /workspace/nightwing
export HF_HOME=/workspace/hf

echo "== [5/7] $(date -u +%H:%M:%S) dev checkpoint sweep (8 contracts, batch 32) =="
python -u -m training.eval_checkpoints --contracts 8 --batch 32

echo "== [6/7] $(date -u +%H:%M:%S) select best dev checkpoint =="
python - <<'PY'
import json, re, sys
from pathlib import Path
curve = json.loads(Path("results/pilot/qwen14b-cuad-lora_curve.json").read_text())["curve"]
best = max(curve, key=lambda p: p["aupr"])
print(f"dev curve: {[(p['steps'], round(p['aupr'],4)) for p in curve]}")
print(f"best: steps={best['steps']} AUPR={best['aupr']:.4f}")
if best["aupr"] < 0.05:
    sys.exit("ABORT: best dev AUPR < 0.05 - something is broken; NOT burning the test run.")
name = "final_adapter" if best["steps"] == "final" else f"checkpoint-{best['steps']}"
path = f"outputs/qwen14b-cuad-lora/{name}"
cfg = Path("config/models.yaml")
s = cfg.read_text(encoding="utf-8")
s2 = re.sub(r'adapter_path: ""[^\n]*', f'adapter_path: {path}', s, count=1)
assert s2 != s, "adapter_path placeholder not found in config/models.yaml"
cfg.write_text(s2, encoding="utf-8")
print(f"config/models.yaml -> adapter_path: {path}")
PY

echo "== [7/7] $(date -u +%H:%M:%S) TEST split - touched exactly once =="
python -u -m harness.run_eval --backend local --split test
python -m analysis.build_comparison

echo "== DONE $(date -u +%H:%M:%S) =="
head -40 results/pilot/local_test_full.json
echo "----------------------------------------"
cat results/comparison.md
