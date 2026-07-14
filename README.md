# nightwing

*The sidekick that trained up.*

**Can a fine-tuned open 14B model match frontier LLMs on contract clause extraction — at ~$0 per contract instead of ~$1.60?**

This repo is a reproducible benchmark harness + training pipeline for
[CUAD](https://www.atticusprojectai.org/cuad) (Contract Understanding Atticus
Dataset, 510 contracts, 41 clause categories), scored with the **official CUAD
metric** (AUPR / precision-at-recall). It evaluates frontier APIs and local
fine-tuned models through one identical pipeline, so the comparison is
apples-to-apples — and every prediction file is committed, so every number here
can be re-scored by anyone.

## Results (official CUAD AUPR, full 102-contract test split)

| Model | AUPR | Cost/run |
|---|---|---|
| claude-opus-4-8 | **0.561** | ~$165 |
| claude-opus-4-6 | 0.498 | ~$154 |
| gpt-5.2 | 0.423 | ~$13 |
| gpt-4o | 0.421 | ~$58 |
| nightwing-14b (Qwen3-14B + LoRA, ckpt-1250) | 0.291 | ~$0 after ~$41 training |

**The punchline: the $40 specialist could not touch Claude overall, but it won
11 of 40 categories outright against gpt-5.2 AND 11 of 40 against gpt-4o**
(Agreement Date 0.687 vs 0.054/0.079, Third Party Beneficiary 0.585 vs 0.355/0.102,
Document Name 0.711 vs 0.471, and more) — beating **all four** frontier models
simultaneously on the annotation-convention categories: Agreement Date (+0.52) and
Effective Date (+0.26). Overall it lands RED on the pre-committed bands (0.291 vs 0.561).
Note the two GPT rows: **gpt-4o (0.421) and gpt-5.2 (0.423) are statistically tied a
full model generation apart** — strong evidence this task is bottlenecked by learned
annotation convention, not frontier scale. The gap between frontier and specialist is
not a wall, it is category-shaped. Full story: [docs/RUN_JOURNAL.md](docs/RUN_JOURNAL.md) ·
per-category table: [results/comparison.md](results/comparison.md) ·
frontier weakness map: [results/category_breakdown.md](results/category_breakdown.md).
Raw predictions for every model are committed — every number is re-scorable.

## Install

Python 3.11.

```bash
pip install -r requirements.txt
cp .env.example .env        # add API keys (frontier baselines only)
python -m data.download_cuad
```

## Use

```bash
# score any backend on the official metric (swap is config-only)
python -m harness.run_eval --backend claude48 --limit 5     # 5-contract smoke first
python -m harness.run_eval --backend claude48 --split test --workers 5

# sanity: the official scorer must give ground-truth-vs-itself a perfect score
python -m harness.scoring --selfcheck
```

Backends live in [config/models.yaml](config/models.yaml) — frontier APIs
(Anthropic/OpenAI, incl. Azure-hosted, with prompt caching) and local
fine-tuned models implement one interface.

## Train the specialist (Linux GPU box, A100 80GB)

```bash
pip install -r training/requirements-train.txt
python -m data.prepare_train        # windowed instruction data + dev carve-out
python -m data.verify_split         # HARD GATE: proves zero train/test contamination
python -m training.smoke_test       # 10 steps, no OOM, sane loss
python -m training.train_qlora      # bf16 LoRA, checkpointed + resumable
# select best checkpoint on DEV (test is touched once):
python -m harness.run_eval --backend local --split dev
python -m harness.run_eval --backend local --split test
python -m analysis.build_comparison # -> results/comparison.md
```

Integrity rules, enforced in code: training refuses to start unless
`verify_split.py` proves zero train/test contamination; checkpoint selection
uses a dev split carved from train; the metric is the unmodified official CUAD
scorer (never reimplemented); local-model confidence comes from token logprobs
(a real PR curve, not self-reported numbers).

## Fine-tune a different model (3 steps)

The pipeline is model-agnostic — data prep, chat templating, and logprob
confidence adapt to any HF causal LM automatically:

```bash
# 1. train (any HF model id)
python -m training.train_qlora --base-model meta-llama/Llama-3.1-8B-Instruct \
    --output-dir outputs/llama8b
# 2. add a backend entry in config/models.yaml (copy the commented template)
# 3. evaluate on the identical harness/metric
python -m harness.run_eval --backend local_llama8b --split dev    # select checkpoint
python -m harness.run_eval --backend local_llama8b --split test   # final, ONCE
```

Only constraint: the window size (`win_chars`, set at data prep) must fit the
model's `max_seq_length` (~2.85 chars/token + room for prompt and answer).

## Why N contracts of smoke first?

Frontier runs cost real money and long runs die (spot GPUs, rate limits,
laptops). Everything here is built for that: per-contract checkpointing with
resume, cost tracking with a spend alert, retry with backoff, and a
44-check offline test suite (`python -m tests.test_hardening`) covering the
failure paths — malformed model output, corrupt checkpoints, concurrent cost
ledger writes, stale locks.

## Roadmap

- [x] Frontier baselines (Claude Opus 4.8 / 4.6, GPT-5.2, GPT-4o) on the official metric
- [x] 41-category weakness analysis
- [x] Qwen3-14B bf16 LoRA specialist + head-to-head comparison (v1: RED overall, 2 category wins — see the run journal)
- [ ] v2: extractive span head on the 14B backbone (framing fix; 0.9B extractive already scores ~0.47 published)
- [ ] AUPR vs model size vs cost curve
- [ ] HuggingFace release (adapter + merged + model card) and demo Space

## License & attribution

Apache-2.0. CUAD is released by [The Atticus Project](https://www.atticusprojectai.org/)
under CC BY 4.0 — cite [Hendrycks et al., 2021](https://arxiv.org/abs/2103.06268).
Base model: [Qwen](https://huggingface.co/Qwen) (Apache-2.0). This project
benchmarks frontier models but is trained **only** on CUAD ground truth — never
on frontier model outputs. Not legal advice.
