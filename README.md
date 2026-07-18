# nightwing

*The sidekick that trained up.*

[Essays on this experiment](https://migrationassurance.com/) |
[Model on Hugging Face](https://huggingface.co/ashish24142/nightwing-14b-cuad-extractive) |
[Author on LinkedIn](https://in.linkedin.com/in/ashish-kr-singh-ml)

**Can a fine-tuned open 14B model match frontier LLMs on contract clause extraction, at ~$0 per contract instead of ~$1.60?**

This repo is a reproducible benchmark harness + training pipeline for
[CUAD](https://www.atticusprojectai.org/cuad) (Contract Understanding Atticus
Dataset, 510 contracts, 41 clause categories), scored with the **official CUAD
metric** (AUPR / precision-at-recall). It evaluates frontier APIs and local
fine-tuned models through one identical pipeline, so the comparison is
apples-to-apples, and every prediction file is committed, so every number here
can be re-scored by anyone.

## Results (official CUAD AUPR, full 102-contract test split)

| Model | AUPR | Cost/run |
|---|---|---|
| claude-opus-4-8 | **0.561** | ~$165 |
| claude-opus-4-6 | 0.498 | ~$154 |
| gpt-5.2 | 0.423 | ~$13 |
| gpt-4o | 0.421 | ~$58 |
| **nightwing-v2-14b (extractive span head)** | **0.389** | ~$0 after ~$32 training |
| nightwing-v1-14b (generative, ckpt-1250) | 0.291 | ~$0 after ~$41 training |

**The punchline, v2 edition: changing the task framing, not the model, not the
data, not the budget, moved the specialist from 0.291 to 0.389, and it now beats
gpt-4o in 25 of 40 categories and gpt-5.2 in 22 of 40** while trailing them
overall by ~3 points. It beats all four frontier models simultaneously in 6
categories (Agreement Date 0.829 vs best-frontier 0.168, Expiration Date 0.853
vs 0.708, Effective Date, and more), and v1's worst reasoning collapses recovered
hard (Covenant Not To Sue 0.107 -> 0.539). The scaling curve is the other finding:
0.5B -> 14B under the identical recipe moves dev AUPR only 0.284 -> 0.303:
**framing does the work; scale is almost decorative** (and gpt-4o vs gpt-5.2,
tied a generation apart, says the same about frontier scale). The v2
pre-registered prediction (clear 0.50) missed, published as committed; the
measured deficit is the deliberately lean recipe (1 epoch, subsampled negatives,
zero tuning), which is v3's controlled variable.
Full story: [v1 journal](docs/RUN_JOURNAL.md) | [v2 journal](docs/RUN_JOURNAL_V2.md) |
per-category table: [results/comparison.md](results/comparison.md) |
frontier weakness map: [results/category_breakdown.md](results/category_breakdown.md).
Raw predictions for every model are committed, every number is re-scorable.

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

Backends live in [config/models.yaml](config/models.yaml), frontier APIs
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

The pipeline is model-agnostic, data prep, chat templating, and logprob
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
failure paths, malformed model output, corrupt checkpoints, concurrent cost
ledger writes, stale locks.

## Roadmap

- [x] Frontier baselines (Claude Opus 4.8 / 4.6, GPT-5.2, GPT-4o) on the official metric
- [x] 41-category weakness analysis
- [x] Qwen3-14B bf16 LoRA specialist + head-to-head comparison (v1: RED overall, 2 category wins, see the run journal)
- [x] v2: extractive span head, 0.389 test (+9.8 pts over v1; pre-registered 0.50 missed, published as committed)
- [x] AUPR vs model size curve (0.5B/1.5B/7B/14B: 0.284/0.289/0.300/0.303 dev, nearly flat; framing, not scale)
- [ ] v3: full training recipe (3 epochs, full negatives, dev-swept LR, calibration), the measured bottleneck
- [x] HuggingFace release: [nightwing-14b-cuad-extractive](https://huggingface.co/ashish24142/nightwing-14b-cuad-extractive) (v2 adapter + model card; card includes the required backbone-loading recipe)
- [ ] Demo Space

## Why "nightwing"?

Nightwing is the codename Dick Grayson takes when he stops being Batman's Robin
and becomes his own hero, the sidekick who trained up and started winning fights
the mentor couldn't. That is exactly this project's bet: a small, cheap specialist
that grows out of the frontier models' shadow and beats them on its own turf. It
doesn't out-punch the giant (Claude) across the board, no sidekick does on day
one, but on the categories it trained for, it lands clean hits on models many
times its size and cost. The name is the thesis: not a frontier replacement, a
specialist that earns its own name in the fields it owns.

## License & attribution

By [Ashish Kumar Singh](https://in.linkedin.com/in/ashish-kr-singh-ml). The full
write-ups live at [migrationassurance.com](https://migrationassurance.com/):
[the v1 experiment](https://migrationassurance.com/forty-dollar-specialist-vs-frontier/)
and [the v2 sequel](https://migrationassurance.com/one-change-ten-points/). The
trained adapter is on
[Hugging Face](https://huggingface.co/ashish24142/nightwing-14b-cuad-extractive).

Apache-2.0. CUAD is released by [The Atticus Project](https://www.atticusprojectai.org/)
under CC BY 4.0, cite [Hendrycks et al., 2021](https://arxiv.org/abs/2103.06268).
Base model: [Qwen](https://huggingface.co/Qwen) (Apache-2.0). This project
benchmarks frontier models but is trained **only** on CUAD ground truth, never
on frontier model outputs. Not legal advice.
