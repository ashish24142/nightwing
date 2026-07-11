# ContractLM Pilot — Claude Code Working Brief

> **What this is:** A $1,000, ~2-week pilot to decide whether a fine-tuned 14B specialist model can beat frontier models (Claude, GPT, Gemini) on the CUAD contract-clause-classification benchmark. This produces a **go/no-go signal** before committing real money to a full training project — it is **not** the final model.
>
> **Your job (Claude Code):** Build the evaluation harness, run the frontier baselines, train one QLoRA pilot on free CUAD data, and produce an apples-to-apples comparison. Then stop and let a human apply the pre-committed decision criteria.

---

## 0. Read this first — operating rules

These are non-negotiable. Violating any of them invalidates the result.

1. **NEVER let the test set leak into training.** CUAD has a defined train/test split. Train ONLY on train. Evaluate ONLY on test. Before any training run, write and run a script that programmatically asserts zero overlap between train and test contract IDs. If you cannot prove zero overlap, STOP and report.
2. **NEVER invent a custom metric.** Use the official CUAD evaluation methodology (AUPR / precision-at-recall) so results are comparable to published numbers. If you must add a secondary metric, the official one is still primary and always reported.
3. **The harness MUST be model-agnostic from the start.** Swapping between a frontier API backend and a local-model backend must require only a config/flag change — never a code rewrite. This harness is reused for the entire future project; it is not throwaway.
4. **Measure cost as you go.** Log token counts and dollar cost for every frontier API run. The total project budget is $1,000; alert if cumulative spend crosses $700.
5. **Run a 5-contract smoke test before any full run.** Never kick off a full-test-split frontier run (expensive) or a full training run without first proving the pipeline works on 5 contracts. Measure per-contract cost on the smoke test and extrapolate before committing to the full run.
6. **Checkpoint training frequently.** Spot GPUs get preempted. Save checkpoints so a preemption never loses a full run.
7. **Do not move the goalposts.** The decision thresholds in Section 7 are fixed. Your job is to produce the numbers, not to interpret them favorably.
8. **Ask before spending on anything ambiguous.** If a step's cost is unclear or could exceed its budget line, stop and surface it rather than proceeding.

---

## 1. Goal & success definition

**Goal of the pilot:** Determine, cheaply, whether an unoptimized fine-tuned Qwen-14B (QLoRA, free CUAD training data, single run) gets *into the neighborhood* of frontier-model performance on CUAD clause classification.

**This pilot succeeds if** it produces, with full methodological integrity:
- Frontier baseline scores (Claude + GPT) on the CUAD test split, overall and per-category.
- A trained 14B pilot model scored on the identical test split with the identical harness.
- A clean comparison table enabling a human to classify the result as Green / Yellow / Red (Section 7).

**This pilot does NOT need to:** produce a production-quality model, use data augmentation, run multiple CRAFT iteration cycles, or beat frontier. It needs to produce an honest signal.

---

## 2. Background: what CUAD is

- **CUAD** (Contract Understanding Atticus Dataset): 510 commercial contracts, 13,000+ expert annotations, **41 clause categories**. Public, free. Source: HuggingFace (`theatticusproject/cuad-qa`) and GitHub (`TheAtticusProject/cuad`).
- **Task framing for this pilot:** clause classification — for each contract and each of the 41 categories, identify whether the clause is present and (where applicable) the supporting span(s).
- **Scoring:** Use the official CUAD evaluation (AUPR, precision-at-recall thresholds). The original CUAD repo ships an evaluation script — use it as the scoring foundation.
- **Split:** Use CUAD's defined train/test split. Do not create your own random split (risks contamination and breaks comparability).

---

## 3. Tech stack (decisions already made — don't re-litigate)

| Component | Choice | Notes |
|---|---|---|
| Base model | Qwen 2.5 14B or Qwen 3 14B | Permissive license; strong reasoning at 14B |
| Fine-tuning | QLoRA | Use **Unsloth** (fastest, most memory-efficient single-GPU) or Axolotl |
| GPU | Single A100 80GB or H100, **spot/interruptible** | RunPod / Lambda / Vast.ai. 14B QLoRA needs ~24–40GB |
| Frontier baselines | Claude Opus 4.x + GPT-5-class (Gemini optional) | Use prompt caching to cut cost ~80–90% |
| Scoring | Official CUAD eval script | Do not reimplement the metric from scratch |
| Language | Python | — |
| Experiment tracking | Weights & Biases (free tier) or local logs | Track loss, eval scores, cost |

**Cost-control note on frontier baselines:** CUAD test contracts each get asked up to 41 category questions. **Cache the contract once, vary the question.** The contract is the bulk of the tokens; caching it across all 41 questions is the single biggest cost saver.

---

## 4. Repository structure to create

```
contractlm-pilot/
├── README.md                    # how to run everything
├── CLAUDE.md                    # this file
├── requirements.txt
├── config/
│   ├── models.yaml              # backend configs (claude / gpt / local) — the swap point
│   └── training.yaml            # QLoRA hyperparameters
├── data/
│   ├── download_cuad.py         # fetch CUAD
│   ├── prepare_train.py         # CUAD train split -> instruction format
│   ├── verify_split.py          # ASSERT zero train/test contamination
│   └── .gitignore               # do NOT commit raw data / model weights
├── harness/
│   ├── backends/
│   │   ├── base.py              # abstract backend interface
│   │   ├── frontier_api.py      # Claude / GPT / Gemini via API (with caching)
│   │   └── local_model.py       # local fine-tuned model inference
│   ├── run_eval.py              # main harness entry point (model-agnostic)
│   ├── scoring.py               # wraps official CUAD metric
│   └── cost_tracker.py          # logs tokens + $ per run
├── training/
│   ├── train_qlora.py           # the pilot fine-tune
│   └── smoke_test.py            # 10-step training sanity check
├── analysis/
│   ├── category_breakdown.py    # 41-category weakness table
│   └── build_comparison.py      # pilot vs frontier, overall + per-category
└── results/
    ├── baselines/               # frontier scores (committed — small JSON)
    ├── pilot/                   # pilot model scores (committed — small JSON)
    └── comparison.md            # the final output table
```

**Commit policy:** Commit code, configs, and small JSON result files. NEVER commit raw CUAD data, model weights, or API keys. Use `.gitignore` and environment variables for secrets.

---

## 5. Task breakdown (execute in order)

### PHASE 1 — Evaluation harness & frontier baseline

**P1.1 — Acquire & understand CUAD**
- Download CUAD (train + test). Confirm the defined split. Document, in the README, exactly what the model must output and how it's scored.
- **DoD:** CUAD downloaded; train/test split identified; task framing documented in one paragraph.

**P1.2 — Stand up official-metric scoring**
- Wire the official CUAD evaluation script into `harness/scoring.py`. Sanity-check: scoring ground-truth-against-itself returns a perfect score.
- **DoD:** Official scoring runs locally; ground-truth-vs-ground-truth = perfect; any predictions file can be scored.

**P1.3 — Build the model-agnostic harness**
- Implement `harness/backends/base.py` (abstract interface), `frontier_api.py`, and `local_model.py`. Implement `run_eval.py` that: loads test contracts + categories → calls the configured backend → parses responses into CUAD format → scores via P1.2.
- **DoD:** Harness runs end-to-end on 5 test contracts with a frontier backend and produces scored output. Backend swap is config-only.

**P1.4 — Frontier baseline: Claude**
- Smoke test on 5 contracts first; measure per-contract cost; extrapolate. Then run the full CUAD test split through Claude Opus 4.x with prompt caching. Record overall + per-category scores and cost.
- **DoD:** Claude scored on full test split; overall + 41-category scores saved to `results/baselines/`; cost logged.

**P1.5 — Frontier baseline: GPT**
- Same as P1.4 for a GPT-5-class model. Identical prompts/harness. (Gemini optional — skip to save budget.)
- **DoD:** GPT scored; overall + per-category saved alongside Claude; cost logged.

**P1.6 — Category-level weakness analysis**
- Build the 41-category × frontier-model table in `analysis/category_breakdown.py`. Identify the top ~10 categories where frontier is weakest (most room for a specialist to win).
- **DoD:** 41-row ranking exists; "top 10 winnable categories" listed; one paragraph of interpretation.

**🏁 PHASE 1 MILESTONE:** Reusable harness built; frontier baselines measured; target categories identified; Phase 1 spend < $400.

### PHASE 2 — Proof-of-concept training run

**P2.1 — Set up training environment**
- Provision spot GPU. Install Unsloth/Axolotl. Download Qwen-14B base. Run a 10-step smoke test (`training/smoke_test.py`).
- **DoD:** GPU running; framework installed; base weights loaded; 10-step smoke test completes without OOM/errors.

**P2.2 — Prepare CUAD training data**
- Convert CUAD **train** split into instruction format (input: clause/section + question; output: labeled answer). Run `data/verify_split.py` to PROVE zero test contamination.
- **DoD:** Train split converted; `verify_split.py` passes (zero overlap, checked in code); dataset loads into the framework.

**P2.3 — Train the 14B QLoRA pilot**
- One QLoRA run, 3–4 epochs, standard hyperparameters. Checkpoint frequently. Monitor loss. If the first run is poor, attempt one more with adjusted hyperparameters (within budget).
- **DoD:** ≥1 run completes; loss decreased sensibly; usable adapter/checkpoint saved.

**P2.4 — Evaluate pilot on the SAME harness**
- Run the fine-tuned model through `run_eval.py` with the local backend. Same test split, same metric.
- **DoD:** Pilot scored on identical test split; overall + per-category scores in the same format as baselines, saved to `results/pilot/`.

**P2.5 — Build the comparison**
- `analysis/build_comparison.py`: pilot vs Claude vs GPT, overall + per-category. Flag categories where the pilot wins/matches. Compute gap-to-frontier overall and per category. Write `results/comparison.md`.
- **DoD:** `comparison.md` exists with the full table; winning categories flagged; gaps computed.

**🏁 PHASE 2 MILESTONE:** Pilot trained and scored; comparison complete; total spend < $1,000.

### DECISION (human-led — Claude Code stops here)

**D.1 — Surface the result against the criteria**
- Present the comparison and state which signal band (Section 7) the result falls into. **Do not decide for the human** — present the numbers and the band, and let the human make the funding call.
- **DoD:** Result classified Green/Yellow/Red against fixed thresholds; numbers presented clearly.

---

## 6. Budget guardrails

| Line | Budget |
|---|---|
| Phase 1 frontier API calls (Claude + GPT, with caching) | ~$300 |
| Phase 1 harness compute | ~$20 |
| Phase 2 training (1–2 QLoRA runs) | ~$80 |
| Phase 2 own-model eval | ~$50 |
| Phase 2 one frontier subset comparison run | ~$100 |
| Contingency | ~$200 |
| **Total** | **< $1,000** |

**Alert the human if cumulative spend crosses $700.** Always smoke-test (5 contracts) and measure per-contract cost before any full frontier run.

---

## 7. Decision criteria (FIXED — do not alter)

These were committed **before** the experiment. Claude Code presents the result against them; the human decides.

| Signal | Pilot result vs frontier | Meaning | Action |
|---|---|---|---|
| 🟢 **GREEN** | Within **3–4 pts** of frontier overall, OR beats frontier in a meaningful number of individual categories | An unoptimized pilot on free data is already this close. The full project's hard-negative mining, synthetic augmentation, and iteration will very likely close the gap and pull ahead. | **Fund** the full $5–7K CUAD project; seriously consider the $28K three-benchmark project. |
| 🟡 **YELLOW** | **5–8 pts** below frontier overall, but strong in specific categories | Gap is real but plausibly closable. Worth one more cheap experiment before committing. | **Iterate once** more cheaply (~$150); re-apply criteria. |
| 🔴 **RED** | **>8–10 pts** below frontier across the board, no competitive categories | If a 14B can't get close on the EASY benchmark with clean data, the hard benchmarks are out of reach. | **Do not fund** as designed. Rethink (bigger base model? different framing?). A Red result is a *successful* pilot outcome — it saved ~$27K. |

**Caveat for interpretation:** The pilot is deliberately underpowered (no augmentation, single run). A Red result means "rethink," not "specialists categorically can't work." Diagnose *why* before abandoning or scaling.

---

## 8. Final deliverables checklist

- [ ] `harness/` — model-agnostic CUAD evaluation harness (reusable asset)
- [ ] `results/baselines/` — Claude + GPT scores, overall + per-category (JSON)
- [ ] `analysis/category_breakdown.py` output — 41-category weakness table
- [ ] Trained pilot adapter/checkpoint (not committed; path documented)
- [ ] `results/pilot/` — pilot scores, overall + per-category (JSON)
- [ ] `results/comparison.md` — pilot vs frontier, overall + per-category, gaps, flagged wins
- [ ] Cost log — total spend with per-run breakdown
- [ ] Result classified Green/Yellow/Red, presented to human for the funding decision

---

## 9. What NOT to do (common failure modes)

- ❌ Don't reimplement the CUAD metric — use the official one.
- ❌ Don't create a custom train/test split — use CUAD's defined split.
- ❌ Don't skip the contamination check — prove zero overlap in code.
- ❌ Don't run a full frontier eval without a 5-contract smoke test and cost extrapolation first.
- ❌ Don't add data augmentation, multi-task training, or CRAFT cycles — those are full-project scope, not pilot scope.
- ❌ Don't commit raw data, weights, or API keys.
- ❌ Don't tune the result to look good, and don't soften the Red threshold. The value of this pilot is an honest signal.
- ❌ Don't make the funding decision autonomously — present the numbers and the band; the human decides.

---

## 10. Environment / secrets

- API keys via environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.). Never hardcode. Never commit.
- GPU credentials per provider; keep out of the repo.
- `requirements.txt` pinned. Document Python version in README.
