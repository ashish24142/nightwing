"""
extractive.py — v2 span-extraction path (SQuAD-native), shared by train + eval.

v1 reframed CUAD as *generation*; v2 uses it as what it already is: extractive
QA. We reuse transformers' QA head (`AutoModelForQuestionAnswering`, which
resolves to Qwen{2,3}ForQuestionAnswering for the Qwen family) + sliding-window
tokenization. NO custom span head, NO reimplemented metric — postprocessed
n-best feeds the SAME official scorer used for every frontier baseline
(harness/scoring.py), so v1 and v2 numbers are directly comparable.

Features are built once into a disk-cached HF `datasets` Dataset (arrow,
memory-mapped) so full CUAD (~450k windows) never has to fit in RAM, and the
0.5B/1.5B runs (identical Qwen2.5 tokenizer) reuse the same cache.

Windowing: long contracts are tokenized with `stride` overflow so a gold span
that straddles one window boundary still lands whole inside a neighbour.
Null anchor: token index 0 (Qwen has no [CLS]); impossible answers train to
(0,0), and P(present)=sigmoid(best_span_score - min_null_score).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = ROOT / "outputs" / "ext_cache"


# ---------------------------------------------------------------------------
# Model loading — CRITICAL correctness fix
# ---------------------------------------------------------------------------
def load_qa_model(base_model: str, dtype=None):
    """`AutoModelForQuestionAnswering` for the Qwen family names its backbone
    `transformer.*`, but the pretrained checkpoint stores it as `model.*`.
    HF cannot remap a foreign prefix, so the backbone silently loads RANDOM
    (only qa_outputs is expected-random). Left unfixed, training/eval run on a
    randomly-initialised backbone and every number is meaningless. We inject the
    real pretrained weights explicitly and ASSERT they landed."""
    import torch
    from transformers import AutoModel, AutoModelForQuestionAnswering
    if dtype is None:
        dtype = torch.bfloat16
    qa = AutoModelForQuestionAnswering.from_pretrained(base_model, dtype=dtype)
    bb = getattr(qa, qa.base_model_prefix, None)
    if bb is not None:
        src = AutoModel.from_pretrained(base_model, dtype=dtype)
        missing, _ = bb.load_state_dict(src.state_dict(), strict=False)
        del src
        broken = [m for m in missing if "embed" in m or "layers" in m]
        assert not broken, f"backbone failed to load (still random): {broken[:3]}"
    return qa


# ---------------------------------------------------------------------------
# Data loading (CUAD is already SQuAD-nested JSON)
# ---------------------------------------------------------------------------
def _titles(split_json: str | Path) -> set[str]:
    return {c["title"] for c in
            json.loads(Path(split_json).read_text(encoding="utf-8"))["data"]}


def load_examples(split_json: str | Path, limit: int | None = None,
                  exclude_json: str | Path | None = None) -> list[dict]:
    """Flatten CUAD SQuAD JSON -> [{id, question, context, answers, is_impossible}].
    `limit` caps the number of CONTRACTS (for smoke runs). `exclude_json` drops
    any contract whose title appears in that split (rule #1: dev is carved from
    train, so training on train.json must exclude the dev contracts)."""
    data = json.loads(Path(split_json).read_text(encoding="utf-8"))["data"]
    if exclude_json is not None:
        drop = _titles(exclude_json)
        data = [c for c in data if c["title"] not in drop]
    if limit is not None:
        data = data[:limit]
    out = []
    for contract in data:
        for para in contract["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                out.append({
                    "id": qa["id"],
                    "title": contract["title"],
                    "question": qa["question"],
                    "context": ctx,
                    "answers": qa.get("answers", []),
                    "is_impossible": qa.get("is_impossible", False),
                })
    return out


# ---------------------------------------------------------------------------
# Char span -> token span (the error-prone bit; one impl, used everywhere)
# ---------------------------------------------------------------------------
def _char_to_token_span(offsets, seq_ids, a0: int, a1: int) -> tuple[int, int]:
    """Token (start, end) of char span [a0,a1) within this window, or (0,0) if
    the answer is not fully inside the window's context tokens (null anchor)."""
    ctx = [i for i, s in enumerate(seq_ids) if s == 1]
    if not ctx:
        return 0, 0
    c0, c1 = ctx[0], ctx[-1]
    if not (offsets[c0][0] <= a0 and offsets[c1][1] >= a1):
        return 0, 0
    s = c0
    while s <= c1 and offsets[s][1] <= a0:
        s += 1
    e = c1
    while e >= c0 and offsets[e][0] >= a1:
        e -= 1
    return (s, e) if c0 <= s <= e <= c1 else (0, 0)


def _tokenize_batch(batch, tokenizer, max_len, stride, is_training):
    """datasets.map(batched=True) function: (question, context) -> N windows."""
    enc = tokenizer(
        batch["question"], batch["context"],
        truncation="only_second", max_length=max_len, stride=stride,
        return_overflowing_tokens=True, return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = enc.pop("overflow_to_sample_mapping")
    offset_map = enc.pop("offset_mapping")
    if is_training:
        starts, ends = [], []
        for i, offsets in enumerate(offset_map):
            si = sample_map[i]
            ans = batch["answers"][si]
            if batch["is_impossible"][si] or not ans:
                starts.append(0); ends.append(0); continue
            a = ans[0]; a0 = a["answer_start"]; a1 = a0 + len(a["text"])
            s, e = _char_to_token_span(offsets, enc.sequence_ids(i), a0, a1)
            starts.append(s); ends.append(e)
        enc["start_positions"] = starts
        enc["end_positions"] = ends
    else:
        ex_ids, new_off = [], []
        for i, offsets in enumerate(offset_map):
            seq = enc.sequence_ids(i)
            ex_ids.append(batch["id"][sample_map[i]])
            new_off.append([list(o) if seq[k] == 1 else None
                            for k, o in enumerate(offsets)])
        enc["example_id"] = ex_ids
        enc["offset_mapping"] = new_off
    return enc


def make_dataset(examples, tokenizer, max_len, stride, is_training,
                 neg_per_pos=None, seed=42, cache_dir=None):
    """Build (or load) a disk-cached arrow Dataset of windowed features.
    Training features carry start/end positions (+ optional negative
    subsampling); eval features carry offset_mapping + example_id."""
    from datasets import Dataset, load_from_disk
    if cache_dir is not None and Path(cache_dir).exists():
        return load_from_disk(str(cache_dir))
    raw = Dataset.from_list(examples)
    feat = raw.map(
        lambda b: _tokenize_batch(b, tokenizer, max_len, stride, is_training),
        batched=True, remove_columns=raw.column_names,
        desc="tokenize->windows",
    )
    if is_training and neg_per_pos is not None:
        is_pos = [s != 0 for s in feat["start_positions"]]
        pos = [i for i, p in enumerate(is_pos) if p]
        neg = [i for i, p in enumerate(is_pos) if not p]
        random.Random(seed).shuffle(neg)
        keep = sorted(set(pos) | set(neg[:len(pos) * neg_per_pos]))
        feat = feat.select(keep)
    if cache_dir is not None:
        feat.save_to_disk(str(cache_dir))
    return feat


def build_features(examples, tokenizer, max_len, stride, is_training) -> list[dict]:
    """Small in-memory list of features (self-check / tiny smokes). Full runs
    use make_dataset() directly to keep features on disk."""
    return make_dataset(examples, tokenizer, max_len, stride, is_training).to_list()


# ---------------------------------------------------------------------------
# Postprocess start/end logits -> official n-best {qid: [{text, probability}]}
# ---------------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def postprocess(examples, example_ids, offsets_of, start_logits, end_logits,
                n_best: int = 20, max_answer_len: int = 512) -> dict:
    """Reduce per-window logits to one best span per qid, with a calibrated
    P(present)=sigmoid(best_span_score - min_null_score) for the PR sweep.

    example_ids: per-feature qid (list, len == #features).
    offsets_of:  callable(feature_index) -> that feature's offset_mapping
                 (context-token offsets; None elsewhere). Lets the caller read
                 offsets lazily from disk instead of materialising them all.
    Every qid in `examples` gets exactly one n-best entry."""
    ctx_by_id = {ex["id"]: ex["context"] for ex in examples}
    feats_by_id: dict[str, list[int]] = {}
    for idx, qid in enumerate(example_ids):
        feats_by_id.setdefault(qid, []).append(idx)

    nbest: dict[str, list[dict]] = {}
    for ex in examples:
        qid = ex["id"]
        best_score, best_text, min_null = -1e9, "", 1e9
        for fi in feats_by_id.get(qid, []):
            sl, el = start_logits[fi], end_logits[fi]
            offsets = offsets_of(fi)
            min_null = min(min_null, float(sl[0]) + float(el[0]))
            top_s = sorted(range(len(sl)), key=lambda i: sl[i], reverse=True)[:n_best]
            top_e = sorted(range(len(el)), key=lambda i: el[i], reverse=True)[:n_best]
            for s in top_s:
                if offsets[s] is None:
                    continue
                for e in top_e:
                    if offsets[e] is None or e < s or (e - s + 1) > max_answer_len:
                        continue
                    score = float(sl[s]) + float(el[e])
                    if score > best_score:
                        best_score = score
                        best_text = ctx_by_id[qid][offsets[s][0]:offsets[e][1]]
        prob = _sigmoid(best_score - min_null) if best_text else 0.0
        nbest[qid] = [{"text": best_text, "probability": prob}]
    return nbest


# ---------------------------------------------------------------------------
# Self-check (CPU, no model weights) — the two error-prone pieces:
# char->token span mapping (train) and logit->span selection (eval).
#   python -m harness.extractive
# ---------------------------------------------------------------------------
def _selfcheck() -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ctx = "This Supply Agreement is dated March 3, 2020 between Acme and Beta."
    gold = "March 3, 2020"
    ex = [{"id": "T__Agreement Date", "question": 'parts related to "Agreement Date"',
           "context": ctx, "answers": [{"text": gold, "answer_start": ctx.index(gold)}],
           "is_impossible": False}]

    tf = build_features(ex, tok, max_len=64, stride=16, is_training=True)
    hit = next((tok.decode(f["input_ids"][f["start_positions"]:f["end_positions"] + 1])
                for f in tf if f["start_positions"] != 0), "")
    assert gold in hit, f"span mapping broke: {hit!r} lacks {gold!r}"

    ef = build_features(ex, tok, max_len=64, stride=16, is_training=False)
    f0 = ef[0]
    s, e = tf[0]["start_positions"], tf[0]["end_positions"]
    starts = [[0.0] * len(f0["input_ids"])]
    ends = [[0.0] * len(f0["input_ids"])]
    starts[0][s], ends[0][e] = 10.0, 10.0
    nb = postprocess(ex, [f0["example_id"]], lambda fi: f0["offset_mapping"],
                     starts, ends)
    got = nb["T__Agreement Date"][0]
    assert gold in got["text"], f"postprocess picked {got['text']!r}, not {gold!r}"
    assert got["probability"] > 0.9, f"confidence too low: {got['probability']}"
    print(f"PASS: span maps to {hit!r}; postprocess recovers {got['text']!r} "
          f"@ p={got['probability']:.3f}")


if __name__ == "__main__":
    _selfcheck()
