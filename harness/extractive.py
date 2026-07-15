"""
extractive.py — v2 span-extraction path (SQuAD-native), shared by train + eval.

v1 reframed CUAD as *generation*; v2 uses it as what it already is: extractive
QA. We reuse transformers' QA head (`AutoModelForQuestionAnswering`, which
resolves to Qwen{2,3}ForQuestionAnswering for the Qwen family) + sliding-window
tokenization. NO custom span head, NO reimplemented metric — postprocessed
n-best feeds the SAME official scorer used for every frontier baseline
(harness/scoring.py), so v1 and v2 numbers are directly comparable.

Windowing: long contracts are tokenized with `stride` overflow so a gold span
that straddles one window boundary still lands whole inside a neighbour.
Null anchor: token index 0 (Qwen has no [CLS]); impossible answers train to
(0,0), and P(present)=sigmoid(best_span_score - min_null_score).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


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
def load_examples(split_json: str | Path, limit: int | None = None) -> list[dict]:
    """Flatten CUAD SQuAD JSON -> [{id, question, context, answers, is_impossible}].
    `limit` caps the number of CONTRACTS (for smoke runs)."""
    data = json.loads(Path(split_json).read_text(encoding="utf-8"))["data"]
    if limit is not None:
        data = data[:limit]
    out = []
    for contract in data:
        for para in contract["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                out.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "context": ctx,
                    "answers": qa.get("answers", []),
                    "is_impossible": qa.get("is_impossible", False),
                })
    return out


# ---------------------------------------------------------------------------
# Feature building (sliding-window tokenization)
# ---------------------------------------------------------------------------
def build_features(examples: list[dict], tokenizer, max_len: int, stride: int,
                   is_training: bool) -> list[dict]:
    """One example -> one or more windowed features.

    Training features carry start/end token positions (answer not fully inside a
    window -> (0,0) null). Eval features carry offset_mapping (context tokens
    only) + example_id for postprocessing back to char spans."""
    feats: list[dict] = []
    for ex in examples:
        enc = tokenizer(
            ex["question"], ex["context"],
            truncation="only_second", max_length=max_len, stride=stride,
            return_overflowing_tokens=True, return_offsets_mapping=True,
            padding="max_length",
        )
        n_windows = len(enc["input_ids"])
        # char span of the (first) answer, if any
        ans_start = ans_end = None
        if not ex["is_impossible"] and ex["answers"]:
            a = ex["answers"][0]
            ans_start = a["answer_start"]
            ans_end = ans_start + len(a["text"])

        for w in range(n_windows):
            seq_ids = enc.sequence_ids(w)
            offsets = enc["offset_mapping"][w]
            input_ids = enc["input_ids"][w]
            attn = enc["attention_mask"][w]
            # context token index range within this window
            ctx_tok = [i for i, s in enumerate(seq_ids) if s == 1]

            if is_training:
                start_pos = end_pos = 0  # null anchor
                if ans_start is not None and ctx_tok:
                    c0, c1 = ctx_tok[0], ctx_tok[-1]
                    # answer fully inside this window's context span?
                    if offsets[c0][0] <= ans_start and offsets[c1][1] >= ans_end:
                        s = c0
                        while s <= c1 and offsets[s][1] <= ans_start:
                            s += 1
                        e = c1
                        while e >= c0 and offsets[e][0] >= ans_end:
                            e -= 1
                        if c0 <= s <= e <= c1:
                            start_pos, end_pos = s, e
                feats.append({
                    "input_ids": input_ids, "attention_mask": attn,
                    "start_positions": start_pos, "end_positions": end_pos,
                })
            else:
                # keep only context offsets (mask out question/special = None)
                ctx_offsets = [tuple(o) if seq_ids[i] == 1 else None
                               for i, o in enumerate(offsets)]
                feats.append({
                    "input_ids": input_ids, "attention_mask": attn,
                    "offset_mapping": ctx_offsets, "example_id": ex["id"],
                })
    return feats


# ---------------------------------------------------------------------------
# Postprocess start/end logits -> official n-best {qid: [{text, probability}]}
# ---------------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def postprocess(examples: list[dict], eval_feats: list[dict],
                start_logits, end_logits, n_best: int = 20,
                max_answer_len: int = 512) -> dict:
    """Reduce per-window logits to one best span per qid, with a calibrated
    P(present)=sigmoid(best_span_score - min_null_score) confidence for the PR
    sweep. Every qid in `examples` gets exactly one n-best entry."""
    ctx_by_id = {ex["id"]: ex["context"] for ex in examples}
    feats_by_id: dict[str, list[int]] = {}
    for idx, f in enumerate(eval_feats):
        feats_by_id.setdefault(f["example_id"], []).append(idx)

    nbest: dict[str, list[dict]] = {}
    for ex in examples:
        qid = ex["id"]
        best_score = -1e9
        best_text = ""
        min_null = 1e9
        for fi in feats_by_id.get(qid, []):
            sl = start_logits[fi]
            el = end_logits[fi]
            offsets = eval_feats[fi]["offset_mapping"]
            min_null = min(min_null, float(sl[0]) + float(el[0]))
            start_idx = sorted(range(len(sl)), key=lambda i: sl[i], reverse=True)[:n_best]
            end_idx = sorted(range(len(el)), key=lambda i: el[i], reverse=True)[:n_best]
            for s in start_idx:
                if offsets[s] is None:
                    continue
                for e in end_idx:
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

    # (1) training span maps back to the gold text
    tf = build_features(ex, tok, max_len=64, stride=16, is_training=True)
    hit = next((tok.decode(f["input_ids"][f["start_positions"]:f["end_positions"] + 1])
                for f in tf if f["start_positions"] != 0), "")
    assert gold in hit, f"span mapping broke: {hit!r} lacks {gold!r}"

    # (2) postprocess picks the gold span when logits peak on its tokens
    ef = build_features(ex, tok, max_len=64, stride=16, is_training=False)
    f0 = ef[0]
    s = tf[0]["start_positions"]; e = tf[0]["end_positions"]
    starts = [[0.0] * len(f0["input_ids"])]
    ends = [[0.0] * len(f0["input_ids"])]
    starts[0][s] = 10.0
    ends[0][e] = 10.0
    nb = postprocess(ex, [f0], starts, ends)
    got = nb["T__Agreement Date"][0]
    assert gold in got["text"], f"postprocess picked {got['text']!r}, not {gold!r}"
    assert got["probability"] > 0.9, f"confidence too low: {got['probability']}"
    print(f"PASS: span maps to {hit!r}; postprocess recovers {got['text']!r} "
          f"@ p={got['probability']:.3f}")


if __name__ == "__main__":
    _selfcheck()
