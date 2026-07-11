"""
prompt.py — model-agnostic prompt construction + response parsing.

Every backend (frontier or local) transports the SAME instructions and output
contract; only the HTTP/inference plumbing differs. This keeps the harness
swap-only (operating rule #3) and guarantees apples-to-apples comparison.

The model is asked, per (contract, category), to decide presence and extract the
supporting span(s) with a confidence. We convert that to the official CUAD
"nbest" format: {qid: [{"text", "probability"}, ...]} (see harness/scoring.py).
"""
from __future__ import annotations

import json
import re

# Static task instructions. Identical across all 41 questions for a contract,
# so it sits in the cacheable prefix (with the contract) for both providers.
SYSTEM_PROMPT = (
    "You are a contract-analysis expert performing clause extraction on the CUAD "
    "benchmark. You are given the full text of a commercial contract and asked "
    "about ONE clause category at a time.\n\n"
    "Your job: decide whether the contract contains a clause of the requested "
    "category, and if so, extract the EXACT verbatim span(s) of text from the "
    "contract that support it (copy the text exactly as it appears; do not "
    "paraphrase). Many categories are absent in any given contract — if the "
    "category is not present, say so. Do not invent spans.\n\n"
    "Respond with ONLY a JSON object, no prose, no code fences:\n"
    '{\n'
    '  "present": true | false,\n'
    '  "spans": ["<exact verbatim quote>", ...],   // [] if absent\n'
    '  "confidence": <number 0.0-1.0>              // your confidence the clause IS present\n'
    "}\n"
    "Rules: if present=false, spans must be []. confidence reflects how sure you "
    "are the clause is present. Keep spans tight and verbatim."
)

CONTRACT_HEADER = "<CONTRACT>\n"
CONTRACT_FOOTER = "\n</CONTRACT>"


def build_contract_block(contract_text: str) -> str:
    """The large, cacheable, per-contract prefix (shared across 41 questions)."""
    return CONTRACT_HEADER + contract_text + CONTRACT_FOOTER


def build_question_text(question: str, category: str) -> str:
    """The small, varying suffix — one of the 41 category questions."""
    return (
        f'Clause category: "{category}"\n'
        f"Question: {question}\n\n"
        "Return the JSON object now."
    )


_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


def _first_json_object(raw: str) -> dict | None:
    """First complete JSON object in raw, via raw_decode from each '{'.
    Unlike a greedy regex, trailing text containing '}' cannot break parsing."""
    dec = json.JSONDecoder()
    i = raw.find("{")
    while i != -1:
        try:
            obj, _ = dec.raw_decode(raw, i)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        i = raw.find("{", i + 1)
    return None


def parse_response(text: str) -> dict:
    """Robustly extract {present, spans, confidence} from a model response.
    On any parse failure, fall back to 'absent' (safe: contributes no spans).
    Strips reasoning-model <think> blocks (e.g. Qwen3) before parsing."""
    if not text:
        return {"present": False, "spans": [], "confidence": 0.0}
    raw = _THINK_RE.sub("", text).strip()
    obj = _first_json_object(raw)
    if obj is None:
        return {"present": False, "spans": [], "confidence": 0.0}
    present = bool(obj.get("present", False))
    spans = obj.get("spans") or []
    if not isinstance(spans, list):
        spans = [str(spans)]
    spans = [str(s) for s in spans if str(s).strip()]
    try:
        conf = float(obj.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = min(1.0, max(0.0, conf))
    if not present or not spans:
        return {"present": False, "spans": [], "confidence": conf}
    return {"present": True, "spans": spans, "confidence": conf}


def to_nbest(parsed: dict) -> list[dict]:
    """Convert parsed answer to official nbest entries for one qid.
    Absent -> [] (no predictions). Present -> each span at the question's
    confidence (the metric sweeps this confidence to trace the PR curve)."""
    if not parsed["present"] or not parsed["spans"]:
        return []
    conf = parsed["confidence"] if parsed["confidence"] > 0 else 0.5
    return [{"text": s, "probability": conf} for s in parsed["spans"]]
