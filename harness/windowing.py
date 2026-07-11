"""
windowing.py — shared sliding-window logic for CUAD long contracts.

Contracts are far longer than a 14B QLoRA's training sequence length, so we
follow CUAD's SQuAD-style approach: split each contract into overlapping
character windows. The SAME function is used by:
  - data/prepare_train.py   (build training examples per window)
  - harness/backends/local_model.py (P2.4: query each window, aggregate spans)
so the pilot is trained and evaluated on identical context units (rule #3).

Windows are sized in CHARACTERS; calibration from the frontier runs is
~2.85 chars/token, so the default 8000-char window ≈ 2800 tokens, leaving room
for the instructions + question + answer inside a 4096-token sequence.
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_WIN_CHARS = 8000
DEFAULT_OVERLAP_CHARS = 2000


@dataclass
class Window:
    start: int          # char offset (inclusive)
    end: int            # char offset (exclusive)
    text: str


def iter_windows(text: str, win_chars: int = DEFAULT_WIN_CHARS,
                 overlap_chars: int = DEFAULT_OVERLAP_CHARS) -> list[Window]:
    """Overlapping char windows covering `text`. Overlap avoids splitting a
    gold span across a window boundary so it never falls in any window."""
    if win_chars <= overlap_chars:
        raise ValueError("win_chars must exceed overlap_chars")
    n = len(text)
    if n <= win_chars:
        return [Window(0, n, text)]
    step = win_chars - overlap_chars
    windows = []
    start = 0
    while start < n:
        end = min(start + win_chars, n)
        windows.append(Window(start, end, text[start:end]))
        if end == n:
            break
        start += step
    return windows


def spans_in_window(answers: list[dict], win: Window) -> list[str]:
    """Gold spans that lie FULLY inside `win` (by char offset). Each answer is
    {"text", "answer_start"}. Returns the span texts (deduped, order-preserved)."""
    out, seen = [], set()
    for a in answers:
        s = a.get("answer_start", -1)
        t = a.get("text", "")
        if s < 0 or not t:
            continue
        if s >= win.start and (s + len(t)) <= win.end:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out
