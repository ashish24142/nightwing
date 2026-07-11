"""
test_hardening.py — offline failure-path tests (no GPU, no API, no money).

Exercises the hardened code paths: malformed model output, windowing edges,
API retry/giveup logic, corrupt-ledger recovery, stale-lock stealing, concurrent
ledger commits, and missing-file errors.

Run:  python -m tests.test_hardening      (exit 0 = all pass)
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

from harness import prompt as P
from harness import windowing as W
from harness.backends import frontier_api as FA

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------------------
# 1. prompt.parse_response / to_nbest — malformed model output
# ---------------------------------------------------------------------------
def test_parse():
    cases = [
        ("", False),                                            # empty
        ("not json at all", False),                             # garbage
        ('{"present": true, "spans": ["X"], "confidence": 0.9}', True),
        ('```json\n{"present": false, "spans": [], "confidence": 0.1}\n```', False),
        ('{"present": true, "spans": []}', False),              # present but no spans
        ('{"present": true, "spans": "single", "confidence": 5}', True),  # non-list span, conf clamp
        ('prefix {"present": true, "spans": ["A"], "confidence": "oops"} suffix', True),
        ('{"present": true, "spans": [123, "", "B"]}', True),   # mixed/empty spans
        # trailing text containing '}' must not break parsing (greedy-regex bug)
        ('{"present": true, "spans": ["X"], "confidence": 0.7} see clause 7.2 {sic}', True),
        # Qwen3-style <think> reasoning stripped before parsing
        ('<think>hmm {"present": false}</think>{"present": true, "spans": ["Y"], "confidence": 0.8}', True),
        ('<think>never closed, no json after', False),          # unterminated think
    ]
    for text, expect_present in cases:
        p = P.parse_response(text)
        check(f"parse present={expect_present!s:5} <- {text[:30]!r}",
              p["present"] == expect_present, f"got {p}")
        # confidence always in range, spans always list of non-empty str
        check("  conf in [0,1]", 0.0 <= p["confidence"] <= 1.0, str(p["confidence"]))
        check("  spans are clean", all(isinstance(s, str) and s for s in p["spans"]))
    # to_nbest never raises and is empty when absent
    check("to_nbest absent->[]", P.to_nbest({"present": False, "spans": [], "confidence": 0}) == [])
    nb = P.to_nbest({"present": True, "spans": ["A", "B"], "confidence": 0.0})
    check("to_nbest present default conf>0", all(e["probability"] > 0 for e in nb))


# ---------------------------------------------------------------------------
# 2. windowing — edges
# ---------------------------------------------------------------------------
def test_windowing():
    check("empty text -> no windows", W.iter_windows("") == [])
    short = W.iter_windows("abc", 100, 10)
    check("short text -> single window", len(short) == 1 and short[0].text == "abc")
    text = "x" * 25000
    wins = W.iter_windows(text, 8000, 2000)
    check("long text -> multiple windows", len(wins) > 1)
    check("windows cover end", wins[-1].end == len(text))
    check("windows overlap", wins[1].start < wins[0].end)
    try:
        W.iter_windows("abc", 100, 100)  # win<=overlap must raise
        check("bad overlap raises", False)
    except ValueError:
        check("bad overlap raises", True)
    # spans_in_window: only fully-contained spans
    win = W.Window(10, 20, "0123456789")
    ans = [{"text": "abc", "answer_start": 12},   # inside
           {"text": "zzz", "answer_start": 18},   # crosses end (18+3>20) -> out
           {"text": "qq", "answer_start": 5}]      # before window -> out
    got = W.spans_in_window(ans, win)
    check("spans_in_window keeps only contained", got == ["abc"], str(got))


# ---------------------------------------------------------------------------
# 3. frontier_api._post_with_retries — retry / give-up / bad bodies
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, status, body, is_json=True, headers=None):
        self.status_code = status
        self._body = body
        self._is_json = is_json
        self.headers = headers or {}
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        if not self._is_json:
            raise ValueError("no json")
        return self._body


def test_retries():
    orig_post, orig_sleep = FA.requests.post, FA.time.sleep
    FA.time.sleep = lambda *_: None  # no real waiting
    try:
        # transient 500 then 200 -> succeeds
        seq = [_FakeResp(500, "err"), _FakeResp(200, {"ok": 1})]
        FA.requests.post = lambda *a, **k: seq.pop(0)
        check("retry 500 then 200", FA._post_with_retries("u", {}, {}) == {"ok": 1})

        # 200 but non-JSON repeatedly -> gives up (RuntimeError)
        FA.requests.post = lambda *a, **k: _FakeResp(200, "<html>", is_json=False)
        try:
            FA._post_with_retries("u", {}, {})
            check("non-JSON 200 gives up", False)
        except RuntimeError:
            check("non-JSON 200 gives up", True)

        # non-retryable 400 -> fails fast
        FA.requests.post = lambda *a, **k: _FakeResp(400, "bad request")
        try:
            FA._post_with_retries("u", {}, {})
            check("400 fails fast", False)
        except RuntimeError as e:
            check("400 fails fast", "non-retryable" in str(e))

        # network exception every time -> gives up
        def boom(*a, **k):
            raise FA.requests.RequestException("conn reset")
        FA.requests.post = boom
        try:
            FA._post_with_retries("u", {}, {})
            check("network error gives up", False)
        except RuntimeError:
            check("network error gives up", True)
    finally:
        FA.requests.post, FA.time.sleep = orig_post, orig_sleep


# ---------------------------------------------------------------------------
# 4. cost_tracker — corrupt ledger, stale lock, concurrent commits
#    (uses a TEMP ledger — never touches the real results/cost_log.json)
# ---------------------------------------------------------------------------
def test_cost_tracker():
    from harness import cost_tracker as CT
    real_ledger = CT.LEDGER_PATH
    tmpdir = Path(tempfile.mkdtemp())
    CT.LEDGER_PATH = tmpdir / "cost_log.json"
    pricing = {"m": {"input": 1.0, "cache_write": 1.0, "cache_read": 1.0, "output": 1.0}}
    try:
        # corrupt ledger -> recovered (backed up) and commit still succeeds
        CT.LEDGER_PATH.write_text("{ this is not json", encoding="utf-8")
        ct = CT.CostTracker("m", pricing)
        ct.add(output_tokens=1_000_000)  # = $1.0
        res = ct.commit_to_ledger("run1")
        check("corrupt ledger recovered", abs(res["cumulative_usd"] - 1.0) < 1e-6, str(res))
        check("corrupt backup written", (CT.LEDGER_PATH.with_suffix(".corrupt.bak")).exists())

        # stale lock is stolen (not a 60s hang)
        lock = CT.LEDGER_PATH.with_suffix(".lock")
        lock.write_text("x")
        old = time.time() - 1000
        os.utime(lock, (old, old))  # make it stale
        t0 = time.time()
        ct2 = CT.CostTracker("m", pricing)
        ct2.add(output_tokens=1_000_000)
        ct2.commit_to_ledger("run2")
        check("stale lock stolen (fast)", time.time() - t0 < 5)

        # concurrent commits -> no lost updates
        def worker(i):
            c = CT.CostTracker("m", pricing)
            c.add(output_tokens=1_000_000)
            c.commit_to_ledger(f"c{i}")
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        ledger = json.loads(CT.LEDGER_PATH.read_text(encoding="utf-8"))
        # run1 + run2 + 8 concurrent = 10 entries, $10 total
        check("concurrent commits: no lost runs", len(ledger["runs"]) == 10, str(len(ledger["runs"])))
        check("concurrent commits: total correct", abs(ledger["cumulative_usd"] - 10.0) < 1e-6)
    finally:
        CT.LEDGER_PATH = real_ledger  # restore so we never touch real data


# ---------------------------------------------------------------------------
# 5. missing-file errors are clear (not raw tracebacks)
# ---------------------------------------------------------------------------
def test_missing_files():
    from harness import scoring
    try:
        scoring.load_gt("does/not/exist.json")
        check("scoring missing file -> clear error", False)
    except FileNotFoundError:
        check("scoring missing file -> clear error", True)
    from harness.backends.base import load_config
    try:
        load_config("nope.yaml")
        check("config missing file -> clear error", False)
    except FileNotFoundError:
        check("config missing file -> clear error", True)


def main() -> int:
    for fn in (test_parse, test_windowing, test_retries, test_cost_tracker,
               test_missing_files):
        print(f"\n== {fn.__name__} ==")
        fn()
    print(f"\n{'='*40}\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
