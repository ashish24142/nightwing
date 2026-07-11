"""
cost_tracker.py — token + dollar accounting (operating rule #4).

Every frontier call reports usage; we accumulate tokens, convert to $ using the
`pricing` block in config/models.yaml, and persist a cumulative ledger so the
$700 project-spend alert (rule #4) survives across runs.

Usage:
    ct = CostTracker(model_id="claude-opus-4-6", pricing=cfg_pricing)
    ct.add(input_tokens=..., cache_write_tokens=..., cache_read_tokens=..., output_tokens=...)
    ct.summary()                      # dict for this run
    ct.commit_to_ledger(run_label)    # append to results/cost_log.json, check $700
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

LEDGER_PATH = Path(__file__).resolve().parent.parent / "results" / "cost_log.json"
ALERT_THRESHOLD_USD = 700.0


class CostTracker:
    def __init__(self, model_id: str, pricing: dict):
        self.model_id = model_id
        # pricing rates are $ per 1M tokens
        rates = pricing.get(model_id, {})
        self.rate_input = rates.get("input", 0.0)
        self.rate_cache_write = rates.get("cache_write", rates.get("input", 0.0))
        self.rate_cache_read = rates.get("cache_read", rates.get("input", 0.0))
        self.rate_output = rates.get("output", 0.0)
        self.missing_rates = model_id not in pricing
        self.input_tokens = 0
        self.cache_write_tokens = 0
        self.cache_read_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def add(self, input_tokens: int = 0, cache_write_tokens: int = 0,
            cache_read_tokens: int = 0, output_tokens: int = 0) -> None:
        # `input_tokens` here = fresh (non-cached) input tokens.
        self.input_tokens += int(input_tokens)
        self.cache_write_tokens += int(cache_write_tokens)
        self.cache_read_tokens += int(cache_read_tokens)
        self.output_tokens += int(output_tokens)
        self.calls += 1

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens / 1e6 * self.rate_input
            + self.cache_write_tokens / 1e6 * self.rate_cache_write
            + self.cache_read_tokens / 1e6 * self.rate_cache_read
            + self.output_tokens / 1e6 * self.rate_output
        )

    def summary(self) -> dict:
        cached = self.cache_read_tokens + self.cache_write_tokens
        total_in = self.input_tokens + cached
        return {
            "model": self.model_id,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "output_tokens": self.output_tokens,
            "cache_hit_rate": round(cached / total_in, 3) if total_in else 0.0,
            "cost_usd": round(self.cost_usd, 4),
            "pricing_known": not self.missing_rates,
        }

    def per_unit(self, n_units: int) -> float:
        """$ per unit (e.g. per contract) — for smoke-test extrapolation."""
        return self.cost_usd / n_units if n_units else 0.0

    _STALE_LOCK_SEC = 120  # a lock older than this is from a crashed process

    def commit_to_ledger(self, run_label: str) -> dict:
        """Append this run to the cumulative ledger; return alert status.
        Cross-process safe: parallel runs serialize via an exclusive lock file.
        Hardened against: stale locks (crashed holder), corrupt ledger JSON."""
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        lock = LEDGER_PATH.with_suffix(".lock")
        acquired = False
        for _ in range(600):  # up to ~60s
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                acquired = True
                break
            except FileExistsError:
                # steal a stale lock left by a crashed process
                try:
                    if time.time() - os.path.getmtime(lock) > self._STALE_LOCK_SEC:
                        os.unlink(lock)
                        continue
                except FileNotFoundError:
                    continue
                time.sleep(0.1)
        try:
            ledger = {"runs": [], "cumulative_usd": 0.0}
            if LEDGER_PATH.exists():
                try:
                    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
                        ledger = json.load(f)
                    assert isinstance(ledger.get("runs"), list)
                except (json.JSONDecodeError, AssertionError, OSError):
                    # never lose a paid run to a corrupt ledger: back it up, restart
                    backup = LEDGER_PATH.with_suffix(".corrupt.bak")
                    try:
                        os.replace(LEDGER_PATH, backup)
                        print(f"   WARNING: ledger was corrupt -> backed up to {backup.name}")
                    except OSError:
                        pass
                    ledger = {"runs": [], "cumulative_usd": 0.0}
            entry = {"run": run_label, **self.summary()}
            # same-label re-run (resume/retry) REPLACES its entry — appending
            # would double-book the replayed checkpoint usage on every retry
            ledger["runs"] = [r for r in ledger["runs"] if r.get("run") != run_label]
            ledger["runs"].append(entry)
            ledger["cumulative_usd"] = round(
                sum(r.get("cost_usd", 0.0) for r in ledger["runs"]), 4
            )
            # atomic write: tmp then replace (no half-written ledger on crash)
            tmp = LEDGER_PATH.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(ledger, f, indent=2)
            os.replace(tmp, LEDGER_PATH)
        finally:
            if acquired:
                try:
                    os.unlink(lock)
                except FileNotFoundError:
                    pass
        alert = ledger["cumulative_usd"] >= ALERT_THRESHOLD_USD
        return {
            "cumulative_usd": ledger["cumulative_usd"],
            "alert": alert,
            "threshold": ALERT_THRESHOLD_USD,
        }
