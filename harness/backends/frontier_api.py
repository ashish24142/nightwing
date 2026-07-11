"""
frontier_api.py — Azure-hosted Claude + GPT frontier backends.

Validated wire formats (see creds smoke test):
  - azure_anthropic        : POST {endpoint}/v1/messages, header x-api-key,
                             native Anthropic Messages API. Prompt caching via
                             cache_control on the system (instructions+contract).
  - azure_openai_responses : POST {endpoint}/openai/responses?api-version=...,
                             header api-key. Prompt caching is automatic on the
                             identical leading prefix (system+contract).

Cost saver: the contract is the bulk of the tokens; we put it in
the cached prefix and vary only the ~per-question suffix across all 41 calls.

Implemented with `requests` (exact validated shapes) rather than the vendor SDKs
to stay robust to Azure's nonstandard routing.
"""
from __future__ import annotations

import os
import random
import threading
import time

import requests

from .. import prompt as P
from .base import Backend, ContractResult, Question, Usage, _env

# generous read timeout: retrying a timed-out POST can DOUBLE-BILL (the server
# may finish + bill the first attempt after the client gives up), so prefer
# waiting longer over retrying sooner.
_TIMEOUT = (10, 300)   # (connect, read) seconds
_MAX_RETRIES = 6
_RETRYABLE = (408, 409, 425, 429, 500, 502, 503, 504, 529)


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter (avoids thundering-herd across workers)."""
    return min(2 ** attempt, 30) + random.uniform(0, 0.75)


def _post_with_retries(url: str, headers: dict, body: dict) -> dict:
    """POST with retry on transient errors. Returns a dict (never None). Raises
    RuntimeError only after exhausting retries or on a non-retryable status."""
    last = None
    for attempt in range(_MAX_RETRIES):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=_TIMEOUT)
            if r.status_code == 200:
                try:
                    data = r.json()
                except ValueError:  # 200 but body isn't JSON -> treat as transient
                    last = f"200 but non-JSON body: {r.text[:200]}"
                    time.sleep(_backoff(attempt))
                    continue
                if not isinstance(data, dict):
                    last = f"200 but JSON is {type(data).__name__}, not object"
                    time.sleep(_backoff(attempt))
                    continue
                return data
            if r.status_code in _RETRYABLE:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                # honor Retry-After if present
                ra = r.headers.get("retry-after")
                time.sleep(float(ra) if ra and ra.isdigit() else _backoff(attempt))
                continue
            # non-retryable (e.g. 400 bad request, 401 auth) -> fail fast & loud
            raise RuntimeError(f"HTTP {r.status_code} (non-retryable): {r.text[:500]}")
        except requests.ConnectionError as e:  # request likely never delivered
            last = str(e)
            time.sleep(_backoff(attempt))
        except requests.RequestException as e:  # timeout AFTER send: the server
            # may still complete and bill the abandoned attempt — make the
            # possible double-charge visible instead of silent.
            last = str(e)
            print(f"      ! timeout after send ({type(e).__name__}); retrying — "
                  "abandoned attempt may still be billed server-side")
            time.sleep(_backoff(attempt))
    raise RuntimeError(f"request failed after {_MAX_RETRIES} retries: {last}")


class FrontierBackend(Backend):
    def __init__(self, cfg: dict, pricing: dict):
        self.cfg = cfg
        self.pricing = pricing
        self.provider = cfg["provider"]
        # model name: literal `model:` in config, else from `model_env`
        self.model_id = cfg.get("model") or _env("model_env", cfg)
        # key into config `pricing:` (deployment name may differ from priced id)
        self.pricing_model_id = cfg.get("pricing_model_id", self.model_id)
        self.base_url = _env("base_url_env", cfg).rstrip("/")
        self.api_key = _env("api_key_env", cfg)
        self.rps = float(cfg.get("rps", 2))
        self._min_interval = 1.0 / self.rps if self.rps > 0 else 0.0
        self._last_call = 0.0
        self._throttle_lock = threading.Lock()  # global rate cap across worker threads
        if self.provider == "azure_anthropic":
            self.max_tokens = int(cfg.get("max_tokens", 1024))
            # temperature is deprecated on newer Opus models -> send only if set
            t = cfg.get("temperature", None)
            self.temperature = float(t) if t is not None else None
            self.anthropic_version = cfg.get("anthropic_version", "2023-06-01")
        elif self.provider == "azure_openai_responses":
            self.max_output_tokens = int(cfg.get("max_output_tokens", 2048))
            self.api_version = os.environ[cfg["api_version_env"]]
        else:
            raise ValueError(f"unknown frontier provider '{self.provider}'")

    # -- rate limit --------------------------------------------------------
    def _throttle(self) -> None:
        # global requests/sec cap, thread-safe so N workers don't exceed rps
        if not self._min_interval:
            return
        with self._throttle_lock:
            dt = time.time() - self._last_call
            if dt < self._min_interval:
                time.sleep(self._min_interval - dt)
            self._last_call = time.time()

    # -- per-provider single question call ---------------------------------
    def _ask_anthropic(self, contract_block: str, question_text: str):
        url = f"{self.base_url}/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }
        # NOTE: the Azure AI Foundry Anthropic passthrough only honors
        # cache_control on a USER content block (it ignores it in `system`).
        # So instructions+contract go in a cached user block; the question is a
        # second, uncached user block that varies across the 41 calls.
        body = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text",
                 "text": P.SYSTEM_PROMPT + "\n\n" + contract_block,
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": question_text},
            ]}],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        data = _post_with_retries(url, headers, body)
        content = data.get("content") or []
        text = "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
        u = data.get("usage") or {}
        usage = Usage(
            input_tokens=int(u.get("input_tokens", 0) or 0),
            cache_write_tokens=int(u.get("cache_creation_input_tokens", 0) or 0),
            cache_read_tokens=int(u.get("cache_read_input_tokens", 0) or 0),
            output_tokens=int(u.get("output_tokens", 0) or 0),
        )
        return text, usage

    def _ask_openai_responses(self, contract_block: str, question_text: str):
        url = f"{self.base_url}/openai/responses?api-version={self.api_version}"
        headers = {"api-key": self.api_key, "content-type": "application/json"}
        body = {
            "model": self.model_id,
            "input": [
                # system (instructions + contract) = identical leading prefix
                # across all 41 questions -> auto prefix cache
                {"role": "system",
                 "content": P.SYSTEM_PROMPT + "\n\n" + contract_block},
                {"role": "user", "content": question_text},
            ],
            "max_output_tokens": self.max_output_tokens,
        }
        data = _post_with_retries(url, headers, body)
        text = data.get("output_text") or ""
        if not text:
            for item in (data.get("output") or []):
                if not isinstance(item, dict):
                    continue
                for c in (item.get("content") or []):
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        text += c.get("text", "") or ""
        # NOTE: empty text can mean a content-filter block or status="incomplete"
        # (reasoning consumed max_output_tokens). Either way -> parses to "absent",
        # a safe degradation; the run still completes and is scoreable.
        u = data.get("usage") or {}
        cached = int((u.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0)
        usage = Usage(
            input_tokens=max(0, int(u.get("input_tokens", 0) or 0) - cached),
            cache_write_tokens=0,            # OpenAI bills no separate cache write
            cache_read_tokens=cached,
            output_tokens=int(u.get("output_tokens", 0) or 0),
        )
        return text, usage

    def _ask(self, contract_block: str, question_text: str):
        self._throttle()
        if self.provider == "azure_anthropic":
            return self._ask_anthropic(contract_block, question_text)
        return self._ask_openai_responses(contract_block, question_text)

    # -- public ------------------------------------------------------------
    def predict_contract(self, contract_text: str,
                         questions: list[Question]) -> ContractResult:
        contract_block = P.build_contract_block(contract_text)
        res = ContractResult()
        for q in questions:
            qtext = P.build_question_text(q.question, q.category)
            try:
                text, usage = self._ask(contract_block, qtext)
            except Exception as e:
                # one question's terminal failure must NOT lose the contract:
                # record an abstain for it, count the error, keep going.
                res.nbest[q.qid] = []
                res.errors += 1
                print(f"      ! question failed ({q.category}): "
                      f"{str(e)[:120]} -> abstain")
                continue
            parsed = P.parse_response(text)
            res.nbest[q.qid] = P.to_nbest(parsed)
            res.usage.input_tokens += usage.input_tokens
            res.usage.cache_write_tokens += usage.cache_write_tokens
            res.usage.cache_read_tokens += usage.cache_read_tokens
            res.usage.output_tokens += usage.output_tokens
        return res
