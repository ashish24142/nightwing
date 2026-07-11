"""
base.py — abstract backend interface + config loader + factory.

THE SWAP POINT (operating rule #3): run_eval picks a backend purely from
config/models.yaml. Adding/swapping a backend never touches harness logic.

A backend takes one contract's text + its 41 questions and returns official
nbest predictions for each qid. Backends own provider-specific concerns
(HTTP, auth, prompt caching, local inference) but share prompt.py for the
task framing and output contract.
"""
from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


@dataclass
class Question:
    qid: str
    question: str
    category: str


@dataclass
class Usage:
    """Per-call token usage, normalized across providers."""
    input_tokens: int = 0          # fresh (non-cached) input — full price
    cache_write_tokens: int = 0    # cache creation — ~1.25x input (Anthropic)
    cache_read_tokens: int = 0     # cache hit — ~0.1x input
    output_tokens: int = 0


@dataclass
class ContractResult:
    """nbest predictions for one contract: {qid: [{"text","probability"}, ...]}."""
    nbest: dict = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    errors: int = 0          # questions that failed after retries (-> empty nbest)


class Backend(abc.ABC):
    """Abstract model backend. One method: predict a single contract."""

    #: model id as the provider reports it (used as the pricing key)
    model_id: str = "unknown"

    @abc.abstractmethod
    def predict_contract(self, contract_text: str,
                         questions: list[Question]) -> ContractResult:
        """Return nbest predictions for all `questions` against `contract_text`.
        Implementations SHOULD cache the contract once and vary the question."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Config + factory
# ---------------------------------------------------------------------------
def load_config(path: str | Path = CONFIG_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "backends" not in cfg:
        raise ValueError(f"malformed config (missing 'backends'): {path}")
    if "active" not in cfg:
        raise ValueError(f"config missing 'active' backend: {path}")
    return cfg


def _env(name_key: str, cfg: dict, required: bool = True) -> str | None:
    """Resolve a value whose env-var NAME is stored under cfg[name_key]."""
    env_name = cfg.get(name_key)
    if not env_name:
        if required:
            raise KeyError(f"config missing '{name_key}'")
        return None
    val = os.environ.get(env_name)
    if required and not val:
        raise EnvironmentError(
            f"environment variable {env_name} (for {name_key}) is not set — "
            f"check your .env"
        )
    return val


def build_backend(backend_name: str, config: dict | None = None) -> Backend:
    """Factory: instantiate the backend named in config/models.yaml.
    Loads .env first so API keys/endpoints resolve."""
    try:
        from dotenv import load_dotenv
        load_dotenv(CONFIG_PATH.parents[1] / ".env")
    except ImportError:
        pass

    config = config or load_config()
    backends = config["backends"]
    if backend_name not in backends:
        raise KeyError(f"unknown backend '{backend_name}'. "
                       f"available: {list(backends)}")
    cfg = backends[backend_name]
    btype = cfg["type"]

    if btype == "frontier_api":
        from .frontier_api import FrontierBackend
        return FrontierBackend(cfg, pricing=config.get("pricing", {}))
    if btype == "local_model":
        from .local_model import LocalBackend
        return LocalBackend(cfg)
    raise ValueError(f"unsupported backend type '{btype}'")
