"""Router classifiers: Laya (self-hosted decision model) and a keyword heuristic.

Laya is served by `laya-serve` (pip install "laya[serve]"), which speaks the
Jev /v1/systemone wire protocol:
  POST {LAYA_URL}/v1/systemone  {"state": <prompt>, "model": "typed-decisions", "questions": {id: q}}
Answers come back typed under answers[id]; score and choice answers carry
`probabilities` (score keyed by level index, choice by option name). A variant
file (variants/*.yaml) holds the question and a tier_map from answer keys to
tiers, so every variant reduces to a probability distribution over tiers.

Overflow: Laya reads at most ~1024 tokens. laya-serve reports `usage.truncated`
when a prompt didn't fit; such prompts are marked `overflow` and the router sends
them to the flagship instead of routing on a partial read. Prompts longer than
LAYA_MAX_CHARS skip the call entirely (they can't fit, and the server caps state size).

Errors are raised, never silently replaced by the heuristic.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Mapping

import httpx
import yaml

TIERS = ("cheap", "balanced", "flagship")
DEFAULT_LAYA_MODEL = "typed-decisions"
# 1024 tokens is ~4K chars of English; anything past this cannot fit, so don't send it.
DEFAULT_MAX_CHARS = 6000

COMPLEX_HINTS = re.compile(
    r"\b(prove|theorem|distributed|concurrency|architect|refactor large|microservice|"
    r"security audit|vulnerability|optimize complex|multi-step|chain of|rigorous)\b",
    re.I,
)
MID_HINTS = re.compile(
    r"\b(write|explain|debug|crash|bug|fails?|broken|safari|function|class|api|test|summar|analy[sz]e|compare|plan|email|essay)\b",
    re.I,
)


@dataclass
class Classification:
    dist: dict[str, float]  # tier -> probability, sums to 1 (empty when overflow)
    confidence: float
    overflow: bool = False  # prompt didn't fit the router; route to flagship
    raw: dict = field(default_factory=dict)
    router_model: str = ""
    heuristic: bool = False

    @property
    def top_tier(self) -> str:
        if self.overflow:
            return "flagship"
        return max(TIERS, key=lambda t: self.dist.get(t, 0.0))


class RouterError(RuntimeError):
    pass


def overflow_classification(reason: str, raw: dict | None = None) -> Classification:
    return Classification({}, 0.0, overflow=True, raw={"overflow": reason, **(raw or {})})


def load_variant(path: str) -> dict:
    with open(path) as f:
        v = yaml.safe_load(f) or {}
    for key in ("name", "question_id", "question", "tier_map"):
        if key not in v:
            raise ValueError(f"variant {path} missing '{key}'")
    bad = set(v["tier_map"].values()) - set(TIERS)
    if bad:
        raise ValueError(f"variant {path} maps to unknown tiers: {bad}")
    return v


def rotate_choice(variant: dict, k: int) -> dict:
    """Copy of a choice variant with its options rotated left by k (option-order bias check)."""
    q = variant["question"]
    if q.get("type") != "choice":
        raise ValueError("rotate_choice only applies to choice questions")
    items = list(q["criteria"].items())
    k %= len(items)
    rotated = dict(items[k:] + items[:k])
    return {**variant, "name": f"{variant['name']}@r{k}", "question": {**q, "criteria": rotated}}


def dist_from_probabilities(probabilities: Mapping, tier_map: Mapping[str, str]) -> dict[str, float]:
    """Sum answer-option probabilities into tiers. Keys are compared as strings
    (score levels may arrive as int 0/1/2 or "0"/"1"/"2")."""
    tm = {str(k): t for k, t in tier_map.items()}
    dist = {t: 0.0 for t in TIERS}
    for key, p in probabilities.items():
        tier = tm.get(str(key))
        if tier is None:
            raise RouterError(f"answer option {key!r} not in tier_map {sorted(tm)}")
        dist[tier] += float(p)
    total = sum(dist.values())
    if total <= 0:
        raise RouterError("answer probabilities sum to 0")
    return {t: p / total for t, p in dist.items()}


def classification_from_response(resp: Mapping, variant: dict) -> Classification:
    """Build a Classification from a /v1/systemone JSON response."""
    usage = resp.get("usage") or {}
    if usage.get("truncated"):  # laya-serve sums this across windows; > 0 means cut
        return overflow_classification("router context exceeded",
                                       {"usage": usage, "model": resp.get("model", "")})
    qid = variant["question_id"]
    try:
        answer = resp["answers"][qid]
    except (KeyError, TypeError) as e:
        raise RouterError(f"response has no answer for '{qid}'") from e
    return Classification(
        dist=dist_from_probabilities(answer["probabilities"], variant["tier_map"]),
        confidence=float(answer.get("confidence", 0.0)),
        raw=dict(resp),
        router_model=str(resp.get("model", "")),
    )


class LayaClient:
    """HTTP client for laya-serve. Config from LAYA_URL / LAYA_API_KEY / LAYA_MODEL."""

    def __init__(self, url: str | None = None, api_key: str | None = None, model: str | None = None,
                 max_chars: int | None = None, timeout: float = 60.0, transport: httpx.BaseTransport | None = None):
        self.url = (url or os.getenv("LAYA_URL", "")).rstrip("/")
        if not self.url:
            raise RouterError("LAYA_URL is not set (e.g. http://localhost:8000 via SSH tunnel)")
        key = api_key if api_key is not None else os.getenv("LAYA_API_KEY", "")
        self.model = model or os.getenv("LAYA_MODEL") or DEFAULT_LAYA_MODEL
        self.max_chars = max_chars or int(os.getenv("LAYA_MAX_CHARS", DEFAULT_MAX_CHARS))
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._http = httpx.Client(timeout=timeout, headers=headers, transport=transport)

    def classify(self, prompt: str, variant: dict) -> Classification:
        if len(prompt) > self.max_chars:
            return overflow_classification(f"prompt over {self.max_chars} chars; not sent")
        body = {"state": prompt, "model": self.model,
                "questions": {variant["question_id"]: variant["question"]}}
        try:
            r = self._http.post(f"{self.url}/v1/systemone", json=body)
            r.raise_for_status()
            return classification_from_response(r.json(), variant)
        except httpx.HTTPStatusError as e:
            raise RouterError(f"Laya returned {e.response.status_code}: {e.response.text[:300]}") from e
        except httpx.HTTPError as e:
            raise RouterError(f"Laya unreachable at {self.url}: {e}") from e

    def close(self) -> None:
        self._http.close()


def heuristic_classify(prompt: str) -> Classification:
    """Length/keyword heuristic, kept as a free baseline. One-hot over tiers."""
    p = prompt.lower()
    if (len(p.split()) <= 8 and "?" in prompt) or p.startswith(("what is", "who is", "define", "translate", "fix typo")):
        tier, conf, why = "cheap", 0.75, "short-factual"
    elif COMPLEX_HINTS.search(prompt) or len(prompt) > 1200:
        tier, conf, why = "flagship", 0.80, "complex-keyword/length"
    elif MID_HINTS.search(prompt) or len(prompt) > 200:
        tier, conf, why = "balanced", 0.70, "mid-keyword/length"
    else:
        tier, conf, why = "cheap", 0.65, "default-short"
    dist = {t: 1.0 if t == tier else 0.0 for t in TIERS}
    return Classification(dist, conf, raw={"reason": why}, router_model="heuristic", heuristic=True)
