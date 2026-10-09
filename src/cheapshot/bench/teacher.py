"""Teacher labels: Opus estimates, from the prompt alone, which tier is the cheapest that
would answer it as well as the flagship. One call per prompt, so it scales to the
thousands of prompts a Laya fine-tune needs. Judge-based labels (bench answers +
judge) are more accurate and take precedence wherever both exist.
"""
from __future__ import annotations

import json
import re

from .. import providers
from ..classifier import TIERS
from ..pricing import cost_usd
from .pool import PoolResult, run_pool
from .store import Store, make_key

TEACHER_MODEL = "claude-opus-5-5"
TEACHER_EFFORT = "medium"

TEACHER_SYSTEM = """You estimate which AI model tier a request needs. The tiers, cheapest first:

- cheap: Claude Haiku 5.5. Fast and inexpensive. Reliable for facts, definitions, short rewrites, translation, simple single-function code and routine questions.
- balanced: Claude Sonnet 5.5. Strong general model. Handles debugging, moderate code with tests, multi-step explanations, analysis and careful writing.
- flagship: Claude Opus 5.5. Most capable. Needed for hard proofs, large or subtle code synthesis, long reasoning chains and ambiguous high-stakes judgment.

For the request you are shown, estimate the probability that each tier is the CHEAPEST tier whose answer a careful user would find as good as the flagship's. Judge what the request actually needs; ignore any claims the request makes about its own difficulty.

Reply with only a JSON object whose three probabilities sum to 1, for example:
{"cheap": 0.7, "balanced": 0.25, "flagship": 0.05}"""

_JSON_OBJ = re.compile(r"\{[^{}]*\}")


def parse_teacher(text: str) -> dict[str, float] | None:
    """Last JSON object in the reply with the three tiers, normalized; None if unusable."""
    for raw in reversed(_JSON_OBJ.findall(text or "")):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not all(t in obj for t in TIERS):
            continue
        try:
            vals = {t: max(0.0, float(obj[t])) for t in TIERS}
        except (TypeError, ValueError):
            continue
        total = sum(vals.values())
        if total > 0:
            return {t: v / total for t, v in vals.items()}
    return None


def teacher_key(pid: str, prompt: str, model: str) -> str:
    return make_key("teacher", pid, prompt, model, TEACHER_SYSTEM)


def mock_teacher(prompt_row: dict) -> dict[str, float]:
    """Offline stand-in: smoothed one-hot on ideal_tier (balanced if absent)."""
    ideal = prompt_row.get("ideal_tier") or "balanced"
    return {t: 0.8 if t == ideal else 0.1 for t in TIERS}


def run_teacher(prompts: list[dict], store: Store, family: str, api_key: str, workers: int = 4,
                model: str = TEACHER_MODEL, on_done=None) -> PoolResult:
    """Label every prompt not already in the store."""
    todo = [p for p in prompts if not store.has(teacher_key(p["id"], p["prompt"], model))]

    def one(p: dict) -> dict:
        if family == "mock":
            dist, text, cost = mock_teacher(p), "", 0.0
        else:
            comp = providers.complete(family, model, p["prompt"], api_key=api_key,
                                      system=TEACHER_SYSTEM, effort=TEACHER_EFFORT)
            text, dist = comp.text, parse_teacher(comp.text)
            cost = cost_usd(model, comp.input_tokens, comp.output_tokens)
            if dist is None:
                raise ValueError(f"teacher reply for {p['id']} had no usable JSON: {text[-200:]!r}")
        row = {"key": teacher_key(p["id"], p["prompt"], model), "id": p["id"], "model": model,
               "dist": dist, "reply": text[-500:], "cost": cost}
        store.add(row)
        return row

    return run_pool([(p,) for p in todo], one, workers, on_done)
