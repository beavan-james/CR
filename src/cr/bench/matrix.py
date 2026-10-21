"""Build the cached matrix: every prompt answered by every tier, plus router classifications.

Answers are the expensive part and are generated once. Classifications are cheap
(Laya is self-hosted) and stored separately, so new variants or a newly-available
Laya server only need `bench classify`, never new Claude calls.
"""
from __future__ import annotations

from typing import Callable

from .. import classifier, providers
from ..classifier import TIERS
from ..pricing import FAMILY_MODELS, cost_usd
from .pool import PoolResult, run_pool
from .store import Store, make_key


def answer_key(pid: str, prompt: str, family: str, model: str, effort: str | None, max_tokens: int) -> str:
    # family is part of the key: API and subscription (claude-code) answers come from different harnesses
    return make_key("answer", pid, prompt, family, model, effort, max_tokens)


def answer_lookup(store: Store, family: str, pid: str, prompt: str, tier: str) -> dict | None:
    model = FAMILY_MODELS[family][tier]
    return store.get(answer_key(pid, prompt, family, model, providers.MODEL_EFFORT.get(model), providers.DEFAULT_MAX_TOKENS))


def pending_answers(prompts: list[dict], family: str, store: Store, tiers=TIERS) -> list[tuple[dict, str]]:
    return [(p, t) for p in prompts for t in tiers if answer_lookup(store, family, p["id"], p["prompt"], t) is None]


def run_answers(prompts: list[dict], family: str, api_key: str, store: Store, workers: int = 4,
                on_done: Callable[[dict | None, Exception | None], None] | None = None, tiers=TIERS) -> PoolResult:
    """Answer every (prompt, tier) not already in the store."""
    todo = pending_answers(prompts, family, store, tiers)

    def one(p: dict, tier: str) -> dict:
        model = FAMILY_MODELS[family][tier]
        effort = providers.MODEL_EFFORT.get(model)
        comp = providers.complete(family, model, p["prompt"], api_key=api_key, effort=effort)
        row = {"key": answer_key(p["id"], p["prompt"], family, model, effort, providers.DEFAULT_MAX_TOKENS),
               "id": p["id"], "tier": tier, "model": model, "effort": effort, "text": comp.text,
               "input_tokens": comp.input_tokens, "output_tokens": comp.output_tokens,
               "cost": cost_usd(model, comp.input_tokens, comp.output_tokens),
               "stop_reason": comp.stop_reason, "latency_s": round(comp.latency_s, 3)}
        store.add(row)
        return row

    return run_pool(todo, one, workers, on_done)


def policy_variants(variants: list[dict], rotations: int) -> list[dict]:
    """Expand choice variants into their rotated option orders (r0 = original order)."""
    out = []
    for v in variants:
        if v["question"].get("type") == "choice" and rotations > 1:
            out += [classifier.rotate_choice(v, k) for k in range(rotations)]
        else:
            out.append(v)
    return out


def classification_key(pid: str, prompt: str, policy: str, question: dict, router_model: str) -> str:
    return make_key("classify", pid, prompt, policy, question, router_model)


def run_classify(prompts: list[dict], backend: str, variants: list[dict], store: Store,
                 client: classifier.LayaClient | None = None) -> int:
    """Classify every prompt under every policy. Returns the number of new rows."""
    new = 0
    if backend == "heuristic":
        jobs = [("heuristic", None)]
    elif backend == "laya":
        if client is None:
            client = classifier.LayaClient()
        jobs = [(v["name"], v) for v in variants]
    else:
        raise ValueError("backend must be laya|heuristic")
    router_model = client.model if client else "heuristic"
    for p in prompts:
        for policy, v in jobs:
            key = classification_key(p["id"], p["prompt"], policy, v["question"] if v else None, router_model)
            if store.has(key):
                continue
            c = classifier.heuristic_classify(p["prompt"]) if v is None else client.classify(p["prompt"], v)
            store.add({"key": key, "id": p["id"], "policy": policy, "backend": backend,
                       "dist": c.dist, "confidence": c.confidence, "overflow": c.overflow,
                       "heuristic": c.heuristic, "router_model": c.router_model or router_model})
            new += 1
    return new
