"""Benchmark: route N prompts, compare routed cost vs always-flagship baseline.

No ground truth needed (per user): we report
  - Jev label distribution (is Jev classifying sensibly?)
  - routed cost vs baseline cost -> savings %
  - latency
If dataset rows include ideal_tier or ideal_label, we also report accuracy.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from . import jev_client, providers, router
from .pricing import FAMILY_MODELS, cost_usd


@dataclass
class RowResult:
    prompt: str
    label: str
    confidence: float
    tier: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    baseline_cost: float
    latency_s: float
    mocked: bool
    ideal_tier: str = ""
    correct: bool | None = None


def run(dataset: Path, family: str, api_key: str, config_path: str = "router_config.yaml") -> list[RowResult]:
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    flagship = FAMILY_MODELS[family]["flagship"]
    out: list[RowResult] = []
    for r in rows:
        prompt = r.get("prompt", "")
        c = jev_client.classify(prompt, config_path=config_path)
        rt = router.route(c.label, c.confidence, family, config_path)
        comp = providers.complete(family, rt.model, prompt, api_key=api_key)
        cost = cost_usd(rt.model, comp.input_tokens, comp.output_tokens)
        # Baseline: same token counts priced at flagship (fair apples-to-apples)
        base = cost_usd(flagship, comp.input_tokens, comp.output_tokens)
        ideal = r.get("ideal_tier", "")
        correct = (ideal == rt.tier) if ideal else None
        out.append(RowResult(prompt, c.label, c.confidence, rt.tier, rt.model,
                             comp.input_tokens, comp.output_tokens, cost, base,
                             comp.latency_s, c.mocked, ideal, correct))
    return out


def summarize(results: list[RowResult]) -> dict:
    n = len(results)
    routed = sum(r.cost for r in results)
    base = sum(r.baseline_cost for r in results)
    savings = (base - routed) / base * 100 if base else 0.0
    dist: dict[str, int] = {}
    for r in results:
        dist[r.tier] = dist.get(r.tier, 0) + 1
    judged = [r for r in results if r.correct is not None]
    acc = sum(1 for r in judged if r.correct) / len(judged) * 100 if judged else None
    return {
        "n": n,
        "routed_cost": round(routed, 6),
        "baseline_cost": round(base, 6),
        "savings_pct": round(savings, 2),
        "avg_latency_s": round(sum(r.latency_s for r in results) / n, 3) if n else 0,
        "tier_distribution": dist,
        "accuracy_pct": round(acc, 1) if acc is not None else None,
        "accuracy_n": len(judged),
    }


def to_jsonl(results: list[RowResult]) -> str:
    return "\n".join(json.dumps(asdict(r)) for r in results)
