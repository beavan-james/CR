"""Pure routing: tier distribution + risk dial -> model. No network here.

Risk dial: pick the cheapest tier t with P(prompt needs a tier above t) <= epsilon.
epsilon=0 only routes down when the router is certain; epsilon=1 always routes cheap.
Prompts that overflowed the router's context go to the flagship.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .classifier import TIERS, Classification
from .pricing import FAMILY_MODELS


@dataclass
class Route:
    tier: str  # cheap | balanced | flagship
    model: str
    p_exceed: float  # P(needs a tier above the chosen one)
    epsilon: float
    reason: str


def exceed_probs(dist: Mapping[str, float]) -> dict[str, float]:
    """P(needed tier is strictly above t) for each tier t."""
    out, above = {}, 0.0
    for t in reversed(TIERS):
        out[t] = above
        above += dist.get(t, 0.0)
    return out


def pick_tier(dist: Mapping[str, float], epsilon: float) -> tuple[str, float]:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
    ex = exceed_probs(dist)
    for t in TIERS:
        # small tolerance so float noise in router probabilities doesn't flip a tie
        if ex[t] <= epsilon + 1e-9:
            return t, ex[t]
    return TIERS[-1], 0.0


def tier_for(c: Classification, epsilon: float) -> str:
    return "flagship" if c.overflow else pick_tier(c.dist, epsilon)[0]


def route(c: Classification, epsilon: float, family: str = "anthropic") -> Route:
    models = FAMILY_MODELS[family]
    if c.overflow:
        return Route("flagship", models["flagship"], 0.0, epsilon,
                     f"overflow ({c.raw.get('overflow', 'router context exceeded')}) -> flagship")
    tier, p = pick_tier(c.dist, epsilon)
    dist_s = " ".join(f"{t}={c.dist.get(t, 0.0):.2f}" for t in TIERS)
    return Route(tier, models[tier], p, epsilon, f"({dist_s}) P(>{tier})={p:.2f} <= eps {epsilon}")
