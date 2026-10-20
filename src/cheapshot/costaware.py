"""Cost-aware routing: weigh what a cheaper tier saves on THIS prompt against the risk it falls short.

Laya answers two questions per prompt: how hard it is (tier probabilities) and how long a
complete answer will be. A small regression, fit on the benchmark's tune split, turns the
length answer plus prompt length into an estimated flagship cost. Then

    tier = argmax_t  est_cost * (1 - price_t / price_flagship)  -  lam * P(t falls short)

so expensive prompts are worth the risk of a cheaper tier and cheap ones are not; most of a
bill sits in a few long answers. lam is the quality knob, picked from the benchmark curve.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping

from .classifier import TIERS
from .pricing import FAMILY_MODELS, PRICING

LENGTH_VARIANT = "variants/length_score.yaml"
UNKNOWN_LEVEL = 1.0  # length level used when Laya couldn't read the prompt


def level(dist: Mapping[str, float] | None) -> float:
    """Expected level 0..2 of a three-way score (cheap/balanced/flagship keys)."""
    if not dist:
        return UNKNOWN_LEVEL
    return dist.get("balanced", 0.0) + 2 * dist.get("flagship", 0.0)


def features(length_dist: Mapping[str, float] | None, prompt_chars: int) -> list[float]:
    return [1.0, level(length_dist), math.log(prompt_chars + 1)]


def fit_cost_model(rows: Iterable[tuple[Mapping[str, float] | None, int, float]], ridge: float = 0.1) -> list[float]:
    """Least squares of log flagship cost on features; rows = (length dist, prompt chars, cost)."""
    X, Y = [], []
    for dist, chars, cost in rows:
        X.append(features(dist, chars))
        Y.append(math.log(cost))
    k = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) + (ridge if i == j and i else 0.0) for j in range(k)] for i in range(k)]
    b = [sum(r[i] * y for r, y in zip(X, Y)) for i in range(k)]
    M = [A[i] + [b[i]] for i in range(k)]
    for c in range(k):  # Gauss-Jordan; A is symmetric positive definite
        for r in range(k):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [u - f * v for u, v in zip(M[r], M[c])]
    return [M[i][k] / M[i][i] for i in range(k)]


def predict_cost(w: list[float], length_dist: Mapping[str, float] | None, prompt_chars: int) -> float:
    return math.exp(sum(a * x for a, x in zip(w, features(length_dist, prompt_chars))))


def price_ratios(family: str) -> dict[str, float]:
    """Each tier's output price relative to the flagship (prices scale together per tier)."""
    fm = FAMILY_MODELS[family]
    return {t: PRICING[fm[t]]["output"] / PRICING[fm["flagship"]]["output"] for t in TIERS}


def shortfall(dist: Mapping[str, float]) -> dict[str, float]:
    """P(tier is not enough) per tier."""
    return {"cheap": dist.get("balanced", 0.0) + dist.get("flagship", 0.0),
            "balanced": dist.get("flagship", 0.0), "flagship": 0.0}


def pick_tier_cost(dist: Mapping[str, float], est_cost: float, lam: float,
                   ratios: Mapping[str, float]) -> tuple[str, float]:
    """(tier, P(it falls short)). Ties go to the more capable tier."""
    fail = shortfall(dist)
    score = {t: est_cost * (1 - ratios[t]) - lam * fail[t] for t in TIERS}
    tier = max(reversed(TIERS), key=lambda t: score[t])
    return tier, fail[tier]
