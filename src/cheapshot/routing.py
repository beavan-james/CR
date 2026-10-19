"""Routing as used interactively: Laya (or the heuristic) + a quality target -> model per turn.

The quality target is translated to epsilon with the router model's benchmark curve
(curves.py). Without a curve, epsilon comes from the variant (epsilon_by_model / epsilon).
If Laya can't be reached the turn goes to the flagship, so an outage costs money, never quality.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import classifier, curves, router

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VARIANT = "variants/a_score.yaml"
FOLLOW_UP_CHARS = 160  # shorter turns ("yes, do it") are routed with the previous request for context


def resolve_path(path: str) -> str:
    """Variant paths work from any directory: as given, else relative to the Cheapshot repo."""
    p = Path(path).expanduser()
    return str(p if p.exists() or p.is_absolute() else REPO_ROOT / p)


def routing_text(prompt: str, previous: str | None) -> str:
    if previous and len(prompt) < FOLLOW_UP_CHARS:
        return f"Previous request: {previous[:2000]}\n\nFollow-up: {prompt}"
    return prompt


@dataclass
class Decision:
    route: router.Route
    classification: classifier.Classification
    note: str = ""


class Routing:
    def __init__(self, variant_path: str = DEFAULT_VARIANT, quality: float | None = None,
                 which: str = "laya", family: str = "claude-code"):
        self.variant = classifier.load_variant(resolve_path(variant_path))
        self.which, self.family = which, family
        self.client = classifier.LayaClient() if which == "laya" else None
        self.router_model = self.client.model if self.client else "heuristic"
        self.curve = curves.load_curve(self.variant["name"], self.router_model) if self.client else None
        self.forced_tier: str | None = None
        self.set_quality(quality if quality is not None else float(os.getenv("CHEAPSHOT_QUALITY", curves.DEFAULT_QUALITY)))

    def set_quality(self, quality: float) -> None:
        self.quality = quality
        if self.curve:
            self.epsilon, self.expected = curves.eps_for_quality(self.curve, quality)
        else:
            v = self.variant
            self.epsilon = float((v.get("epsilon_by_model") or {}).get(self.router_model, v.get("epsilon", 0.2)))
            self.expected = None

    def decide(self, prompt: str, previous: str | None = None) -> Decision:
        models = router.FAMILY_MODELS[self.family]
        if self.forced_tier:
            c = classifier.Classification({}, 1.0, router_model="manual")
            return Decision(router.Route(self.forced_tier, models[self.forced_tier], 0.0, self.epsilon,
                                         f"pinned with /model {self.forced_tier}"), c)
        text = routing_text(prompt, previous)
        if self.client is None:
            c = classifier.heuristic_classify(text)
            tier = c.top_tier
            return Decision(router.Route(tier, models[tier], 0.0, self.epsilon, "heuristic top tier"), c)
        try:
            c = self.client.classify(text, self.variant)
        except classifier.RouterError as e:
            c = classifier.overflow_classification(f"router unavailable: {e}")
            return Decision(router.route(c, self.epsilon, self.family), c, note="Laya unreachable; used the flagship")
        return Decision(router.route(c, self.epsilon, self.family), c)

    def close(self) -> None:
        if self.client:
            self.client.close()
