"""Pure routing: Jev label -> tier -> model_id. No network here."""
from __future__ import annotations

from dataclasses import dataclass

import yaml

from .pricing import FAMILY_MODELS


@dataclass
class Route:
    tier: str  # cheap | balanced | flagship
    model: str
    label: str
    confidence: float
    fallback_used: bool
    reason: str


def load_router_config(path: str = "router_config.yaml") -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def route(label: str, confidence: float, family: str = "anthropic", config_path: str = "router_config.yaml") -> Route:
    cfg = load_router_config(config_path)
    tier_map = cfg.get("tier_map", {"simple": "cheap", "standard": "balanced", "complex": "flagship"})
    fallback_tier = cfg.get("fallback_tier", "flagship")
    threshold = float(cfg.get("confidence_threshold", 0.4))

    label = (label or "standard").lower()
    tier = tier_map.get(label, "balanced")
    fallback_used = False
    reason = f"jev={label} (conf {confidence:.2f}) -> {tier}"

    if confidence < threshold:
        tier = fallback_tier
        fallback_used = True
        reason = f"low confidence ({confidence:.2f} < {threshold}) -> fallback {tier}"

    models = FAMILY_MODELS.get(family, FAMILY_MODELS["anthropic"])
    model = models.get(tier, models["flagship"])
    return Route(tier, model, label, confidence, fallback_used, reason)
