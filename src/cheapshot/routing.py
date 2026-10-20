"""Routing as used interactively: Laya (or the heuristic) + a quality target -> model per turn.

The quality target is translated to epsilon with the router model's benchmark curve
(curves.py). Without a curve, epsilon comes from the variant (epsilon_by_model / epsilon).
If Laya can't be reached the turn goes to the flagship, so an outage costs money, never quality.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from . import classifier, costaware, curves, router

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VARIANT = "variants/a_score.yaml"
FOLLOW_UP_CHARS = 160  # shorter turns ("yes, do it") are routed with the previous request for context

# Pure thanks / sign-offs go straight to the cheapest tier: they ask for no work.
# "ok", "yes", "go ahead" are NOT trivial: after a plan they mean "do the hard thing".
_ACK_WORDS = {"ok", "okay", "great", "perfect", "nice", "cool", "awesome", "excellent", "thanks", "thank", "you",
              "thx", "ty", "cheers", "bye", "goodbye", "much", "so", "very", "a", "lot", "many", "appreciated",
              "appreciate", "it", "that", "all", "good", "for", "now", "help", "the"}
_CLOSERS = {"thanks", "thank", "thx", "ty", "cheers", "bye", "goodbye", "appreciated", "appreciate"}
TRIVIAL_MAX_CHARS = 60


def is_trivial(prompt: str) -> bool:
    """Thanks/sign-off with nothing else in it, e.g. "ok thanks!", "great, thank you so much"."""
    if len(prompt) > TRIVIAL_MAX_CHARS:
        return False
    words = re.findall(r"[a-z']+", prompt.lower())
    return bool(words) and set(words) <= _ACK_WORDS and bool(set(words) & _CLOSERS) and not re.search(r"[?/`]", prompt)


def resolve_path(path: str) -> str:
    """Variant paths work from any directory: as given, else relative to the Cheapshot repo."""
    p = Path(path).expanduser()
    return str(p if p.exists() or p.is_absolute() else REPO_ROOT / p)


def routing_text(prompt: str, previous: str | None) -> str:
    if previous and len(prompt) < FOLLOW_UP_CHARS:
        return f"Previous request: {previous[:2000]}\n\nFollow-up: {prompt}"
    return prompt


RULES = ("cost", "threshold")


@dataclass
class Decision:
    route: router.Route
    classification: classifier.Classification
    note: str = ""
    rule: str = ""  # set when a rule, not the router, chose the tier
    why: str = ""  # one-line routing summary for the REPL
    length: classifier.Classification | None = None
    est_cost: float | None = None


class Routing:
    """rule "threshold" (default): cheapest tier with P(needs more) <= epsilon. Best for agent
    sessions, where each turn re-sends the whole context and so costs about the same.
    rule "cost": cost-aware (costaware.py); saves most where answer length drives cost (chat)."""

    def __init__(self, variant_path: str = DEFAULT_VARIANT, quality: float | None = None,
                 which: str = "laya", family: str = "claude-code", rule: str | None = None):
        self.variant = classifier.load_variant(resolve_path(variant_path))
        self.length_variant = classifier.load_variant(resolve_path(costaware.LENGTH_VARIANT))
        self.which, self.family = which, family
        self.client = classifier.LayaClient() if which == "laya" else None
        self.router_model = self.client.model if self.client else "heuristic"
        self.curves = {}
        if self.client:
            self.curves = {"threshold": curves.load_curve(self.variant["name"], self.router_model),
                           "cost": curves.load_curve(curves.COST_CURVE_NAME, self.router_model)}
        self.ratios = costaware.price_ratios(family)
        self.forced_tier: str | None = None
        self.trivial_rule = os.getenv("CHEAPSHOT_TRIVIAL", "1") != "0"
        self.quality = quality if quality is not None else float(os.getenv("CHEAPSHOT_QUALITY", curves.DEFAULT_QUALITY))
        rule = rule or os.getenv("CHEAPSHOT_RULE") or "threshold"
        self.set_rule(rule)

    @property
    def curve(self) -> dict | None:
        return self.curves.get(self.rule)

    def set_rule(self, rule: str) -> None:
        if rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}")
        if rule == "cost" and not self.curves.get("cost"):
            raise ValueError(f"no cost-aware curve for router {self.router_model}; build one with "
                             "`cheapshot bench curve --rule cost`")
        self.rule = rule
        self.set_quality(self.quality)

    def set_quality(self, quality: float) -> None:
        self.quality = quality
        self.lam = None
        if self.curve:
            self.expected = curves.point_for_quality(self.curve, quality)
            self.setting = curves.knob_label(self.curve, self.expected)
            if self.rule == "cost":
                self.lam = curves.knob(self.curve, self.expected)
                self.epsilon = 0.0
            else:
                self.epsilon = curves.knob(self.curve, self.expected)
        else:
            v = self.variant
            self.epsilon = float((v.get("epsilon_by_model") or {}).get(self.router_model, v.get("epsilon", 0.2)))
            self.expected = None
            self.setting = f"ε {self.epsilon:.2f}"

    def decide(self, prompt: str, previous: str | None = None) -> Decision:
        models = router.FAMILY_MODELS[self.family]
        if self.forced_tier:
            c = classifier.Classification({}, 1.0, router_model="manual")
            return Decision(router.Route(self.forced_tier, models[self.forced_tier], 0.0, self.epsilon,
                                         f"pinned with /model {self.forced_tier}"), c, rule="pinned", why="pinned")
        if self.trivial_rule and is_trivial(prompt):
            c = classifier.Classification({}, 1.0, router_model="rule")
            return Decision(router.Route("cheap", models["cheap"], 0.0, self.epsilon,
                                         "thanks/sign-off: no work requested -> cheapest tier"), c,
                            rule="trivial turn", why="trivial turn")
        text = routing_text(prompt, previous)
        if self.client is None:
            c = classifier.heuristic_classify(text)
            tier = c.top_tier
            return Decision(router.Route(tier, models[tier], 0.0, self.epsilon, "heuristic top tier"), c, why="heuristic")
        cost_rule = self.rule == "cost"
        try:
            if cost_rule:
                c, length = self.client.classify_many(text, [self.variant, self.length_variant])
            else:
                c, length = self.client.classify(text, self.variant), None
        except classifier.RouterError as e:
            c = classifier.overflow_classification(f"router unavailable: {e}")
            return Decision(router.route(c, self.epsilon, self.family), c, note="Laya unreachable; used the flagship",
                            why="router unavailable")
        if c.overflow:
            return Decision(router.route(c, self.epsilon, self.family), c, why="too long for the router -> flagship")
        if not cost_rule:
            rt = router.route(c, self.epsilon, self.family)
            return Decision(rt, c, why=f"P(needs more)={rt.p_exceed:.2f} ≤ ε {self.epsilon:.2f}")
        w = self.curve["cost_model"]["w"]
        est = costaware.predict_cost(w, None if length.overflow else length.dist, len(text))
        tier, fail = costaware.pick_tier_cost(c.dist, est, self.lam, self.ratios)
        size = est / self.curve["cost_model"]["mean_cost"]
        dist_s = " ".join(f"{t}={c.dist.get(t, 0.0):.2f}" for t in classifier.TIERS)
        reason = (f"({dist_s}) answer length level {costaware.level(length.dist):.2f}, est. cost {size:.1f}× average; "
                  f"P({tier} falls short)={fail:.2f}; {self.setting}")
        rt = router.Route(tier, models[tier], fail, self.epsilon, reason)
        return Decision(rt, c, why=f"P(falls short)={fail:.2f} · est. size {size:.1f}× avg", length=length, est_cost=est)

    def close(self) -> None:
        if self.client:
            self.client.close()
