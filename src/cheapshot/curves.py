"""Savings-vs-quality curves: let users pick a quality target instead of epsilon.

A curve is built once from a benchmark run (`cheapshot bench curve`) for one router model
and variant. For each epsilon it stores tune and test metrics. A quality target maps to
the epsilon with the most per-prompt savings whose TUNE quality meets it, and the test
numbers at that epsilon are what we show the user as the expected outcome.

Cost-aware curves (rule "cost", costaware.py) work the same way with lam in place of
epsilon, stored relative to the mean tune flagship cost, and pick the most TOTAL savings:
that rule exists to cut the bill, which a few long answers dominate.
Points are ordered safest first, so an unreachable target falls back to points[0].
"""
from __future__ import annotations

import json
from pathlib import Path

from . import costaware
from .bench import replay

CURVE_DIR = Path(__file__).parent / "curves"
DEFAULT_QUALITY = 0.95
COST_CURVE_NAME = "cost_aware"
LAM_GRID = tuple(2 ** (k / 8) for k in range(64, -65, -1))  # x mean flagship cost; large lam = safe
_FIELDS = ("quality", "savings", "savings_per_prompt", "mix")


def _point(m: replay.Matrix, a: dict[str, str], tune: list[str], test: list[str], **knob) -> dict:
    tm, xm = replay.evaluate(m, a, tune), replay.evaluate(m, a, test)
    rnd = replay.random_at_mix(m, xm.mix, test)
    return {**knob, "tune": {k: getattr(tm, k) for k in _FIELDS}, "test": {k: getattr(xm, k) for k in _FIELDS},
            "random_test": {"quality": rnd.quality, "savings_per_prompt": rnd.savings_per_prompt}}


def curve_path(variant_name: str, router_model: str) -> Path:
    return CURVE_DIR / f"{variant_name}__{router_model}.json"


def build_curve(m: replay.Matrix, policy: str, router_model: str, source: str) -> dict:
    tune, test = m.ids("tune"), m.ids("test")
    points = [_point(m, replay.assign(m, policy, eps), tune, test, eps=eps) for eps in replay.EPS_GRID]
    return {"variant": policy, "router_model": router_model, "source": source, "rule": "threshold",
            "objective": "savings_per_prompt", "n_tune": len(tune), "n_test": len(test), "points": points}


def cost_inputs(m: replay.Matrix, tier_policy: str, length_policy: str) -> tuple[dict, dict]:
    """(tier dists, length dists) per prompt id; None marks overflow."""
    tiers = {pid: (None if r.overflow else r.dist) for pid, r in m.policies[tier_policy].items()}
    lengths = {pid: (None if r.overflow else r.dist) for pid, r in m.policies[length_policy].items()}
    return tiers, lengths


def assign_cost(m: replay.Matrix, tiers: dict, lengths: dict, w: list[float], lam: float,
                ratios: dict[str, float]) -> dict[str, str]:
    out = {}
    for pid, dist in tiers.items():
        if dist is None:
            out[pid] = "flagship"
            continue
        est = costaware.predict_cost(w, lengths.get(pid), len(m.prompts[pid]["prompt"]))
        out[pid] = costaware.pick_tier_cost(dist, est, lam, ratios)[0]
    return out


def build_cost_curve(m: replay.Matrix, tier_policy: str, length_policy: str, router_model: str,
                     family: str, source: str) -> dict:
    tune, test = m.ids("tune"), m.ids("test")
    tiers, lengths = cost_inputs(m, tier_policy, length_policy)
    w = costaware.fit_cost_model((lengths.get(p), len(m.prompts[p]["prompt"]), m.cost[(p, "flagship")]) for p in tune)
    mean_cost = sum(m.cost[(p, "flagship")] for p in tune) / len(tune)
    ratios = costaware.price_ratios(family)
    points = [_point(m, assign_cost(m, tiers, lengths, w, k * mean_cost, ratios), tune, test, lam=k) for k in LAM_GRID]
    return {"variant": COST_CURVE_NAME, "router_model": router_model, "source": source, "rule": "cost",
            "objective": "savings", "tier_variant": tier_policy, "length_variant": length_policy,
            "cost_model": {"w": w, "mean_cost": mean_cost}, "n_tune": len(tune), "n_test": len(test), "points": points}


def save_curve(curve: dict) -> Path:
    path = curve_path(curve["variant"], curve["router_model"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(curve, indent=1))
    return path


def load_curve(variant_name: str, router_model: str) -> dict | None:
    path = curve_path(variant_name, router_model)
    return json.loads(path.read_text()) if path.exists() else None


def knob(curve: dict, p: dict) -> float:
    """The routing parameter at a point: epsilon, or absolute lam for a cost curve."""
    return p["lam"] * curve["cost_model"]["mean_cost"] if curve.get("rule") == "cost" else p["eps"]


def knob_label(curve: dict, p: dict) -> str:
    return f"λ {p['lam']:.2f}×" if curve.get("rule") == "cost" else f"ε {p['eps']:.2f}"


def point_for_quality(curve: dict, quality: float) -> dict:
    """Most savings (the curve's objective, on tune) among points whose tune quality meets
    the target; the safest point if none does."""
    pts, obj = curve["points"], curve.get("objective", "savings_per_prompt")
    ok = [(i, p) for i, p in enumerate(pts) if p["tune"]["quality"] >= quality]
    if not ok:
        return pts[0]
    return max(ok, key=lambda ip: (ip[1]["tune"][obj], ip[1]["tune"]["quality"], -ip[0]))[1]


def eps_for_quality(curve: dict, quality: float) -> tuple[float, dict]:
    p = point_for_quality(curve, quality)
    return knob(curve, p), p


def frontier(curve: dict, targets=(1.0, 0.97, 0.95, 0.92, 0.90, 0.85, 0.80)) -> list[dict]:
    """One row per quality target: the chosen epsilon and its held-out (test) outcome."""
    rows = []
    for q in targets:
        p = point_for_quality(curve, q)
        rows.append({"target": q, "knob": knob_label(curve, p), **p["test"], "random_quality": p["random_test"]["quality"]})
    return rows
