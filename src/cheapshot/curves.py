"""Savings-vs-quality curves: let users pick a quality target instead of epsilon.

A curve is built once from a benchmark run (`cheapshot bench curve`) for one router model
and variant. For each epsilon it stores tune and test metrics. A quality target maps to
the epsilon with the most per-prompt savings whose TUNE quality meets it, and the test
numbers at that epsilon are what we show the user as the expected outcome.
"""
from __future__ import annotations

import json
from pathlib import Path

from .bench import replay

CURVE_DIR = Path(__file__).parent / "curves"
DEFAULT_QUALITY = 0.95
_FIELDS = ("quality", "savings", "savings_per_prompt", "mix")


def curve_path(variant_name: str, router_model: str) -> Path:
    return CURVE_DIR / f"{variant_name}__{router_model}.json"


def build_curve(m: replay.Matrix, policy: str, router_model: str, source: str) -> dict:
    tune, test = m.ids("tune"), m.ids("test")
    points = []
    for eps in replay.EPS_GRID:
        a = replay.assign(m, policy, eps)
        tm, xm = replay.evaluate(m, a, tune), replay.evaluate(m, a, test)
        rnd = replay.random_at_mix(m, xm.mix, test)
        points.append({"eps": eps,
                       "tune": {k: getattr(tm, k) for k in _FIELDS},
                       "test": {k: getattr(xm, k) for k in _FIELDS},
                       "random_test": {"quality": rnd.quality, "savings_per_prompt": rnd.savings_per_prompt}})
    return {"variant": policy, "router_model": router_model, "source": source,
            "n_tune": len(tune), "n_test": len(test), "points": points}


def save_curve(curve: dict) -> Path:
    path = curve_path(curve["variant"], curve["router_model"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(curve, indent=1))
    return path


def load_curve(variant_name: str, router_model: str) -> dict | None:
    path = curve_path(variant_name, router_model)
    return json.loads(path.read_text()) if path.exists() else None


def eps_for_quality(curve: dict, quality: float) -> tuple[float, dict]:
    """Most per-prompt savings among epsilons whose tune quality meets the target;
    epsilon 0 (always the safest tier) if none does."""
    pts = curve["points"]
    ok = [p for p in pts if p["tune"]["quality"] >= quality]
    if not ok:
        return 0.0, pts[0]
    best = max(ok, key=lambda p: (p["tune"]["savings_per_prompt"], p["tune"]["quality"], -p["eps"]))
    return best["eps"], best


def frontier(curve: dict, targets=(1.0, 0.97, 0.95, 0.92, 0.90, 0.85, 0.80)) -> list[dict]:
    """One row per quality target: the chosen epsilon and its held-out (test) outcome."""
    rows = []
    for q in targets:
        eps, p = eps_for_quality(curve, q)
        rows.append({"target": q, "eps": eps, **p["test"], "random_quality": p["random_test"]["quality"]})
    return rows
