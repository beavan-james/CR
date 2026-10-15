"""Offline replay: score any routing policy against the cached matrix, no LLM calls.

quality(policy) = share of prompts whose routed answer is non-inferior to the flagship's
savings(policy) = 1 - routed answer cost / always-flagship cost   (Laya router cost is $0)
savings_per_prompt = mean over prompts of (1 - routed cost / flagship cost for that prompt)

Total savings is what the bill sees, but a few long answers dominate it (on v1 test the top
5 prompts are 44% of flagship cost), so it swings with a handful of routing decisions.
Per-prompt savings counts each prompt equally and is the steadier comparison.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field

from ..classifier import TIERS
from ..router import pick_tier
from .judge import CANDIDATE_TIERS, ORDERS, combine, judgment_key
from .matrix import answer_lookup
from .store import Store

EPS_GRID = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
OBJECTIVES = ("per-prompt", "total")
TEMP_GRID = tuple(round(0.2 * (50 ** (k / 39)), 4) for k in range(40))  # 0.2 .. 10, geometric


@dataclass
class PolicyRow:
    dist: dict[str, float]
    overflow: bool
    heuristic: bool


@dataclass
class Matrix:
    prompts: dict[str, dict]
    cost: dict[tuple[str, str], float]
    ok: dict[tuple[str, str], bool]
    policies: dict[str, dict[str, PolicyRow]] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def ids(self, split: str | None = None) -> list[str]:
        return [pid for pid, p in self.prompts.items() if split is None or p["split"] == split]


@dataclass
class Metrics:
    n: int
    quality: float
    cost: float
    baseline: float
    savings: float
    mix: dict[str, float]
    overflow_share: float = 0.0
    savings_per_prompt: float = 0.0

    def objective(self, name: str) -> float:
        return self.savings_per_prompt if name == "per-prompt" else self.savings

    def to_dict(self) -> dict:
        return asdict(self)


def build_matrix(prompts: list[dict], answers: Store, judgments: Store, classifications: Store,
                 family: str, judge_model: str) -> Matrix:
    m = Matrix({}, {}, {})
    dropped, truncated, refused, disagree, judged = [], 0, 0, 0, 0
    for p in prompts:
        if p["split"] == "train":  # fine-tuning data; never scored
            continue
        ans = {t: answer_lookup(answers, family, p["id"], p["prompt"], t) for t in TIERS}
        if any(a is None for a in ans.values()):
            dropped.append(p["id"])
            continue
        oks, complete = {}, True
        for tier in CANDIDATE_TIERS:
            outs = []
            for order in ORDERS:
                j = judgments.get(judgment_key(p["id"], tier, order, ans[tier]["key"], ans["flagship"]["key"], judge_model))
                outs.append(j["outcome"] if j else None)
            combined = combine(outs)
            if combined is None:
                complete = False
                break
            judged += 1
            disagree += outs[0] != outs[1]
            oks[tier] = combined in ("win", "tie")
        if not complete:
            dropped.append(p["id"])
            continue
        m.prompts[p["id"]] = p
        for t in TIERS:
            m.cost[(p["id"], t)] = ans[t]["cost"]
            truncated += ans[t]["stop_reason"] == "max_tokens"
            refused += ans[t]["stop_reason"] == "refusal"
        m.ok.update({(p["id"], t): v for t, v in oks.items()})
        m.ok[(p["id"], "flagship")] = True

    by_policy: dict[str, dict[str, PolicyRow]] = {}
    for r in classifications.rows():  # file order; later rows override earlier ones
        if r["id"] in m.prompts:
            by_policy.setdefault(r["policy"], {})[r["id"]] = PolicyRow(r["dist"], r["overflow"], r.get("heuristic", False))
    incomplete = []
    for name, rows in by_policy.items():
        if len(rows) == len(m.prompts):
            m.policies[name] = rows
        else:
            incomplete.append(name)
    m.stats = {"dropped": dropped, "truncated_answers": truncated, "refused_answers": refused,
               "judge_order_disagreement": round(disagree / judged, 3) if judged else None,
               "incomplete_policies": incomplete}
    return m


def temper(dist: dict[str, float], T: float) -> dict[str, float]:
    w = {t: max(dist.get(t, 0.0), 1e-9) ** (1.0 / T) for t in TIERS}
    s = sum(w.values())
    return {t: v / s for t, v in w.items()}


def assign(m: Matrix, policy: str, eps: float, T: float = 1.0) -> dict[str, str]:
    out = {}
    for pid, row in m.policies[policy].items():
        if row.overflow:
            out[pid] = "flagship"
        elif row.heuristic:
            out[pid] = max(TIERS, key=lambda t: row.dist.get(t, 0.0))
        else:
            out[pid] = pick_tier(temper(row.dist, T) if T != 1.0 else row.dist, eps)[0]
    return out


def constant(m: Matrix, tier: str) -> dict[str, str]:
    return {pid: tier for pid in m.prompts}


def oracle(m: Matrix) -> dict[str, str]:
    return {pid: next(t for t in TIERS if m.ok[(pid, t)]) for pid in m.prompts}


def evaluate(m: Matrix, assignment: dict[str, str], ids: list[str], policy: str | None = None) -> Metrics:
    n = len(ids)
    if not n:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0, {})
    cost = sum(m.cost[(pid, assignment[pid])] for pid in ids)
    base = sum(m.cost[(pid, "flagship")] for pid in ids)
    quality = sum(m.ok[(pid, assignment[pid])] for pid in ids) / n
    mix = {t: sum(assignment[pid] == t for pid in ids) / n for t in TIERS}
    over = sum(m.policies[policy][pid].overflow for pid in ids) / n if policy in m.policies else 0.0
    per = sum(1 - m.cost[(pid, assignment[pid])] / m.cost[(pid, "flagship")] for pid in ids) / n
    return Metrics(n, quality, cost, base, 1 - cost / base if base else 0.0, mix, over, per)


def random_at_mix(m: Matrix, mix: dict[str, float], ids: list[str]) -> Metrics:
    """Expected metrics of routing each prompt to a random tier with the given mix."""
    n = len(ids)
    if not n:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0, {})
    base = sum(m.cost[(pid, "flagship")] for pid in ids)
    cost = sum(mix[t] * m.cost[(pid, t)] for pid in ids for t in TIERS)
    quality = sum(mix[t] * m.ok[(pid, t)] for pid in ids for t in TIERS) / n
    per = sum(mix[t] * (1 - m.cost[(pid, t)] / m.cost[(pid, "flagship")]) for pid in ids for t in TIERS) / n
    return Metrics(n, quality, cost, base, 1 - cost / base if base else 0.0, dict(mix), 0.0, per)


def choose_eps(m: Matrix, policy: str, T: float, ids: list[str], floor: float,
               objective: str = "per-prompt") -> tuple[float, bool]:
    """Highest-savings epsilon whose quality clears the floor; else the highest-quality epsilon."""
    scored = [(eps, evaluate(m, assign(m, policy, eps, T), ids)) for eps in EPS_GRID]
    passing = [(e, x) for e, x in scored if x.quality >= floor]
    if passing:
        return max(passing, key=lambda ex: (ex[1].objective(objective), ex[1].quality))[0], True
    return max(scored, key=lambda ex: (ex[1].quality, ex[1].objective(objective)))[0], False


def _calib_rows(m: Matrix, policy: str, ids: list[str]):
    labels = oracle(m)
    for pid in ids:
        row = m.policies[policy][pid]
        if not row.overflow and not row.heuristic:
            yield row.dist, labels[pid]


def nll_brier(m: Matrix, policy: str, ids: list[str], T: float) -> tuple[float, float]:
    rows = list(_calib_rows(m, policy, ids))
    if not rows:
        return float("nan"), float("nan")
    nll = brier = 0.0
    for dist, label in rows:
        p = temper(dist, T)
        nll -= math.log(max(p[label], 1e-9))
        brier += sum((p[t] - (t == label)) ** 2 for t in TIERS)
    return nll / len(rows), brier / len(rows)


def fit_temperature(m: Matrix, policy: str, ids: list[str]) -> float:
    if not list(_calib_rows(m, policy, ids)):
        return 1.0
    return min(TEMP_GRID, key=lambda T: nll_brier(m, policy, ids, T)[0])


def bootstrap(m: Matrix, assignment: dict[str, str], ids: list[str], B: int = 2000, seed: int = 0) -> dict:
    """95% percentile intervals for savings (total and per-prompt) and quality, resampling prompts."""
    fields = ("savings", "savings_per_prompt", "quality")
    if not ids:
        return {f: [0.0, 0.0] for f in fields}
    rng = random.Random(seed)
    draws = {f: [] for f in fields}
    for _ in range(B):
        x = evaluate(m, assignment, [rng.choice(ids) for _ in ids])
        for f in fields:
            draws[f].append(getattr(x, f))
    lo, hi = int(0.025 * B), int(0.975 * B) - 1
    return {f: [sorted(v)[lo], sorted(v)[hi]] for f, v in draws.items()}
