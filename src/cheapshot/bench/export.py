"""Export fine-tuning data for `laya-train` (JSONL with per-question `gold` probabilities).

Only `train`-split prompts are exported: `tune` picks epsilon and calibration and
`test` is the held-out result, so training on either would inflate the report.

Label per prompt, best source first:
  judge   - from the cached benchmark. Each judge order yields its own oracle tier
            (cheapest tier the judge accepted); gold is the average of the two, so a
            disagreement between orders becomes a split label instead of a coin flip.
  teacher - Opus's probability over tiers from the prompt alone.
"""
from __future__ import annotations

from ..classifier import TIERS
from .judge import CANDIDATE_TIERS, ORDERS, judgment_key
from .matrix import answer_lookup
from .store import Store
from .teacher import teacher_key


def judge_gold(answers: Store, judgments: Store, family: str, judge_model: str, p: dict) -> dict[str, float] | None:
    ans = {t: answer_lookup(answers, family, p["id"], p["prompt"], t) for t in TIERS}
    if any(a is None for a in ans.values()):
        return None
    gold = {t: 0.0 for t in TIERS}
    for order in ORDERS:
        label = "flagship"
        for tier in CANDIDATE_TIERS:  # cheapest first; stop at the first accepted tier (lazy judging skips the rest)
            j = judgments.get(judgment_key(p["id"], tier, order, ans[tier]["key"], ans["flagship"]["key"], judge_model))
            if not j or j["outcome"] is None:
                return None
            if j["outcome"] in ("win", "tie"):
                label = tier
                break
        gold[label] += 1 / len(ORDERS)
    return gold


def tier_gold_to_levels(gold: dict[str, float], tier_map: dict) -> dict[str, float]:
    """Map tier probabilities onto the variant's answer keys (score levels or choice options)."""
    by_tier = {}
    for key, tier in tier_map.items():
        by_tier.setdefault(tier, []).append(str(key))
    out = {}
    for tier, keys in by_tier.items():
        for k in keys:  # split a tier's mass evenly if several keys map to it
            out[k] = gold.get(tier, 0.0) / len(keys)
    return out


def build_training_rows(prompts: list[dict], variant: dict, answers: Store, judgments: Store, teacher: Store,
                        family: str, judge_model: str, teacher_model: str,
                        teacher_map: list[tuple[float, dict]] | None = None,
                        judge_weight: int = 1) -> tuple[list[dict], dict]:
    """judge_weight > 1 repeats judge-labeled rows (ids suffixed #2, #3, ...) so the more
    accurate labels count more against the teacher-labeled majority."""
    rows, counts = [], {"judge": 0, "teacher": 0, "unlabeled": 0, "not_train_split": 0}
    qid = variant["question_id"]
    for p in prompts:
        if p["split"] != "train":
            counts["not_train_split"] += 1
            continue
        gold = judge_gold(answers, judgments, family, judge_model, p)
        source = "judge"
        if gold is None:
            t = teacher.get(teacher_key(p["id"], p["prompt"], teacher_model))
            gold, source = (t["dist"], "teacher") if t else (None, None)
            if gold is not None and teacher_map:
                gold = apply_teacher_map(gold, teacher_map)
        if gold is None:
            counts["unlabeled"] += 1
            continue
        counts[source] += 1
        row = {"id": p["id"], "label_source": source, "state": p["prompt"],
               "questions": {qid: variant["question"]},
               "gold": {qid: {"probabilities": tier_gold_to_levels(gold, variant["tier_map"])}}}
        rows.append(row)
        if source == "judge":
            rows += [{**row, "id": f"{p['id']}#{k}"} for k in range(2, judge_weight + 1)]
    return rows, counts


# ---------------------------------------------------------------- teacher recalibration
# The teacher is systematically optimistic (on v1 tune: mean level 0.45 vs judge 1.19,
# correlation 0.49). Map teacher level -> the tier mix the judge actually gave at that
# level, fitted on tune-split prompts that have both, and apply it to train labels.

def _level(d: dict[str, float]) -> float:
    return d.get("balanced", 0.0) + 2 * d.get("flagship", 0.0)


def fit_teacher_map(pairs: list[tuple[dict, dict]], bins: int = 5, prior_weight: float = 3.0) -> list[tuple[float, dict]]:
    """pairs = (teacher dist, judge gold). Returns [(bin center level, judge mix)], sorted.
    Quantile bins on teacher level; each bin's mix is shrunk toward the overall judge mix."""
    if not pairs:
        raise ValueError("no (teacher, judge) pairs to fit on")
    overall = {t: sum(g[t] for _, g in pairs) / len(pairs) for t in TIERS}
    ranked = sorted(pairs, key=lambda p: _level(p[0]))
    size = max(1, len(ranked) // bins)
    chunks = [ranked[i:i + size] for i in range(0, len(ranked), size)]
    if len(chunks) > 1 and len(chunks[-1]) < size:
        chunks[-2] += chunks.pop()
    out = []
    for ch in chunks:
        n = len(ch)
        mix = {t: (sum(g[t] for _, g in ch) + prior_weight * overall[t]) / (n + prior_weight) for t in TIERS}
        out.append((sum(_level(t) for t, _ in ch) / n, mix))
    return out


def apply_teacher_map(dist: dict[str, float], mapping: list[tuple[float, dict]]) -> dict[str, float]:
    """Linear interpolation between bin centers by the teacher's level."""
    x = _level(dist)
    if x <= mapping[0][0]:
        return dict(mapping[0][1])
    if x >= mapping[-1][0]:
        return dict(mapping[-1][1])
    for (x0, m0), (x1, m1) in zip(mapping, mapping[1:]):
        if x0 <= x <= x1:
            w = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
            return {t: (1 - w) * m0[t] + w * m1[t] for t in TIERS}
    return dict(mapping[-1][1])


def teacher_calibration_pairs(prompts, answers, judgments, teacher, family, judge_model, teacher_model):
    """(teacher, judge) pairs from tune prompts and judge-labeled train prompts; never test."""
    pairs = []
    for p in prompts:
        if p["split"] not in ("tune", "train"):
            continue
        t = teacher.get(teacher_key(p["id"], p["prompt"], teacher_model))
        g = judge_gold(answers, judgments, family, judge_model, p)
        if t and g:
            pairs.append((t["dist"], g))
    return pairs
