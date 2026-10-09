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
        ok = {"flagship": True}
        for tier in CANDIDATE_TIERS:
            j = judgments.get(judgment_key(p["id"], tier, order, ans[tier]["key"], ans["flagship"]["key"], judge_model))
            if not j or j["outcome"] is None:
                return None
            ok[tier] = j["outcome"] in ("win", "tie")
        gold[next(t for t in TIERS if ok[t])] += 1 / len(ORDERS)
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
                        family: str, judge_model: str, teacher_model: str) -> tuple[list[dict], dict]:
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
        if gold is None:
            counts["unlabeled"] += 1
            continue
        counts[source] += 1
        rows.append({"id": p["id"], "label_source": source, "state": p["prompt"],
                     "questions": {qid: variant["question"]},
                     "gold": {qid: {"probabilities": tier_gold_to_levels(gold, variant["tier_map"])}}})
    return rows, counts
