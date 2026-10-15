"""Pairwise judge: is the cheaper tier's answer at least as good as the flagship's?

Each (prompt, candidate tier) is judged twice with the answers in swapped
positions. Outcomes are from the candidate's point of view (win/tie/loss); if the
two orders disagree the result is a tie. Non-inferior = win or tie.
"""
from __future__ import annotations

import re

from .. import providers
from ..classifier import TIERS
from ..pricing import cost_usd
from .matrix import answer_lookup
from .pool import PoolResult, run_pool
from .store import Store, make_key

JUDGE_MODEL = "claude-opus-5-5"
JUDGE_EFFORT = "medium"
CANDIDATE_TIERS = ("cheap", "balanced")
ORDERS = ("candidate_first", "flagship_first")

JUDGE_SYSTEM = """You compare two AI responses to the same user request and decide which serves the user better.

Judge on correctness first, then completeness and usefulness for what was actually asked. Length, formatting and tone matter only when they change how well the user is served. Do not prefer a response because it is longer or appears first.

Answer TIE when both responses would serve the user about equally well, including when both are fully adequate and differ only in style, or when both fail in similar ways.

Explain your reasoning briefly, then end with exactly one final line:
VERDICT: A
VERDICT: B
or
VERDICT: TIE"""

_VERDICT = re.compile(r"VERDICT:\s*(A|B|TIE)\b", re.I)


def build_user(prompt: str, a: str, b: str) -> str:
    return (f"<request>\n{prompt}\n</request>\n\n<response_a>\n{a}\n</response_a>\n\n"
            f"<response_b>\n{b}\n</response_b>")


def parse_verdict(text: str) -> str | None:
    found = _VERDICT.findall(text or "")
    return found[-1].upper() if found else None


def candidate_outcome(verdict: str | None, order: str) -> str | None:
    if verdict is None:
        return None
    if verdict == "TIE":
        return "tie"
    cand_slot = "A" if order == "candidate_first" else "B"
    return "win" if verdict == cand_slot else "loss"


def combine(outcomes: list[str | None]) -> str | None:
    """Both orders agree -> that outcome; disagree -> tie; any missing -> None."""
    if len(outcomes) != len(ORDERS) or any(o is None for o in outcomes):
        return None
    return outcomes[0] if len(set(outcomes)) == 1 else "tie"


def judgment_key(pid: str, tier: str, order: str, cand_key: str, flag_key: str, judge_model: str) -> str:
    return make_key("judge", pid, tier, order, cand_key, flag_key, judge_model, JUDGE_SYSTEM)


def mock_verdict(prompt_row: dict, tier: str, order: str) -> str:
    """Deterministic stand-in for offline runs: candidate ties if its tier >= ideal_tier, else flagship wins."""
    ideal = prompt_row.get("ideal_tier")
    if not ideal or TIERS.index(tier) >= TIERS.index(ideal):
        return "TIE"
    return "B" if order == "candidate_first" else "A"


def is_lazy(p: dict) -> bool:
    """Train prompts only need a label (cheapest accepted tier per order), so balanced is judged
    in an order only after cheap lost there. Scored splits are judged fully: replay needs both."""
    return p["split"] == "train"


def pending_judgments(prompts: list[dict], answers: Store, judgments: Store, family: str,
                      judge_model: str = JUDGE_MODEL) -> tuple[list[tuple], int, int]:
    """(jobs runnable now, lazy jobs waiting on a cheap verdict, prompt-tiers missing answers)."""
    jobs, waiting, skipped = [], 0, 0
    for p in prompts:
        flag = answer_lookup(answers, family, p["id"], p["prompt"], "flagship")
        cheap_outcome = {}
        for tier in CANDIDATE_TIERS:
            cand = answer_lookup(answers, family, p["id"], p["prompt"], tier)
            if not flag or not cand:
                skipped += 1
                continue
            for order in ORDERS:
                key = judgment_key(p["id"], tier, order, cand["key"], flag["key"], judge_model)
                done = judgments.get(key)
                if tier == "cheap":
                    cheap_outcome[order] = done["outcome"] if done else None
                if done:
                    continue
                if tier != "cheap" and is_lazy(p):
                    prev = cheap_outcome.get(order)
                    if prev is None:
                        waiting += 1
                        continue
                    if prev != "loss":  # cheap accepted in this order: balanced can't change the label
                        continue
                jobs.append((p, tier, order, cand, flag, key))
    return jobs, waiting, skipped


def run_judge(prompts: list[dict], answers: Store, judgments: Store, family: str, api_key: str,
              workers: int = 4, judge_model: str = JUDGE_MODEL, on_done=None) -> tuple[PoolResult, int]:
    """Judge every candidate answer that has both itself and the flagship answer cached
    (lazily for train prompts, see is_lazy). Returns (pool result, prompt-tiers skipped for missing answers)."""

    def one(p, tier, order, cand, flag, key) -> dict:
        a, b = (cand["text"], flag["text"]) if order == "candidate_first" else (flag["text"], cand["text"])
        if family == "mock":
            verdict, text, cost = mock_verdict(p, tier, order), "", 0.0
        else:
            comp = providers.complete(family, judge_model, build_user(p["prompt"], a, b), api_key=api_key,
                                      system=JUDGE_SYSTEM, effort=JUDGE_EFFORT)
            text = comp.text
            verdict = parse_verdict(text)
            cost = cost_usd(judge_model, comp.input_tokens, comp.output_tokens)
        row = {"key": key, "id": p["id"], "tier": tier, "order": order, "verdict": verdict,
               "outcome": candidate_outcome(verdict, order), "judge_model": judge_model,
               "reasoning": text[-2000:], "cost": cost}
        judgments.add(row)
        return row

    total = PoolResult()
    while True:  # a second pass picks up lazy balanced jobs unlocked by fresh cheap verdicts
        jobs, waiting, skipped = pending_judgments(prompts, answers, judgments, family, judge_model)
        if not jobs:
            total.not_run += waiting
            return total, skipped
        res = run_pool(jobs, one, workers, on_done)
        total.done += res.done
        total.failures += res.failures
        if res.stop_reason or res.done == 0:
            total.stop_reason = res.stop_reason
            total.not_run += res.not_run + waiting
            return total, skipped
