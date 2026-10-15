"""Build evals/datasets/v2.jsonl = v1 unchanged + long-answer eval prompts + a judge-label subset.

v1 rows keep their ids, prompts and splits, so everything already cached for v1 stays valid.
Additions (deterministic, seed 11):
  - 150 new eval prompts (50 tune, 100 test) weighted toward long, expensive answers, where
    most of the routing savings are: unused Arena-Hard prompts first, then oasst2 prompts that
    ask for code / writing / analysis. On v1, Arena-Hard flagship answers cost ~5x oasst2's.
  - "subset": "judge" on 300 train prompts (200 likely-long, 100 random) to get judge-based
    training labels (`bench answers/judge --split train --subset judge`).

Needs evals/datasets/v1.jsonl (python evals/build_v1.py) and the raw files it uses.
Run from the repo root: python evals/build_v2.py
"""
from __future__ import annotations

import json
import random
import re

import build_v1 as v1

OUT = v1.ROOT / "datasets" / "v2.jsonl"
SEED = 11
EVAL_TUNE, EVAL_TEST = 50, 100
EVAL_ARENA = 90  # rest from oasst2
JUDGE_LONG, JUDGE_RANDOM = 200, 100
MAX_EVAL_CHARS = 6000  # longer prompts overflow Laya and always go to the flagship: no routing signal

LONG = re.compile(r"\b(code|function|implement|script|class|python|javascript|typescript|sql|program|regex|write|"
                  r"draft|compose|story|essay|poem|article|blog|outline|plan|explain|describe|compare|analy[sz]e|"
                  r"step[- ]by[- ]step|design|create)\b|```", re.I)


def likely_long(r: dict) -> bool:
    return r["source"].startswith("arena") or bool(LONG.search(r["prompt"]))


def main() -> None:
    rng = random.Random(SEED)
    base = [json.loads(l) for l in v1.OUT.read_text().split("\n") if l.strip()]
    seen = {v1.norm(r["prompt"]) for r in base}

    def fresh(rows):
        out = []
        for r in rows:
            k = v1.norm(r["prompt"])
            if len(k) >= v1.MIN_CHARS and k not in seen and len(r["prompt"]) <= MAX_EVAL_CHARS:
                seen.add(k)
                out.append(r)
        return out

    arena = fresh(v1.load_arena())
    rng.shuffle(arena)
    arena.sort(key=lambda r: not likely_long(r))  # stable: likely-long first, shuffled within
    oasst = [r for r in fresh(v1.load_oasst2()) if LONG.search(r["prompt"])]
    rng.shuffle(oasst)
    n_arena = min(EVAL_ARENA, len(arena))
    evals = arena[:n_arena] + oasst[: EVAL_TUNE + EVAL_TEST - n_arena]
    rng.shuffle(evals)
    added = [{**r, "split": "tune" if i < EVAL_TUNE else "test", "batch": "v2"} for i, r in enumerate(evals)]

    train = [r for r in base if r["split"] == "train"]
    longs = [r for r in train if likely_long(r)]
    rng.shuffle(longs)
    picked = {r["id"] for r in longs[:JUDGE_LONG]}
    others = [r for r in train if r["id"] not in picked]
    rng.shuffle(others)
    picked |= {r["id"] for r in others[:JUDGE_RANDOM]}
    rows = [{**r, "subset": "judge"} if r["id"] in picked else r for r in base] + added

    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    by = lambda rs, k: {v: sum(r.get(k) == v for r in rs) for v in sorted({r.get(k) for r in rs})}
    print(f"wrote {len(rows)} prompts -> {OUT}")
    print(f"  new eval prompts: {len(added)} {by(added, 'split')} sources {by(added, 'source')}")
    print(f"  judge subset: {len(picked)} train prompts ({min(JUDGE_LONG, len(longs))} likely-long)")


if __name__ == "__main__":
    main()
