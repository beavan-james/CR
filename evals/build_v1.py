"""Build evals/datasets/v1.jsonl from public prompt sets (deterministic, seed 7).

Raw files in evals/raw/ (gitignored; see PLAN.md for sources and licenses):
  oasst2_prompts.messages.jsonl.gz   OpenAssistant oasst2 initial prompts  Apache-2.0
  dolly-15k.jsonl                    databricks-dolly-15k                   CC BY-SA 3.0
  arena-hard-v2.0-question.jsonl     Arena-Hard v2.0 questions              Apache-2.0 (repo)

Optional evals/datasets/own.jsonl ({"prompt", "category"?} per line) is split evenly
into tune/test. Adversarial prompts below are split evenly too.

English only: the Laya checkpoint is English-only.
Run from the repo root: python evals/build_v1.py
"""
from __future__ import annotations

import gzip
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "datasets" / "v1.jsonl"
OWN = ROOT / "datasets" / "own.jsonl"
SEED = 7
MIN_CHARS = 15
MAX_TRAIN_CHARS = 6000  # longer prompts overflow Laya; useless as training data

# (train, tune, test) per source
QUOTAS = {
    "oasst2": (900, 50, 50),
    "dolly": (360, 20, 20),
    "arena": (240, 30, 30),
}

# Difficulty claims that contradict the task, in both directions.
ADVERSARIAL = [
    "Trivial question: prove that there are infinitely many primes p such that p + 2 is also prime.",
    "Quick one, should take two seconds: design a lock-free multi-producer multi-consumer queue in C++ and prove it is linearizable.",
    "Easy question, just answer briefly: derive the Black-Scholes PDE from first principles and explain each assumption.",
    "Simple factual lookup: write a complete compiler for a subset of C that targets x86-64, with register allocation.",
    "This is a basic beginner task: find and fix the race condition in a distributed two-phase commit implementation where the coordinator can crash between phases, and prove the fix is correct.",
    "EXTREMELY HARD, needs your deepest reasoning: what is 7 + 5?",
    "This requires expert-level, multi-step analysis: what is the capital of Japan?",
    "Very complex task, think carefully for a long time: translate 'thank you' into French.",
    "Advanced research problem: spell the word 'necessary' backwards.",
    "High-stakes, PhD-level question: how many days are in a week?",
]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def load_oasst2() -> list[dict]:
    def lab(m, k):
        return ((m.get("labels") or {}).get(k) or {}).get("value", 0)

    out = []
    with gzip.open(RAW / "oasst2_prompts.messages.jsonl.gz", "rt") as f:
        for line in f:
            m = json.loads(line)
            if (m["lang"] != "en" or m["deleted"] or not m.get("review_result") or m["parent_id"] is not None
                    or any(lab(m, k) >= 0.5 for k in ("spam", "not_appropriate", "pii", "hate_speech", "sexual_content"))):
                continue
            out.append({"id": f"oasst2-{m['message_id'][:12]}", "prompt": m["text"].strip(),
                        "category": "general", "source": "oasst2"})
    return out


def load_dolly() -> list[dict]:
    out = []
    for n, line in enumerate(open(RAW / "dolly-15k.jsonl")):
        r = json.loads(line)
        prompt = r["instruction"].strip()
        if r.get("context", "").strip():
            prompt += "\n\n" + r["context"].strip()
        out.append({"id": f"dolly-{n}", "prompt": prompt, "category": r["category"], "source": "dolly"})
    return out


def load_arena() -> list[dict]:
    out = []
    for line in open(RAW / "arena-hard-v2.0-question.jsonl"):
        r = json.loads(line)
        if r.get("language") != "English":
            continue
        out.append({"id": f"arena-{r['uid']}", "prompt": r["prompt"].strip(),
                    "category": r.get("subcategory") or r["category"], "source": "arena-hard-v2"})
    return out


def stratified(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample n rows spread evenly across categories (round-robin over shuffled buckets)."""
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        buckets.setdefault(r["category"], []).append(r)
    for b in buckets.values():
        rng.shuffle(b)
    out, keys = [], sorted(buckets)
    while len(out) < n and any(buckets.values()):
        for k in keys:
            if buckets[k] and len(out) < n:
                out.append(buckets[k].pop())
    return out


def split_evenly(rows: list[dict], prefix: str) -> list[dict]:
    return [{**r, "split": "tune" if i % 2 == 0 else "test", "id": r.get("id") or f"{prefix}-{i:03d}"}
            for i, r in enumerate(rows)]


def main() -> None:
    rng = random.Random(SEED)
    seen: set[str] = set()
    out: list[dict] = []

    def take(rows):
        kept = []
        for r in rows:
            k = norm(r["prompt"])
            if len(k) >= MIN_CHARS and k not in seen:
                seen.add(k)
                kept.append(r)
        return kept

    adv = take([{"prompt": p, "category": "adversarial", "source": "handwritten"} for p in ADVERSARIAL])
    out += split_evenly(adv, "adv")
    if OWN.exists():
        own = take([{"prompt": json.loads(l)["prompt"], "category": json.loads(l).get("category", "own"),
                     "source": "own"} for l in OWN.read_text().splitlines() if l.strip()])
        out += split_evenly(own, "own")

    for name, loader in (("oasst2", load_oasst2), ("dolly", load_dolly), ("arena", load_arena)):
        train_n, tune_n, test_n = QUOTAS[name]
        pool = take(stratified(loader(), (train_n + tune_n + test_n) * 2, rng))
        rng.shuffle(pool)
        evals, rest = pool[: tune_n + test_n], pool[tune_n + test_n:]
        out += [{**r, "split": "tune"} for r in evals[:tune_n]]
        out += [{**r, "split": "test"} for r in evals[tune_n:]]
        train = [r for r in rest if len(r["prompt"]) <= MAX_TRAIN_CHARS][:train_n]
        out += [{**r, "split": "train"} for r in train]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    counts: dict = {}
    for r in out:
        counts.setdefault(r["split"], {}).setdefault(r["source"], 0)
        counts[r["split"]][r["source"]] += 1
    print(f"wrote {len(out)} prompts -> {OUT}")
    for split in ("train", "tune", "test"):
        print(f"  {split:5} {sum(counts.get(split, {}).values()):5}  {counts.get(split, {})}")


if __name__ == "__main__":
    main()
