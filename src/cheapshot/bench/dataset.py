"""Benchmark dataset: JSONL rows {id, prompt, category, source, split[, ideal_tier]}.

split is "train" (Laya fine-tuning data only), "tune" (choose epsilon / calibration) or
"test" (held out, reported). The benchmark report ignores train; export ignores tune/test.
ideal_tier is optional and only drives the mock judge.
"""
from __future__ import annotations

import json
from pathlib import Path

SPLITS = ("train", "tune", "test")


def load_dataset(path: Path) -> list[dict]:
    rows, seen = [], set()
    for n, line in enumerate(Path(path).read_text().split("\n"), 1):  # not splitlines(): U+2028 inside prompts
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("prompt"):
            raise ValueError(f"{path}:{n} has no prompt")
        r.setdefault("id", f"row{n}")
        if r["id"] in seen:
            raise ValueError(f"{path}:{n} duplicate id {r['id']!r}")
        seen.add(r["id"])
        r.setdefault("category", "uncategorized")
        r.setdefault("source", "unknown")
        r.setdefault("split", "tune")
        if r["split"] not in SPLITS:
            raise ValueError(f"{path}:{n} split must be one of {SPLITS}")
        rows.append(r)
    return rows
