"""Benchmark dataset: JSONL rows {id, prompt, category, source, split[, ideal_tier]}.

split is "tune" (choose epsilon / wording / calibration) or "test" (held out, reported).
ideal_tier is optional and only drives the mock judge.
"""
from __future__ import annotations

import json
from pathlib import Path

SPLITS = ("tune", "test")


def load_dataset(path: Path) -> list[dict]:
    rows, seen = [], set()
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
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
