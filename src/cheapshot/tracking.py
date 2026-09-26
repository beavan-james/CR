"""JSONL usage log for post-hoc cost analysis."""
from __future__ import annotations

import json
import time
from pathlib import Path

LOG = Path("usage_log.jsonl")


def log(entry: dict) -> None:
    entry = {"ts": time.time(), **entry}
    with LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")
