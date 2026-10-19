"""JSONL usage log for post-hoc cost analysis."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

LOG = Path(os.getenv("CHEAPSHOT_LOG", Path.home() / ".cheapshot" / "usage.jsonl"))


def log(entry: dict) -> None:
    entry = {"ts": time.time(), **entry}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")
