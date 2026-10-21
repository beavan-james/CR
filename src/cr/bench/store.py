"""Append-only JSONL store keyed by `key`, so benchmark steps resume after a crash."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path


def make_key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._rows: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().split("\n"):  # not splitlines(): U+2028 inside text
                if line.strip():
                    row = json.loads(line)
                    self._rows[row["key"]] = row  # last write wins

    def has(self, key: str) -> bool:
        return key in self._rows

    def get(self, key: str) -> dict | None:
        return self._rows.get(key)

    def add(self, row: dict) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            self._rows[row["key"]] = row

    def rows(self) -> list[dict]:
        return list(self._rows.values())
