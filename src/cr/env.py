"""Settings from the environment. CR_* names win; the pre-rename CHEAPSHOT_* names still work."""
from __future__ import annotations

import os
from pathlib import Path


def env(name: str, default: str | None = None) -> str | None:
    """CR_<name>, else the legacy CHEAPSHOT_<name>, else default."""
    for prefix in ("CR_", "CHEAPSHOT_"):
        value = os.getenv(prefix + name)
        if value is not None:
            return value
    return default


def home() -> Path:
    """Config and history directory: CR_HOME, else ~/.cr (or ~/.cheapshot while only that exists)."""
    if override := env("HOME"):
        return Path(override)
    new, old = Path.home() / ".cr", Path.home() / ".cheapshot"
    return old if old.exists() and not new.exists() else new
