from pathlib import Path

from cr.env import env, home


def _clear(monkeypatch, name):
    monkeypatch.delenv("CR_" + name, raising=False)
    monkeypatch.delenv("CHEAPSHOT_" + name, raising=False)


def test_cr_name_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("CR_QUALITY", "0.9")
    monkeypatch.setenv("CHEAPSHOT_QUALITY", "0.8")
    assert env("QUALITY") == "0.9"


def test_legacy_name_still_works(monkeypatch):
    _clear(monkeypatch, "QUALITY")
    monkeypatch.setenv("CHEAPSHOT_QUALITY", "0.8")
    assert env("QUALITY", "0.95") == "0.8"


def test_default_when_unset(monkeypatch):
    _clear(monkeypatch, "QUALITY")
    assert env("QUALITY", "0.95") == "0.95"


def test_home_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CR_HOME", str(tmp_path / "x"))
    assert home() == tmp_path / "x"


def test_home_prefers_cr_then_falls_back_to_legacy(monkeypatch, tmp_path):
    _clear(monkeypatch, "HOME")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert home() == tmp_path / ".cr"  # neither exists
    (tmp_path / ".cheapshot").mkdir()
    assert home() == tmp_path / ".cheapshot"  # only the old one exists
    (tmp_path / ".cr").mkdir()
    assert home() == tmp_path / ".cr"  # both exist
