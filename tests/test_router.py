import pytest

from cheapshot import router
from cheapshot.classifier import Classification, overflow_classification
from cheapshot.pricing import cost_usd

DIST = {"cheap": 0.5, "balanced": 0.3, "flagship": 0.2}


def test_exceed_probs():
    ex = router.exceed_probs(DIST)
    assert ex["cheap"] == pytest.approx(0.5)
    assert ex["balanced"] == pytest.approx(0.2)
    assert ex["flagship"] == 0.0


@pytest.mark.parametrize("eps,tier", [(0.0, "flagship"), (0.2, "balanced"), (0.49, "balanced"), (0.5, "cheap"), (1.0, "cheap")])
def test_risk_dial(eps, tier):
    assert router.pick_tier(DIST, eps)[0] == tier


def test_certain_cheap_routes_cheap_even_at_zero():
    assert router.pick_tier({"cheap": 1.0, "balanced": 0.0, "flagship": 0.0}, 0.0)[0] == "cheap"


def test_epsilon_bounds():
    with pytest.raises(ValueError):
        router.pick_tier({"cheap": 1.0}, 1.5)


def test_route_models():
    r = router.route(Classification({"cheap": 1.0, "balanced": 0.0, "flagship": 0.0}, 1.0), 0.2)
    assert r.model == "claude-haiku-5-5"
    r = router.route(Classification({"cheap": 0.0, "balanced": 0.0, "flagship": 1.0}, 1.0), 0.2)
    assert r.model == "claude-opus-5-5"


def test_overflow_routes_flagship_at_any_epsilon():
    c = overflow_classification("too long")
    assert router.route(c, 1.0).model == "claude-opus-5-5"
    assert router.tier_for(c, 1.0) == "flagship"


def test_costs():
    assert cost_usd("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.0)
    assert cost_usd("claude-haiku-5-5", 0, 1_000_000) == pytest.approx(0.5)
    with pytest.raises(KeyError):
        cost_usd("unknown-model", 1, 1)


# ---------------------------------------------------------------- claude-code (Pro/Max) provider

import json
import subprocess

from cheapshot import providers


def test_claude_code_command_disables_tools_and_sets_effort():
    cmd = providers.claude_code_command("claude-opus-5-5", None, "medium")
    assert cmd[:2] == ["claude", "-p"] and "--no-session-persistence" in cmd
    assert cmd[cmd.index("--tools") + 1] == "" and cmd[cmd.index("--effort") + 1] == "medium"
    assert cmd[cmd.index("--system-prompt") + 1] == providers.CLAUDE_CODE_SYSTEM


def test_parse_claude_code_output():
    out = json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "pong",
                      "usage": {"input_tokens": 10, "cache_creation_input_tokens": 600,
                                "cache_read_input_tokens": 17, "output_tokens": 4}})
    c = providers.parse_claude_code_output(out, "claude-haiku-5-5", 1.0)
    assert c.text == "pong" and c.input_tokens == 627 and c.output_tokens == 4
    err = json.dumps({"subtype": "success", "is_error": True, "result": "usage limit reached"})
    with pytest.raises(providers.ClaudeCodeError, match="usage limit"):
        providers.parse_claude_code_output(err, "m", 0.0)
    with pytest.raises(providers.ClaudeCodeError):
        providers.parse_claude_code_output("not json", "m", 0.0)


def test_claude_code_strips_api_keys(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["env"], seen["input"], seen["cwd"] = kw["env"], kw["input"], kw["cwd"]
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"subtype": "success", "result": "ok", "usage": {}}), "")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setattr(providers.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(providers.subprocess, "run", fake_run)
    c = providers.complete("claude-code", "claude-haiku-5-5", "hi")
    assert c.text == "ok" and "ANTHROPIC_API_KEY" not in seen["env"] and seen["input"] == "hi"
    assert "cheapshot-cc-" in seen["cwd"]
