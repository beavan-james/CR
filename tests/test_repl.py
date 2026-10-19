import json

from cheapshot import curves, repl, routing, session


def test_handle_event_streams_text_tools_and_usage():
    res, texts, tools = session.TurnResult(model="m"), [], []
    events = [
        {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hel"}}},
        {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "lo"}}},
        {"type": "assistant", "message": {"model": "claude-sonnet-5-5", "content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls -la"}}]}},
        {"type": "result", "subtype": "success", "total_cost_usd": 0.02, "duration_ms": 1500,
         "usage": {"input_tokens": 2, "cache_read_input_tokens": 100, "output_tokens": 7}},
    ]
    for e in events:
        session.handle_event(e, res, texts.append, tools.append)
    assert res.text == "Hello" and texts == ["Hel", "lo"] and tools == ["Bash(ls -la)"]
    assert res.model == "claude-sonnet-5-5" and res.cost_usd == 0.02 and res.input_tokens == 102
    assert res.output_tokens == 7 and not res.is_error
    session.handle_event({"type": "result", "subtype": "error_max_turns", "is_error": True}, res, texts.append, tools.append)
    assert res.is_error


def test_turn_command_first_then_resume():
    first = session.turn_command("claude-haiku-5-5", "sid", True, "acceptEdits", "medium")
    later = session.turn_command("claude-opus-5-5", "sid", False, "plan")
    assert first[first.index("--session-id") + 1] == "sid" and "--resume" not in first
    assert later[later.index("--resume") + 1] == "sid" and later[later.index("--permission-mode") + 1] == "plan"
    assert later[later.index("--model") + 1] == "claude-opus-5-5"


def _curve():
    pts = []
    for eps, tq, sp in ((0.0, 1.0, 0.0), (0.3, 0.97, 0.05), (0.4, 0.95, 0.11), (0.5, 0.88, 0.25)):
        m = {"quality": tq, "savings": sp / 2, "savings_per_prompt": sp, "mix": {"cheap": 0, "balanced": 0.2, "flagship": 0.8}}
        pts.append({"eps": eps, "tune": m, "test": m, "random_test": {"quality": tq - 0.05, "savings_per_prompt": sp}})
    return {"variant": "a_score", "router_model": "x", "n_tune": 10, "n_test": 10, "points": pts}


def test_eps_for_quality_picks_most_savings_meeting_target():
    c = _curve()
    assert curves.eps_for_quality(c, 0.95)[0] == 0.4
    assert curves.eps_for_quality(c, 0.96)[0] == 0.3
    assert curves.eps_for_quality(c, 0.80)[0] == 0.5
    assert curves.eps_for_quality(c, 1.01)[0] == 0.0  # unreachable target -> safest
    assert [r["target"] for r in curves.frontier(c, (0.95, 0.9))] == [0.95, 0.9]


def test_follow_up_turns_carry_previous_request():
    assert routing.routing_text("Prove the Riemann hypothesis in full detail please", None).startswith("Prove")
    t = routing.routing_text("yes do it", "Refactor the auth module to use OAuth")
    assert "Refactor the auth module" in t and t.endswith("yes do it")
    long = "x" * 500
    assert routing.routing_text(long, "previous") == long


def test_opus_equivalent_and_pretty():
    assert repl.pretty("claude-sonnet-5-5") == "Sonnet 5.5"
    assert repl.opus_equivalent(1.0, "claude-sonnet-5-5") == 2.0
    assert repl.opus_equivalent(1.0, "claude-opus-5-5") == 1.0


def test_handle_event_relative_paths_and_denials():
    TurnResult, handle_event = session.TurnResult, session.handle_event
    res = TurnResult()
    tools = []
    handle_event({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Edit", "input": {"file_path": "/proj/src/a.py"}}]}},
        res, lambda s: None, tools.append, cwd="/proj")
    assert tools == ["Edit(src/a.py)"]
    handle_event({"type": "result", "subtype": "success", "total_cost_usd": 0.01, "usage": {},
                  "permission_denials": [{"tool_name": "Bash"}, {"tool_name": "Bash"}]},
                 res, lambda s: None, lambda s: None)
    assert res.denied == ["Bash", "Bash"] and not res.is_error
