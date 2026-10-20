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


def test_trivial_turns():
    for s in ("ok thanks", "Thanks!", "great, thank you so much", "perfect, thanks for the help", "bye", "ty"):
        assert routing.is_trivial(s), s
    for s in ("ok", "yes", "go ahead", "yes do it", "thanks, now fix the tests", "thanks?", "lgtm",
              "thank you " * 10):
        assert not routing.is_trivial(s), s


def test_length_gold():
    from cheapshot.bench.export import length_gold
    for n, top in ((50, "0"), (1100, "1"), (8000, "2")):
        g = length_gold(n)
        assert abs(sum(g.values()) - 1) < 1e-9 and max(g, key=g.get) == top
    near = length_gold(700)
    assert 0.4 < near["0"] < 0.6


def test_cost_model_recovers_weights_and_picks_by_size():
    from cheapshot import costaware
    import math
    rows = []
    for lvl_d in ({"cheap": 1.0}, {"balanced": 1.0}, {"flagship": 1.0}, {"cheap": .5, "balanced": .5}):
        for chars in (20, 200, 2000):
            rows.append((lvl_d, chars, math.exp(-5 + 1.5 * costaware.level(lvl_d) + 0.1 * math.log(chars + 1))))
    w = costaware.fit_cost_model(rows, ridge=0.0)
    assert [round(x, 3) for x in w] == [-5.0, 1.5, 0.1]
    ratios = costaware.price_ratios("claude-code")
    dist = {"cheap": 0.3, "balanced": 0.3, "flagship": 0.4}
    # same risk: a big answer is worth sending to Haiku, a small one is not
    assert costaware.pick_tier_cost(dist, est_cost=1.0, lam=0.5, ratios=ratios)[0] == "cheap"
    assert costaware.pick_tier_cost(dist, est_cost=0.01, lam=0.5, ratios=ratios)[0] == "flagship"


def test_cost_curve_point_selection_uses_total_savings_and_safest_fallback():
    def pt(lam, q, tot, pp):
        m = {"quality": q, "savings": tot, "savings_per_prompt": pp, "mix": {}}
        return {"lam": lam, "tune": m, "test": m, "random_test": {"quality": q, "savings_per_prompt": pp}}
    c = {"rule": "cost", "objective": "savings", "cost_model": {"w": [0, 0, 0], "mean_cost": 0.1},
         "points": [pt(8, 1.0, 0.0, 0.0), pt(2, 0.96, 0.25, 0.05), pt(1.5, 0.95, 0.20, 0.15), pt(1, 0.85, 0.5, 0.3)]}
    assert curves.point_for_quality(c, 0.95)["lam"] == 2  # most TOTAL savings, not per-prompt
    assert abs(curves.knob(c, curves.point_for_quality(c, 0.95)) - 0.2) < 1e-12
    assert curves.point_for_quality(c, 1.01)["lam"] == 8
    assert curves.knob_label(c, c["points"][1]) == "λ 2.00×"


def test_turn_cost_is_delta_of_cumulative_total():
    s = session.ClaudeCodeSession(cwd="/tmp")
    assert [round(s.turn_cost(x), 4) for x in (0.1394, 0.6399, 0.0, 0.6467)] == [0.1394, 0.5005, 0.0, 0.0068]
    s.reset()
    assert s.turn_cost(0.05) == 0.05
