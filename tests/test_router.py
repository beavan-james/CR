from cheapshot import jev_client, router
from cheapshot.pricing import cost_usd


def test_mock_classify_tiers():
    assert jev_client.mock_classify("What is 2+2?").label == "simple"
    c = jev_client.mock_classify("Design a distributed rate limiter with proof of correctness under clock skew")
    assert c.label == "complex"


def test_router_maps_and_fallback():
    r = router.route("simple", 0.9, "anthropic")
    assert r.tier == "cheap" and "claude" in r.model
    r2 = router.route("complex", 0.9, "openai")
    assert r2.tier == "flagship"
    r3 = router.route("simple", 0.0, "gemini")  # low confidence -> flagship fallback
    assert r3.tier == "flagship" and r3.fallback_used


def test_cost_positive():
    assert cost_usd("gpt-4o-mini", 1000, 500) > 0
