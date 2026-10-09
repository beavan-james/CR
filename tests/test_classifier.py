import json

import httpx
import pytest

from cheapshot import classifier

A = classifier.load_variant("variants/a_score.yaml")
C = classifier.load_variant("variants/c_tier_choice.yaml")

# laya-serve /v1/systemone shape (Jev-compatible, plus Laya's extended usage block)
SCORE_RESPONSE = {
    "model": "laya-typed-decisions",
    "answers": {"complexity": {"type": "score", "score": 1.43, "confidence": 0.35,
                               "probabilities": {"0": 0.07, "1": 0.43, "2": 0.50},
                               "legend": {"0": "s", "1": "m", "2": "c"}}},
    "usage": {"input_tokens": 328, "output_tokens": 34, "truncated": 0, "state_tokens": 40, "state_tokens_dropped": 0},
}
CHOICE_RESPONSE = {
    "model": "laya-typed-decisions",
    "answers": {"tier": {"type": "choice", "choice": "small", "confidence": 0.9,
                         "probabilities": {"mid": 0.05, "small": 0.95, "frontier": 0.0}}},
    "usage": {"input_tokens": 400, "output_tokens": 30},
}


def test_score_response_to_dist():
    c = classifier.classification_from_response(SCORE_RESPONSE, A)
    assert c.dist == pytest.approx({"cheap": 0.07, "balanced": 0.43, "flagship": 0.50})
    assert c.top_tier == "flagship" and c.confidence == 0.35 and not c.overflow


def test_int_score_keys():
    d = classifier.dist_from_probabilities({0: 0.2, 1: 0.8, 2: 0.0}, A["tier_map"])
    assert d["balanced"] == pytest.approx(0.8)


def test_choice_response_to_dist():
    c = classifier.classification_from_response(CHOICE_RESPONSE, C)
    assert c.dist == pytest.approx({"cheap": 0.95, "balanced": 0.05, "flagship": 0.0})


def test_truncated_usage_is_overflow():
    resp = {**SCORE_RESPONSE, "usage": {"truncated": 1, "state_tokens_dropped": 300}}
    c = classifier.classification_from_response(resp, A)
    assert c.overflow and c.top_tier == "flagship" and c.dist == {}


def test_missing_answer_and_unknown_option_raise():
    with pytest.raises(classifier.RouterError):
        classifier.classification_from_response({"answers": {}}, A)
    with pytest.raises(classifier.RouterError):
        classifier.dist_from_probabilities({"huge": 1.0}, C["tier_map"])


def _client(handler, **kw):
    return classifier.LayaClient(url="http://laya.test", api_key="k", transport=httpx.MockTransport(handler), **kw)


def test_laya_client_request_shape():
    seen = {}

    def handler(req: httpx.Request):
        seen["auth"] = req.headers.get("authorization")
        seen["body"] = json.loads(req.content)
        seen["path"] = req.url.path
        return httpx.Response(200, json=CHOICE_RESPONSE)

    c = _client(handler).classify("hello", C)
    assert c.dist["cheap"] == pytest.approx(0.95)
    assert seen["auth"] == "Bearer k" and seen["path"] == "/v1/systemone"
    assert seen["body"]["model"] == "typed-decisions" and seen["body"]["state"] == "hello"
    assert list(seen["body"]["questions"]) == ["tier"]


def test_laya_client_skips_overlong_prompt():
    def handler(req):
        raise AssertionError("should not be called")

    c = _client(handler, max_chars=10).classify("x" * 11, A)
    assert c.overflow


def test_laya_client_errors_raise():
    with pytest.raises(classifier.RouterError, match="422"):
        _client(lambda r: httpx.Response(422, json={"detail": "bad"})).classify("hi", A)

    def down(req):
        raise httpx.ConnectError("refused")

    with pytest.raises(classifier.RouterError, match="unreachable"):
        _client(down).classify("hi", A)


def test_laya_client_requires_url(monkeypatch):
    monkeypatch.delenv("LAYA_URL", raising=False)
    with pytest.raises(classifier.RouterError, match="LAYA_URL"):
        classifier.LayaClient()


def test_rotate_choice():
    r1 = classifier.rotate_choice(C, 1)
    assert list(r1["question"]["criteria"]) == ["mid", "frontier", "small"]
    assert list(C["question"]["criteria"]) == ["small", "mid", "frontier"]
    assert r1["name"] == "c_tier_choice@r1"
    with pytest.raises(ValueError):
        classifier.rotate_choice(A, 1)


def test_heuristic_is_one_hot():
    c = classifier.heuristic_classify("What is 2+2?")
    assert c.top_tier == "cheap" and c.heuristic and sum(c.dist.values()) == 1.0
    c = classifier.heuristic_classify("Design a distributed rate limiter with proof of correctness under clock skew")
    assert c.top_tier == "flagship"
