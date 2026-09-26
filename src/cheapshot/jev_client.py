"""Jev classification client.

Expected Jev API shape (from user example):
  POST {JEV_API_URL}  headers {Authorization: Bearer $JEV_API_KEY}
  body {"state": "<prompt>", "questions": {<from router_config.yaml>}}
  response: one of (we accept all):
    {"answers": {"complexity": {"score": 1, "confidence": 0.9}}}
    {"results": {"complexity": 1}}
    {"complexity": {"label": "standard", "confidence": 0.8}}
    {"complexity": 2}

If JEV_API_URL is unset, falls back to deterministic mock heuristics so the
CLI/benchmarks work offline. Mock returns (label, confidence).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import httpx
import yaml

COMPLEX_HINTS = re.compile(
    r"\b(prove|theorem|distributed|concurrency|architect|refactor large|microservice|"
    r"security audit|vulnerability|optimize complex|multi-step|chain of|rigorous)\b",
    re.I,
)
MID_HINTS = re.compile(
    r"\b(write|explain|debug|crash|bug|fails?|broken|safari|function|class|api|test|summar|analy[sz]e|compare|plan|email|essay)\b",
    re.I,
)


@dataclass
class Classification:
    label: str  # simple | standard | complex
    confidence: float
    raw: dict
    mocked: bool = False


def load_questions(config_path: str = "router_config.yaml") -> dict:
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("questions", {})
    except FileNotFoundError:
        return {}


def mock_classify(prompt: str) -> Classification:
    p = prompt.lower()
    # very short factual prompts -> simple
    if len(p.split()) <= 8 and "?" in prompt or p.startswith(("what is", "who is", "define", "translate", "fix typo")):
        return Classification("simple", 0.75, {"reason": "short-factual-heuristic"}, True)
    if COMPLEX_HINTS.search(prompt) or len(prompt) > 1200:
        return Classification("complex", 0.80, {"reason": "complex-keyword/length-heuristic"}, True)
    if MID_HINTS.search(prompt) or len(prompt) > 200:
        return Classification("standard", 0.70, {"reason": "mid-keyword/length-heuristic"}, True)
    return Classification("simple", 0.65, {"reason": "default-short"}, True)


def _parse_label(value) -> tuple[str, float]:
    """Normalize the many plausible Jev answer shapes to (label, confidence)."""
    conf = 0.7
    if isinstance(value, dict):
        conf = float(value.get("confidence", value.get("score_confidence", 0.7)))
        if "score" in value and isinstance(value["score"], int):
            idx = max(0, min(2, value["score"]))
            return (["simple", "standard", "complex"][idx], conf)
        for key in ("label", "answer", "value", "tier", "classification"):
            if key in value:
                return (_norm_label(value[key]), conf)
        return ("standard", 0.4)
    if isinstance(value, int):
        idx = max(0, min(2, value))
        return (["simple", "standard", "complex"][idx], conf)
    return (_norm_label(str(value)), conf)


def _norm_label(s) -> str:
    s = str(s).lower()
    if "complex" in s or "hard" in s or s.strip() == "2":
        return "complex"
    if "standard" in s or "medium" in s or "moderate" in s or s.strip() == "1":
        return "standard"
    return "simple"


def classify(prompt: str, config_path: str = "router_config.yaml", timeout: float = 15.0) -> Classification:
    url = os.getenv("JEV_API_URL", "").strip()
    if not url:
        return mock_classify(prompt)
    questions = load_questions(config_path)
    headers = {}
    if os.getenv("JEV_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['JEV_API_KEY']}"
    try:
        r = httpx.post(url, json={"state": prompt, "questions": questions}, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        # Unwrap common envelopes
        payload = data.get("answers", data.get("results", data))
        # Prefer 'complexity' question, else first question
        if isinstance(payload, dict) and "complexity" in payload:
            label, conf = _parse_label(payload["complexity"])
        elif isinstance(payload, dict) and payload:
            label, conf = _parse_label(next(iter(payload.values())))
        else:
            label, conf = "standard", 0.4
        return Classification(label, conf, {"response": data})
    except Exception as e:  # network/API error -> safe fallback, flagged
        m = mock_classify(prompt)
        m.raw = {"error": str(e), "fallback_heuristic": m.label}
        return m
