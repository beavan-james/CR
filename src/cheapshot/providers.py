"""Provider adapters (BYOK). Each exposes complete(model, prompt) -> Completion."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str


def _estimate_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _mock_text(model: str, prompt: str) -> str:
    return f"[mock:{model}] Response to: {prompt[:220]}"


def complete(family: str, model: str, prompt: str, api_key: str = "", timeout: float = 60.0) -> Completion:
    family = family.lower()
    if family == "mock" or os.getenv("JEV_MOCK_LLM") == "1" or not api_key:
        # Offline-safe path: if no key provided we still return mock rather than crash.
        # CLI warns the user in this case.
        t0 = time.time()
        text = _mock_text(model, prompt)
        return Completion(text, _estimate_tokens(prompt), _estimate_tokens(text), time.time() - t0, model)
    if family == "anthropic":
        return _anthropic(model, prompt, api_key, timeout)
    if family == "openai":
        return _openai(model, prompt, api_key, timeout)
    if family == "gemini":
        return _gemini(model, prompt, api_key, timeout)
    raise ValueError(f"unknown family: {family}")


def _anthropic(model: str, prompt: str, key: str, timeout: float) -> Completion:
    t0 = time.time()
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": model, "max_tokens": 512, "messages": [{"role": "user", "content": prompt}]},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    usage = data.get("usage", {})
    return Completion(text, usage.get("input_tokens", _estimate_tokens(prompt)),
                      usage.get("output_tokens", _estimate_tokens(text)), time.time() - t0, model)


def _openai(model: str, prompt: str, key: str, timeout: float) -> Completion:
    t0 = time.time()
    r = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "max_tokens": 512, "messages": [{"role": "user", "content": prompt}]},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    text = choice.get("message", {}).get("content", "")
    usage = data.get("usage", {})
    return Completion(text, usage.get("prompt_tokens", _estimate_tokens(prompt)),
                      usage.get("completion_tokens", _estimate_tokens(text)), time.time() - t0, model)


def _gemini(model: str, prompt: str, key: str, timeout: float) -> Completion:
    t0 = time.time()
    r = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": 512}},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
    text = "".join(p.get("text", "") for p in parts)
    usage = data.get("usageMetadata", {})
    return Completion(text, usage.get("promptTokenCount", _estimate_tokens(prompt)),
                      usage.get("candidatesTokenCount", _estimate_tokens(text)), time.time() - t0, model)
