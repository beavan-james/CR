"""Provider adapters (BYOK). Each exposes complete(model, prompt) -> Completion.

Anthropic goes through the official SDK with streaming. Effort is set explicitly
per model (MODEL_EFFORT) because defaults differ: Opus 5.5 and Haiku 5.5 default
to medium, Sonnet 5.5 to high. Thinking is adaptive (always on for Opus 5.5) and
its tokens are billed as output, so they're included in output_tokens.
No refusal fallback: in a benchmark it would silently swap which model answered.

The "claude-code" family uses a Claude Pro/Max subscription instead of an API
key, by running the official Claude Code CLI headless (`claude -p`) under the
user's own login. Cheapshot never touches the subscription's OAuth token. Tools,
MCP servers and session persistence are off, and the coding-agent system prompt is
replaced. Usage counts against the subscription's limits; there is no per-token bill.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass

import httpx

DEFAULT_MAX_TOKENS = 16000  # includes thinking tokens; hitting it is flagged, not hidden

MODEL_EFFORT = {
    "claude-haiku-5-5": "medium",
    "claude-sonnet-5-5": "high",
    "claude-opus-5-5": "medium",
}


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str
    stop_reason: str = "end_turn"

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "max_tokens"

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


def _estimate_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _mock_text(model: str, prompt: str) -> str:
    return f"[mock:{model}] Response to: {prompt[:220]}"


def complete(family: str, model: str, prompt: str, api_key: str = "", timeout: float = 600.0,
             system: str | None = None, max_tokens: int = DEFAULT_MAX_TOKENS, effort: str | None = None) -> Completion:
    family = family.lower()
    if family == "claude-code":
        return _claude_code(model, prompt, timeout, system, effort or MODEL_EFFORT.get(model))
    if family == "mock" or os.getenv("CHEAPSHOT_MOCK_LLM") == "1" or not api_key:
        # Offline-safe path: if no key provided we still return mock rather than crash.
        # CLI warns the user in this case.
        t0 = time.time()
        text = _mock_text(model, prompt)
        return Completion(text, _estimate_tokens(prompt), _estimate_tokens(text), time.time() - t0, model)
    if family == "anthropic":
        return _anthropic(model, prompt, api_key, timeout, system, max_tokens, effort or MODEL_EFFORT.get(model))
    if family == "openai":
        return _openai(model, prompt, api_key, timeout)
    if family == "gemini":
        return _gemini(model, prompt, api_key, timeout)
    raise ValueError(f"unknown family: {family}")


_anthropic_clients: dict[str, object] = {}
_anthropic_lock = threading.Lock()


def _anthropic_client(key: str):
    import anthropic

    with _anthropic_lock:
        if key not in _anthropic_clients:
            _anthropic_clients[key] = anthropic.Anthropic(api_key=key, max_retries=4)
        return _anthropic_clients[key]


def _anthropic(model: str, prompt: str, key: str, timeout: float, system: str | None,
               max_tokens: int, effort: str | None) -> Completion:
    client = _anthropic_client(key)
    kwargs: dict = {"model": model, "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}]}
    if system:
        kwargs["system"] = system
    if effort:
        kwargs["output_config"] = {"effort": effort}
    t0 = time.time()
    with client.with_options(timeout=timeout).messages.stream(**kwargs) as stream:
        msg = stream.get_final_message()
    text = "".join(b.text for b in msg.content if b.type == "text")
    return Completion(text, msg.usage.input_tokens, msg.usage.output_tokens, time.time() - t0,
                      model, msg.stop_reason or "")


class ClaudeCodeError(RuntimeError):
    pass


CLAUDE_CODE_SYSTEM = "You are Claude, a helpful AI assistant."
# Keys that would make the CLI bill an API account instead of the subscription.
_API_AUTH_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


def claude_code_available() -> bool:
    return shutil.which("claude") is not None


def claude_code_command(model: str, system: str | None, effort: str | None) -> list[str]:
    cmd = ["claude", "-p", "--output-format", "json", "--model", model,
           "--system-prompt", system or CLAUDE_CODE_SYSTEM,
           "--tools", "", "--strict-mcp-config", "--no-session-persistence"]
    if effort:
        cmd += ["--effort", effort]
    return cmd


def parse_claude_code_output(stdout: str, model: str, latency_s: float) -> Completion:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise ClaudeCodeError(f"claude -p returned non-JSON output: {stdout[:300]!r}") from e
    if data.get("is_error") or data.get("subtype", "success") != "success":
        raise ClaudeCodeError(f"claude -p failed ({data.get('subtype')}): {str(data.get('result', ''))[:300]}")
    u = data.get("usage") or {}
    # Cache reads/writes still consume the model; count them as input for API-equivalent cost.
    input_tokens = (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0) \
        + (u.get("cache_read_input_tokens") or 0)
    return Completion(data.get("result") or "", input_tokens, u.get("output_tokens") or 0, latency_s, model,
                      data.get("stop_reason") or "end_turn")


def _claude_code(model: str, prompt: str, timeout: float, system: str | None, effort: str | None) -> Completion:
    if not claude_code_available():
        raise ClaudeCodeError("`claude` CLI not found. Install Claude Code and log in with your Pro/Max account.")
    env = {k: v for k, v in os.environ.items() if k not in _API_AUTH_ENV}
    t0 = time.time()
    # Empty working dir: no project CLAUDE.md or files leak into the answer.
    with tempfile.TemporaryDirectory(prefix="cheapshot-cc-") as cwd:
        try:
            proc = subprocess.run(claude_code_command(model, system, effort), input=prompt, capture_output=True,
                                  text=True, timeout=timeout, cwd=cwd, env=env)
        except subprocess.TimeoutExpired as e:
            raise ClaudeCodeError(f"claude -p timed out after {timeout}s") from e
    if proc.returncode != 0 and not proc.stdout.strip():
        raise ClaudeCodeError(f"claude -p exited {proc.returncode}: {proc.stderr.strip()[:300]}")
    return parse_claude_code_output(proc.stdout, model, time.time() - t0)


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
