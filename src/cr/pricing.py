"""Pricing table ($ per 1M tokens). Update as vendors change prices."""
# Sources: vendor pricing pages; override via pricing.yaml if needed.
PRICING = {
    # Anthropic family (first-party API rates, checked 2026-10-08).
    # Haiku 5.5 rate applies to prompts up to 100K tokens ($0.50/$2.50 beyond).
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0},
    "claude-opus-5-5": {"input": 4.0, "output": 20.0},
    # OpenAI / Gemini: not verified, out of v1 scope (Anthropic only)
    # OpenAI family
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-5": {"input": 5.0, "output": 15.0},
    # Gemini family
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    # Mock (offline)
    "mock-cheap": {"input": 0.10, "output": 0.40},
    "mock-balanced": {"input": 1.0, "output": 4.0},
    "mock-flagship": {"input": 5.0, "output": 15.0},
}

# Tier -> model per provider family. User picks family on startup.
FAMILY_MODELS = {
    "anthropic": {
        "cheap": "claude-haiku-5-5",
        "balanced": "claude-sonnet-5-5",
        "flagship": "claude-opus-5-5",
    },
    "openai": {
        "cheap": "gpt-4o-mini",
        "balanced": "gpt-4o",
        "flagship": "gpt-5",
    },
    "gemini": {
        "cheap": "gemini-2.0-flash",
        "balanced": "gemini-2.5-flash",
        "flagship": "gemini-2.5-pro",
    },
    # Pro/Max subscription via the Claude Code CLI. Same models; costs are
    # API-equivalent (a proxy for usage-limit consumption, not a bill).
    "claude-code": {
        "cheap": "claude-haiku-5-5",
        "balanced": "claude-sonnet-5-5",
        "flagship": "claude-opus-5-5",
    },
    "mock": {
        "cheap": "mock-cheap",
        "balanced": "mock-balanced",
        "flagship": "mock-flagship",
    },
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model)
    if p is None:
        raise KeyError(f"no pricing for model {model!r}; add it to PRICING")
    return (input_tokens / 1_000_000) * p["input"] + (output_tokens / 1_000_000) * p["output"]

