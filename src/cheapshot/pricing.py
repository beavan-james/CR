"""Pricing table ($ per 1M tokens). Update as vendors change prices."""
# Sources: vendor pricing pages; override via pricing.yaml if needed.
PRICING = {
    # Anthropic family
    "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.0},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-opus-4-1": {"input": 15.0, "output": 75.0},
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
        "cheap": "claude-3-5-haiku-latest",
        "balanced": "claude-sonnet-4-5",
        "flagship": "claude-opus-4-1",
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
    "mock": {
        "cheap": "mock-cheap",
        "balanced": "mock-balanced",
        "flagship": "mock-flagship",
    },
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model, {"input": 1.0, "output": 4.0})
    return (input_tokens / 1_000_000) * p["input"] + (output_tokens / 1_000_000) * p["output"]
