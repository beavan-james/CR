# Cheapshot — Jev Agent Harness

Classify the **initial prompt** with Jev, route to the cheapest suitable model (BYOK), show **cost optimization %**.

## Quickstart

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in keys you use

# Choose family on startup (or pass --family); key via --api-key, env, or prompt
cheapshot run "The export button crashes the settings page in Safari..." --family anthropic --dry-run
cheapshot run "Write a debounce function in TS" --family openai
cheapshot models
cheapshot classify "What is the capital of France?"

# Benchmark (cost optimization % vs always-flagship)
cheapshot benchmark --family mock --dataset evals/datasets/seed.jsonl --out report.md
```
(`jev` still works as an alias for the CLI.)

## How it works

```
prompt -> POST {JEV_API_URL} {"state": prompt, "questions": {...router_config.yaml...}}
       -> Jev label (simple|standard|complex) + confidence
       -> router (tier_map + confidence fallback) -> model for chosen family
       -> complete(model, prompt) with YOUR api key -> answer + tokens/$/latency
```

- **Jev API shape** supported per your spec: flexible response parsing (`answers`/`results`/bare), score-int or label-string.
- **No `JEV_API_URL`?** → deterministic mock heuristics so CLI/benchmarks work offline (flagged `(mock)`).
- **BYOK:** keys never stored; resolution order `--api-key` > env (`ANTHROPIC_API_KEY`/`OPENAI_API_KEY`/`GEMINI_API_KEY`) > interactive prompt. Blank = mock mode, no charges.
- **Benchmark:** routed cost vs always-flagship baseline on same tokens → `savings %`. Tier distribution + optional `ideal_tier` accuracy (see `evals/datasets/seed.jsonl`).

## Config

- `router_config.yaml` — Jev `questions`, `tier_map`, `confidence_threshold`, `fallback_tier`
- `src/jev_harness/pricing.py` — `FAMILY_MODELS` + `PRICING` ($/1M). Edit as vendors change.
