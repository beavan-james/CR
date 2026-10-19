# Cheapshot

A cost-aware LLM router. A self-hosted [Laya](https://huggingface.co/convaiinnovations/laya-typed-decisions) decision model reads each prompt and estimates which model tier it needs. Cheapshot then sends it to the cheapest tier that's likely good enough (BYOK) and benchmarks the savings against always using the flagship.

| tier | model (Anthropic) | effort |
|---|---|---|
| cheap | `claude-haiku-5-5` | medium |
| balanced | `claude-sonnet-5-5` | high |
| flagship | `claude-opus-5-5` | medium |

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env   # ANTHROPIC_API_KEY, LAYA_URL, LAYA_API_KEY

cheapshot classify "What is the capital of France?"                 # Laya -> tier
cheapshot classify "Write a debounce in TS" --classifier heuristic  # no Laya needed
cheapshot run "Explain this Safari crash..." --family anthropic -e 0.1
cheapshot models
```

## Interactive CLI: Claude Code with per-turn routing

Run `cheapshot` in any project directory. It works like Claude Code: it reads and edits files and runs commands. The difference is that Laya picks Haiku, Sonnet or Opus for **every turn**. Each turn runs `claude -p --model <routed> --resume <session>` under your Pro/Max login, so the conversation keeps its context when the model switches.

```bash
cd ~/my-project
cheapshot                      # quality target 95% (default)
cheapshot -q 0.90              # trade some quality for more savings
cheapshot --mode plan          # Claude Code permission mode (acceptEdits by default)
```

```
❯ There is a bug in calc.py. Fix it.
→ Sonnet 5.5 · P(needs more)=0.31 ≤ ε 0.40
  ⏺ Read(calc.py)
  ⏺ Edit(calc.py)
Fixed: `subtract` returned a + b ...
Sonnet 5.5 · $0.0412 (Opus est. $0.0687) · 9s · session $0.041, 40% saved
```

| command | what it does |
|---|---|
| `/quality 0.9` (or `90%`) | set the quality target and show the trade-off table |
| `/curve` | savings vs quality for the current router (held-out benchmark) |
| `/model auto\|haiku\|sonnet\|opus` | pin a model, or go back to routing |
| `/why` | how the last turn was routed |
| `/cost` | session spend vs an always-Opus estimate |
| `/mode acceptEdits\|plan\|default\|bypassPermissions` | Claude Code permission mode |
| `/new`, `/exit` | fresh conversation, quit |

- **Quality target, not ε.** "Quality" means the share of prompts answered as well as Opus, as scored by the Opus judge on the benchmark. Each router model has a stored curve (`src/cheapshot/curves/`). A target maps to the ε with the most per-prompt savings that met it on the tune split. The banner shows what that ε achieved on the held-out test split. `cheapshot curve` prints the whole table, and `cheapshot bench curve -r <run>` rebuilds it after a new benchmark.
- **Follow-ups** shorter than 160 characters ("yes, do it") are routed together with the previous request, so a short reply to a hard task doesn't drop to Haiku.
- **Thanks and sign-offs** ("ok thanks", "great, thank you") skip the router and go to Haiku, since they ask for no work. "ok", "yes" and "go ahead" are still routed, because after a plan they mean "do it". Turn this off with `CHEAPSHOT_TRIVIAL=0`.
- **If Laya is unreachable**, the turn goes to Opus and the routing line says so. An outage costs money, never quality.
- **Permissions:** headless Claude Code can't stop to ask for approval. With `acceptEdits`, file edits go through but shell commands are blocked, and Cheapshot tells you which tools were blocked. Allow the commands you trust in your project's `.claude/settings.json` (`permissions.allow`, e.g. `"Bash(pytest:*)"`), or use `/mode bypassPermissions` in a sandbox.
- **Usage log:** every turn is appended to `~/.cheapshot/usage.jsonl` (override with `CHEAPSHOT_LOG`). It records model, tier, routing probabilities, cost and the Opus estimate, but not prompt text.
- **Config:** put `LAYA_URL`, `LAYA_API_KEY`, `LAYA_MODEL` and optionally `CHEAPSHOT_QUALITY` in `~/.cheapshot/config.env` so `cheapshot` works from any directory.
- `cheapshot run "..." --quality 0.9` and `cheapshot classify --quality 0.9` accept the same target.

## Using a Claude Pro/Max subscription instead of an API key

Use `--family claude-code`. Cheapshot then runs the official Claude Code CLI headless (`claude -p`) under your own login. It never reads or reuses the subscription's OAuth token itself.

```bash
claude auth status                       # should show authMethod "claude.ai"
cheapshot run "Explain this Safari crash..." --family claude-code
cheapshot bench answers -r runs/v1-sub -d evals/datasets/v1.jsonl -f claude-code -w 2
```

- Tools, MCP servers and session saving are off, and Claude Code's coding-agent system prompt is replaced with a neutral one. Each call runs in an empty temp directory.
- `ANTHROPIC_API_KEY` is stripped from the CLI's environment so it can't silently bill your API account.
- No per-token bill: reported costs are **API-equivalent**, which serves as a proxy for how much of your usage limit each tier consumes. The report labels this.
- A full benchmark (about 450 answers plus about 600 Opus judge calls) is heavy for a subscription and will likely span several usage-limit windows on Pro. Calls that fail at a limit are cached as missing; re-run the same command later to resume. Use fewer workers (`-w 2`).
- Each call carries ~600 tokens of CLI overhead, so per-call numbers are higher than the API path. API and subscription answers are cached separately (the family is part of the cache key); separate run directories keep their reports apart.

## How routing works

```
prompt -> Laya (laya-serve /v1/systemone, question from variants/*.yaml)
       -> probability per tier {cheap, balanced, flagship}
       -> risk dial: cheapest tier with P(needs a higher tier) <= epsilon
       -> routed Claude model, with YOUR api key
```

- **Overflow:** Laya reads at most ~1024 tokens. If a prompt doesn't fit (laya-serve reports `usage.truncated`, or the prompt is over `LAYA_MAX_CHARS`), it goes straight to the flagship instead of being routed on a partial read.
- **No silent fallback:** if Laya is unreachable, `run` and `classify` fail, and the interactive CLI visibly sends the turn to the flagship. The keyword heuristic only runs with `--classifier heuristic`.
- **Variants:** `variants/a_score.yaml` (complexity Score, the control) and `variants/c_tier_choice.yaml` (pick a tier directly).

## Benchmark

Every prompt is answered by all three tiers once. An Opus judge compares the cheap and balanced answers against the flagship's, in both orders. Everything is cached under `runs/<name>/`, so any routing policy (variant × ε × calibration) is scored offline without new Claude calls.

```bash
cheapshot bench answers  -r runs/v1 -d evals/datasets/v1.jsonl -f anthropic   # paid; asks first
cheapshot bench classify -r runs/v1 -d evals/datasets/v1.jsonl                  # Laya, free
cheapshot bench classify -r runs/v1 -d evals/datasets/v1.jsonl -b heuristic     # baseline
cheapshot bench judge    -r runs/v1 -d evals/datasets/v1.jsonl -f anthropic    # paid; asks first
cheapshot bench report   -r runs/v1 -d evals/datasets/v1.jsonl -f anthropic --floor 0.95
```

Re-running any step resumes where it stopped. Use `-f mock` (the default) to run the whole pipeline offline.

Dataset rows: `{"id", "prompt", "category", "source", "split": "train"|"tune"|"test"}`. `train` is fine-tuning data only and never scored; epsilon and calibration are chosen on `tune`; headline numbers come from `test`.

## Fine-tuning data for Laya

Zero-shot Laya barely separates easy from hard prompts (see `PLAN.md`), so it gets fine-tuned. Labels are hybrid: judge-based where the benchmark has run on a prompt, otherwise an Opus "teacher" probability over tiers (one call per prompt).

```bash
cheapshot bench teacher      -r runs/ft -d evals/datasets/v1.jsonl -f anthropic   # paid; train split only
cheapshot bench export-train -r runs/ft -d evals/datasets/v1.jsonl -f anthropic   # -> runs/ft/laya_train.jsonl
```

Check the file with `laya-train --data laya_train.jsonl --base typed-decisions --dry-run` (works on the A1, no GPU needed). Train on a 16 GB GPU (Kaggle T4), then load the checkpoint on the server via `LAYA_EXTRA_MODELS`.

Judge labels for part of the train split (`evals/build_v2.py` tags 300 prompts `"subset": "judge"`):

```bash
cheapshot bench answers -r runs/v2 -d evals/datasets/v2.jsonl -f claude-code --split train --subset judge
cheapshot bench judge   -r runs/v2 -d evals/datasets/v2.jsonl -f claude-code --split train --subset judge
cheapshot bench export-train -r runs/v2 -d evals/datasets/v2.jsonl -f claude-code --judge-weight 2
```

Train prompts are judged lazily: in each judge order, Sonnet is judged only if Haiku lost, since a Haiku pass already fixes the label. `--judge-weight` writes each judge-labeled row that many times. Judge-labeled train prompts also join the tune split in fitting the teacher recalibration.

The report shows **savings per prompt** (each prompt weighted equally) next to total savings, and picks ε by per-prompt savings by default (`--objective total` to switch). A few long answers dominate total cost, so total savings swings with a handful of routing decisions.

## Running Laya on a server

`laya-serve` (from the `laya` package) already serves the `/v1/systemone` protocol. On the server (e.g. an Oracle Ampere A1):

```bash
python3 -m venv ~/laya && ~/laya/bin/pip install "laya[serve]"
LAYA_HOST=127.0.0.1 LAYA_PORT=8000 LAYA_MODELS=typed-decisions LAYA_API_KEY=<secret> \
  LAYA_THREADS=4 USE_TF=0 ~/laya/bin/laya-serve
```

- Keep `LAYA_HOST=127.0.0.1`. The default is `0.0.0.0` (all interfaces).
- Reach it from your machine with an SSH tunnel, then set `LAYA_URL=http://localhost:8000`:

```bash
ssh -N -L 8000:127.0.0.1:8000 <user>@<server>
```

- Don't set `LAYA_JEV_STRICT`; it strips the `usage.truncated` field that overflow detection reads.

See `PLAN.md` for the v1 plan.
