# Cheapshot v1 — Implementation Plan

**Goal:** prove Cheapshot's routing schema, run on a self-hosted Laya decision model, routes Anthropic prompts to cheaper models without losing quality.
Headline result shape:

> At ≥95% non-inferior to Opus 5.5, variant X saves Y% (95% CI). Router cost: $0 per call (self-hosted).

## Locked decisions

| Area | Decision |
|---|---|
| Scope | Benchmark first. Anthropic only: cheap = Haiku 5.5, balanced = Sonnet 5.5, flagship = Opus 5.5 |
| Router model | **Laya** (`convaiinnovations/laya-typed-decisions`, Apache 2.0) self-hosted on an Oracle Ampere A1. Replaces Jev entirely; TypeSafe SDK removed |
| Under test | Two schemas. **A**: single 3-level complexity Score (control). **C**: Choice over tiers with capabilities described, run in 3 rotated option orders (checks option-order bias) |
| Routing rule | Risk dial: cheapest tier with P(needs higher tier) ≤ ε. Sweep ε → one curve per variant |
| Quality | Pairwise judge (Opus 5.5), both A/B orders; disagreement = tie. Non-inferior = win or tie. Same-family judge bias makes savings err low |
| Benchmark design | Full matrix: every prompt answered by all 3 tiers + judged once, cached. Policies and baselines scored offline by replay |
| Dataset | ~150 prompts: ~100 public set + ~50 own, category-tagged, plus ~10 adversarial ("trivial question:" on hard tasks) |
| Headline | Savings at configurable quality floor (95%), bootstrap CIs; full curves below |
| Later | Fine-tune Laya on oracle labels to close the domain gap (M6) |

## Laya facts that shape the plan (from model card)

- ~0.4B-parameter ModernBERT-large encoder, non-generative. CPU inference is fine on the A1.
- Same question types as Jev: `noul`, `choice` (keep < ~20 options), `score`, with probabilities.
- **Context: 1024 tokens.** Prompts that don't fit are **not trimmed**: they go straight to the flagship (Opus 5.5, medium effort). Detected from `usage.truncated` in laya-serve's response, plus a client-side char cap (`LAYA_MAX_CHARS`, default 6000) that skips the call for prompts that can't possibly fit.
- **Domain gap:** trained on 4 workflows (invoices, customer service, security incidents, agent traces). The card expects base-checkpoint performance or worse elsewhere; base accuracy is 0.36 vs 0.32 random. Prompt-difficulty routing is outside its training, so routing quality is unproven until M4.
- **Over-confident probabilities** (ECE 0.213); card advises refitting temperatures on your own held-out data. The risk dial depends on calibration → calibration step in M4.
- Ships `laya-serve` (`pip install "laya[serve]"`): a Jev-compatible `POST /v1/systemone` server with Bearer auth (`LAYA_API_KEY`). No custom service needed. Checkpoint name `typed-decisions`.
- Known gotcha: set `USE_TF=0` if `laya.load()` hangs.

## Defaults (change if needed)

- **Hold-out split:** 100 tune / 50 test. Tune wording, ε and calibration on tune split only; report test split.
- **Baselines:** always-flagship, always-cheap, always-balanced, random at same tier mix, length/keyword heuristic, oracle (cheapest non-inferior tier per prompt = savings upper bound).
- **`max_tokens`:** 16000 (includes thinking tokens), streamed. Truncated/refused answers are judged as-is and counted in the report.
- **Effort per tier:** Haiku 5.5 medium, Sonnet 5.5 high (its default), Opus 5.5 medium. Judge: Opus 5.5 medium.
- **No refusal fallback** on Claude calls: it would silently change which model answered a tier.
- **No silent fallback:** router errors stop the benchmark. Heuristic only via explicit `--classifier heuristic`.
- **Network:** `laya-serve` binds to 127.0.0.1 on the A1 (`LAYA_HOST`; its default is 0.0.0.0); Cheapshot reaches it over an SSH tunnel or Tailscale. No public port.

## Milestones

### M0 — Laya client (code done; live check pending server setup)
- `classifier.py`: `LayaClient` (HTTP to laya-serve via `LAYA_URL`, Bearer `LAYA_API_KEY`, model `typed-decisions`), overflow detection, keyword heuristic as explicit baseline. Jev/TypeSafe removed.
- `router.py`: risk dial + overflow → flagship.
- **Remaining (user):** install `laya[serve]` on the A1, run it on 127.0.0.1, open an SSH tunnel. See README "Running Laya on a server".
- **Done when:** `cheapshot classify` hits Laya for 10 prompts; raw responses saved as fixtures (`--save-fixture`); per-call latency recorded.

### M1 — Dataset
- `evals/datasets/v1.jsonl`: `{id, prompt, category, source, split}`.
- Confirm public set license (MT-Bench / Arena-Hard-style candidates).
- User supplies ~50 own prompts.

### M2 — Matrix generation (code done, mock-verified)
- `cheapshot bench matrix`: each prompt × 3 tiers + Laya (A + 3 C rotations, via `predict_batch`).
- Disk cache keyed by hash(model, prompt, params); resumable.
- ~450 Claude answers; Laya calls are free.
- `providers.py` uses the official `anthropic` SDK with streaming and explicit per-tier effort; thinking tokens are billed as output and counted.
- Commands: `bench answers` (paid, asks first), `bench classify` (Laya or heuristic). Cache in `runs/<name>/*.jsonl`.

### M3 — Judge (code done, mock-verified)
- `cheapshot bench judge`: cheap-vs-flagship and balanced-vs-flagship, both orders (~600 Opus calls).
- Fixed rubric; parsed verdict `A` / `B` / `tie`.

### M4 — Calibrate, replay, report (code done, mock-verified)
- Oracle tier per prompt (from judge results) = label. Fit a temperature on Laya's tier distribution using tune-split labels (minimize NLL); report ECE before/after.
- `cheapshot bench report`: every policy × ε (raw and calibrated) and every baseline, scored from cache.
- Markdown report: headline, curves, per-category breakdown, C order sensitivity, adversarial results, Laya vs heuristic vs random.
- **Decision gate:** if Laya doesn't beat random-at-same-mix and the heuristic, skip M5 wording tuning and go straight to M6.

### M5 — Tune wording
- Iterate A and C wording on tune split only; one final run on test split.

### M6 — Fine-tune Laya
- Training data: (prompt, oracle tier) pairs. 100 tune prompts is too few, so expand the labeled set (e.g. 500+ prompts through M2–M3; Claude cost scales linearly) and keep the 50 test prompts untouched.
- Fine-tune from `laya-typed-decisions` with the variant's question; compare against the base checkpoint on the test split.
- Training is not done on the A1 (CPU too slow). Use a free GPU (e.g. Kaggle 2×T4, as the card's authors did), then copy weights to the A1.

### Tests
- Risk-dial unit tests.
- Tier-distribution parsing against Laya fixtures from M0.
- LayaClient request shape, auth, overflow and error handling (httpx MockTransport).
- End-to-end mock pipeline: answers → classify → judge → replay → report.
- Replay and calibration math on a small synthetic matrix.

## Subscription mode (added 2026-10-08)
- `--family claude-code` uses a Claude Pro/Max subscription via the official `claude -p` CLI (tools off, neutral system prompt, empty cwd, API keys stripped from its env). Works for `run` and every `bench` step, including the judge.
- Savings are reported on an API-equivalent cost basis (usage-limit proxy), labeled in the report.
- Expect the full matrix + judge to span several usage-limit windows on Pro; the cache makes resuming free.

## Status (2026-10-08)
- 35 tests passing. Whole pipeline runs offline with `-f mock`.
- Blocked on: Laya server (M0 live check), ANTHROPIC_API_KEY (first paid run), M1 dataset.

## Open items
- Optional cost lever: Message Batches API (50% off) for `bench answers` / `bench judge`; results arrive async.
- Which public prompt set (pending license check).
- A1 access method: SSH tunnel or Tailscale.
