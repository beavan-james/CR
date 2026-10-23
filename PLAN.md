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

### M0 — Laya client (done 2026-10-09)
- `classifier.py`: `LayaClient` (HTTP to laya-serve via `LAYA_URL`, Bearer `LAYA_API_KEY`, model `typed-decisions`), overflow detection, keyword heuristic as explicit baseline. Jev/TypeSafe removed.
- `router.py`: risk dial + overflow → flagship.
- **Remaining (user):** install `laya[serve]` on the A1, run it on 127.0.0.1, open an SSH tunnel. See README "Running Laya on a server".
- **Done when:** `cheapshot classify` hits Laya for 10 prompts; raw responses saved as fixtures (`--save-fixture`); per-call latency recorded.

### M1 — Dataset (done 2026-10-09)
- `evals/build_v1.py` → `evals/datasets/v1.jsonl` (seed 7, deterministic; file gitignored because it includes CC BY-SA Dolly text, so rebuild from `evals/raw/`):
  - train 1,500 (oasst2 900, Dolly 360 stratified by category, Arena-Hard v2 English 240; ≤6,000 chars)
  - tune 105 / test 105 (each: oasst2 50, Dolly 20, Arena-Hard 30, adversarial 5). Test grew from 50 to ~100 for tighter CIs.
  - Optional `evals/datasets/own.jsonl` merges your own prompts into tune/test.
- Original M1 notes:
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

## M0 findings: zero-shot Laya (2026-10-09, 9 seed prompts, A1 with 2 cores)
- Server: laya-serve on the A1 (127.0.0.1, systemd, Nice=10, MemoryMax=4G), reached over an SSH tunnel. Checkpoint confirmed `convaiinnovations/laya/typed-decisions`; `usage.truncated` present.
- **Variant A (Score):** expected level orders all 8 non-adversarial prompts correctly (cheap 0.70–0.71, balanced 0.81–1.13, flagship 1.25–1.39), but probabilities are nearly flat. Even temperature-sharpened, P(needs more than cheap) stays ~0.5 for trivial prompts, so the risk dial never picks Haiku. Weak, correctly ordered signal.
- **Variant C (Choice):** no signal (~0.35/0.42/0.23 regardless of prompt) and order-sensitive.
- **Adversarial:** "Trivial question:" pulled a twin-primes proof toward cheap (0.40).
- **Latency:** median 2.66 s, p90 3.08 s per call. Fine for benchmarks; adds noticeable delay in interactive `run`.
- **Conclusion:** matches the Laya docs (base checkpoints near chance zero-shot; fine-tuning is where the value is). M6 moves ahead of M5. Fine-tune on A's Score question (ordinal, already ordered). C stays as a zero-shot baseline only.

## Fine-tuning notes (from Laya docs)
- `laya-train --data train.jsonl --base typed-decisions --out ./ckpt`: JSONL rows `{state, questions, expected}` (hard label; score levels may be fractional) or `gold` (probabilities per option). Fits calibration temperatures on a held-out slice (≥10 items) automatically.
- Small datasets collapse: ~1,100 items gave too few optimizer updates in the docs' tests; repeat runs across seeds before judging a result. Target 1,000+ labeled prompts.
- Hardware: full fine-tune needs a 16 GB GPU (Kaggle free T4). Not the M3/8 GB Mac (too little memory) and not the 2-core A1 (too slow). `--freeze-encoder` fits smaller devices but gains much less.
- **Labels (decided: hybrid):** judge-based gold where a train prompt has answers + judgments (each judge order gives an oracle tier; gold averages them), else Opus teacher probability over tiers. Test split always judge-labeled.
- Built: `bench teacher` (Opus, 1 call/prompt, train split by default), `bench export-train` (laya-train JSONL with `gold` on variant A's Score levels). New `train` split; the benchmark report never scores it.
- Candidate prompt sources (licenses checked 2026-10-09): OpenAssistant oasst2 initial prompts (Apache-2.0, 15.6 MB), Dolly-15k (CC BY-SA 3.0, 13.1 MB), Arena-Hard v2.0 questions (repo Apache-2.0, 0.96 MB, 500 hard + 250 creative). WildChat-1M (ODC-BY) is an optional later addition; it is large and needs toxicity filtering.

## Subscription mode (added 2026-10-08)
- `--family claude-code` uses a Claude Pro/Max subscription via the official `claude -p` CLI (tools off, neutral system prompt, empty cwd, API keys stripped from its env). Works for `run` and every `bench` step, including the judge.
- Savings are reported on an API-equivalent cost basis (usage-limit proxy), labeled in the report.
- Expect the full matrix + judge to span several usage-limit windows on Pro; the cache makes resuming free.

## Next runs (rough estimates on the API, not quotes)
1. `bench teacher` on 1,500 train prompts: ~1,500 Opus calls, roughly $10–30.
2. `bench answers` + `bench judge` on tune+test (210 prompts): ~630 answers + ~840 Opus judge calls, roughly $20–50.
3. Optional judge labels for a few hundred train prompts (hybrid): roughly $0.10–0.20 per prompt.
4. `bench classify` with zero-shot Laya on tune+test (free, ~30–40 min on the A1), then `bench report` = zero-shot baseline.
5. `bench export-train` → `laya-train --dry-run` on the A1 → fine-tune on Kaggle T4 → serve the checkpoint via `LAYA_EXTRA_MODELS` → re-run `bench classify` + `bench report`.
On a Pro/Max subscription (`-f claude-code`) these are ~3,000 calls with no per-token bill, spread over several usage-limit windows.

## Status (2026-10-08)
- 41 tests passing. Whole pipeline runs offline with `-f mock`.
- Blocked on: Laya server (M0 live check), ANTHROPIC_API_KEY (first paid run), M1 dataset.

## Open items
- Optional cost lever: Message Batches API (50% off) for `bench answers` / `bench judge`; results arrive async.
- Which public prompt set (pending license check).
- A1 access method: SSH tunnel or Tailscale.

## Status (2026-10-10)
- **Benchmark v1 on subscription (`runs/v1-sub`, local only):** 630/630 answers, 840/840 judgments, Laya classifications for A, C×3 and heuristic.
- **First report (test split, 105 prompts):** Haiku non-inferior to Opus on 34%, Sonnet 62%; oracle savings ceiling 27%. Zero-shot Laya A: 11% savings at 93% quality, no better than random routing at the same mix (10%/92%). C: almost everything to Opus. Heuristic: 51% savings at 48% quality.
- **Teacher labels:** 1,499/1,500 train (1 refused by Opus safeguards) + 105 tune. Teacher is optimistic vs judge (mean level 0.45 vs 1.19, corr 0.49).
- **Recalibration:** `export-train` maps teacher level → judge tier mix (fit on tune, 5 quantile bins, shrunk toward overall). Train labels now average 29% cheap / 28% balanced / 42% flagship.
- **laya-train dry run on A1:** 1,499 valid items, none skipped; 1,350 train after calibration slice. Default 4 epochs = 88 updates (too few) → use `--epochs 16` (~350).
- **Next:** fine-tune on Kaggle T4, two seeds; copy checkpoints to A1 via `LAYA_EXTRA_MODELS`; re-run `bench classify` + `bench report` vs zero-shot baseline.

## Status (2026-10-15)
- **Fine-tune 1 (teacher labels, `cheapshot-s0`, Kaggle T4, 16 epochs):** served on the A1 via `LAYA_EXTRA_MODELS`. Seeds 0 and 1 gave near-identical calibration-slice numbers (accuracy 0.18 → 0.86).
- **Held-out ranking improved, savings didn't:** test AUC for "Haiku fails" 0.47 → 0.70, "Sonnet fails" 0.53 → 0.69. Total savings at the 95% floor 1.7% (zero-shot 10.7%). Per-prompt savings 8.0% at 98.1% quality vs zero-shot 9.3% at 93.3%; vs random at the same mix the fine-tune keeps ~5 more quality points.
- **Why savings is noisy:** flagship cost is concentrated; on v1 test the top 5 prompts are 44% of cost and one prompt is 21%. Cost-aware routing with a prompt-only cost estimate (corr 0.35 with true cost) didn't help; with true costs the ceiling is ~45% savings at 93% quality. Not merged.
- **Changes:** per-prompt savings in the report (default ε objective); lazy judging for train prompts; `--subset`; `export-train --judge-weight` (default 2); teacher recalibration also fits on judged train prompts.
- **v2 dataset (`evals/build_v2.py`):** v1 unchanged + 150 long-answer eval prompts (90 Arena-Hard, 60 oasst2; 50 tune / 100 test) + 300 train prompts tagged for judge labels. Running on subscription in `runs/v2-sub` (`run.sh`); fine-tuned classifications in `runs/v2-sub-ft` (shares the answer/judge cache).
- **Next:** judge labels → export → fine-tune 2 → compare zero-shot / teacher-trained / judge-trained on the 255-prompt test split. Use real answer lengths from judged prompts to fit a cost estimator.

## Status (2026-10-19)
- **v2 run complete (`runs/v2-sub`):** 360 tune+test prompts fully answered and judged (test = 205). 298/300 judge-subset train prompts labeled (2 answers time out at 600s; they keep teacher labels).
- **Judge vs teacher on 298 train prompts:** mean level 1.29 vs 0.47, corr 0.40; judge mix 21% cheap / 30% balanced / 50% flagship. Recalibration now fit on 403 pairs.
- **Three routers on v2 test (per-prompt savings, ε on tune with 0.01 steps):**
  - zero-shot: 8.4% @ 91.7% (95% floor missed), AUC ~0.50; no better than random at the same mix.
  - **ft1 `cheapshot-s0` (teacher labels): 11.0% @ 95.6% (95% floor), 23.1% @ 87.8% (90% floor); +5–8 quality points vs random at the same mix; AUC 0.68.** Default router.
  - ft2 `cheapshot-v2` (teacher + 2× judge labels): 9.1% @ 96.1%, 20.0% @ 87.8%; statistically tied with ft1. More judge labels are not worth ~7 calls/prompt.
- Neither fine-tune routes anything to Haiku; total-bill savings stay 3–5% because long answers still go to Opus.
- **Changes:** ε grid now 0.00–1.00 in 0.01 steps; `epsilon_by_model` in variants (looked up by `LAYA_MODEL`); `variants/length_score.yaml` for the cost-signal experiment.
- **Next:** zero-shot "answer length" question vs real flagship cost on the 360 eval prompts → cost-aware rule combining tier risk and predicted length. If promising, Opus answers for the remaining ~1,200 train prompts as length labels and a two-question fine-tune. Target: ≥20% total-bill savings at 95% quality.

## Status (2026-10-19, later): interactive CLI
- **Length signal (zero-shot, scratch experiment):** cost-aware routing with Laya's answer-length score reached 15.2% total-bill savings at 92.2% quality (90% floor), vs 10.1% for the ε rule; the true-cost ceiling is 38–50%. Haiku vs Opus answer length correlate 0.90 (log tokens), so ~1,200 **Haiku** answers would be cheap length labels. Deferred.
- **`cheapshot` REPL:** Claude Code with per-turn routing. Each turn = `claude -p --model <routed> --session-id/--resume <uuid>` in the user's directory with tools on, so context carries across model switches (verified). Streams text and tool calls, prints per-turn cost vs an Opus estimate, logs to `~/.cheapshot/usage.jsonl`.
- **Quality target instead of ε:** per-router curves in `src/cheapshot/curves/` (`cheapshot bench curve`); a target picks the ε with the most per-prompt savings that met it on tune, and the test numbers are shown. `/quality`, `/curve`, `cheapshot curve`, `run --quality`.
- Short follow-ups (<160 chars) are routed with the previous request; Laya outage → flagship with a visible note.
- Permissions: `-p` mode can't prompt, so `acceptEdits` blocks Bash; denials are surfaced with a pointer to `permissions.allow`.
- **Next:** live routed REPL test with Laya up; then decide on the Haiku length labels + two-question fine-tune.
- **Haiku length labels (2026-10-19):** 1,498 of 1,500 train prompts answered by Haiku (2 failed). `runs/v2-sub/laya_train_len.jsonl`: 1,797 rows (judge ×2 + teacher) with both `complexity` and `answer_length` gold; length mass 39% short / 37% medium / 24% long. Next: Kaggle fine-tune on it, classify tune/test with both questions, then compare cost-aware routing vs the ε rule.
- **Not doing (for now):** Opus judge labels for the other 1,200 train prompts (~6,000 mostly-Opus calls; ft2 showed 300 judged prompts didn't move routing). If revisited, first test Sonnet-as-judge agreement on the 360 already-judged eval prompts.

## Status (2026-10-20): two-question router (cheapshot-v3)
- Fine-tune on `laya_train_len.jsonl` (complexity + Haiku-labeled answer length). Length level vs log Opus cost: corr 0.77 on test (prompt length: 0.25). Complexity head unchanged (threshold rule 11.5% per-prompt @ 95.6%).
- **Cost-aware rule** (`costaware.py`, curve `cost_aware__cheapshot-v3.json`) on v2 test, total-bill savings: 9.2% @ 97.1%, 24.8% @ 95.6%, 28.0% @ 93.7%, 32.0% @ 90.7% (threshold rule: 5.0% @ 95.6%, 7.7% @ 90.2%). At 95% the tune split ties λ 2.38× and 2.18× (test 9% vs 25%); the safer one is kept.
- **Agent sessions differ:** each turn re-sends ~38K+ tokens of context, so turn cost is mostly fixed and the length signal doesn't help short turns. The REPL defaults to `threshold`; `cost` is opt-in. Revisit with real `usage.jsonl` data.
- **Bug fixed:** Claude Code's `total_cost_usd` is cumulative over a resumed session; per-turn cost is now the delta (earlier REPL cost lines overstated later turns).
- Idea: switching models loses the prompt cache (the new model re-writes the whole context), so frequent up-switches cost extra; consider a stickiness term.

## Status (2026-10-20, later): renamed to CR
- Package, command and paths are now `cr` (`src/cr/`, `~/.cr/`, `CR_*`). Earlier sections keep the old names. The `cheapshot` command, `CHEAPSHOT_*` variables and `~/.cheapshot/` (used while `~/.cr/` doesn't exist) still work. Router checkpoints keep their `cheapshot-*` names on the Laya server.

## Status (2026-10-23): where the losses land
- Replayed the cost-aware curve on v2 test (cheapshot-v3, from cache). Haiku's loss rate vs Opus rises with prompt size: 41% / 65% / 80% / 94% by quarter of Opus cost. The cost rule sends the biggest prompts to Haiku, so its misses are big tasks: at λ 2.0 (93.7%, 28.0% saved), 12 of 13 losses are in the top cost quarter and 25.0 of the 28.0 points saved come from prompts judged worse. The #1 prompt by cost (adversarial "simple lookup: write a compiler", $1.16 on Opus) alone is 9.3 points.
- New metric, **weighted quality** (each prompt counted by its Opus cost): cost-aware 91.1% @ 97.1%, 77.5% @ 95.6%, 74.5% @ 93.7%; risk threshold ε 0.38: 96.4% @ 95.6%.
- Against random routing at the same mix: the threshold rule is +5.4 pts (95.6 vs 90.2); the cost rule is level on quality (97.1 vs 97.3) and saves 2.4× (9.2% vs 3.8%). The cost rule never picks Sonnet at these settings.
- Headline changed to 9.2% @ 97.1% (91.1% weighted) in the README, case study and video. Idea: make risk depend on predicted length, since loss rate climbs with size.
