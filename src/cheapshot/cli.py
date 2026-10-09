"""Typer + Rich CLI. Provider family chosen on startup (flag or prompt). BYOK."""
from __future__ import annotations

import getpass
import json
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.prompt import Prompt
from rich.table import Table

from . import classifier, providers, router, tracking
from .bench import judge as bench_judge
from .bench import matrix as bench_matrix
from .bench import replay as bench_replay
from .bench import report as bench_report
from .bench.dataset import load_dataset
from .bench.store import Store
from .pricing import FAMILY_MODELS, PRICING, cost_usd

load_dotenv()
app = typer.Typer(add_completion=False, help="Cheapshot — a Laya decision model routes each prompt to the cheapest capable model.")
bench = typer.Typer(add_completion=False, help="Benchmark: cached answers + judge, replayed offline for any routing policy.")
app.add_typer(bench, name="bench")
console = Console()

FAMILIES = ["anthropic", "claude-code", "openai", "gemini", "mock"]
ENV_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "claude-code": "", "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY", "mock": ""}
KEYLESS = ("mock", "claude-code")  # claude-code uses your Pro/Max login via the `claude` CLI


def resolve_family(family: str | None) -> str:
    if family and family.lower() in FAMILIES:
        return family.lower()
    return Prompt.ask("Choose model family", choices=FAMILIES, default="anthropic")


def resolve_key(family: str, api_key: str | None) -> str:
    if api_key:
        return api_key
    if family in KEYLESS:
        return ""
    env = os.getenv(ENV_KEYS[family], "")
    if env:
        return env
    console.print(f"[yellow]No {ENV_KEYS[family]} found — enter your key (BYOK, not stored).[/yellow]")
    try:
        return getpass.getpass(f"{family} API key (blank=mock mode): ")
    except Exception:
        return ""


VARIANT_OPT = typer.Option("variants/a_score.yaml", "--variant", "-v", help="Router variant YAML")
EPS_OPT = typer.Option(None, "--epsilon", "-e", help="Risk dial in [0,1]; default from variant")
CLASSIFIER_OPT = typer.Option("laya", "--classifier", help="laya|heuristic")


def classify_prompt(prompt: str, variant_path: str, which: str) -> tuple[classifier.Classification, dict | None]:
    if which == "heuristic":
        return classifier.heuristic_classify(prompt), None
    if which != "laya":
        raise typer.BadParameter("classifier must be laya|heuristic")
    variant = classifier.load_variant(variant_path)
    try:
        client = classifier.LayaClient()
        try:
            return client.classify(prompt, variant), variant
        finally:
            client.close()
    except classifier.RouterError as e:
        console.print(f"[red]{e}[/red]\n[dim]Set LAYA_URL (and LAYA_API_KEY), or pass --classifier heuristic.[/dim]")
        raise typer.Exit(1)


def effective_epsilon(epsilon: float | None, variant: dict | None) -> float:
    if epsilon is not None:
        return epsilon
    return float((variant or {}).get("epsilon", 0.2))


def _routing_text(c: classifier.Classification, rt: router.Route) -> str:
    return (f"router={c.router_model or 'laya'} conf={c.confidence:.2f}\n"
            f"route: {rt.tier} -> [bold]{rt.model}[/bold]\n{rt.reason}")


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Initial prompt to classify + answer"),
    family: str | None = typer.Option(None, "--family", "-f", help="anthropic|claude-code|openai|gemini|mock"),
    api_key: str | None = typer.Option(None, "--api-key", help="BYOK key (else env/prompt)"),
    variant: str = VARIANT_OPT,
    epsilon: float | None = EPS_OPT,
    which: str = CLASSIFIER_OPT,
    dry_run: bool = typer.Option(False, "--dry-run", help="Classify + route only, no generation call"),
    explain: bool = typer.Option(True, "--explain/--no-explain", help="Show routing panel"),
):
    """Classify the prompt, route with the risk dial, call the routed model."""
    fam = resolve_family(family)
    c, v = classify_prompt(prompt, variant, which)
    rt = router.route(c, effective_epsilon(epsilon, v), fam)
    if dry_run:
        console.print(Panel(_routing_text(c, rt), title=f"dry-run ({fam})"))
        return
    key = resolve_key(fam, api_key)
    eff_fam = fam if key or fam in KEYLESS else "mock"
    if fam not in KEYLESS and not key:
        console.print("[yellow]No key given — running in mock mode (no charges).[/yellow]")
    model = FAMILY_MODELS[eff_fam][rt.tier]
    comp = providers.complete(eff_fam, model, prompt, api_key=key)
    cost = cost_usd(model, comp.input_tokens, comp.output_tokens)
    tracking.log({"mode": "run", "family": fam, "variant": (v or {}).get("name", "heuristic"),
                  "epsilon": rt.epsilon, "dist": c.dist, "overflow": c.overflow, "tier": rt.tier,
                  "model": model, "cost": cost, "mocked": eff_fam == "mock"})
    if explain:
        console.print(Panel(_routing_text(c, rt) + f"\ntokens {comp.input_tokens} in / {comp.output_tokens} out | "
                            f"${cost:.6f} | {comp.latency_s:.2f}s | stop={comp.stop_reason}", title="routing"))
    console.print(Panel(comp.text, title=model))


@app.command()
def classify(
    prompt: str = typer.Argument(...),
    variant: str = VARIANT_OPT,
    epsilon: float | None = EPS_OPT,
    which: str = CLASSIFIER_OPT,
    save_fixture: Path | None = typer.Option(None, "--save-fixture", help="Write raw router response JSON here"),
):
    """Classify only (no generation): tier distribution, routed tier, raw response."""
    c, v = classify_prompt(prompt, variant, which)
    rt = router.route(c, effective_epsilon(epsilon, v))
    console.print(Panel(_routing_text(c, rt) + f"\nraw={json.dumps(c.raw)[:500]}", title="classification"))
    if save_fixture:
        save_fixture.parent.mkdir(parents=True, exist_ok=True)
        save_fixture.write_text(json.dumps({"prompt": prompt, "variant": v, "response": c.raw}, indent=2))
        console.print(f"Saved {save_fixture}")


@app.command()
def models():
    """Show tier -> model map + pricing."""
    t = Table(title="families / tiers / $/1M in-out")
    t.add_column("family"); t.add_column("cheap"); t.add_column("balanced"); t.add_column("flagship")
    for fam, m in FAMILY_MODELS.items():
        def fmt(mid: str) -> str:
            p = PRICING.get(mid, {})
            return f"{mid}\n${p.get('input')}/${p.get('output')}"
        t.add_row(fam, fmt(m["cheap"]), fmt(m["balanced"]), fmt(m["flagship"]))
    console.print(t)


# ---------------------------------------------------------------- bench

DATASET_OPT = typer.Option(Path("evals/datasets/seed.jsonl"), "--dataset", "-d")
RUN_OPT = typer.Option(Path("runs/dev"), "--run", "-r", help="Run directory (cache lives here)")
BENCH_FAMILY_OPT = typer.Option("mock", "--family", "-f", help="anthropic (API key) | claude-code (Pro/Max) | mock")


def _bench_key(family: str, api_key: str | None) -> str:
    if family == "mock":
        return ""
    if family == "claude-code":
        if not providers.claude_code_available():
            console.print("[red]`claude` CLI not found. Install Claude Code and log in with your Pro/Max account.[/red]")
            raise typer.Exit(1)
        return ""
    if family != "anthropic":
        raise typer.BadParameter("bench supports anthropic|claude-code|mock in v1")
    key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        console.print("[red]ANTHROPIC_API_KEY not set. Benchmarks never fall back to mock answers.[/red]")
        raise typer.Exit(1)
    return key


def _confirm(n_calls: int, what: str, family: str, yes: bool) -> None:
    if family == "mock" or n_calls == 0 or yes:
        return
    kind = ("Claude calls on your Pro/Max subscription (they count against your usage limits; "
            "if you hit a limit, re-run later to resume)") if family == "claude-code" else "paid Claude API calls"
    if not typer.confirm(f"This makes {n_calls} {kind} ({what}). Continue?"):
        raise typer.Exit(0)


def _progress_run(total: int, label: str, fn) -> int:
    errors: list[str] = []
    with Progress(console=console) as prog:
        task = prog.add_task(label, total=total)

        def on_done(row, err):
            if err:
                errors.append(str(err))
            prog.advance(task)

        failures = fn(on_done)
    for e in errors[:5]:
        console.print(f"[red]error:[/red] {e[:300]}")
    return failures


@bench.command("answers")
def bench_answers(
    dataset: Path = DATASET_OPT, run_dir: Path = RUN_OPT, family: str = BENCH_FAMILY_OPT,
    api_key: str | None = typer.Option(None, "--api-key"),
    workers: int = typer.Option(4, "--workers", "-w"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the cost confirmation"),
):
    """Answer every prompt with every tier (cached; re-run to resume)."""
    key = _bench_key(family, api_key)
    prompts = load_dataset(dataset)
    store = Store(run_dir / "answers.jsonl")
    todo = bench_matrix.pending_answers(prompts, family, store)
    _confirm(len(todo), "3 tiers per prompt", family, yes)
    failures = _progress_run(len(todo), "answers", lambda cb: bench_matrix.run_answers(prompts, family, key, store, workers, cb))
    console.print(f"answers: {len(todo) - failures} new, {failures} failed (re-run to retry) -> {store.path}")


@bench.command("classify")
def bench_classify(
    dataset: Path = DATASET_OPT, run_dir: Path = RUN_OPT,
    backend: str = typer.Option("laya", "--backend", "-b", help="laya|heuristic"),
    variants: list[str] = typer.Option(["variants/a_score.yaml", "variants/c_tier_choice.yaml"], "--variant", "-v"),
    rotations: int = typer.Option(3, "--rotations", help="Option orders per choice variant"),
):
    """Classify every prompt with the router (Laya is free, so this is cheap to redo)."""
    prompts = load_dataset(dataset)
    store = Store(run_dir / "classifications.jsonl")
    vs = bench_matrix.policy_variants([classifier.load_variant(p) for p in variants], rotations)
    try:
        new = bench_matrix.run_classify(prompts, backend, vs, store)
    except classifier.RouterError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print(f"classifications: {new} new -> {store.path}")


@bench.command("judge")
def bench_judge_cmd(
    dataset: Path = DATASET_OPT, run_dir: Path = RUN_OPT, family: str = BENCH_FAMILY_OPT,
    api_key: str | None = typer.Option(None, "--api-key"),
    workers: int = typer.Option(4, "--workers", "-w"),
    yes: bool = typer.Option(False, "--yes", "-y"),
):
    """Judge cheap/balanced answers against the flagship, in both orders."""
    key = _bench_key(family, api_key)
    prompts = load_dataset(dataset)
    answers, judgments = Store(run_dir / "answers.jsonl"), Store(run_dir / "judgments.jsonl")
    n = sum(1 for p in prompts for t in bench_judge.CANDIDATE_TIERS for o in bench_judge.ORDERS
            if (a := bench_matrix.answer_lookup(answers, family, p["id"], p["prompt"], t))
            and (f := bench_matrix.answer_lookup(answers, family, p["id"], p["prompt"], "flagship"))
            and not judgments.has(bench_judge.judgment_key(p["id"], t, o, a["key"], f["key"], bench_judge.JUDGE_MODEL)))
    _confirm(n, f"judge = {bench_judge.JUDGE_MODEL}", family, yes)
    result = {}

    def go(cb):
        result["v"] = bench_judge.run_judge(prompts, answers, judgments, family, key, workers, on_done=cb)
        return result["v"][0]

    failures = _progress_run(n, "judge", go)
    skipped = result["v"][1]
    console.print(f"judgments: {n - failures} new, {failures} failed, {skipped} skipped (missing answers) -> {judgments.path}")


@bench.command("report")
def bench_report_cmd(
    dataset: Path = DATASET_OPT, run_dir: Path = RUN_OPT, family: str = BENCH_FAMILY_OPT,
    floor: float = typer.Option(0.95, "--floor", help="Quality floor (share non-inferior)"),
):
    """Replay every policy and baseline from the cache; write report.md + report.json."""
    prompts = load_dataset(dataset)
    m = bench_replay.build_matrix(prompts, Store(run_dir / "answers.jsonl"), Store(run_dir / "judgments.jsonl"),
                                  Store(run_dir / "classifications.jsonl"), family, bench_judge.JUDGE_MODEL)
    if not m.prompts:
        console.print("[red]No prompt has all answers and judgments yet. Run `bench answers` and `bench judge`.[/red]")
        raise typer.Exit(1)
    r = bench_report.build_report(m, floor)
    r["meta"]["cost_basis"] = "api-equivalent (subscription usage proxy)" if family == "claude-code" else "api"
    md = bench_report.render_markdown(r)
    (run_dir / "report.json").write_text(json.dumps(r, indent=2))
    (run_dir / "report.md").write_text(md)
    console.print(md)
    console.print(f"Saved {run_dir / 'report.md'} and report.json")
