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

from . import classifier, costaware, curves, providers, router, tracking
from .classifier import TIERS
from .routing import resolve_path
from .bench import judge as bench_judge
from .bench import matrix as bench_matrix
from .bench import replay as bench_replay
from .bench import export as bench_export
from .bench import report as bench_report
from .bench import teacher as bench_teacher
from .bench.dataset import load_dataset
from .bench.store import Store
from .pricing import FAMILY_MODELS, PRICING, cost_usd

load_dotenv(Path(os.getenv("CHEAPSHOT_HOME", Path.home() / ".cheapshot")) / "config.env")  # user config wins
load_dotenv()  # then the repo's .env
app = typer.Typer(add_completion=False, invoke_without_command=True,
                  help="Cheapshot — Claude Code where a Laya decision model routes every turn to the cheapest capable model.")
bench = typer.Typer(add_completion=False, help="Benchmark: cached answers + judge, replayed offline for any routing policy.")
app.add_typer(bench, name="bench")
console = Console()

FAMILIES = ["anthropic", "claude-code", "openai", "gemini", "mock"]
ENV_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "claude-code": "", "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY", "mock": ""}
KEYLESS = ("mock", "claude-code")  # claude-code uses your Pro/Max login via the `claude` CLI


@app.callback()
def main(
    ctx: typer.Context,
    quality: float | None = typer.Option(None, "--quality", "-q", help="Quality target 0-1 (default CHEAPSHOT_QUALITY or 0.95)"),
    mode: str = typer.Option("acceptEdits", "--mode", help="Claude Code permission mode: acceptEdits|plan|default|bypassPermissions"),
    which: str = typer.Option("laya", "--classifier", help="laya|heuristic"),
    variant: str = typer.Option("variants/a_score.yaml", "--variant", "-v", help="Router variant YAML"),
    rule: str | None = typer.Option(None, "--rule", help="threshold (default) | cost"),
):
    """With no command: start an interactive Claude Code session routed per turn (Pro/Max subscription)."""
    if ctx.invoked_subcommand is not None:
        return
    from .repl import Repl
    from .routing import Routing
    from .session import PERMISSION_MODES, ClaudeCodeSession
    if not providers.claude_code_available():
        console.print("[red]`claude` CLI not found. Install Claude Code and log in with your Pro/Max account.[/red]")
        raise typer.Exit(1)
    if mode not in PERMISSION_MODES:
        raise typer.BadParameter(f"--mode must be one of {', '.join(PERMISSION_MODES)}")
    try:
        routing = Routing(variant, quality, which, rule=rule)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    Repl(console, routing, ClaudeCodeSession(permission_mode=mode)).loop()


@app.command("curve")
def curve_cmd(
    variant: str = typer.Option("variants/a_score.yaml", "--variant", "-v"),
    router_model: str | None = typer.Option(None, "--router-model", help="Default: LAYA_MODEL"),
    rule: str = typer.Option("threshold", "--rule", help="threshold | cost"),
):
    """Show the savings-vs-quality trade-off measured for the router, one row per quality target."""
    from .repl import curve_table
    name = curves.COST_CURVE_NAME if rule == "cost" else classifier.load_variant(resolve_path(variant))["name"]
    model = router_model or os.getenv("LAYA_MODEL") or classifier.DEFAULT_LAYA_MODEL
    c = curves.load_curve(name, model)
    if not c:
        console.print(f"[red]No curve for {name} / {model}. Build it with `cheapshot bench curve`.[/red]")
        raise typer.Exit(1)
    console.print(curve_table(c))
    console.print("[dim]quality = share of prompts the Opus judge rated as good as Opus's answer. "
                  "savings/prompt counts each prompt equally; total-bill savings weights by cost.[/dim]")


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
QUALITY_OPT = typer.Option(None, "--quality", "-q", help="Quality target 0-1, mapped to epsilon via the benchmark curve")
CLASSIFIER_OPT = typer.Option("laya", "--classifier", help="laya|heuristic")


def classify_prompt(prompt: str, variant_path: str, which: str) -> tuple[classifier.Classification, dict | None]:
    if which == "heuristic":
        return classifier.heuristic_classify(prompt), None
    if which != "laya":
        raise typer.BadParameter("classifier must be laya|heuristic")
    variant = classifier.load_variant(resolve_path(variant_path))
    try:
        client = classifier.LayaClient()
        try:
            return client.classify(prompt, variant), variant
        finally:
            client.close()
    except classifier.RouterError as e:
        console.print(f"[red]{e}[/red]\n[dim]Set LAYA_URL (and LAYA_API_KEY), or pass --classifier heuristic.[/dim]")
        raise typer.Exit(1)


def effective_epsilon(epsilon: float | None, variant: dict | None, quality: float | None = None) -> float:
    """--epsilon; else --quality via the router's benchmark curve; else the variant's epsilon for
    the configured LAYA_MODEL (tuned on the benchmark); else its default."""
    if epsilon is not None:
        return epsilon
    v = variant or {}
    model = os.getenv("LAYA_MODEL") or classifier.DEFAULT_LAYA_MODEL
    if quality is not None and v.get("name"):
        c = curves.load_curve(v["name"], model)
        if c:
            return curves.eps_for_quality(c, quality)[0]
        console.print(f"[yellow]No benchmark curve for {model}; ignoring --quality.[/yellow]")
    return float((v.get("epsilon_by_model") or {}).get(model, v.get("epsilon", 0.2)))


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
    quality: float | None = QUALITY_OPT,
    which: str = CLASSIFIER_OPT,
    dry_run: bool = typer.Option(False, "--dry-run", help="Classify + route only, no generation call"),
    explain: bool = typer.Option(True, "--explain/--no-explain", help="Show routing panel"),
):
    """Classify the prompt, route with the risk dial, call the routed model."""
    fam = resolve_family(family)
    c, v = classify_prompt(prompt, variant, which)
    rt = router.route(c, effective_epsilon(epsilon, v, quality), fam)
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
    quality: float | None = QUALITY_OPT,
    which: str = CLASSIFIER_OPT,
    save_fixture: Path | None = typer.Option(None, "--save-fixture", help="Write raw router response JSON here"),
):
    """Classify only (no generation): tier distribution, routed tier, raw response."""
    c, v = classify_prompt(prompt, variant, which)
    rt = router.route(c, effective_epsilon(epsilon, v, quality))
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


SPLIT_OPT = typer.Option("tune,test", "--split", help="Comma list of splits (train,tune,test) or 'all'")
LIMIT_OPT = typer.Option(None, "--limit", help="Only the first N prompts after the split filter")
SUBSET_OPT = typer.Option(None, "--subset", help="Only rows tagged with this dataset subset (e.g. judge)")


def _select(dataset: Path, split: str, limit: int | None = None, subset: str | None = None) -> list[dict]:
    wanted = None if split == "all" else {s.strip() for s in split.split(",") if s.strip()}
    rows = [p for p in load_dataset(dataset) if (wanted is None or p["split"] in wanted)
            and (subset is None or p.get("subset") == subset)]
    return rows[:limit] if limit else rows


def _progress_run(total: int, label: str, fn):
    """Run fn(on_done) under a progress bar; fn returns a PoolResult."""
    errors: list[str] = []
    with Progress(console=console) as prog:
        task = prog.add_task(label, total=total)

        def on_done(row, err):
            if err:
                errors.append(str(err))
            prog.advance(task)

        res = fn(on_done)
    for e in errors[:3]:
        console.print(f"[red]error:[/red] {e[:300]}")
    return res


def _summary(what: str, res, path: Path, extra: str = "") -> None:
    console.print(f"{what}: {res.done} new, {res.failures} failed, {res.not_run} not attempted{extra} -> {path}")
    if res.stop_reason:
        console.print(f"[yellow]Stopped early ({res.stop_reason}).[/yellow]\n"
                      "[yellow]If that's a usage limit, re-run the same command after it resets; finished work is cached.[/yellow]")


@bench.command("answers")
def bench_answers(
    dataset: Path = DATASET_OPT, run_dir: Path = RUN_OPT, family: str = BENCH_FAMILY_OPT,
    api_key: str | None = typer.Option(None, "--api-key"),
    workers: int = typer.Option(4, "--workers", "-w"),
    split: str = SPLIT_OPT, limit: int | None = LIMIT_OPT, subset: str | None = SUBSET_OPT,
    tiers: str = typer.Option("cheap,balanced,flagship", "--tiers", help="Comma list of tiers to answer with"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the cost confirmation"),
):
    """Answer prompts with every tier (cached; re-run to resume). Default: tune+test splits."""
    key = _bench_key(family, api_key)
    prompts = _select(dataset, split, limit, subset)
    tier_list = tuple(t.strip() for t in tiers.split(",") if t.strip())
    bad = [t for t in tier_list if t not in TIERS]
    if bad:
        raise typer.BadParameter(f"unknown tiers {bad}; use {', '.join(TIERS)}")
    store = Store(run_dir / "answers.jsonl")
    todo = bench_matrix.pending_answers(prompts, family, store, tier_list)
    _confirm(len(todo), f"{len(tier_list)} tier(s) x {len(prompts)} prompts", family, yes)
    res = _progress_run(len(todo), "answers",
                        lambda cb: bench_matrix.run_answers(prompts, family, key, store, workers, cb, tier_list))
    _summary("answers", res, store.path)


@bench.command("classify")
def bench_classify(
    dataset: Path = DATASET_OPT, run_dir: Path = RUN_OPT,
    backend: str = typer.Option("laya", "--backend", "-b", help="laya|heuristic"),
    variants: list[str] = typer.Option(["variants/a_score.yaml", "variants/c_tier_choice.yaml"], "--variant", "-v"),
    rotations: int = typer.Option(3, "--rotations", help="Option orders per choice variant"),
    split: str = SPLIT_OPT, limit: int | None = LIMIT_OPT,
):
    """Classify prompts with the router (Laya is free, so this is cheap to redo). Default: tune+test."""
    prompts = _select(dataset, split, limit)
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
    split: str = SPLIT_OPT, limit: int | None = LIMIT_OPT, subset: str | None = SUBSET_OPT,
    yes: bool = typer.Option(False, "--yes", "-y"),
):
    """Judge cheap/balanced answers against the flagship, in both orders. Default: tune+test.
    Train prompts are judged lazily: balanced only in an order where cheap lost."""
    key = _bench_key(family, api_key)
    prompts = _select(dataset, split, limit, subset)
    answers, judgments = Store(run_dir / "answers.jsonl"), Store(run_dir / "judgments.jsonl")
    jobs, waiting, _ = bench_judge.pending_judgments(prompts, answers, judgments, family)
    n = len(jobs) + waiting
    _confirm(n, f"judge = {bench_judge.JUDGE_MODEL}" + (f"; at most, {waiting} only if cheap loses" if waiting else ""),
             family, yes)
    result = {}

    def go(cb):
        result["res"], result["skipped"] = bench_judge.run_judge(prompts, answers, judgments, family, key, workers, on_done=cb)
        return result["res"]

    res = _progress_run(n, "judge", go)
    _summary("judgments", res, judgments.path, f", {result['skipped']} skipped (missing answers)")


@bench.command("teacher")
def bench_teacher_cmd(
    dataset: Path = DATASET_OPT, run_dir: Path = RUN_OPT, family: str = BENCH_FAMILY_OPT,
    api_key: str | None = typer.Option(None, "--api-key"),
    workers: int = typer.Option(4, "--workers", "-w"),
    split: str = typer.Option("train", "--split", help="Comma list of splits or 'all'"),
    limit: int | None = LIMIT_OPT,
    yes: bool = typer.Option(False, "--yes", "-y"),
):
    """Opus teacher labels (probability per tier) for fine-tuning data. One call per prompt. Default: train."""
    key = _bench_key(family, api_key)
    prompts = _select(dataset, split, limit)
    store = Store(run_dir / "teacher.jsonl")
    n = sum(not store.has(bench_teacher.teacher_key(p["id"], p["prompt"], bench_teacher.TEACHER_MODEL)) for p in prompts)
    _confirm(n, f"teacher = {bench_teacher.TEACHER_MODEL}", family, yes)
    res = _progress_run(n, "teacher", lambda cb: bench_teacher.run_teacher(prompts, store, family, key, workers, on_done=cb))
    _summary("teacher labels", res, store.path)


@bench.command("export-train")
def bench_export_cmd(
    dataset: Path = DATASET_OPT, run_dir: Path = RUN_OPT, family: str = BENCH_FAMILY_OPT,
    variant: str = VARIANT_OPT,
    out: Path = typer.Option(None, "--out", "-o", help="Default: <run>/laya_train.jsonl"),
    calibrate_teacher: bool = typer.Option(True, "--calibrate-teacher/--no-calibrate-teacher",
                                           help="Map teacher labels onto judge outcomes (fit on tune + judged train)"),
    judge_weight: int = typer.Option(2, "--judge-weight", min=1, help="Times each judge-labeled row is written"),
    length_variant: str = typer.Option(None, "--length-variant",
                                       help="Also label this answer-length question from Haiku answers "
                                            "(e.g. variants/length_score.yaml)"),
):
    """Write laya-train JSONL from train-split prompts (judge labels first, teacher labels otherwise)."""
    v = classifier.load_variant(variant)
    prompts = load_dataset(dataset)
    stores = (Store(run_dir / "answers.jsonl"), Store(run_dir / "judgments.jsonl"), Store(run_dir / "teacher.jsonl"))
    tmap = None
    if calibrate_teacher:
        pairs = bench_export.teacher_calibration_pairs(prompts, *stores, family, bench_judge.JUDGE_MODEL,
                                                       bench_teacher.TEACHER_MODEL)
        if len(pairs) < 30:
            console.print(f"[red]Only {len(pairs)} tune/train prompts have both teacher and judge labels; need 30+ to "
                          "recalibrate. Run `bench teacher --split tune`, or pass --no-calibrate-teacher.[/red]")
            raise typer.Exit(1)
        tmap = bench_export.fit_teacher_map(pairs)
        console.print(f"teacher recalibrated on {len(pairs)} tune/train prompts:")
        for lvl, mix in tmap:
            console.print(f"  teacher level {lvl:.2f} -> " + " ".join(f"{t}={mix[t]:.2f}" for t in mix))
    rows, counts = bench_export.build_training_rows(prompts, v, *stores, family, bench_judge.JUDGE_MODEL,
                                                    bench_teacher.TEACHER_MODEL, tmap, judge_weight,
                                                    classifier.load_variant(length_variant) if length_variant else None)
    out = out or run_dir / "laya_train.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    console.print(f"{len(rows)} training rows -> {out}  {counts}")
    if len(rows) < 1000:
        console.print("[yellow]Under ~1,000 rows: the Laya docs report small fine-tunes collapsing. "
                      "Add more train prompts or more epochs, and repeat runs across seeds.[/yellow]")


@bench.command("curve")
def bench_curve_cmd(
    dataset: Path = DATASET_OPT, run_dir: Path = RUN_OPT, family: str = BENCH_FAMILY_OPT,
    variant: str = VARIANT_OPT,
    router_model: str = typer.Option(..., "--router-model", help="Laya model these classifications came from"),
    rule: str = typer.Option("threshold", "--rule",
                             help="threshold (epsilon on --variant) | cost (also needs length_score classifications)"),
):
    """Save the savings-vs-quality curve of a run's classifications, so --quality can pick a setting."""
    v = classifier.load_variant(resolve_path(variant))
    m = bench_replay.build_matrix(load_dataset(dataset), Store(run_dir / "answers.jsonl"), Store(run_dir / "judgments.jsonl"),
                                  Store(run_dir / "classifications.jsonl"), family, bench_judge.JUDGE_MODEL)
    length_name = classifier.load_variant(resolve_path(costaware.LENGTH_VARIANT))["name"]
    need = [v["name"]] + ([length_name] if rule == "cost" else [])
    missing = [n for n in need if n not in m.policies]
    if missing:
        console.print(f"[red]{run_dir} has no complete classifications for {', '.join(missing)}.[/red]")
        raise typer.Exit(1)
    source = f"{dataset.name} / {run_dir.name}"
    curve = (curves.build_cost_curve(m, v["name"], length_name, router_model, family, source) if rule == "cost"
             else curves.build_curve(m, v["name"], router_model, source))
    path = curves.save_curve(curve)
    console.print(f"Saved {path}")
    from .repl import curve_table
    console.print(curve_table(curves.load_curve(curve["variant"], router_model)))


@bench.command("report")
def bench_report_cmd(
    dataset: Path = DATASET_OPT, run_dir: Path = RUN_OPT, family: str = BENCH_FAMILY_OPT,
    floor: float = typer.Option(0.95, "--floor", help="Quality floor (share non-inferior)"),
    objective: str = typer.Option("per-prompt", "--objective",
                                  help="Savings that epsilon maximizes: per-prompt (each prompt counts equally) | total"),
):
    """Replay every policy and baseline from the cache; write report.md + report.json."""
    prompts = load_dataset(dataset)
    m = bench_replay.build_matrix(prompts, Store(run_dir / "answers.jsonl"), Store(run_dir / "judgments.jsonl"),
                                  Store(run_dir / "classifications.jsonl"), family, bench_judge.JUDGE_MODEL)
    if not m.prompts:
        console.print("[red]No prompt has all answers and judgments yet. Run `bench answers` and `bench judge`.[/red]")
        raise typer.Exit(1)
    if objective not in bench_replay.OBJECTIVES:
        raise typer.BadParameter(f"--objective must be one of {', '.join(bench_replay.OBJECTIVES)}")
    r = bench_report.build_report(m, floor, objective=objective)
    r["meta"]["cost_basis"] = "api-equivalent (subscription usage proxy)" if family == "claude-code" else "api"
    md = bench_report.render_markdown(r)
    (run_dir / "report.json").write_text(json.dumps(r, indent=2))
    (run_dir / "report.md").write_text(md)
    console.print(md)
    console.print(f"Saved {run_dir / 'report.md'} and report.json")
