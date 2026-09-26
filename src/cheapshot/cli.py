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
from rich.prompt import Prompt
from rich.table import Table

from . import jev_client, providers, router, tracking
from .eval import run as eval_run, summarize
from .pricing import FAMILY_MODELS, cost_usd

load_dotenv()
app = typer.Typer(add_completion=False, help="Cheapshot — classify prompt with Jev, route to cheapest suitable model.")
console = Console()

FAMILIES = ["anthropic", "openai", "gemini", "mock"]
ENV_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "mock": ""}


def resolve_family(family: str | None) -> str:
    if family and family.lower() in FAMILIES:
        return family.lower()
    # Startup choice (requirement #2)
    choice = Prompt.ask("Choose model family", choices=FAMILIES, default="anthropic")
    return choice


def resolve_key(family: str, api_key: str | None) -> str:
    if api_key:
        return api_key
    if family == "mock":
        return ""
    env = os.getenv(ENV_KEYS[family], "")
    if env:
        return env
    # BYOK: prompt without echo
    console.print(f"[yellow]No {ENV_KEYS[family]} found — enter your key (BYOK, not stored).[/yellow]")
    try:
        return getpass.getpass(f"{family} API key (blank=mock mode): ")
    except Exception:
        return ""


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Initial prompt to classify + answer"),
    family: str | None = typer.Option(None, "--family", "-f", help="anthropic|openai|gemini|mock"),
    api_key: str | None = typer.Option(None, "--api-key", help="BYOK key (else env/prompt)"),
    config: str = typer.Option("router_config.yaml", "--config"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Classify + route only, no generation call"),
    explain: bool = typer.Option(True, "--explain/--no-explain", help="Show routing panel"),
):
    """Classify initial prompt with Jev, call routed model with the prompt."""
    fam = resolve_family(family)
    key = resolve_key(fam, api_key)
    eff_fam = fam if key or fam == "mock" else "mock"
    if fam != "mock" and not key:
        console.print("[yellow]No key given — running in mock mode (no charges).[/yellow]")

    c = jev_client.classify(prompt, config_path=config)
    rt = router.route(c.label, c.confidence, eff_fam, config)
    if dry_run:
        console.print(Panel(f"jev=[bold]{c.label}[/bold] conf={c.confidence:.2f}{' (mock)' if c.mocked else ''}\n"
                            f"route: {rt.tier} -> [bold]{rt.model}[/bold]\n{rt.reason}",
                            title=f"dry-run ({fam})"))
        return
    comp = providers.complete(eff_fam, rt.model, prompt, api_key=key)
    cost = cost_usd(rt.model, comp.input_tokens, comp.output_tokens)
    tracking.log({"mode": "run", "family": fam, "label": c.label, "tier": rt.tier,
                  "model": rt.model, "cost": cost, "mocked": c.mocked or eff_fam == "mock"})
    if explain:
        console.print(Panel(f"jev=[bold]{c.label}[/bold] conf={c.confidence:.2f}{' (mock)' if c.mocked else ''}\n"
                            f"route: {rt.tier} -> [bold]{rt.model}[/bold]  family={fam}\n{rt.reason}\n"
                            f"tokens {comp.input_tokens} in / {comp.output_tokens} out | "
                            f"${cost:.6f} | {comp.latency_s:.2f}s",
                            title="routing"))
    console.print(Panel(comp.text, title=rt.model))


@app.command()
def classify(
    prompt: str = typer.Argument(...),
    config: str = typer.Option("router_config.yaml", "--config"),
):
    """Jev classify only (no generation). Shows raw request shape compat."""
    c = jev_client.classify(prompt, config_path=config)
    console.print(Panel(f"label=[bold]{c.label}[/bold] conf={c.confidence:.2f} mocked={c.mocked}\n"
                        f"raw={json.dumps(c.raw)[:500]}", title="jev classification"))


@app.command()
def models():
    """Show tier -> model map + pricing."""
    t = Table(title="families / tiers / $/1M in-out")
    t.add_column("family"); t.add_column("cheap"); t.add_column("balanced"); t.add_column("flagship")
    from .pricing import PRICING
    for fam, m in FAMILY_MODELS.items():
        def fmt(mid: str) -> str:
            p = PRICING.get(mid, {})
            return f"{mid}\n${p.get('input')}/${p.get('output')}"
        t.add_row(fam, fmt(m["cheap"]), fmt(m["balanced"]), fmt(m["flagship"]))
    console.print(t)


@app.command()
def benchmark(
    dataset: Path = typer.Option(Path("evals/datasets/seed.jsonl"), "--dataset", "-d"),
    family: str | None = typer.Option(None, "--family", "-f"),
    api_key: str | None = typer.Option(None, "--api-key"),
    config: str = typer.Option("router_config.yaml", "--config"),
    out: Path = typer.Option(Path("report.md"), "--out", "-o"),
):
    """Run dataset, print + save cost-optimization report."""
    fam = resolve_family(family)
    key = resolve_key(fam, api_key)
    eff_fam = fam if key or fam == "mock" else "mock"
    if not dataset.exists():
        console.print(f"[red]Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)
    with console.status("Benchmarking..."):
        results = eval_run(dataset, eff_fam, key, config)
    s = summarize(results)
    t = Table(title=f"benchmark ({fam}, n={s['n']})")
    t.add_column("metric"); t.add_column("value")
    t.add_row("routed cost", f"${s['routed_cost']:.6f}")
    t.add_row("always-flagship baseline", f"${s['baseline_cost']:.6f}")
    t.add_row("cost optimization (savings %)", f"[bold green]{s['savings_pct']}%[/bold green]")
    t.add_row("tier distribution", json.dumps(s["tier_distribution"]))
    t.add_row("avg latency", f"{s['avg_latency_s']}s")
    if s["accuracy_pct"] is not None:
        t.add_row("routing accuracy", f"{s['accuracy_pct']}% (n={s['accuracy_n']})")
    console.print(t)
    md = (f"# Jev benchmark report\n\nfamily={fam} n={s['n']}\n\n"
          f"- routed cost: ${s['routed_cost']:.6f}\n- baseline (always flagship): ${s['baseline_cost']:.6f}\n"
          f"- **cost optimization: {s['savings_pct']}%**\n- tier distribution: {s['tier_distribution']}\n"
          f"- avg latency: {s['avg_latency_s']}s\n"
          + (f"- routing accuracy: {s['accuracy_pct']}% (n={s['accuracy_n']})\n" if s["accuracy_pct"] is not None else ""))
    out.write_text(md)
    console.print(f"Saved {out}")
