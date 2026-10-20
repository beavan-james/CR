"""Interactive Cheapshot: a Claude Code session where every turn is routed to the cheapest
capable model. Run `cheapshot` in a project directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import curves, tracking
from .classifier import TIERS
from .pricing import FAMILY_MODELS, PRICING
from .routing import Routing
from .session import PERMISSION_MODES, ClaudeCodeSession

FAMILY = "claude-code"
TIER_ALIASES = {"haiku": "cheap", "sonnet": "balanced", "opus": "flagship",
                "cheap": "cheap", "balanced": "balanced", "flagship": "flagship"}
HISTORY = Path(os.getenv("CHEAPSHOT_HOME", Path.home() / ".cheapshot")) / "history"

HELP = """[bold]Commands[/bold]
  /quality [0.80-1.00]   quality target (share of turns as good as Opus); shows the trade-off
  /curve                 savings vs quality for the current router
  /rule [threshold|cost] risk threshold (default) or cost-aware routing (saves on long answers)
  /model [auto|haiku|sonnet|opus]   pin a model, or back to routing
  /why                   how the last turn was routed
  /cost                  spend this session vs always-Opus
  /mode [acceptEdits|plan|default|bypassPermissions]   Claude Code permission mode
  /new                   start a fresh conversation
  /exit                  quit (Ctrl-D also works)
End a line with \\ to keep typing on the next line. Ctrl-C stops the current turn."""


def pretty(model: str) -> str:
    """claude-sonnet-5-5 -> Sonnet 5.5"""
    parts = model.replace("claude-", "").split("-")
    return f"{parts[0].capitalize()} {'.'.join(parts[1:])}" if len(parts) > 1 else model


def opus_equivalent(cost: float, model: str) -> float:
    """What the same tokens would have cost on the flagship (prices scale together per tier)."""
    flag = FAMILY_MODELS[FAMILY]["flagship"]
    return cost * PRICING[flag]["output"] / PRICING[model]["output"] if model in PRICING else cost


def curve_table(curve: dict, current: float | None = None) -> Table:
    rule = "cost-aware" if curve.get("rule") == "cost" else "threshold"
    t = Table(title=f"Savings vs quality — {curve['router_model']}, {rule} routing "
                    f"(held-out {curve['n_test']} prompts; setting chosen on {curve['n_tune']} tune prompts)")
    for col in ("quality target", "setting", "quality", "savings/prompt", "total-bill savings", "mix H/S/O", "random @ same mix"):
        t.add_column(col, justify="right")
    for r in curves.frontier(curve):
        mark = " ◀" if current is not None and abs(r["target"] - current) < 1e-9 else ""
        mix = "/".join(f"{100 * r['mix'].get(k, 0):.0f}" for k in TIERS)
        t.add_row(f"{r['target']:.0%}{mark}", r["knob"], f"{r['quality']:.1%}", f"{r['savings_per_prompt']:.1%}",
                  f"{r['savings']:.1%}", mix, f"{r['random_quality']:.1%}")
    return t


class Repl:
    def __init__(self, console: Console, routing: Routing, session: ClaudeCodeSession):
        self.c, self.routing, self.session = console, routing, session
        self.previous: str | None = None
        self.last = None
        self.spent = self.opus_est = 0.0
        self.turn_tiers: list[str] = []

    # ------------------------------------------------------------------ display
    def banner(self) -> None:
        r = self.routing
        self.c.print(f"[bold]Cheapshot[/bold] · Claude Code with per-turn routing · [dim]{self.session.cwd}[/dim]")
        self.c.print(f"router [cyan]{r.router_model}[/cyan] · {self.quality_line()} · permissions [cyan]{self.session.permission_mode}[/cyan]")
        self.c.print("[dim]/help for commands[/dim]\n")

    def quality_line(self) -> str:
        r = self.routing
        if r.forced_tier:
            return f"model pinned to [bold]{pretty(FAMILY_MODELS[FAMILY][r.forced_tier])}[/bold]"
        s = f"{r.rule} routing · quality target [bold]{r.quality:.0%}[/bold] ({r.setting})"
        if r.expected:
            x = r.expected["test"]
            s += (f" · benchmark: {x['quality']:.0%} as good as Opus, {x['savings']:.0%} of the bill saved "
                  f"({x['savings_per_prompt']:.0%} per prompt)")
        elif r.client:
            s += " · [yellow]no benchmark curve for this router; ε from the variant[/yellow]"
        return s

    # ------------------------------------------------------------------ commands
    def command(self, line: str) -> bool:
        """Handle a slash command. Returns False to exit."""
        cmd, _, arg = line[1:].partition(" ")
        arg = arg.strip()
        r = self.routing
        if cmd in ("exit", "quit", "q"):
            return False
        if cmd == "help":
            self.c.print(HELP)
        elif cmd == "quality":
            if arg:
                try:
                    q = float(arg.rstrip("%"))
                    q = q / 100 if q > 1 else q
                except ValueError:
                    self.c.print("[red]usage: /quality 0.95  (or 95%)[/red]")
                    return True
                r.set_quality(min(max(q, 0.0), 1.0))
            self.c.print(self.quality_line())
            if r.curve:
                self.c.print(curve_table(r.curve, r.quality))
        elif cmd == "curve":
            if r.curve:
                self.c.print(curve_table(r.curve))
            else:
                self.c.print("[yellow]No curve for this router. Build one with `cheapshot bench curve`.[/yellow]")
        elif cmd == "rule":
            if arg:
                try:
                    r.set_rule(arg)
                except ValueError as e:
                    self.c.print(f"[red]{e}[/red]")
                    return True
            self.c.print(self.quality_line())
        elif cmd == "model":
            if not arg or arg == "auto":
                r.forced_tier = None
            elif arg in TIER_ALIASES:
                r.forced_tier = TIER_ALIASES[arg]
            else:
                self.c.print("[red]usage: /model auto|haiku|sonnet|opus[/red]")
                return True
            self.c.print(self.quality_line())
        elif cmd == "why":
            if not self.last:
                self.c.print("[dim]No turn yet.[/dim]")
            else:
                d = self.last
                self.c.print(f"{pretty(d.route.model)}: {d.route.reason}" + (f"\n[yellow]{d.note}[/yellow]" if d.note else ""))
        elif cmd == "cost":
            self.print_cost()
        elif cmd == "mode":
            if arg in PERMISSION_MODES:
                self.session.permission_mode = arg
            elif arg:
                self.c.print(f"[red]modes: {', '.join(PERMISSION_MODES)}[/red]")
            self.c.print(f"permissions: {self.session.permission_mode}")
        elif cmd == "new":
            self.session.reset()
            self.previous = None
            self.c.print("[dim]New conversation.[/dim]")
        else:
            self.c.print(f"[red]Unknown command /{cmd}[/red] — /help")
        return True

    def print_cost(self) -> None:
        if not self.turn_tiers:
            self.c.print("[dim]No turns yet.[/dim]")
            return
        saved = 1 - self.spent / self.opus_est if self.opus_est else 0.0
        mix = ", ".join(f"{pretty(FAMILY_MODELS[FAMILY][t])} {self.turn_tiers.count(t)}" for t in TIERS if t in self.turn_tiers)
        self.c.print(f"{len(self.turn_tiers)} turns ({mix}) · ${self.spent:.3f} API-equivalent · "
                     f"always-Opus est. ${self.opus_est:.3f} · [bold]{saved:.0%} saved[/bold]")

    # ------------------------------------------------------------------ turns
    def turn(self, prompt: str) -> None:
        d = self.routing.decide(prompt, self.previous)
        self.last = d
        model = d.route.model
        why = d.rule or d.why
        self.c.print(f"[dim]→ {pretty(model)} · {why}[/dim]" + (f" [yellow]{d.note}[/yellow]" if d.note else ""))

        out = sys.stdout
        state = {"at_line_start": True}

        def on_text(chunk: str) -> None:
            out.write(chunk)
            out.flush()
            state["at_line_start"] = chunk.endswith("\n")

        def on_tool(label: str) -> None:
            if not state["at_line_start"]:
                out.write("\n")
            self.c.print(f"[dim]  ⏺ {label}[/dim]", highlight=False)
            state["at_line_start"] = True

        res = self.session.run_turn(prompt, model, on_text, on_tool)
        if not state["at_line_start"]:
            out.write("\n")
        if res.is_error:
            self.c.print(f"[red]{res.error}[/red]")
        if res.denied:
            self.c.print(f"[yellow]Blocked by permission mode {self.session.permission_mode}: {', '.join(sorted(set(res.denied)))}. "
                         "Allow tools in .claude/settings.json (permissions.allow), or /mode bypassPermissions.[/yellow]")
        est = opus_equivalent(res.cost_usd, model)
        self.spent += res.cost_usd
        self.opus_est += est
        self.turn_tiers.append(d.route.tier)
        self.previous = prompt
        saved = 1 - self.spent / self.opus_est if self.opus_est else 0.0
        self.c.print(f"[dim]{pretty(model)} · ${res.cost_usd:.4f} (Opus est. ${est:.4f}) · {res.duration_s:.0f}s · "
                     f"session ${self.spent:.3f}, {saved:.0%} saved[/dim]\n")
        tracking.log({"mode": "repl", "session": self.session.session_id, "cwd": self.session.cwd,
                      "router": self.routing.router_model, "quality_target": self.routing.quality,
                      "rule": self.routing.rule, "epsilon": d.route.epsilon, "lam": self.routing.lam,
                      "dist": d.classification.dist, "length_dist": d.length.dist if d.length else None,
                      "est_cost": d.est_cost, "overflow": d.classification.overflow,
                      "pinned": bool(self.routing.forced_tier), "override": d.rule or None, "tier": d.route.tier, "model": model,
                      "cost": res.cost_usd, "opus_est": est, "input_tokens": res.input_tokens,
                      "output_tokens": res.output_tokens, "tools": len(res.tools), "error": res.error or None})

    # ------------------------------------------------------------------ loop
    def read(self) -> str | None:
        lines = []
        while True:
            try:
                line = input("❯ " if not lines else "… ")
            except EOFError:
                return None
            if line.endswith("\\"):
                lines.append(line[:-1])
                continue
            lines.append(line)
            return "\n".join(lines).strip()

    def loop(self) -> None:
        _setup_history()
        self.banner()
        while True:
            try:
                text = self.read()
            except KeyboardInterrupt:
                self.c.print()
                continue
            if text is None:
                break
            if not text:
                continue
            if text.startswith("/"):
                if not self.command(text):
                    break
                continue
            self.turn(text)
        self.print_cost()
        self.routing.close()


def _setup_history() -> None:
    try:
        import readline
    except ImportError:
        return
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    try:
        readline.read_history_file(HISTORY)
    except OSError:
        pass
    import atexit
    atexit.register(lambda: readline.write_history_file(HISTORY))
