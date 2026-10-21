"""A routed Claude Code conversation: each turn runs `claude -p` with the model the router
picked, resuming the same session so context carries across model switches.

Claude Code does the agent work (reading and editing files, running commands) in the
user's working directory, under the user's own Pro/Max login. CR only chooses
the model per turn and accounts for cost.
"""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Callable

from .providers import _API_AUTH_ENV, MODEL_EFFORT

PERMISSION_MODES = ("acceptEdits", "plan", "default", "bypassPermissions")


@dataclass
class TurnResult:
    text: str = ""
    model: str = ""
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_s: float = 0.0
    is_error: bool = False
    error: str = ""
    tools: list[str] = field(default_factory=list)
    denied: list[str] = field(default_factory=list)


def turn_command(model: str, session_id: str, first: bool, permission_mode: str,
                 effort: str | None = None) -> list[str]:
    cmd = ["claude", "-p", "--output-format", "stream-json", "--verbose", "--include-partial-messages",
           "--model", model, "--permission-mode", permission_mode]
    cmd += ["--session-id", session_id] if first else ["--resume", session_id]
    if effort:
        cmd += ["--effort", effort]
    return cmd


def _tool_label(block: dict, cwd: str = "") -> str:
    name, inp = block.get("name", "tool"), block.get("input") or {}
    detail = inp.get("command") or inp.get("file_path") or inp.get("pattern") or inp.get("url") or inp.get("description") or ""
    detail = str(detail).replace("\n", " ")
    if cwd and detail.startswith(cwd.rstrip("/") + "/"):
        detail = detail[len(cwd.rstrip("/")) + 1:]
    return f"{name}({detail[:80]}{'…' if len(detail) > 80 else ''})" if detail else name


def handle_event(e: dict, res: TurnResult, on_text: Callable[[str], None], on_tool: Callable[[str], None],
                 cwd: str = "") -> None:
    """Fold one stream-json event into the turn result, calling back for live output."""
    t = e.get("type")
    if t == "stream_event":
        ev = e.get("event") or {}
        if ev.get("type") == "content_block_delta" and (ev.get("delta") or {}).get("type") == "text_delta":
            chunk = ev["delta"].get("text", "")
            res.text += chunk
            on_text(chunk)
    elif t == "assistant":
        msg = e.get("message") or {}
        res.model = msg.get("model") or res.model
        for block in msg.get("content") or []:
            if block.get("type") == "tool_use":
                label = _tool_label(block, cwd)
                res.tools.append(label)
                on_tool(label)
    elif t == "result":
        usage = e.get("usage") or {}
        res.input_tokens = sum(int(usage.get(k) or 0) for k in
                               ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        res.output_tokens = int(usage.get("output_tokens") or 0)
        res.cost_usd = float(e.get("total_cost_usd") or 0.0)
        res.duration_s = float(e.get("duration_ms") or 0) / 1000
        res.denied = [d.get("tool_name", "tool") for d in e.get("permission_denials") or []]
        if e.get("is_error") or e.get("subtype") not in (None, "success"):
            res.is_error = True
            res.error = str(e.get("result") or e.get("subtype") or "claude -p failed")


class ClaudeCodeSession:
    def __init__(self, cwd: str | None = None, permission_mode: str = "acceptEdits"):
        self.cwd = cwd or os.getcwd()
        self.permission_mode = permission_mode
        self.session_id = str(uuid.uuid4())
        self.turns = 0
        self.reported_total = 0.0  # Claude Code's total_cost_usd is cumulative over a resumed session

    def reset(self) -> None:
        self.session_id = str(uuid.uuid4())
        self.turns = 0
        self.reported_total = 0.0

    def run_turn(self, prompt: str, model: str, on_text: Callable[[str], None] = lambda s: None,
                 on_tool: Callable[[str], None] = lambda s: None, effort: str | None = None) -> TurnResult:
        env = {k: v for k, v in os.environ.items() if k not in _API_AUTH_ENV}  # bill the subscription, never an API key
        cmd = turn_command(model, self.session_id, self.turns == 0, self.permission_mode,
                           effort or MODEL_EFFORT.get(model))
        res = TurnResult(model=model)
        proc = subprocess.Popen(cmd, cwd=self.cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, bufsize=1)
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    handle_event(json.loads(line), res, on_text, on_tool, self.cwd)
                except json.JSONDecodeError:
                    continue
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            res.is_error, res.error = True, "interrupted"
        if proc.returncode not in (0, None) and not res.is_error:
            res.is_error, res.error = True, (proc.stderr.read() or f"claude exited {proc.returncode}").strip()[:500]
        if not res.is_error or res.text:
            self.turns += 1  # the session exists on disk once a turn produced output
        res.cost_usd = self.turn_cost(res.cost_usd)
        return res

    def turn_cost(self, reported: float) -> float:
        """This turn's cost from the session's running total (0 when the turn reported none)."""
        if not reported:
            return 0.0
        if reported < self.reported_total:  # counter restarted; take the report as this turn's cost
            self.reported_total = 0.0
        cost, self.reported_total = reported - self.reported_total, reported
        return cost
