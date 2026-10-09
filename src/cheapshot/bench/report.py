"""Assemble the benchmark report (JSON for tools/UI, Markdown for reading).

Epsilon and calibration are chosen on the tune split; headline numbers come from
the test split. Without both splits the report says so and uses all prompts.
"""
from __future__ import annotations

from collections import defaultdict

from ..classifier import TIERS
from . import replay
from .replay import Matrix


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def build_report(m: Matrix, floor: float, B: int = 2000) -> dict:
    tune, test = m.ids("tune"), m.ids("test")
    held_out = bool(tune and test)
    if not held_out:
        tune = test = m.ids()

    baselines = {f"always-{t}": replay.evaluate(m, replay.constant(m, t), test).to_dict() for t in TIERS}
    baselines["oracle"] = replay.evaluate(m, replay.oracle(m), test).to_dict()

    entries, calibration = [], {}
    for name in sorted(m.policies):
        rows = m.policies[name]
        if all(r.heuristic for r in rows.values()):
            baselines[name] = replay.evaluate(m, replay.assign(m, name, 0.0), test, name).to_dict()
            continue
        T = replay.fit_temperature(m, name, tune)
        nb, bb = replay.nll_brier(m, name, tune, 1.0)
        na, ba = replay.nll_brier(m, name, tune, T)
        calibration[name] = {"T": T, "nll_before": nb, "nll_after": na, "brier_before": bb, "brier_after": ba}
        for calibrated, temp in ((False, 1.0), (True, T)):
            eps, met = replay.choose_eps(m, name, temp, tune, floor)
            a = replay.assign(m, name, eps, temp)
            tm = replay.evaluate(m, a, test, name)
            entries.append({
                "name": name + (" (calibrated)" if calibrated else ""), "policy": name,
                "calibrated": calibrated, "T": temp, "eps": eps, "floor_met_on_tune": met,
                "tune": replay.evaluate(m, a, tune, name).to_dict(), "test": tm.to_dict(),
                "ci": replay.bootstrap(m, a, test, B),
                "random_at_mix": replay.random_at_mix(m, tm.mix, test).to_dict(),
                "curve": [{"eps": e, **{k: v for k, v in replay.evaluate(m, replay.assign(m, name, e, temp), test).to_dict().items()
                                         if k in ("savings", "quality")}} for e in replay.EPS_GRID],
            })

    passing = [e for e in entries if e["floor_met_on_tune"]]
    headline = max(passing, key=lambda e: (e["tune"]["savings"], e["tune"]["quality"])) if passing else None

    by_category = {}
    if headline:
        a = replay.assign(m, headline["policy"], headline["eps"], headline["T"])
        cats = defaultdict(list)
        for pid, p in m.prompts.items():
            cats[p["category"]].append(pid)
        by_category = {c: replay.evaluate(m, a, ids).to_dict() for c, ids in sorted(cats.items())}

    rotations = {}
    groups = defaultdict(list)
    for name in m.policies:
        if "@r" in name:
            groups[name.split("@r")[0]].append(name)
    for base, names in groups.items():
        if len(names) < 2:
            continue
        chosen = {e["policy"]: e for e in entries if not e["calibrated"]}
        assigns = [replay.assign(m, n, chosen[n]["eps"]) for n in names if n in chosen]
        agree = sum(len({a[pid] for a in assigns}) == 1 for pid in m.prompts) / len(m.prompts)
        rotations[base] = {"tier_agreement": agree, "rotations": sorted(names)}

    return {
        "meta": {"n_prompts": len(m.prompts), "n_tune": len(m.ids("tune")), "n_test": len(m.ids("test")),
                 "held_out": held_out, "floor": floor, **m.stats},
        "headline": headline["name"] if headline else None,
        "baselines": baselines, "policies": entries, "calibration": calibration,
        "by_category": by_category, "rotations": rotations,
    }


def render_markdown(r: dict) -> str:
    meta = r["meta"]
    split = "test split" if meta["held_out"] else "all prompts (no held-out split!)"
    out = ["# Cheapshot benchmark report", ""]
    head = next((e for e in r["policies"] if e["name"] == r["headline"]), None)
    if head:
        t, ci, rnd = head["test"], head["ci"], head["random_at_mix"]
        out += [f"**At ≥{_pct(meta['floor'])} non-inferior to the flagship, `{head['name']}` saves "
                f"{_pct(t['savings'])}** (95% CI {_pct(ci['savings'][0])}–{_pct(ci['savings'][1])}) on the {split}; "
                f"quality {_pct(t['quality'])} (CI {_pct(ci['quality'][0])}–{_pct(ci['quality'][1])}), ε={head['eps']}.",
                "",
                f"Random routing with the same tier mix: savings {_pct(rnd['savings'])}, quality {_pct(rnd['quality'])}.",
                ""]
    elif not r["policies"]:
        out += ["**No router classifications yet.** Run `cheapshot bench classify --backend laya` once Laya is up.", ""]
    else:
        out += [f"**No policy met the {_pct(meta['floor'])} quality floor on the tune split.**", ""]

    out += ["## Policies", "", "| policy | ε | T | test savings | test quality | mix c/b/f | overflow | tune floor met |",
            "|---|---|---|---|---|---|---|---|"]
    for e in r["policies"]:
        t = e["test"]
        mix = "/".join(_pct(t["mix"].get(k, 0)) for k in TIERS)
        out.append(f"| {e['name']} | {e['eps']} | {e['T']:.2f} | {_pct(t['savings'])} | {_pct(t['quality'])} | "
                   f"{mix} | {_pct(t['overflow_share'])} | {'yes' if e['floor_met_on_tune'] else 'no'} |")
    out += ["", "## Baselines", "", "| baseline | savings | quality |", "|---|---|---|"]
    for name, b in r["baselines"].items():
        out.append(f"| {name} | {_pct(b['savings'])} | {_pct(b['quality'])} |")

    if r["calibration"]:
        out += ["", "## Calibration (tune split, label = oracle tier)", "",
                "| policy | T | NLL before → after | Brier before → after |", "|---|---|---|---|"]
        for name, c in r["calibration"].items():
            out.append(f"| {name} | {c['T']:.2f} | {c['nll_before']:.3f} → {c['nll_after']:.3f} | "
                       f"{c['brier_before']:.3f} → {c['brier_after']:.3f} |")
    if r["by_category"]:
        out += ["", f"## By category (`{r['headline']}`, all prompts)", "", "| category | n | savings | quality |",
                "|---|---|---|---|"]
        for c, x in r["by_category"].items():
            out.append(f"| {c} | {x['n']} | {_pct(x['savings'])} | {_pct(x['quality'])} |")
    if r["rotations"]:
        out += ["", "## Option-order sensitivity", ""]
        for base, x in r["rotations"].items():
            out.append(f"- `{base}`: routed tier identical across {len(x['rotations'])} orders on {_pct(x['tier_agreement'])} of prompts")

    out += ["", "## Run notes", "",
            f"- cost basis: {meta.get('cost_basis', 'api')}",
            f"- prompts: {meta['n_prompts']} scored (tune {meta['n_tune']}, test {meta['n_test']}); "
            f"{len(meta['dropped'])} dropped for missing answers/judgments",
            f"- judge order disagreement: {meta['judge_order_disagreement']}",
            f"- truncated answers: {meta['truncated_answers']}, refused: {meta['refused_answers']}"]
    if meta["incomplete_policies"]:
        out.append(f"- skipped incomplete policies: {', '.join(meta['incomplete_policies'])}")
    return "\n".join(out) + "\n"
