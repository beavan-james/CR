import pytest

from cheapshot import classifier
from cheapshot.bench import judge, matrix, replay, report
from cheapshot.bench.dataset import load_dataset
from cheapshot.bench.replay import Matrix, PolicyRow
from cheapshot.bench.store import Store

A = classifier.load_variant("variants/a_score.yaml")
C = classifier.load_variant("variants/c_tier_choice.yaml")


# ---------------------------------------------------------------- store / judge parsing

def test_store_resumes(tmp_path):
    s = Store(tmp_path / "x.jsonl")
    s.add({"key": "a", "v": 1})
    s.add({"key": "a", "v": 2})
    s2 = Store(tmp_path / "x.jsonl")
    assert s2.get("a")["v"] == 2 and len(s2.rows()) == 1


def test_parse_verdict_takes_last():
    assert judge.parse_verdict("A mentions VERDICT: A as an example...\nVERDICT: TIE") == "TIE"
    assert judge.parse_verdict("verdict: b") == "B"
    assert judge.parse_verdict("no decision") is None


def test_outcomes_and_combine():
    assert judge.candidate_outcome("A", "candidate_first") == "win"
    assert judge.candidate_outcome("A", "flagship_first") == "loss"
    assert judge.candidate_outcome("TIE", "flagship_first") == "tie"
    assert judge.combine(["win", "win"]) == "win"
    assert judge.combine(["win", "loss"]) == "tie"
    assert judge.combine(["loss", None]) is None


# ---------------------------------------------------------------- replay math

def synthetic() -> Matrix:
    # p1: cheap is fine. p2: needs balanced. p3: needs flagship. p4: cheap fine.
    need = {"p1": "cheap", "p2": "balanced", "p3": "flagship", "p4": "cheap"}
    order = ("cheap", "balanced", "flagship")
    cost = {"cheap": 1.0, "balanced": 4.0, "flagship": 10.0}
    m = Matrix({}, {}, {})
    for pid, n in need.items():
        m.prompts[pid] = {"id": pid, "split": "tune", "category": "x"}
        for t in order:
            m.cost[(pid, t)] = cost[t]
            m.ok[(pid, t)] = order.index(t) >= order.index(n)
    # an over-confident but well-ordered router
    m.policies["laya"] = {
        "p1": PolicyRow({"cheap": 0.9, "balanced": 0.08, "flagship": 0.02}, False, False),
        "p2": PolicyRow({"cheap": 0.1, "balanced": 0.85, "flagship": 0.05}, False, False),
        "p3": PolicyRow({}, True, False),  # overflow
        "p4": PolicyRow({"cheap": 0.7, "balanced": 0.2, "flagship": 0.1}, False, False),
    }
    return m


def test_evaluate_constant_and_oracle():
    m = synthetic()
    ids = m.ids()
    flag = replay.evaluate(m, replay.constant(m, "flagship"), ids)
    assert flag.quality == 1.0 and flag.savings == 0.0
    cheap = replay.evaluate(m, replay.constant(m, "cheap"), ids)
    assert cheap.quality == 0.5 and cheap.savings == pytest.approx(1 - 4 / 40)
    o = replay.evaluate(m, replay.oracle(m), ids)
    assert o.quality == 1.0 and o.cost == 1 + 4 + 10 + 1


def test_assign_risk_dial_and_overflow():
    m = synthetic()
    a = replay.assign(m, "laya", 0.15)
    assert a == {"p1": "cheap", "p2": "balanced", "p3": "flagship", "p4": "balanced"}
    x = replay.evaluate(m, a, m.ids(), "laya")
    assert x.quality == 1.0 and x.overflow_share == 0.25


def test_random_at_mix_expectation():
    m = synthetic()
    r = replay.random_at_mix(m, {"cheap": 1.0, "balanced": 0.0, "flagship": 0.0}, m.ids())
    assert r.quality == replay.evaluate(m, replay.constant(m, "cheap"), m.ids()).quality


def test_choose_eps_respects_floor():
    m = synthetic()
    eps, met = replay.choose_eps(m, "laya", 1.0, m.ids(), floor=1.0)
    assert met and replay.evaluate(m, replay.assign(m, "laya", eps), m.ids()).quality == 1.0


def test_temperature_softens_overconfident_router():
    m = synthetic()
    m.policies["laya"]["p4"] = PolicyRow({"cheap": 0.02, "balanced": 0.96, "flagship": 0.02}, False, False)
    T = replay.fit_temperature(m, "laya", m.ids())
    assert T > 1.0
    assert replay.nll_brier(m, "laya", m.ids(), T)[0] < replay.nll_brier(m, "laya", m.ids(), 1.0)[0]


def test_bootstrap_bounds_contain_point():
    m = synthetic()
    a = replay.assign(m, "laya", 0.15)
    ci = replay.bootstrap(m, a, m.ids(), B=300)
    point = replay.evaluate(m, a, m.ids())
    assert ci["savings"][0] <= point.savings <= ci["savings"][1]


# ---------------------------------------------------------------- end-to-end (mock family, stub Laya)

class StubLaya:
    model = "typed-decisions"

    def classify(self, prompt, variant):
        h = classifier.heuristic_classify(prompt)
        soft = {t: 0.8 * p + 0.2 / 3 for t, p in h.dist.items()}
        return classifier.Classification(soft, 0.6, router_model="stub")


def test_pipeline_end_to_end(tmp_path):
    prompts = load_dataset("evals/datasets/seed.jsonl")
    answers = Store(tmp_path / "answers.jsonl")
    assert matrix.run_answers(prompts, "mock", "", answers, workers=2) == 0
    assert matrix.run_answers(prompts, "mock", "", answers) == 0  # resume: nothing pending
    assert len(answers.rows()) == 3 * len(prompts)

    cls = Store(tmp_path / "classifications.jsonl")
    vs = matrix.policy_variants([A, C], rotations=3)
    assert [v["name"] for v in vs] == ["a_score", "c_tier_choice@r0", "c_tier_choice@r1", "c_tier_choice@r2"]
    assert matrix.run_classify(prompts, "laya", vs, cls, client=StubLaya()) == 4 * len(prompts)
    assert matrix.run_classify(prompts, "heuristic", vs, cls) == len(prompts)

    judgments = Store(tmp_path / "judgments.jsonl")
    failures, skipped = judge.run_judge(prompts, answers, judgments, "mock", "")
    assert failures == 0 and skipped == 0 and len(judgments.rows()) == 4 * len(prompts)

    m = replay.build_matrix(prompts, answers, judgments, cls, "mock", judge.JUDGE_MODEL)
    assert len(m.prompts) == len(prompts) and set(m.policies) >= {"a_score", "heuristic"}
    r = report.build_report(m, floor=0.9, B=200)
    assert r["meta"]["held_out"] and r["baselines"]["oracle"]["quality"] == 1.0
    assert "c_tier_choice" in r["rotations"]
    md = report.render_markdown(r)
    assert "## Policies" in md and "heuristic" in md
