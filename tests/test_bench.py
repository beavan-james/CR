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
    assert matrix.run_answers(prompts, "mock", "", answers, workers=2).failures == 0
    assert matrix.run_answers(prompts, "mock", "", answers).done == 0  # resume: nothing pending
    assert len(answers.rows()) == 3 * len(prompts)

    cls = Store(tmp_path / "classifications.jsonl")
    vs = matrix.policy_variants([A, C], rotations=3)
    assert [v["name"] for v in vs] == ["a_score", "c_tier_choice@r0", "c_tier_choice@r1", "c_tier_choice@r2"]
    assert matrix.run_classify(prompts, "laya", vs, cls, client=StubLaya()) == 4 * len(prompts)
    assert matrix.run_classify(prompts, "heuristic", vs, cls) == len(prompts)

    judgments = Store(tmp_path / "judgments.jsonl")
    res, skipped = judge.run_judge(prompts, answers, judgments, "mock", "")
    assert res.failures == 0 and skipped == 0 and len(judgments.rows()) == 4 * len(prompts)

    m = replay.build_matrix(prompts, answers, judgments, cls, "mock", judge.JUDGE_MODEL)
    assert len(m.prompts) == len(prompts) and set(m.policies) >= {"a_score", "heuristic"}
    r = report.build_report(m, floor=0.9, B=200)
    assert r["meta"]["held_out"] and r["baselines"]["oracle"]["quality"] == 1.0
    assert "c_tier_choice" in r["rotations"]
    md = report.render_markdown(r)
    assert "## Policies" in md and "heuristic" in md


# ---------------------------------------------------------------- teacher + fine-tuning export

from cheapshot.bench import export, teacher


def test_parse_teacher():
    d = teacher.parse_teacher('Thinking... {"cheap": 2, "balanced": 1, "flagship": 1}')
    assert d == pytest.approx({"cheap": 0.5, "balanced": 0.25, "flagship": 0.25})
    assert teacher.parse_teacher('{"cheap": 0.9}') is None
    assert teacher.parse_teacher("no json") is None
    # last valid object wins
    assert teacher.parse_teacher('{"cheap":1,"balanced":0,"flagship":0} then {"cheap":0,"balanced":0,"flagship":1}')["flagship"] == 1.0


def test_tier_gold_to_levels():
    g = export.tier_gold_to_levels({"cheap": 0.5, "balanced": 0.5, "flagship": 0.0}, A["tier_map"])
    assert g == {"0": 0.5, "1": 0.5, "2": 0.0}


def test_export_prefers_judge_and_skips_non_train(tmp_path):
    prompts = [
        {"id": "j", "prompt": "What is 2+2?", "split": "train", "ideal_tier": "cheap"},
        {"id": "t", "prompt": "Prove Fermat's last theorem.", "split": "train", "ideal_tier": "flagship"},
        {"id": "x", "prompt": "Unlabeled prompt", "split": "train"},
        {"id": "h", "prompt": "Held-out prompt", "split": "test", "ideal_tier": "cheap"},
    ]
    answers, judgments, tstore = (Store(tmp_path / f) for f in ("a.jsonl", "j.jsonl", "t.jsonl"))
    matrix.run_answers(prompts[:1] + prompts[3:], "mock", "", answers)
    judge.run_judge(prompts[:1] + prompts[3:], answers, judgments, "mock", "")
    teacher.run_teacher(prompts[:2], tstore, "mock", "")
    rows, counts = export.build_training_rows(prompts, A, answers, judgments, tstore, "mock",
                                              judge.JUDGE_MODEL, teacher.TEACHER_MODEL)
    assert counts == {"judge": 1, "teacher": 1, "unlabeled": 1, "not_train_split": 1}
    by_id = {r["id"]: r for r in rows}
    assert by_id["j"]["label_source"] == "judge"
    assert by_id["j"]["gold"]["complexity"]["probabilities"] == {"0": 1.0, "1": 0.0, "2": 0.0}
    assert by_id["t"]["gold"]["complexity"]["probabilities"]["2"] == pytest.approx(0.8)
    assert set(by_id["t"]) >= {"state", "questions", "gold"} and "h" not in by_id


def test_judge_disagreement_splits_gold(tmp_path):
    p = {"id": "d", "prompt": "q", "split": "train"}
    answers, judgments = Store(tmp_path / "a.jsonl"), Store(tmp_path / "j.jsonl")
    matrix.run_answers([p], "mock", "", answers)
    a = {t: matrix.answer_lookup(answers, "mock", "d", "q", t) for t in ("cheap", "balanced", "flagship")}
    outcomes = {("cheap", "candidate_first"): "tie", ("cheap", "flagship_first"): "loss",
                ("balanced", "candidate_first"): "tie", ("balanced", "flagship_first"): "tie"}
    for (tier, order), out in outcomes.items():
        judgments.add({"key": judge.judgment_key("d", tier, order, a[tier]["key"], a["flagship"]["key"], judge.JUDGE_MODEL),
                       "outcome": out})
    g = export.judge_gold(answers, judgments, "mock", judge.JUDGE_MODEL, p)
    assert g == {"cheap": 0.5, "balanced": 0.5, "flagship": 0.0}


def test_report_ignores_train_split(tmp_path):
    prompts = load_dataset("evals/datasets/seed.jsonl")
    prompts[0] = {**prompts[0], "split": "train"}
    answers, judgments, cls = (Store(tmp_path / f) for f in ("a.jsonl", "j.jsonl", "c.jsonl"))
    matrix.run_answers(prompts, "mock", "", answers)
    judge.run_judge(prompts, answers, judgments, "mock", "")
    m = replay.build_matrix(prompts, answers, judgments, cls, "mock", judge.JUDGE_MODEL)
    assert prompts[0]["id"] not in m.prompts and len(m.prompts) == len(prompts) - 1


def test_unicode_line_separator_survives_jsonl(tmp_path):
    import json as _json
    p = tmp_path / "d.jsonl"
    p.write_text(_json.dumps({"id": "u", "prompt": "a b", "split": "test"}, ensure_ascii=False) + "\n")
    assert load_dataset(p)[0]["prompt"] == "a b"
    s = Store(tmp_path / "s.jsonl")
    s.add({"key": "k", "text": "x y"})
    assert Store(tmp_path / "s.jsonl").get("k")["text"] == "x y"


def test_pool_stops_after_consecutive_failures():
    from cheapshot.bench.pool import run_pool

    def fn(i):
        raise RuntimeError("usage limit reached")

    res = run_pool([(i,) for i in range(50)], fn, workers=1, max_consecutive_failures=3)
    assert res.failures == 3 and res.not_run == 47 and "usage limit" in res.stop_reason


def test_pool_success_resets_streak():
    from cheapshot.bench.pool import run_pool

    def fn(i):
        if i % 2:
            raise RuntimeError("flaky")
        return i

    res = run_pool([(i,) for i in range(10)], fn, workers=1, max_consecutive_failures=2)
    assert res.done == 5 and res.failures == 5 and res.stop_reason is None
