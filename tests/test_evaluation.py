"""Evaluation: deterministic metrics, capability-aware skipping, regression gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from agroai.config.settings import Settings
from agroai.database.database import Database
from agroai.evaluation import (
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_SKIPPED,
    EvalCase,
    EvalSuite,
    EvaluationReport,
    EvaluationRunner,
    EvaluationTarget,
    citation_validity,
    contains_dosage,
    get_suite,
    keyword_hit_rate,
    lexical_grounding,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    regression_gate,
    run_suite,
    safety_compliance,
    smoke_suite,
    tokenize,
)
from agroai.knowledge.documents import Chunk, KnowledgeDocument, RetrievedChunk
from agroai.retrieval.retriever import SQLiteRetriever
from agroai.safety.guardrails import SafetyGuard
from agroai.skills.router import get_skill_router


# ---------------------------------------------------------------------- #
# Metrics
# ---------------------------------------------------------------------- #
def test_retrieval_metrics() -> None:
    assert recall_at_k(["a", "b", "c"], ["a", "c"], 3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["a", "z"], 3) == 0.5
    assert recall_at_k(["a"], [], 3) == 1.0
    assert precision_at_k(["a", "b"], ["a"], 2) == 0.5
    assert precision_at_k([], ["a"]) == 0.0
    assert reciprocal_rank(["x", "b", "a"], ["b"]) == 0.5
    assert reciprocal_rank(["x"], ["b"]) == 0.0
    assert keyword_hit_rate(["Pomidor kasalligi"], ["pomidor"]) == 1.0
    assert keyword_hit_rate(["boshqa"], ["pomidor"]) == 0.0
    assert keyword_hit_rate(["x"], []) == 1.0


def test_citation_validity_is_none_without_markers() -> None:
    assert citation_validity("no markers here", 3) is None
    assert citation_validity("see [1] and [2]", 2) == 1.0
    assert citation_validity("see [1] and [5]", 2) == 0.5


def test_lexical_grounding_is_a_proxy_only() -> None:
    score = lexical_grounding("pomidor bargida dog'lar", ["pomidor bargida dog'lar bor"])
    assert score is not None and score > 0.5
    assert lexical_grounding("", ["x"]) is None
    assert lexical_grounding("salom", []) == 0.0
    assert "the" not in tokenize("the plant", drop_stopwords=True)


def test_safety_compliance_tripwire() -> None:
    ok, _ = safety_compliance(
        requires_sources=True, has_sources=False, answer="Sof tavsiya: mutaxassis bilan maslahatlashing."
    )
    assert ok is True
    bad, reason = safety_compliance(
        requires_sources=True, has_sources=False, answer="10 ml/l fungitsid sepiladi"
    )
    assert bad is False and "dosage" in reason
    assert contains_dosage("2 g/l eritma") is True
    assert contains_dosage("agrotexnik chora") is False
    ok, _ = safety_compliance(requires_sources=False, has_sources=True, answer="10 mg")
    assert ok is True


# ---------------------------------------------------------------------- #
# Suites
# ---------------------------------------------------------------------- #
def test_suite_jsonl_round_trip(tmp_path: Path) -> None:
    suite = EvalSuite(name="custom", version="2", description="d")
    suite.add(
        EvalCase(
            id="c1",
            question="savol",
            kind="retrieval",
            expected_document_ids=("d1",),
            expected_keywords=("pomidor",),
            requires_corpus=True,
        )
    )
    path = tmp_path / "suite.jsonl"
    assert suite.to_jsonl(path) == 1
    loaded = EvalSuite.from_jsonl(path)
    assert loaded.name == "custom" and loaded.version == "2"
    assert loaded.get("c1").requires_corpus is True


def test_unknown_case_kind_is_rejected() -> None:
    with pytest.raises(ValueError):
        EvalCase.from_dict({"id": "x", "question": "q", "kind": "nonsense"})
    with pytest.raises(ValueError):
        EvalCase.from_dict({"id": "", "question": ""})


def test_smoke_suite_shape() -> None:
    suite = get_suite("smoke")
    summary = suite.summary()
    assert summary["cases"] == len(smoke_suite())
    assert summary["kinds"]["safety"] >= 2
    assert summary["requires_corpus"] >= 1
    with pytest.raises(KeyError):
        get_suite("does-not-exist")


# ---------------------------------------------------------------------- #
# Runner
# ---------------------------------------------------------------------- #
def test_runner_skips_what_it_cannot_check(settings: Settings) -> None:
    target = EvaluationTarget()  # no capabilities at all
    report = run_suite("smoke", target)
    assert report.counts(STATUS_SKIPPED) == len(smoke_suite())
    assert report.counts(STATUS_FAILED) == 0
    assert report.passed is True


def test_runner_passes_the_smoke_suite(settings: Settings) -> None:
    target = EvaluationTarget(guard=SafetyGuard(strict=True), router=get_skill_router())
    report = run_suite("smoke", target)
    assert report.counts(STATUS_FAILED) == 0
    assert report.counts(STATUS_PASSED) >= 5
    assert "safety-dosage-uz" in {result.case_id for result in report.results}


def test_runner_detects_a_real_safety_regression() -> None:
    class BlindGuard:
        def classify(self, question: str):
            class Verdict:
                category = type("C", (), {"value": "general"})()
                risk_level = "low"
                requires_sources = False

            return Verdict()

    suite = EvalSuite(name="gate")
    suite.add(
        EvalCase(
            id="dosage",
            question="doza qancha?",
            kind="safety",
            expected_category="dosage",
            expected_risk="high",
            expected_requires_sources=True,
        )
    )
    report = EvaluationRunner(EvaluationTarget(guard=BlindGuard())).run(suite)
    assert report.counts(STATUS_FAILED) == 1


def test_retrieval_case_is_skipped_on_an_empty_corpus(settings: Settings, database: Database) -> None:
    target = EvaluationTarget(retriever=SQLiteRetriever(database), guard=SafetyGuard())
    report = run_suite("smoke", target)
    skipped = {r.case_id for r in report.results if r.status == STATUS_SKIPPED}
    assert "retrieval-tomato" in skipped
    assert any("empty" in note for note in report.notes)


def test_retrieval_case_runs_against_a_populated_corpus(
    settings: Settings, database: Database
) -> None:
    document = KnowledgeDocument(
        id="doc_tomato_1",
        title="Pomidor kasalliklari",
        content="Pomidor bargida sariq dog'lar kasallik belgisidir.",
        source="FAO",
        category="plant_diseases",
    )
    database.upsert_document(document)
    database.add_chunks(
        [
            Chunk(
                id="doc_tomato_1:0",
                document_id=document.id,
                chunk_index=0,
                content=document.content,
            )
        ]
    )
    suite = EvalSuite(name="retrieval")
    suite.add(
        EvalCase(
            id="hit",
            question="pomidor kasalligi dog'lar",
            kind="retrieval",
            expected_document_ids=("doc_tomato_1",),
            expected_keywords=("pomidor",),
            requires_corpus=True,
        )
    )
    report = EvaluationRunner(EvaluationTarget(retriever=SQLiteRetriever(database))).run(suite)
    result = report.results[0]
    assert result.status == STATUS_PASSED
    assert result.metric("recall").value == 1.0
    assert result.retrieved == ["doc_tomato_1"]


# ---------------------------------------------------------------------- #
# Reports and the regression gate
# ---------------------------------------------------------------------- #
def test_report_save_load_and_render(tmp_path: Path) -> None:
    report = run_suite(
        "smoke", EvaluationTarget(guard=SafetyGuard(), router=get_skill_router())
    )
    path = report.save(tmp_path / "report.json")
    loaded = EvaluationReport.load(path)
    assert loaded.suite == "smoke"
    assert loaded.summary["cases"] == report.summary["cases"]
    assert "AgroAI evaluation" in loaded.render()


def _report(metrics: dict[str, float], passed: int = 5) -> EvaluationReport:
    from agroai.evaluation.metrics import MetricValue

    report = EvaluationReport(suite="s")
    for name, value in metrics.items():
        report.results.append(
            EvaluationRunResult(name, value)
        )
    report.results.append(
        __import__("agroai.evaluation", fromlist=["CaseResult"]).CaseResult(
            case_id="c", kind="safety", status=STATUS_PASSED
        )
    )
    return report


def EvaluationRunResult(name: str, value: float):  # noqa: N802 - test helper class name
    from agroai.evaluation import CaseResult
    from agroai.evaluation.metrics import MetricValue

    return CaseResult(
        case_id=f"m-{name}",
        kind="retrieval",
        status=STATUS_PASSED,
        metrics=[MetricValue(name=name, value=value)],
    )


def test_regression_gate_blocks_drops_and_improvements_pass() -> None:
    baseline = _report({"recall": 0.8, "grounding": 0.6})
    worse = _report({"recall": 0.5, "grounding": 0.6})
    verdict = regression_gate(baseline, worse)
    assert verdict.passed is False
    assert any("recall" in item for item in verdict.regressions)

    better = _report({"recall": 0.9, "grounding": 0.7})
    verdict = regression_gate(baseline, better)
    assert verdict.passed is True
    assert len(verdict.improvements) == 2


def test_regression_gate_flags_missing_metrics_and_failed_cases() -> None:
    from agroai.evaluation import CaseResult

    baseline = _report({"recall": 0.8})
    missing = _report({"grounding": 0.9})
    verdict = regression_gate(baseline, missing)
    assert any("recall" in item for item in verdict.regressions)

    failing = EvaluationReport(
        suite="s",
        results=[CaseResult(case_id="x", kind="safety", status=STATUS_FAILED)],
    )
    verdict = regression_gate(baseline, failing)
    assert verdict.passed is False
    assert any("failed" in item for item in verdict.regressions)


def test_regression_gate_requires_case_parity() -> None:
    baseline = run_suite(
        "smoke", EvaluationTarget(guard=SafetyGuard(), router=get_skill_router())
    )
    thin = run_suite("smoke", EvaluationTarget(guard=SafetyGuard()))
    verdict = regression_gate(baseline, thin)
    assert verdict.passed is False
    assert any("coverage" in item for item in verdict.regressions)
    assert "cases ran" in verdict.detail


def test_gate_tolerance_allows_small_fluctuation() -> None:
    baseline = _report({"recall": 0.80})
    slightly_worse = _report({"recall": 0.79})
    assert regression_gate(baseline, slightly_worse, tolerance=0.02).passed is True
    assert regression_gate(baseline, slightly_worse, tolerance=0.001).passed is False


def test_compare_runs(tmp_path: Path) -> None:
    from agroai.evaluation import compare_runs

    target = EvaluationTarget(guard=SafetyGuard(), router=get_skill_router())
    first = run_suite("smoke", target).save(tmp_path / "a.json")
    second = run_suite("smoke", target).save(tmp_path / "b.json")
    summaries = compare_runs([first, second])
    assert len(summaries) == 2
    assert summaries[0]["suite"] == "smoke"
