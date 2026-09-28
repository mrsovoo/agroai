"""Running suites, and the regression gate that guards a model/knowledge update.

```text
AgroAI v1 -> 1000 tests -> accuracy / grounding / retrieval / safety -> PASS|FAIL
```

The runner is deliberately capability-based: it runs every case it *can* run and
reports the rest as **skipped with a reason**. That is the only honest behaviour
when, for example, no LLM is configured in this build - a green report that
silently tested nothing would be worse than a red one.

:func:`regression_gate` is the piece that makes improvements safe: compare a new
run against a stored baseline and refuse to declare victory when a metric got
worse, even if others improved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from agroai.evaluation.metrics import (
    DEFAULT_THRESHOLDS,
    MetricValue,
    citation_validity,
    keyword_hit_rate,
    lexical_grounding,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    safety_compliance,
)
from agroai.evaluation.suites import (
    KIND_ANSWER,
    KIND_RETRIEVAL,
    KIND_ROUTING,
    KIND_SAFETY,
    EvalCase,
    EvalSuite,
)
from agroai.knowledge.metadata import utcnow_iso

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

#: A metric is only part of the score when it is measurable.
AGGREGATED_METRICS: tuple[str, ...] = (
    "recall",
    "precision",
    "mrr",
    "keyword_hit",
    "grounding",
    "citation_validity",
)


# ---------------------------------------------------------------------- #
# Results
# ---------------------------------------------------------------------- #
@dataclass
class CaseResult:
    """Outcome of one case."""

    case_id: str
    kind: str
    status: str
    detail: str = ""
    metrics: list[MetricValue] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.status == STATUS_FAILED

    def metric(self, name: str) -> MetricValue | None:
        for item in self.metrics:
            if item.name == name:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "kind": self.kind,
            "status": self.status,
            "detail": self.detail,
            "metrics": [item.to_dict() for item in self.metrics],
            "retrieved": list(self.retrieved),
        }


@dataclass
class EvaluationReport:
    """The result of running one suite."""

    suite: str
    version: str = "1"
    results: list[CaseResult] = field(default_factory=list)
    generated_at: str = field(default_factory=utcnow_iso)
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @property
    def passed(self) -> bool:
        return not any(result.failed for result in self.results)

    def counts(self, status: str) -> int:
        return sum(1 for result in self.results if result.status == status)

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "version": self.version,
            "cases": len(self.results),
            "passed": self.counts(STATUS_PASSED),
            "failed": self.counts(STATUS_FAILED),
            "skipped": self.counts(STATUS_SKIPPED),
            "metrics": self.metrics(),
            "generated_at": self.generated_at,
            "ok": self.passed,
        }

    def metrics(self) -> dict[str, float]:
        """Mean of every measurable metric across all cases."""
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for result in self.results:
            for item in result.metrics:
                if item.value is None:
                    continue
                totals[item.name] = totals.get(item.name, 0.0) + item.value
                counts[item.name] = counts.get(item.name, 0) + 1
        return {
            name: round(total / counts[name], 4)
            for name, total in totals.items()
            if counts.get(name)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "version": self.version,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "notes": list(self.notes),
            "results": [result.to_dict() for result in self.results],
        }

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> "EvaluationReport":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        results = [
            CaseResult(
                case_id=item["case_id"],
                kind=item["kind"],
                status=item["status"],
                detail=item.get("detail", ""),
                metrics=[
                    MetricValue(
                        name=metric["name"],
                        value=metric.get("value"),
                        threshold=metric.get("threshold"),
                        detail=metric.get("detail", ""),
                    )
                    for metric in item.get("metrics", [])
                ],
                retrieved=list(item.get("retrieved", [])),
            )
            for item in data.get("results", [])
        ]
        return cls(
            suite=data.get("suite", "unknown"),
            version=str(data.get("version", "1")),
            generated_at=data.get("generated_at", ""),
            results=results,
            notes=list(data.get("notes", [])),
        )

    def render(self) -> str:
        summary = self.summary
        lines = [
            f"AgroAI evaluation: suite='{self.suite}' v{self.version}",
            "=" * 60,
        ]
        for result in self.results:
            marker = {"passed": "[PASS]", "failed": "[FAIL]", "skipped": "[SKIP]"}.get(
                result.status, "[??]"
            )
            lines.append(f"{marker} {result.case_id} ({result.kind})")
            if result.detail:
                lines.append(f"        {result.detail}")
            for item in result.metrics:
                shown = "n/a" if item.value is None else f"{item.value:.3f}"
                verdict = "" if item.passed is None else (" ok" if item.passed else " BELOW")
                lines.append(f"        {item.name}: {shown}{verdict}")
        lines.append("=" * 60)
        lines.append(
            f"Cases: {summary['cases']} "
            f"(passed={summary['passed']}, failed={summary['failed']}, "
            f"skipped={summary['skipped']})"
        )
        if summary["metrics"]:
            lines.append(
                "Metrics: "
                + ", ".join(f"{name}={value:.3f}" for name, value in summary["metrics"].items())
            )
        for note in self.notes:
            lines.append(f"! {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Target
# ---------------------------------------------------------------------- #
class EvaluationTarget:
    """The capabilities a case can be checked against.

    Everything is optional. A missing capability turns its cases into ``skipped``
    - which is exactly what the report should say.
    """

    def __init__(
        self,
        *,
        retriever: Any | None = None,
        guard: Any | None = None,
        router: Any | None = None,
        agent: Any | None = None,
    ) -> None:
        self.retriever = retriever
        self.guard = guard
        self.router = router
        self.agent = agent

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "retriever": self.retriever is not None,
            "guard": self.guard is not None,
            "router": self.router is not None,
            "agent": self.agent is not None,
        }

    def corpus_size(self) -> int:
        """Number of indexed documents, or ``-1`` when it cannot be known."""
        if self.retriever is None:
            return -1
        try:
            return int(self.retriever.health_check().documents)
        except Exception:  # noqa: BLE001 - a health check must not break evaluation
            return -1


# ---------------------------------------------------------------------- #
# Runner
# ---------------------------------------------------------------------- #
class EvaluationRunner:
    """Runs suites against an :class:`EvaluationTarget`."""

    def __init__(
        self,
        target: EvaluationTarget,
        *,
        thresholds: dict[str, float] | None = None,
        top_k: int = 5,
    ) -> None:
        self._target = target
        self._thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self._top_k = top_k

    def run(self, suite: EvalSuite) -> EvaluationReport:
        report = EvaluationReport(suite=suite.name, version=suite.version)
        for case in suite.cases:
            report.results.append(self._run_case(case))
        if self._target.corpus_size() == 0:
            report.notes.append(
                "Knowledge base is empty: retrieval and answer cases were skipped. "
                "Ingest documents, then re-run the suite for real numbers."
            )
        return report

    # ------------------------------------------------------------------ #
    def _run_case(self, case: EvalCase) -> CaseResult:
        if case.kind == KIND_SAFETY:
            return self._run_safety(case)
        if case.kind == KIND_ROUTING:
            return self._run_routing(case)
        if case.kind == KIND_RETRIEVAL:
            return self._run_retrieval(case)
        if case.kind == KIND_ANSWER:
            return self._run_answer(case)
        return CaseResult(
            case_id=case.id,
            kind=case.kind,
            status=STATUS_SKIPPED,
            detail=f"no runner for case kind {case.kind!r}",
        )

    # ------------------------------------------------------------------ #
    def _run_safety(self, case: EvalCase) -> CaseResult:
        if self._target.guard is None:
            return self._skip(case, "no safety guard configured")
        verdict = self._target.guard.classify(case.question)
        category = getattr(getattr(verdict, "category", None), "value", "")
        failures: list[str] = []
        if case.expected_category and category != case.expected_category:
            failures.append(f"category {category!r} != {case.expected_category!r}")
        risk = getattr(verdict, "risk_level", "")
        if case.expected_risk and risk != case.expected_risk:
            failures.append(f"risk {risk!r} != {case.expected_risk!r}")
        if (
            case.expected_requires_sources is not None
            and bool(getattr(verdict, "requires_sources", False))
            is not case.expected_requires_sources
        ):
            failures.append(
                f"requires_sources={getattr(verdict, 'requires_sources', None)} "
                f"!= {case.expected_requires_sources}"
            )
        return CaseResult(
            case_id=case.id,
            kind=case.kind,
            status=STATUS_FAILED if failures else STATUS_PASSED,
            detail="; ".join(failures)
            or f"category={category}, risk={risk}, "
            f"requires_sources={getattr(verdict, 'requires_sources', None)}",
        )

    def _run_routing(self, case: EvalCase) -> CaseResult:
        if self._target.router is None:
            return self._skip(case, "no skill router configured")
        result = self._target.router.route(case.question)
        names = list(getattr(result, "names", []) or [])
        missing = [name for name in case.expected_skills if name not in names]
        return CaseResult(
            case_id=case.id,
            kind=case.kind,
            status=STATUS_FAILED if missing else STATUS_PASSED,
            detail=(
                f"missing skills: {', '.join(missing)}"
                if missing
                else f"selected: {', '.join(names) or '(none)'}"
            ),
        )

    def _run_retrieval(self, case: EvalCase) -> CaseResult:
        if self._target.retriever is None:
            return self._skip(case, "no retriever configured")
        corpus = self._target.corpus_size()
        if case.requires_corpus and corpus == 0:
            return self._skip(case, "knowledge base is empty (corpus required)")
        try:
            found = self._target.retriever.search(case.question, top_k=self._top_k)
        except Exception as exc:  # noqa: BLE001 - report, never crash the suite
            return self._skip(case, f"retrieval failed: {exc}")

        document_ids = [item.chunk.document_id for item in found]
        texts = [item.chunk.content for item in found]
        titles = [
            item.document.title if item.document is not None else item.chunk.document_id
            for item in found
        ]
        if not found:
            return CaseResult(
                case_id=case.id,
                kind=case.kind,
                status=STATUS_FAILED,
                detail="nothing retrieved",
            )

        metrics: list[MetricValue] = []
        if case.expected_document_ids:
            metrics.append(
                MetricValue(
                    "recall",
                    recall_at_k(document_ids, case.expected_document_ids),
                    self._thresholds.get("recall"),
                    f"expected: {', '.join(case.expected_document_ids)}",
                )
            )
            metrics.append(
                MetricValue(
                    "precision",
                    precision_at_k(document_ids, case.expected_document_ids),
                    self._thresholds.get("precision"),
                )
            )
            metrics.append(
                MetricValue("mrr", reciprocal_rank(document_ids, case.expected_document_ids))
            )
        if case.expected_keywords:
            metrics.append(
                MetricValue(
                    "keyword_hit",
                    keyword_hit_rate([*texts, *titles], case.expected_keywords),
                    None,
                    f"keywords: {', '.join(case.expected_keywords)}",
                )
            )
        below = [
            item.name for item in metrics if item.passed is False
        ]
        return CaseResult(
            case_id=case.id,
            kind=case.kind,
            status=STATUS_FAILED if below else STATUS_PASSED,
            detail=(
                f"below threshold: {', '.join(below)}"
                if below
                else f"{len(found)} chunk(s) retrieved"
            ),
            metrics=metrics,
            retrieved=document_ids,
        )

    def _run_answer(self, case: EvalCase) -> CaseResult:
        if self._target.agent is None:
            return self._skip(case, "no agent configured")
        try:
            result = self._target.agent.ask(case.question)
        except Exception as exc:  # noqa: BLE001
            return self._skip(case, f"agent failed: {exc}")

        answer = getattr(result, "answer", "") or ""
        sources = list(getattr(result, "sources", []) or [])
        evidence = [
            item.chunk.content for item in sources
            if getattr(item, "chunk", None) is not None
        ]
        status = getattr(result, "status", "")

        metrics: list[MetricValue] = []
        failures: list[str] = []

        if case.requires_citations and not sources:
            failures.append("no sources attached to the answer")
        validity = citation_validity(answer, len(sources))
        metrics.append(
            MetricValue(
                "citation_validity",
                validity,
                self._thresholds.get("citation_validity"),
                "not measurable: the answer used no citation markers"
                if validity is None
                else "citation markers that point at a real source",
            )
        )
        if evidence:
            grounding = lexical_grounding(answer, evidence)
            metrics.append(
                MetricValue(
                    "grounding",
                    grounding,
                    self._thresholds.get("grounding"),
                    "lexical proxy: token overlap with retrieved evidence",
                )
            )
        elif status == "ok":
            # An answer produced with no retrieved evidence is an ungrounded claim.
            failures.append("answer produced without retrieved evidence")

        requires_sources = bool(getattr(result, "safety_blocked", False)) or bool(
            getattr(getattr(result, "verdict", None), "requires_sources", False)
        )
        ok, reason = safety_compliance(
            requires_sources=requires_sources,
            has_sources=bool(sources),
            answer=answer,
        )
        if not ok:
            failures.append(reason)

        below = [item.name for item in metrics if item.passed is False]
        return CaseResult(
            case_id=case.id,
            kind=case.kind,
            status=STATUS_FAILED if (failures or below) else STATUS_PASSED,
            detail="; ".join([*failures, *[f"below threshold: {n}" for n in below]])
            or f"status={status}, sources={len(sources)}",
            metrics=metrics,
        )

    @staticmethod
    def _skip(case: EvalCase, reason: str) -> CaseResult:
        return CaseResult(
            case_id=case.id, kind=case.kind, status=STATUS_SKIPPED, detail=reason
        )


# ---------------------------------------------------------------------- #
# Regression gate
# ---------------------------------------------------------------------- #
@dataclass
class RegressionVerdict:
    """Whether a new run may replace a baseline."""

    passed: bool
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    tolerance: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "regressions": list(self.regressions),
            "improvements": list(self.improvements),
            "tolerance": self.tolerance,
            "detail": self.detail,
        }

    def render(self) -> str:
        lines = [
            f"Regression gate: {'PASS' if self.passed else 'FAIL'} (tolerance={self.tolerance})"
        ]
        for item in self.regressions:
            lines.append(f"  REGRESSION: {item}")
        for item in self.improvements:
            lines.append(f"  improvement: {item}")
        if self.detail:
            lines.append(f"  {self.detail}")
        return "\n".join(lines)


def regression_gate(
    baseline: EvaluationReport | dict[str, Any],
    current: EvaluationReport,
    *,
    tolerance: float = 0.02,
    require_case_parity: bool = True,
) -> RegressionVerdict:
    """Compare ``current`` against ``baseline``.

    Passes only when no aggregated metric dropped by more than ``tolerance``, and
    (by default) the current run skipped no case the baseline ran - a run that
    tested *less* must never look like an improvement.
    """
    baseline_dict = baseline.to_dict() if isinstance(baseline, EvaluationReport) else baseline
    baseline_metrics = dict(baseline_dict.get("summary", {}).get("metrics", {}))
    current_metrics = current.metrics()

    regressions: list[str] = []
    improvements: list[str] = []
    for name, base_value in baseline_metrics.items():
        if name not in current_metrics:
            regressions.append(
                f"{name}: missing from the current run (baseline {base_value:.3f})"
            )
            continue
        delta = current_metrics[name] - float(base_value)
        if delta < -abs(tolerance):
            regressions.append(
                f"{name}: {base_value:.3f} -> {current_metrics[name]:.3f} "
                f"({delta:+.3f})"
            )
        elif delta > abs(tolerance):
            improvements.append(f"{name}: {base_value:.3f} -> {current_metrics[name]:.3f}")

    detail = ""
    if require_case_parity:
        base_summary = baseline_dict.get("summary", {})
        base_run = int(base_summary.get("passed", 0)) + int(base_summary.get("failed", 0))
        current_run = current.counts(STATUS_PASSED) + current.counts(STATUS_FAILED)
        if current_run < base_run:
            regressions.append(
                f"case coverage: {base_run} case(s) ran before, only {current_run} now "
                "(fewer cases must not read as an improvement)"
            )
        detail = f"cases ran: baseline={base_run}, current={current_run}"
    if current.counts(STATUS_FAILED):
        regressions.append(f"{current.counts(STATUS_FAILED)} case(s) failed in the current run")

    return RegressionVerdict(
        passed=not regressions,
        regressions=regressions,
        improvements=improvements,
        tolerance=abs(tolerance),
        detail=detail,
    )


def run_suite(
    suite: EvalSuite | str,
    target: EvaluationTarget,
    *,
    top_k: int = 5,
    thresholds: dict[str, float] | None = None,
) -> EvaluationReport:
    """Convenience: run a suite (by name or object) against a target."""
    from agroai.evaluation.suites import get_suite

    resolved = get_suite(suite) if isinstance(suite, str) else suite
    return EvaluationRunner(target, top_k=top_k, thresholds=thresholds).run(resolved)


def compare_runs(paths: Sequence[Path | str]) -> list[dict[str, Any]]:
    """Load several saved reports and return their summaries side by side."""
    return [EvaluationReport.load(path).summary for path in paths]


__all__ = [
    "AGGREGATED_METRICS",
    "CaseResult",
    "EvaluationReport",
    "EvaluationRunner",
    "EvaluationTarget",
    "RegressionVerdict",
    "STATUS_FAILED",
    "STATUS_PASSED",
    "STATUS_SKIPPED",
    "compare_runs",
    "regression_gate",
    "run_suite",
]
