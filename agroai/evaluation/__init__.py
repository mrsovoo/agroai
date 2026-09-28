"""Evaluation: the framework that decides whether AgroAI is getting better.

```text
AgroAI v1 -> 1000 tests -> accuracy / grounding / retrieval / safety -> PASS|FAIL
```

Usage:

```python
from agroai.evaluation import EvaluationTarget, run_suite, regression_gate

target = EvaluationTarget(retriever=retriever, guard=guard, router=router)
report = run_suite("smoke", target)
print(report.render())

verdict = regression_gate(baseline_report, report)
```

Design rules:

* **capability-based** - a missing component turns its cases into ``skipped``
  with a reason; it never fabricates a pass;
* **deterministic** - every metric is recomputable from stored artifacts;
* **no invented scores** - there is no heuristic "hallucination score"; lexical
  grounding is labelled a *proxy* everywhere it appears;
* **a gate, not a report** - :func:`regression_gate` blocks a change that lowers
  a metric or tests fewer cases than the baseline.
"""

from __future__ import annotations

from agroai.evaluation.metrics import (
    DEFAULT_THRESHOLDS,
    MetricValue,
    citation_markers,
    citation_validity,
    contains_dosage,
    keyword_hit_rate,
    lexical_grounding,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    safety_compliance,
    tokenize,
)
from agroai.evaluation.runner import (
    CaseResult,
    EvaluationReport,
    EvaluationRunner,
    EvaluationTarget,
    RegressionVerdict,
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_SKIPPED,
    compare_runs,
    regression_gate,
    run_suite,
)
from agroai.evaluation.suites import (
    CASE_KINDS,
    KIND_ANSWER,
    KIND_RETRIEVAL,
    KIND_ROUTING,
    KIND_SAFETY,
    EvalCase,
    EvalSuite,
    default_suites,
    get_suite,
    load_suites,
    smoke_suite,
)

__all__ = [
    "CASE_KINDS",
    "CaseResult",
    "DEFAULT_THRESHOLDS",
    "EvalCase",
    "EvalSuite",
    "EvaluationReport",
    "EvaluationRunner",
    "EvaluationTarget",
    "KIND_ANSWER",
    "KIND_RETRIEVAL",
    "KIND_ROUTING",
    "KIND_SAFETY",
    "MetricValue",
    "RegressionVerdict",
    "STATUS_FAILED",
    "STATUS_PASSED",
    "STATUS_SKIPPED",
    "citation_markers",
    "citation_validity",
    "compare_runs",
    "contains_dosage",
    "default_suites",
    "get_suite",
    "keyword_hit_rate",
    "lexical_grounding",
    "load_suites",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "regression_gate",
    "run_suite",
    "safety_compliance",
    "smoke_suite",
    "tokenize",
]
