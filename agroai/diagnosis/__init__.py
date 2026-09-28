"""Differential diagnosis.

The roadmap's most under-appreciated requirement:

```text
tomato leaf spots -> 5 possible causes -> symptom comparison -> environment
comparison -> image evidence -> knowledge evidence -> confidence -> best matches
```

"the disease is X" is not an acceptable answer shape in agriculture or
veterinary medicine: several causes look identical, and the cost of being wrong
is a lost harvest or a poisoned animal. AgroAI therefore answers with a *ranked
differential*:

* :func:`~agroai.diagnosis.base.detect_symptoms` - what the user described,
* :class:`~agroai.diagnosis.base.Hypothesis` - one possible cause plus evidence,
* :class:`~agroai.diagnosis.base.DifferentialDiagnosis` - the ranked list plus
  the observations that would discriminate between them,
* :class:`~agroai.diagnosis.engine.RetrievalDifferentialEngine` - the working
  engine, grounded in the local knowledge base.

The engine never states a diagnosis and never invents a cause: hypotheses are
documents that exist in the knowledge base, and the confidence values are
explicitly ranking aids, not probabilities.
"""

from __future__ import annotations

from agroai.diagnosis.base import (
    BAND_HIGH,
    BAND_LOW,
    BAND_MEDIUM,
    BAND_NONE,
    DifferentialDiagnosis,
    Hypothesis,
    confidence_band,
    detect_subject,
    detect_symptoms,
    required_observations,
)
from agroai.diagnosis.engine import (
    DifferentialEngine,
    RetrievalDifferentialEngine,
)

__all__ = [
    "BAND_HIGH",
    "BAND_LOW",
    "BAND_MEDIUM",
    "BAND_NONE",
    "DifferentialDiagnosis",
    "DifferentialEngine",
    "Hypothesis",
    "RetrievalDifferentialEngine",
    "confidence_band",
    "detect_subject",
    "detect_symptoms",
    "required_observations",
]
