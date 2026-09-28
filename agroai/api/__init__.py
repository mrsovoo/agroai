"""AgroAI API: one engine, many clients.

```text
AgroAI Web   AgroAI Bot   Mobile   AgrozAI
      \\          |          /        /
       ---------- AgroAIService -----
                     |
     agent + RAG + knowledge + memory + vision + quotas
```

The bot has no AI of its own and neither does the web app: they are clients of
this service. That is the whole point of a central engine - one knowledge base,
one model, one safety layer, one quota system.

Framework-free by design (the core stays standard-library only). The boundaries
are:

* :mod:`agroai.api.contracts` - request/response envelopes, the route table, the
  error model and the quota action per route;
* :mod:`agroai.api.quotas` - plans, per-user limits and the server-side counter;
* :mod:`agroai.api.service` - :class:`~agroai.api.service.AgroAIService`, which
  authenticates, enforces quotas and produces responses.

```python
from agroai.api import AgroAIService, ApiRequest, build_service

service = build_service()
response = service.handle(ApiRequest("POST", "/api/v1/ask", {"question": "..."}))
```
"""

from __future__ import annotations

from agroai.api.contracts import (
    API_PREFIX,
    ROUTES,
    ApiError,
    ApiRequest,
    ApiResponse,
    RouteSpec,
    describe_routes,
    match_route,
    normalize_path,
)
from agroai.api.quotas import (
    ACTIONS,
    ACTION_EXTERNAL_SEARCH,
    ACTION_IMAGE,
    ACTION_KNOWLEDGE_WRITE,
    ACTION_MESSAGE,
    ACTION_SEARCH,
    ACTION_VIDEO,
    DEFAULT_PLANS,
    Plan,
    QuotaDecision,
    QuotaLimits,
    QuotaPolicy,
    UNLIMITED,
    UsageStore,
)
from agroai.api.service import (
    AgroAIService,
    Identity,
    LOCAL_DEV_MODE_NOTE,
    LOCAL_USER,
    build_service,
)

__all__ = [
    "ACTIONS",
    "ACTION_EXTERNAL_SEARCH",
    "ACTION_IMAGE",
    "ACTION_KNOWLEDGE_WRITE",
    "ACTION_MESSAGE",
    "ACTION_SEARCH",
    "ACTION_VIDEO",
    "API_PREFIX",
    "AgroAIService",
    "ApiError",
    "ApiRequest",
    "ApiResponse",
    "DEFAULT_PLANS",
    "Identity",
    "LOCAL_DEV_MODE_NOTE",
    "LOCAL_USER",
    "Plan",
    "QuotaDecision",
    "QuotaLimits",
    "QuotaPolicy",
    "ROUTES",
    "RouteSpec",
    "UNLIMITED",
    "UsageStore",
    "build_service",
    "describe_routes",
    "match_route",
    "normalize_path",
]
