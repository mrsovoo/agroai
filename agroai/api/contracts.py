"""Framework-free API contracts and the route table.

Deliberately no web framework: the core stays standard-library-only, and the
contract is what matters. ``agroai.api`` defines the request/response envelopes,
the routes, the error model and the quota action per route; a thin adapter
(FastAPI/WSGI/Telegram) is then a few dozen lines with **no logic of its own**.

```text
POST /api/v1/chat      POST /api/v1/vision    GET /api/v1/sources
POST /api/v1/ask       POST /api/v1/search    GET /api/v1/health
POST /api/v1/knowledge/candidates             GET /api/v1/models
GET  /api/v1/knowledge                        GET /api/v1/plans
```

Every response carries a ``request_id`` so a log line and a user report can be
matched, and an error always has a machine-readable ``code``.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

HTTP_GET = "GET"
HTTP_POST = "POST"
HTTP_DELETE = "DELETE"
HTTP_PUT = "PUT"
HTTP_PATCH = "PATCH"

API_PREFIX = "/api/v1"

#: Error codes the service can return.
ERROR_BAD_REQUEST = "bad_request"
ERROR_UNAUTHORIZED = "unauthorized"
ERROR_NOT_FOUND = "not_found"
ERROR_METHOD_NOT_ALLOWED = "method_not_allowed"
ERROR_QUOTA_EXCEEDED = "quota_exceeded"
ERROR_NOT_CONFIGURED = "not_configured"
ERROR_INTERNAL = "internal_error"

STATUS_BY_CODE: dict[str, int] = {
    ERROR_BAD_REQUEST: 400,
    ERROR_UNAUTHORIZED: 401,
    ERROR_NOT_FOUND: 404,
    ERROR_METHOD_NOT_ALLOWED: 405,
    ERROR_QUOTA_EXCEEDED: 429,
    ERROR_NOT_CONFIGURED: 503,
    ERROR_INTERNAL: 500,
}


class ApiError(Exception):
    """A structured API failure (never a bare exception trace)."""

    def __init__(self, code: str, message: str, *, detail: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    @property
    def status(self) -> int:
        return STATUS_BY_CODE.get(self.code, 500)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"error": {"code": self.code, "message": self.message}}
        if self.detail:
            payload["error"]["detail"] = self.detail
        return payload


@dataclass
class ApiRequest:
    """One inbound API call."""

    method: str
    path: str
    body: dict[str, Any] = field(default_factory=dict)
    query: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    request_id: str = ""
    api_key: str = ""
    user_id: str = ""

    def __post_init__(self) -> None:
        self.method = (self.method or HTTP_POST).upper()
        self.path = normalize_path(self.path)
        if not self.request_id:
            self.request_id = uuid.uuid4().hex[:16]
        # Header values are case-insensitive; normalise once here.
        if not self.api_key:
            self.api_key = _header(self.headers, "x-api-key") or _header(
                self.headers, "authorization"
            ).removeprefix("Bearer ").strip()
        if not self.user_id:
            self.user_id = _header(self.headers, "x-user-id")

    @property
    def body_text_key(self) -> str:
        """The user's text: accepts 'question', 'message' or 'query'."""
        for key in ("question", "message", "query"):
            value = self.body.get(key)
            if value:
                return str(value).strip()
        return ""

    def get(self, key: str, default: Any = "") -> Any:
        """Read a value from the body, then the query string."""
        if key in self.body:
            return self.body[key]
        return self.query.get(key, default)


@dataclass
class ApiResponse:
    """One outbound API response."""

    status: int = 200
    body: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


@dataclass(frozen=True)
class RouteSpec:
    """One route: what it does and which quota action it consumes."""

    method: str
    path: str
    handler: str
    action: str = ""
    description: str = ""
    #: Vision routes choose ``image`` or ``video`` from the request body.
    action_by_kind: bool = False
    auth_required: bool = False

    @property
    def key(self) -> tuple[str, str]:
        return (self.method, self.path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "path": self.path,
            "description": self.description,
            "quota_action": self.action or None,
            "auth_required": self.auth_required,
        }


#: The complete route table. A path pattern may contain ``{name}`` segments.
ROUTES: tuple[RouteSpec, ...] = (
    RouteSpec(
        HTTP_POST,
        f"{API_PREFIX}/chat",
        "chat",
        action="message",
        description="Chat with the AgroAI agent (retrieval-grounded answer + sources).",
    ),
    RouteSpec(
        HTTP_POST,
        f"{API_PREFIX}/ask",
        "ask",
        action="message",
        description="One-shot question: same pipeline as /chat, without history.",
    ),
    RouteSpec(
        HTTP_POST,
        f"{API_PREFIX}/search",
        "search",
        action="search",
        description="Search the local knowledge base.",
    ),
    RouteSpec(
        HTTP_POST,
        f"{API_PREFIX}/vision",
        "vision",
        action="image",
        action_by_kind=True,
        description=(
            "Inspect an image or video and produce a differential from the "
            "description. Reports 'not configured' when no vision model exists."
        ),
    ),
    RouteSpec(
        HTTP_POST,
        f"{API_PREFIX}/knowledge/candidates",
        "submit_candidate",
        action="knowledge_write",
        description="Submit a knowledge candidate (never auto-approved).",
    ),
    RouteSpec(HTTP_GET, f"{API_PREFIX}/knowledge", "knowledge", description="Knowledge base statistics."),
    RouteSpec(HTTP_GET, f"{API_PREFIX}/knowledge/candidates", "list_candidates", description="List knowledge candidates."),
    RouteSpec(HTTP_GET, f"{API_PREFIX}/sources", "sources", description="List ingested sources."),
    RouteSpec(HTTP_GET, f"{API_PREFIX}/models", "models", description="Model registry and dataset statistics."),
    RouteSpec(HTTP_GET, f"{API_PREFIX}/plans", "plans", description="Plans, limits and current usage."),
    RouteSpec(HTTP_GET, f"{API_PREFIX}/vision", "vision_info", description="Vision engine health."),
    RouteSpec(HTTP_GET, f"{API_PREFIX}/health", "health", description="Health of every layer."),
    RouteSpec(HTTP_GET, f"{API_PREFIX}/version", "version", description="AgroAI version."),
)

_PATH_PARAM_RE = re.compile(r"\{[^/]+\}")


def normalize_path(path: str) -> str:
    """Normalise a path: strip a trailing slash and the query string."""
    cleaned = (path or "/").split("?", 1)[0].strip()
    if not cleaned.startswith("/"):
        cleaned = "/" + cleaned
    if len(cleaned) > 1 and cleaned.endswith("/"):
        cleaned = cleaned.rstrip("/")
    return cleaned


def match_route(method: str, path: str, routes: tuple[RouteSpec, ...] = ROUTES):
    """Return the :class:`RouteSpec` matching ``method``/``path``, or ``None``.

    Path templates with ``{name}`` segments match any single non-empty segment
    (e.g. ``/api/v1/conversations/{conversation_id}`` matches
    ``/api/v1/conversations/c_abc``). Static routes always match first.
    """
    method = (method or "").upper()
    path = normalize_path(path)
    static: RouteSpec | None = None
    for route in routes:
        if route.method != method:
            continue
        if route.path == path:
            return route
        if static is None and "{" in route.path and _template_matches(route.path, path):
            static = route
    return static


def _template_matches(template: str, path: str) -> bool:
    """True when ``path`` fits the ``{name}`` template (segment by segment)."""
    template_parts = template.split("/")
    path_parts = path.split("/")
    if len(template_parts) != len(path_parts):
        return False
    for expected, actual in zip(template_parts, path_parts):
        if expected.startswith("{") and expected.endswith("}"):
            if not actual:
                return False
            continue
        if expected != actual:
            return False
    return True


def allowed_methods(path: str, routes: tuple[RouteSpec, ...] = ROUTES) -> list[str]:
    """HTTP methods available for ``path`` (used for a correct 405)."""
    path = normalize_path(path)
    return sorted({route.method for route in routes if route.path == path})


def path_params(route: RouteSpec, path: str) -> dict[str, str]:
    """Extract ``{name}`` segments as a dict."""
    if "{" not in route.path:
        return {}
    route_parts = normalize_path(route.path).split("/")
    path_parts = normalize_path(path).split("/")
    if len(route_parts) != len(path_parts):
        return {}
    params: dict[str, str] = {}
    for template, value in zip(route_parts, path_parts):
        if template.startswith("{") and template.endswith("}"):
            params[template[1:-1]] = value
    return params


def describe_routes(routes: tuple[RouteSpec, ...] = ROUTES) -> list[dict[str, Any]]:
    """Machine-readable route documentation (used by the README/CLI)."""
    return [route.to_dict() for route in routes]


def error_response(error: ApiError, request_id: str = "") -> ApiResponse:
    """Turn an :class:`ApiError` into a response envelope."""
    body = error.to_dict()
    if request_id:
        body["request_id"] = request_id
    return ApiResponse(status=error.status, body=body)


def ok_response(body: dict[str, Any], request_id: str = "", headers: Mapping[str, str] | None = None) -> ApiResponse:
    payload = dict(body)
    if request_id:
        payload["request_id"] = request_id
    return ApiResponse(status=200, body=payload, headers=dict(headers or {}))


def _header(headers: Mapping[str, str], name: str) -> str:
    for key, value in (headers or {}).items():
        if key.lower() == name.lower():
            return str(value)
    return ""


__all__ = [
    "API_PREFIX",
    "ApiError",
    "ApiRequest",
    "ApiResponse",
    "ERROR_BAD_REQUEST",
    "ERROR_INTERNAL",
    "ERROR_METHOD_NOT_ALLOWED",
    "ERROR_NOT_CONFIGURED",
    "ERROR_NOT_FOUND",
    "ERROR_QUOTA_EXCEEDED",
    "ERROR_UNAUTHORIZED",
    "HTTP_DELETE",
    "HTTP_GET",
    "HTTP_PATCH",
    "HTTP_POST",
    "HTTP_PUT",
    "ROUTES",
    "RouteSpec",
    "STATUS_BY_CODE",
    "allowed_methods",
    "describe_routes",
    "error_response",
    "match_route",
    "normalize_path",
    "ok_response",
    "path_params",
]
