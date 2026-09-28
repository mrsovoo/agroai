"""``AgroAIService``: the single entry point every product talks to.

```text
AgroAI Web   AgroAI Bot   Mobile   AgrozAI
      \\          |          /        /
       ---------- AgroAIService -----
                     |
     agent + RAG + knowledge + memory + vision + quotas
```

One engine, many clients. The service owns the cross-cutting concerns that must
never be re-implemented per client:

* **authentication** - API key -> (user, plan), or explicit local mode;
* **authorisation + quotas** - checked server-side on every call;
* **request ids** - in every response so a report can be traced to a log line;
* **errors** - a structured ``code``, never a stack trace;
* **media path safety** - the vision route reads only from the uploads
  directory unless explicitly allowed otherwise.

It contains no HTTP code: an adapter passes an :class:`ApiRequest` in and returns
an :class:`ApiResponse`.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agroai import __version__
from agroai.agent.agent import Agent
from agroai.api.contracts import (
    ERROR_BAD_REQUEST,
    ERROR_INTERNAL,
    ERROR_METHOD_NOT_ALLOWED,
    ERROR_NOT_CONFIGURED,
    ERROR_NOT_FOUND,
    ERROR_QUOTA_EXCEEDED,
    ERROR_UNAUTHORIZED,
    ROUTES,
    ApiError,
    ApiRequest,
    ApiResponse,
    RouteSpec,
    allowed_methods,
    error_response,
    match_route,
    ok_response,
)
from agroai.api.conversation_patches import register_product_routes
from agroai.api.quotas import (
    ACTION_IMAGE,
    ACTION_MESSAGE,
    ACTION_VIDEO,
    Plan,
    QuotaPolicy,
    UsageStore,
)
from agroai.config.settings import Settings, get_settings
from agroai.database.database import Database
from agroai.knowledge.candidates import EvidenceLevel, KnowledgeCandidate, guess_evidence_level
from agroai.knowledge.manager import KnowledgeManager
from agroai.models import get_llm_provider
from agroai.models.fine_tuning.manager import FineTuningManager
from agroai.models.registry import ModelRegistry
from agroai.datasets.manager import DatasetManager
from agroai.retrieval.retriever import get_retriever
from agroai.safety.guardrails import SafetyGuard
from agroai.skills.router import get_skill_router
from agroai.sources.external import get_external_search
from agroai.sources.registry import build_catalog
from agroai.tools.base import get_default_registry
from agroai.vision.analysis import MediaAnalysis
from agroai.vision.engines import get_vision_engine

logger = logging.getLogger(__name__)

LOCAL_USER = "local"
LOCAL_DEV_MODE_NOTE = (
    "No API keys are configured: this is local development mode. Set "
    "AGROAI_API_KEYS and AGROAI_REQUIRE_API_KEY=true before exposing the API."
)


@dataclass
class Identity:
    """The authenticated caller."""

    user_id: str
    plan: str = ""
    authenticated: bool = False


class AgroAIService:
    """Framework-free AgroAI API implementation."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        agent: Agent | None = None,
        retriever: Any | None = None,
        knowledge: KnowledgeManager | None = None,
        quotas: QuotaPolicy | None = None,
        vision: Any | None = None,
        database: Database | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._owns_database = database is None
        self.database = database or self._build_database()
        self.knowledge = knowledge or KnowledgeManager(self.database, self.settings)
        self.retriever = retriever or get_retriever(self.database, self.settings)
        # The skill router is shared by the agent, the API and the evaluation
        # suite, so a single instance is resolved before the agent is built.
        self.skills = get_skill_router()
        self.agent = agent or self._build_agent()
        self.quotas = quotas or QuotaPolicy(
            store=UsageStore(self.settings.data_dir / "api" / "usage.json"),
            default_plan=getattr(self.settings, "default_plan", "free"),
            enforce=bool(getattr(self.settings, "enforce_quotas", True)),
        )
        self._vision = vision
        self.routes = ROUTES
        # Product routes (conversations) extend the route table once; handlers
        # are attached to this instance so the core service stays untouched.
        register_product_routes(self)

    # ------------------------------------------------------------------ #
    # Composition
    # ------------------------------------------------------------------ #
    def _build_database(self) -> Database:
        self.settings.ensure_directories()
        database = Database(self.settings.database_path)
        database.connect()
        database.initialize()
        return database

    def _build_agent(self) -> Agent:
        return Agent(
            retriever=self.retriever,
            llm=get_llm_provider(self.settings),
            top_k=5,
            guard=SafetyGuard(strict=self.settings.safety_strict),
            tools=get_default_registry(),
            skills=self.skills,
        )

    @property
    def vision(self):
        if self._vision is None:
            self._vision = get_vision_engine(self.settings)
        return self._vision

    def close(self) -> None:
        if self._owns_database:
            self.database.close()

    # ------------------------------------------------------------------ #
    # Authentication / authorisation
    # ------------------------------------------------------------------ #
    def authenticate(self, request: ApiRequest) -> Identity:
        """Resolve the caller: API key -> (user, plan), else local mode."""
        keys = self.settings.api_key_map()
        if not keys:
            # Local development: nothing to authenticate against.
            return Identity(user_id=request.user_id or LOCAL_USER, plan="", authenticated=False)
        if not request.api_key:
            if not self.settings.require_api_key:
                return Identity(
                    user_id=request.user_id or LOCAL_USER, plan="", authenticated=False
                )
            raise ApiError(
                ERROR_UNAUTHORIZED,
                "An API key is required. Send it in the 'X-API-Key' header.",
            )
        entry = keys.get(request.api_key)
        if entry is None:
            # Never echo the key itself.
            raise ApiError(ERROR_UNAUTHORIZED, "The API key is not recognised.")
        user_id, plan = entry
        return Identity(user_id=user_id, plan=plan, authenticated=True)

    def _require_quota(self, identity: Identity, action: str) -> dict[str, str]:
        decision = self.quotas.consume(identity.user_id, action, plan=identity.plan)
        if not decision.allowed:
            raise ApiError(
                ERROR_QUOTA_EXCEEDED,
                decision.reason or "Quota exceeded for this action.",
                detail=(
                    f"action={action}, plan={decision.plan}, used={decision.used}, "
                    f"limit={decision.limit}"
                ),
            )
        return decision.as_headers()

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #
    def handle(self, request: ApiRequest) -> ApiResponse:
        """Route one request. Never raises: failures become error responses."""
        try:
            return self._dispatch(request)
        except ApiError as error:
            return error_response(error, request.request_id)
        except Exception as exc:  # noqa: BLE001 - the API boundary must not leak traces
            logger.exception("Unhandled API error on %s %s", request.method, request.path)
            return error_response(
                ApiError(
                    ERROR_INTERNAL,
                    "Internal error while handling the request.",
                    detail=f"{type(exc).__name__}: {exc}",
                ),
                request.request_id,
            )

    def _dispatch(self, request: ApiRequest) -> ApiResponse:
        route = match_route(request.method, request.path, self.routes)
        if route is None:
            methods = allowed_methods(request.path, self.routes)
            if methods:
                raise ApiError(
                    ERROR_METHOD_NOT_ALLOWED,
                    f"{request.method} is not allowed for {request.path}.",
                    detail=f"allowed: {', '.join(methods)}",
                )
            raise ApiError(
                ERROR_NOT_FOUND,
                f"No such endpoint: {request.path}.",
                detail="See GET /api/v1/version for the service name.",
            )

        identity = self.authenticate(request)
        action = self._action_for(route, request)
        headers: dict[str, str] = {"X-Request-Id": request.request_id}
        if action:
            headers.update(self._require_quota(identity, action))

        body = self._invoke(route.handler, request, identity)
        response = ok_response(body, request.request_id, headers)
        if not identity.authenticated:
            response.headers["X-AgroAI-Mode"] = "local"
        return response

    @staticmethod
    def _action_for(route: RouteSpec, request: ApiRequest) -> str:
        if not route.action:
            return ""
        if route.action_by_kind:
            kind = str(request.body.get("kind") or request.query.get("kind") or "image").lower()
            return ACTION_VIDEO if kind == "video" else ACTION_IMAGE
        return route.action

    # ------------------------------------------------------------------ #
    # Handlers
    # ------------------------------------------------------------------ #
    def _invoke(self, handler: str, request: ApiRequest, identity: Identity) -> dict[str, Any]:
        method = getattr(self, f"_h_{handler}", None)
        if method is None:
            raise ApiError(
                ERROR_NOT_CONFIGURED,
                f"The '{handler}' endpoint is declared but not implemented in this build.",
            )
        return method(request, identity)

    def _h_version(self, request: ApiRequest, identity: Identity) -> dict[str, Any]:
        return {
            "name": "AgroAI",
            "version": __version__,
            "api_version": "v1",
            "domain": "agriculture and veterinary medicine",
        }

    def _h_chat(self, request: ApiRequest, identity: Identity) -> dict[str, Any]:
        use_memory = bool(request.body.get("use_memory", True))
        wants_conversation = bool(
            request.body.get("conversation_id") or request.body.get("create_conversation")
        )
        if wants_conversation:
            # Conversation-aware chat (used by Web and Telegram): persists the
            # user message and the answer in the shared SQLite store.
            from agroai.api.conversations import handle_chat_with_conversation

            return handle_chat_with_conversation(self, request, identity)
        return self._answer(request, identity, use_memory=use_memory)

    def _h_ask(self, request: ApiRequest, identity: Identity) -> dict[str, Any]:
        return self._answer(request, identity, use_memory=False)

    def _answer(self, request: ApiRequest, identity: Identity, *, use_memory: bool) -> dict[str, Any]:
        question = request.body_text_key
        if not question:
            raise ApiError(ERROR_BAD_REQUEST, "A non-empty 'question' is required.")
        result = self.agent.ask(question, use_memory=use_memory)
        return self._serialize_answer(result, question)

    def _serialize_answer(self, result: Any, question: str) -> dict[str, Any]:
        sources = [
            {
                "index": index,
                "title": item.document.title if item.document is not None else "",
                "source": item.document.source if item.document is not None else "",
                "url": item.document.source_url if item.document is not None else "",
                "document_id": item.chunk.document_id,
                "score": round(float(item.score), 4),
            }
            for index, item in enumerate(getattr(result, "sources", []) or [], start=1)
        ]
        plan = getattr(result, "plan", None)
        payload: dict[str, Any] = {
            "question": question,
            "answer": getattr(result, "answer", "") or "",
            "status": getattr(result, "status", ""),
            "grounded": bool(getattr(result, "grounded", False)),
            "sources": sources,
            "safety": {
                "blocked": bool(getattr(result, "safety_blocked", False)),
                "disclaimers": list(getattr(result, "disclaimers", []) or []),
            },
            "limitations": self._limitations(result),
        }
        if plan is not None:
            payload["plan"] = {
                "skills": list(getattr(plan, "skills", []) or []),
                "steps": [step.action for step in getattr(plan, "steps", []) or []],
            }
        return payload

    @staticmethod
    def _limitations(result: Any) -> list[str]:
        """State plainly what this answer could not do."""
        notes: list[str] = []
        if getattr(result, "status", "") == "not_configured":
            notes.append(
                "No language model is configured in this build, so no generated "
                "answer was produced. Retrieval and safety still ran."
            )
        if not getattr(result, "grounded", False) and getattr(result, "status", "") == "ok":
            notes.append("The answer is not backed by retrieved sources.")
        validation = getattr(result, "validation", None)
        if validation is not None and not getattr(validation, "ok", True):
            notes.append("Answer validation flagged problems; see validation details.")
        return notes

    def _h_search(self, request: ApiRequest, identity: Identity) -> dict[str, Any]:
        query = str(request.get("query") or request.body_text_key).strip()
        if not query:
            raise ApiError(ERROR_BAD_REQUEST, "A non-empty 'query' is required.")
        top_k = _positive_int(request.get("top_k", 5), default=5)
        found = self.retriever.search(query, top_k=top_k)
        return {
            "query": query,
            "results": [
                {
                    "index": index,
                    "document_id": item.chunk.document_id,
                    "title": item.document.title if item.document is not None else "",
                    "source": item.document.source if item.document is not None else "",
                    "score": round(float(item.score), 4),
                    "excerpt": " ".join(item.chunk.content.split())[:400],
                }
                for index, item in enumerate(found, start=1)
            ],
            "count": len(found),
        }

    def _h_vision(self, request: ApiRequest, identity: Identity) -> dict[str, Any]:
        path = str(request.body.get("path") or request.query.get("path") or "").strip()
        if not path:
            raise ApiError(ERROR_BAD_REQUEST, "A 'path' to an image or video is required.")
        resolved = self._resolve_media_path(path, bool(request.body.get("allow_any_path", False)))
        question = str(request.body.get("question") or request.body.get("description") or "")
        kind = str(request.body.get("kind") or request.query.get("kind") or "auto")
        analysis = self._analyze_media(resolved, question=question, kind=kind)
        return self._serialize_analysis(analysis)

    def _analyze_media(self, path: Path, *, question: str, kind: str) -> MediaAnalysis:
        from agroai.vision.analysis import MediaAnalyzer

        analyzer = MediaAnalyzer(engine=self.vision, retriever=self.retriever)
        return analyzer.analyze(str(path), question=question, kind=kind)

    @staticmethod
    def _serialize_analysis(analysis: MediaAnalysis) -> dict[str, Any]:
        payload = analysis.to_dict()
        payload["answer"] = analysis.render()
        return payload

    def _resolve_media_path(self, path: str, allow_any_path: bool) -> Path:
        """Restrict media reads to the uploads directory (defence in depth)."""
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.settings.documents_dir / candidate
        resolved = candidate.resolve()
        if allow_any_path:
            return resolved
        uploads = self.settings.documents_dir.resolve()
        try:
            resolved.relative_to(uploads)
        except ValueError as exc:
            raise ApiError(
                ERROR_BAD_REQUEST,
                "Media paths must live under the uploads directory.",
                detail=f"uploads={uploads}",
            ) from exc
        return resolved

    def _h_knowledge(self, request: ApiRequest, identity: Identity) -> dict[str, Any]:
        return {"knowledge": self.knowledge.stats(), "graph": self.knowledge.graph().stats()}

    def _h_submit_candidate(self, request: ApiRequest, identity: Identity) -> dict[str, Any]:
        statement = str(
            request.body.get("statement") or request.body.get("content") or ""
        ).strip()
        if not statement:
            raise ApiError(ERROR_BAD_REQUEST, "A non-empty 'statement' is required.")
        level = _evidence_level(request.body.get("evidence"))
        candidate = KnowledgeCandidate(
            id=request.body.get("id") or f"cand_{uuid.uuid4().hex[:12]}",
            statement=statement,
            category=str(request.body.get("category") or "general"),
            crop=str(request.body.get("crop") or ""),
            animal=str(request.body.get("animal") or ""),
            region=str(request.body.get("region") or ""),
            country=str(request.body.get("country") or ""),
            language=str(request.body.get("language") or self.settings.default_language),
            source=str(request.body.get("source") or ""),
            source_url=str(request.body.get("source_url") or ""),
            author=identity.user_id,
            evidence_level=level,
            metadata={"submitted_via": "api", "user_id": identity.user_id},
        )
        stored = self.knowledge.submit_candidate(candidate)
        return {
            "candidate": stored.to_dict() if hasattr(stored, "to_dict") else vars(stored),
            "note": (
                "Candidates are stored as PENDING and never become knowledge "
                "automatically: a human review decides."
            ),
        }

    def _h_list_candidates(self, request: ApiRequest, identity: Identity) -> dict[str, Any]:
        status = str(request.query.get("status") or "").strip() or None
        limit = _positive_int(request.query.get("limit", 50), default=50)
        found = self.knowledge.list_candidates(status=status, limit=limit)
        return {
            "count": len(found),
            "candidates": [
                item.to_dict() if hasattr(item, "to_dict") else vars(item) for item in found
            ],
        }

    def _h_sources(self, request: ApiRequest, identity: Identity) -> dict[str, Any]:
        catalog = build_catalog(self.database.list_sources())
        return {
            "count": len(catalog),
            "sources": [
                {
                    "name": entry.name,
                    "source_type": entry.source_type,
                    "url": entry.url,
                    "documents": entry.document_count,
                    "trusted": entry.trusted,
                }
                for entry in catalog
            ],
        }

    def _h_models(self, request: ApiRequest, identity: Identity) -> dict[str, Any]:
        registry = ModelRegistry(self.settings.models_dir)
        datasets = DatasetManager(self.settings.datasets_dir)
        training = FineTuningManager(self.settings.models_dir, dataset_manager=datasets)
        return {
            "models": registry.stats(),
            "datasets": datasets.stats(),
            "training": training.stats(),
            "requirements": {
                "models": registry.requirements(),
                "training": training.requirements(),
            },
        }

    def _h_plans(self, request: ApiRequest, identity: Identity) -> dict[str, Any]:
        usage = self.quotas.usage(identity.user_id)
        return {
            "plans": [plan.to_dict() for plan in sorted(self.quotas.plans.values(), key=lambda p: p.name)],
            "current": {
                "user_id": identity.user_id,
                "plan": identity.plan or self.quotas.default_plan,
                "authenticated": identity.authenticated,
            },
            "usage": usage,
            "enforcement": self.quotas.enforce,
            "note": None if self.settings.api_key_map() else LOCAL_DEV_MODE_NOTE,
        }

    def _h_vision_info(self, request: ApiRequest, identity: Identity) -> dict[str, Any]:
        health = self.vision.health_check()
        return {
            "vision": health,
            "configured": self.vision.is_configured,
            "note": (
                "File inspection (format, resolution, duration) works without a "
                "model; content understanding does not."
            ),
        }

    def _h_health(self, request: ApiRequest, identity: Identity) -> dict[str, Any]:
        report: dict[str, Any] = {"status": "ok", "version": __version__}
        try:
            retrieval = self.retriever.health_check()
            report["retrieval"] = {
                "backend": retrieval.backend,
                "ready": retrieval.ready,
                "documents": retrieval.documents,
                "chunks": retrieval.chunks,
                "search_mode": retrieval.search_mode,
            }
        except Exception as exc:  # noqa: BLE001
            report["retrieval"] = {"ready": False, "detail": str(exc)}

        llm = getattr(self.agent, "llm", None)
        if llm is not None:
            health = llm.health_check()
            report["llm"] = {
                "provider": health.provider,
                "configured": health.configured,
                "available": health.available,
                "detail": health.detail,
            }
        report["vision"] = self.vision.health_check()
        report["external_search"] = get_external_search(self.settings).health_check()
        report["knowledge"] = self.knowledge.stats()
        report["quotas"] = {"enforce": self.quotas.enforce, "default_plan": self.quotas.default_plan}

        if not report.get("llm", {}).get("configured"):
            report["status"] = "degraded"
            report["detail"] = (
                "Retrieval, knowledge and safety layers are ready; no language "
                "model is configured, so answers are not generated."
            )
        return report


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _positive_int(value: Any, *, default: int, maximum: int = 100) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    if number <= 0:
        return default
    return min(number, maximum)


def _evidence_level(value: Any) -> EvidenceLevel:
    """Resolve a declared evidence level, guessing honestly when absent."""
    if isinstance(value, EvidenceLevel):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return EvidenceLevel(value.strip().lower())
        except ValueError:
            return guess_evidence_level(value)
    return EvidenceLevel.USER_REPORT


def build_service(settings: Settings | None = None, **kwargs: Any) -> AgroAIService:
    """Compose a ready-to-use service (also used by the CLI and tests)."""
    return AgroAIService(settings=settings, **kwargs)


def default_plan_name(service: AgroAIService) -> str:
    return service.quotas.default_plan


def plan_of(service: AgroAIService, name: str) -> Plan:
    return service.quotas.plan_for(name)


__all__ = [
    "AgroAIService",
    "Identity",
    "LOCAL_DEV_MODE_NOTE",
    "LOCAL_USER",
    "build_service",
    "default_plan_name",
    "plan_of",
    "ACTION_MESSAGE",
]
