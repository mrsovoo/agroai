"""Conversation routes and chat-with-history support on the existing service.

The product requirement is one shared core: Web and Telegram both call the same
``AgroAIService``, and conversation persistence lives here — not in the clients.

New routes (added to the route table in :mod:`agroai.api.contracts`):

```text
GET    /api/v1/conversations            list (newest first)
POST   /api/v1/conversations            create
GET    /api/v1/conversations/{id}       one conversation + its messages
DELETE /api/v1/conversations/{id}       delete with its messages
```

``POST /api/v1/chat`` is extended to accept ``conversation_id`` and ``message``:
when a conversation is supplied (or created implicitly), the user message and the
assistant answer are stored, and the conversation is included in the response.
Ownership is enforced by ``user_id`` — one user can never read another's thread.
"""

from __future__ import annotations

from typing import Any

from agroai.api import contracts as _contracts
from agroai.api.contracts import (
    API_PREFIX,
    ERROR_BAD_REQUEST,
    ERROR_NOT_FOUND,
    HTTP_DELETE,
    HTTP_GET,
    HTTP_POST,
    ROUTES,
    ApiError,
    ApiRequest,
    RouteSpec,
)
from agroai.product.conversations import (
    ROLE_ASSISTANT,
    ROLE_USER,
    ConversationStore,
)

_CONVERSATION_ROUTES_ADDED = False

DEFAULT_HISTORY_LIMIT = 200


def ensure_conversation_routes() -> tuple[RouteSpec, ...]:
    """Append the conversation routes to the route table exactly once."""
    global ROUTES, _CONVERSATION_ROUTES_ADDED
    if _CONVERSATION_ROUTES_ADDED:
        return ROUTES
    if not any(route.handler == "chat" for route in ROUTES):  # pragma: no cover
        return ROUTES
    added = (
        RouteSpec(
            HTTP_GET,
            f"{API_PREFIX}/conversations",
            "list_conversations",
            description="List conversations of the caller (newest first).",
        ),
        RouteSpec(
            HTTP_POST,
            f"{API_PREFIX}/conversations",
            "create_conversation",
            description="Create a new (empty) conversation.",
        ),
        RouteSpec(
            HTTP_GET,
            f"{API_PREFIX}/conversations/{{conversation_id}}",
            "get_conversation",
            description="One conversation with its message history.",
        ),
        RouteSpec(
            HTTP_POST,
            f"{API_PREFIX}/conversations/{{conversation_id}}",
            "update_conversation",
            description="Rename a conversation.",
        ),
        RouteSpec(
            HTTP_DELETE,
            f"{API_PREFIX}/conversations/{{conversation_id}}",
            "delete_conversation",
            description="Delete a conversation and its messages.",
        ),
    )
    # Publish through the owning module so every importer sees the same tuple.
    _contracts.ROUTES = _contracts.ROUTES + added
    ROUTES = _contracts.ROUTES
    _CONVERSATION_ROUTES_ADDED = True
    return ROUTES


def conversations_service(service: Any) -> Any:
    """Lazily build/attach a :class:`ConversationStore` to a service instance."""
    store = getattr(service, "_conversations", None)
    if store is None:
        store = ConversationStore(service.database)
        service._conversations = store
    return store


def resolve_conversation(service: Any, request: ApiRequest, identity: Any) -> Any:
    """Fetch the conversation in the path, enforcing ownership."""
    from agroai.api.contracts import match_route, path_params

    store = conversations_service(service)
    route = match_route(request.method, request.path, service.routes)
    params = path_params(route, request.path) if route else {}
    conversation_id = params.get("conversation_id", "")
    conversation = store.get_conversation(conversation_id, user_id=identity.user_id)
    if conversation is None:
        raise ApiError(
            ERROR_NOT_FOUND,
            "Conversation not found (or it belongs to another user).",
        )
    return conversation


def _auto_title(question: str) -> str:
    """A short title from the first user message (never a generated fake)."""
    cleaned = " ".join(question.split())
    return cleaned[:60] + ("…" if len(cleaned) > 60 else "")


def handle_chat_with_conversation(service: Any, request: ApiRequest, identity: Any) -> dict[str, Any]:
    """Wrap the core chat handler with conversation persistence.

    Behaviour:
    * ``conversation_id`` given -> reuse it (404 if missing/foreign);
    * ``conversation_id`` omitted but ``create_conversation=true`` (default when
      called by the product clients) -> create one titled from the question;
    * otherwise the legacy stateless behaviour is kept unchanged.
    """
    store = conversations_service(service)
    question = request.body_text_key
    if not question:
        raise ApiError(ERROR_BAD_REQUEST, "A non-empty 'question' is required.")

    conversation_id = str(request.body.get("conversation_id") or "").strip()
    create_flag = bool(request.body.get("create_conversation", True))
    title = str(request.body.get("title") or "").strip()

    if not conversation_id and not create_flag:
        # Legacy stateless mode: hand the request to the original handler.
        return service._h_chat(request, identity)

    if conversation_id:
        conversation = store.get_conversation(conversation_id, user_id=identity.user_id)
        if conversation is None:
            raise ApiError(
                ERROR_NOT_FOUND,
                "Conversation not found (or it belongs to another user).",
            )
    else:
        conversation = store.create_conversation(
            user_id=identity.user_id, title=title or _auto_title(question)
        )

    result = service.agent.ask(question, use_memory=True)
    payload = service._serialize_answer(result, question)
    payload["conversation_id"] = conversation.id

    user_message = store.append_message(
        conversation.id, role=ROLE_USER, content=question, metadata={"language": str(request.body.get("language") or "")}
    )
    assistant_message = store.append_message(
        conversation.id,
        role=ROLE_ASSISTANT,
        content=str(payload.get("answer") or ""),
        metadata={
            "status": payload.get("status"),
            "grounded": payload.get("grounded"),
            "source_count": len(payload.get("sources") or []),
        },
    )
    payload["message_id"] = assistant_message.id
    payload["user_message_id"] = user_message.id
    payload["conversation"] = store.get_conversation(conversation.id).to_dict()
    return payload


def serialize_conversation(store: Any, conversation: Any, *, with_messages: bool = False) -> dict[str, Any]:
    """Conversation payload for the API (with optional message history)."""
    payload = conversation.to_dict()
    if with_messages:
        payload["messages"] = [
            message.to_dict() for message in store.list_messages(conversation.id)
        ]
    return payload


def register(service: Any) -> None:
    """Attach conversation handlers to a running :class:`AgroAIService`."""
    ensure_conversation_routes()
    service.routes = ROUTES
    service._conversations = conversations_service(service)

    def _h_list_conversations(instance, request, identity):  # noqa: ANN001
        store = conversations_service(instance)
        limit = _limit_of(request)
        found = store.list_conversations(user_id=identity.user_id, limit=limit)
        return {
            "count": len(found),
            "conversations": [item.to_dict() for item in found],
        }

    def _h_create_conversation(instance, request, identity):  # noqa: ANN001
        store = conversations_service(instance)
        title = str(request.body.get("title") or "").strip()
        conversation = store.create_conversation(user_id=identity.user_id, title=title)
        return {"conversation": conversation.to_dict()}

    def _h_get_conversation(instance, request, identity):  # noqa: ANN001
        store = conversations_service(instance)
        conversation = resolve_conversation(instance, request, identity)
        return serialize_conversation(store, conversation, with_messages=True)

    def _h_update_conversation(instance, request, identity):  # noqa: ANN001
        store = conversations_service(instance)
        conversation = resolve_conversation(instance, request, identity)
        title = str(request.body.get("title") or "").strip()
        if not title:
            raise ApiError(ERROR_BAD_REQUEST, "A non-empty 'title' is required.")
        updated = store.rename_conversation(conversation.id, title)
        return {"conversation": updated.to_dict()}

    def _h_delete_conversation(instance, request, identity):  # noqa: ANN001
        store = conversations_service(instance)
        conversation = resolve_conversation(instance, request, identity)
        store.delete_conversation(conversation.id)
        return {"deleted": True, "conversation_id": conversation.id}

    instance_handlers = {
        "_h_list_conversations": _h_list_conversations,
        "_h_create_conversation": _h_create_conversation,
        "_h_get_conversation": _h_get_conversation,
        "_h_update_conversation": _h_update_conversation,
        "_h_delete_conversation": _h_delete_conversation,
    }
    for name, handler in instance_handlers.items():
        setattr(service, name, handler.__get__(service, type(service)))


def _limit_of(request: ApiRequest) -> int:
    from agroai.api.service import _positive_int

    try:
        return _positive_int(request.query.get("limit", 50), default=50)
    except Exception:  # pragma: no cover - defensive
        return DEFAULT_HISTORY_LIMIT


__all__ = [
    "ensure_conversation_routes",
    "handle_chat_with_conversation",
    "register",
    "resolve_conversation",
    "serialize_conversation",
]
