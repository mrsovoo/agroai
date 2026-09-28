"""Compatibility shim: conversation handlers live in ``agroai.api.conversations``.

Kept so the service import stays stable even if the module layout evolves.
"""

from __future__ import annotations

from agroai.api.conversations import (
    ensure_conversation_routes,
    handle_chat_with_conversation,
    register,
    resolve_conversation,
)

def register_product_routes(service: object) -> None:
    """Attach conversation routes + handlers to a service instance (once)."""
    register(service)


__all__ = ["register_product_routes"]
