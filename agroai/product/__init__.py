"""AgroAI product layer: shared services used by the API, Web and Telegram.

Everything in this package talks to the existing AgroAI Core (agent, retrieval,
safety, knowledge) and to the existing SQLite database. It deliberately adds no
new AI logic — the clients all share one engine.
"""

from agroai.product.conversations import (
    ROLE_ASSISTANT,
    ROLE_USER,
    Conversation,
    ConversationStore,
    Message,
)

__all__ = [
    "ROLE_ASSISTANT",
    "ROLE_USER",
    "Conversation",
    "ConversationStore",
    "Message",
]
