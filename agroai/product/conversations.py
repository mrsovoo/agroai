"""Persistent conversations for the product layer (Web, Telegram, API).

```text
Conversation 1:N Message        (both stored in the existing SQLite database)
```

The existing core keeps its in-process bounded memory (``agroai.agent.memory``);
this module adds **durable storage** so a conversation survives a restart and can
be resumed by any client. The schema lives in the same SQLite file as the
knowledge base (tables ``conversations`` / ``messages``) and the access pattern
reuses :class:`agroai.database.database.Database` — one storage system, not two.

The schema is created with ``CREATE TABLE IF NOT EXISTS``, so opening an existing
AgroAI database is migration-safe: nothing is dropped or rewritten.

PostgreSQL-ready note: every statement here is plain ANSI-ish SQL over a
single connection handle; replacing SQLite later means implementing the same
small surface (create/get/list/delete conversation, append/list messages).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Any

from agroai.database.database import Database

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

_CONVERSATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL DEFAULT 'local',
    title      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    metadata        TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
"""


def _utcnow_iso() -> str:
    """Reuse the core timestamp helper (single definition in the project)."""
    from agroai.knowledge.metadata import utcnow_iso

    return utcnow_iso()


@dataclass
class Conversation:
    """One chat thread owned by a user."""

    id: str
    user_id: str = "local"
    title: str = ""
    created_at: str = ""
    updated_at: str = ""
    message_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
        }


@dataclass
class Message:
    """One message inside a conversation."""

    id: str
    conversation_id: str
    role: str
    content: str
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class ConversationStore:
    """SQLite-backed conversation storage (migration-safe, same database)."""

    def __init__(self, database: Database) -> None:
        self.database = database
        self._ensure_schema()

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def _ensure_schema(self) -> None:
        conn = self.database.connection
        conn.executescript(_CONVERSATION_SCHEMA)
        conn.commit()

    # ------------------------------------------------------------------ #
    # Conversations
    # ------------------------------------------------------------------ #
    def create_conversation(
        self, *, user_id: str = "local", title: str = "", conversation_id: str = ""
    ) -> Conversation:
        """Create a conversation row and return it."""
        conn = self.database.connection
        now = _utcnow_iso()
        conversation = Conversation(
            id=conversation_id or f"c_{uuid.uuid4().hex[:16]}",
            user_id=user_id,
            title=title,
            created_at=now,
            updated_at=now,
        )
        conn.execute(
            "INSERT INTO conversations (id, user_id, title, created_at, updated_at)"
            " VALUES (?,?,?,?,?)",
            (
                conversation.id,
                conversation.user_id,
                conversation.title,
                conversation.created_at,
                conversation.updated_at,
            ),
        )
        conn.commit()
        return conversation

    def get_conversation(
        self, conversation_id: str, *, user_id: str | None = None
    ) -> Conversation | None:
        """Fetch one conversation; when ``user_id`` is set, enforce ownership."""
        row = self.database.connection.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            return None
        if user_id is not None and row["user_id"] != user_id:
            return None
        return self._conversation_from_row(row)

    def list_conversations(
        self, *, user_id: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[Conversation]:
        """List conversations, most recently active first."""
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        params: list[Any] = []
        where = ""
        if user_id is not None:
            where = "WHERE user_id = ?"
            params.append(user_id)
        params.extend([limit, offset])
        rows = self.database.connection.execute(
            f"""
            SELECT c.*, COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            {where}
            GROUP BY c.id
            ORDER BY c.updated_at DESC, c.created_at DESC
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        return [self._conversation_from_row(row) for row in rows]

    def rename_conversation(self, conversation_id: str, title: str) -> Conversation | None:
        conn = self.database.connection
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title.strip(), _utcnow_iso(), conversation_id),
        )
        conn.commit()
        return self.get_conversation(conversation_id)

    def touch_conversation(self, conversation_id: str) -> None:
        """Bump ``updated_at`` so the conversation sorts to the top of history."""
        self.database.connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_utcnow_iso(), conversation_id),
        )
        self.database.connection.commit()

    def delete_conversation(self, conversation_id: str, *, user_id: str | None = None) -> bool:
        """Delete a conversation and its messages. Returns True if a row was removed."""
        existing = self.get_conversation(conversation_id, user_id=user_id)
        if existing is None:
            return False
        conn = self.database.connection
        # Explicit child delete keeps the intent clear even where PRAGMA
        # foreign_keys is unavailable (e.g. older SQLite builds).
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        return True

    def count_conversations(self, *, user_id: str | None = None) -> int:
        params: tuple[Any, ...] = ()
        where = ""
        if user_id is not None:
            where = "WHERE user_id = ?"
            params = (user_id,)
        row = self.database.connection.execute(
            f"SELECT COUNT(*) FROM conversations {where}", params
        ).fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------ #
    # Messages
    # ------------------------------------------------------------------ #
    def append_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        message_id: str = "",
    ) -> Message:
        """Insert one message and touch the parent conversation."""
        if role not in {ROLE_USER, ROLE_ASSISTANT}:
            raise ValueError(f"unknown message role: {role!r}")
        conn = self.database.connection
        now = _utcnow_iso()
        message = Message(
            id=message_id or f"m_{uuid.uuid4().hex[:16]}",
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=now,
            metadata=dict(metadata or {}),
        )
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, metadata, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (
                message.id,
                message.conversation_id,
                message.role,
                message.content,
                json.dumps(message.metadata, ensure_ascii=False),
                message.created_at,
            ),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        conn.commit()
        return message

    def list_messages(
        self, conversation_id: str, *, limit: int = 200, offset: int = 0
    ) -> list[Message]:
        """Messages in chronological order (oldest first)."""
        limit = max(1, min(int(limit), 1000))
        offset = max(0, int(offset))
        rows = self.database.connection.execute(
            "SELECT * FROM messages WHERE conversation_id = ?"
            " ORDER BY created_at ASC, rowid ASC LIMIT ? OFFSET ?",
            (conversation_id, limit, offset),
        ).fetchall()
        return [self._message_from_row(row) for row in rows]

    def count_messages(self, conversation_id: str) -> int:
        row = self.database.connection.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
        return int(row[0]) if row else 0

    def clear_messages(self, conversation_id: str) -> int:
        """Remove messages but keep the conversation itself. Returns removed count."""
        conn = self.database.connection
        removed = self.count_messages(conversation_id)
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.commit()
        return removed

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _conversation_from_row(row: sqlite3.Row) -> Conversation:
        return Conversation(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            title=str(row["title"] or ""),
            created_at=str(row["created_at"] or ""),
            updated_at=str(row["updated_at"] or ""),
            message_count=int(row["message_count"]) if "message_count" in row.keys() else 0,
        )

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> Message:
        raw_metadata = row["metadata"]
        try:
            metadata = json.loads(raw_metadata) if raw_metadata else {}
        except json.JSONDecodeError:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return Message(
            id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            role=str(row["role"]),
            content=str(row["content"]),
            created_at=str(row["created_at"] or ""),
            metadata=metadata,
        )


__all__ = [
    "ROLE_ASSISTANT",
    "ROLE_USER",
    "Conversation",
    "ConversationStore",
    "Message",
]
