"""Agent memory abstraction.

Phase 1 implements only in-memory, short-term conversation context. Nothing is
written to disk and nothing is retained between runs: no sensitive user data is
stored unnecessarily. A persistent store can later implement the same
:class:`Memory` interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from agroai.models.base import Message, Role


class Memory(ABC):
    """Interface for conversation memory."""

    @abstractmethod
    def add(self, message: Message) -> None:
        """Append one message to the conversation."""

    @abstractmethod
    def history(self) -> list[Message]:
        """Return the current conversation messages (oldest first)."""

    @abstractmethod
    def clear(self) -> None:
        """Forget everything currently remembered."""

    def last_user_question(self) -> str:
        """Convenience: the most recent user message (empty string if none)."""
        for message in reversed(self.history()):
            if message.role is Role.USER:
                return message.content
        return ""


@dataclass
class ShortTermMemory(Memory):
    """Bounded, in-process conversation buffer.

    Only the last ``max_messages`` exchanges are kept, so long sessions cannot
    grow without limit. Everything lives in RAM and dies with the process.
    """

    max_messages: int = 20
    _messages: list[Message] = field(default_factory=list, init=False, repr=False)

    def add(self, message: Message) -> None:
        self._messages.append(message)
        if self.max_messages > 0 and len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

    def history(self) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()


__all__ = ["Memory", "ShortTermMemory"]
