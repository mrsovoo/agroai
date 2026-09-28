"""Tests for persistent conversations (product layer) and their API routes."""

from __future__ import annotations

import pytest

from agroai.api.contracts import ApiRequest
from agroai.api.service import build_service
from agroai.database.database import Database
from agroai.product.conversations import (
    ROLE_ASSISTANT,
    ROLE_USER,
    ConversationStore,
)


@pytest.fixture()
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("AGROAI_DATABASE_PATH", str(tmp_path / "conv.db"))
    from agroai.config.settings import reset_settings_cache

    reset_settings_cache()
    svc = build_service()
    yield svc
    svc.close()
    reset_settings_cache()


def _create(service, title="T"):
    response = service.handle(ApiRequest("POST", "/api/v1/conversations", body={"title": title}))
    assert response.ok
    return response.body["conversation"]["id"]


class TestConversationStore:
    def test_create_and_get(self, tmp_path):
        database = Database(tmp_path / "t.db")
        database.connect()
        database.initialize()
        store = ConversationStore(database)
        conversation = store.create_conversation(user_id="u1", title="Hello")
        assert conversation.id.startswith("c_")
        fetched = store.get_conversation(conversation.id)
        assert fetched is not None and fetched.title == "Hello"

    def test_ownership_enforced(self, tmp_path):
        database = Database(tmp_path / "t.db")
        database.connect()
        database.initialize()
        store = ConversationStore(database)
        conversation = store.create_conversation(user_id="alice")
        assert store.get_conversation(conversation.id, user_id="bob") is None
        assert store.get_conversation(conversation.id, user_id="alice") is not None
        assert not store.delete_conversation(conversation.id, user_id="bob")
        assert store.delete_conversation(conversation.id, user_id="alice")

    def test_messages_order_and_roles(self, tmp_path):
        database = Database(tmp_path / "t.db")
        database.connect()
        database.initialize()
        store = ConversationStore(database)
        conversation = store.create_conversation(user_id="u")
        store.append_message(conversation.id, role=ROLE_USER, content="savol")
        store.append_message(conversation.id, role=ROLE_ASSISTANT, content="javob")
        messages = store.list_messages(conversation.id)
        assert [m.role for m in messages] == [ROLE_USER, ROLE_ASSISTANT]
        with pytest.raises(ValueError):
            store.append_message(conversation.id, role="system", content="x")

    def test_reopen_is_migration_safe(self, tmp_path):
        path = tmp_path / "t.db"
        database = Database(path)
        database.connect()
        database.initialize()
        ConversationStore(database).create_conversation(user_id="u")
        database.close()

        database2 = Database(path)
        database2.connect()
        database2.initialize()
        store = ConversationStore(database2)
        assert store.count_conversations() == 1

    def test_clear_messages_keeps_conversation(self, tmp_path):
        database = Database(tmp_path / "t.db")
        database.connect()
        database.initialize()
        store = ConversationStore(database)
        conversation = store.create_conversation(user_id="u")
        store.append_message(conversation.id, role=ROLE_USER, content="x")
        assert store.clear_messages(conversation.id) == 1
        assert store.get_conversation(conversation.id) is not None
        assert store.count_messages(conversation.id) == 0


class TestConversationRoutes:
    def test_create_list_get_delete(self, service):
        conversation_id = _create(service)
        listed = service.handle(ApiRequest("GET", "/api/v1/conversations"))
        assert listed.ok and listed.body["count"] >= 1

        got = service.handle(ApiRequest("GET", f"/api/v1/conversations/{conversation_id}"))
        assert got.ok and got.body["messages"] == []

        renamed = service.handle(
            ApiRequest("POST", f"/api/v1/conversations/{conversation_id}", body={"title": "N"})
        )
        assert renamed.ok and renamed.body["conversation"]["title"] == "N"

        deleted = service.handle(ApiRequest("DELETE", f"/api/v1/conversations/{conversation_id}"))
        assert deleted.ok and deleted.body["deleted"] is True
        gone = service.handle(ApiRequest("GET", f"/api/v1/conversations/{conversation_id}"))
        assert gone.status == 404

    def test_chat_persists_messages(self, service):
        conversation_id = _create(service)
        response = service.handle(
            ApiRequest(
                "POST",
                "/api/v1/chat",
                body={"conversation_id": conversation_id, "message": "pomidor kasalligi"},
            )
        )
        assert response.ok
        assert response.body["conversation_id"] == conversation_id
        history = service.handle(ApiRequest("GET", f"/api/v1/conversations/{conversation_id}"))
        roles = [m["role"] for m in history.body["messages"]]
        assert roles == [ROLE_USER, ROLE_ASSISTANT]

    def test_legacy_stateless_chat_unchanged(self, service):
        response = service.handle(ApiRequest("POST", "/api/v1/chat", body={"question": "salom"}))
        assert response.ok
        assert "conversation_id" not in response.body

    def test_foreign_user_gets_404(self, service):
        conversation_id = _create(service)
        foreign = service.handle(
            ApiRequest(
                "GET", f"/api/v1/conversations/{conversation_id}", headers={"X-User-Id": "other"}
            )
        )
        assert foreign.status == 404

    def test_unknown_conversation_404(self, service):
        response = service.handle(ApiRequest("GET", "/api/v1/conversations/c_missing"))
        assert response.status == 404
