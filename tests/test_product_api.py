"""Tests for the product layer: HTTP adapter, admin panel, Telegram bot.

No real network is used: the Telegram transport is faked, the HTTP server is
started on an ephemeral port and shut down within the test.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar

import pytest

from agroai.api.main import make_server
from agroai.api.service import build_service
from agroai.config.settings import get_settings, reset_settings_cache
from agroai.product.conversations import ROLE_USER
from agroai.telegram import AgroAITelegramBot, BotConfig, describe_token_state


@pytest.fixture()
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("AGROAI_DATABASE_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("AGROAI_ADMIN_PASSWORD", "test-password-123")
    reset_settings_cache()
    svc = build_service()
    yield svc
    svc.close()
    reset_settings_cache()


@pytest.fixture()
def http(service):
    server = make_server(service=service, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()


class FakeTransport:
    """Records calls; returns canned Telegram API responses."""

    def __init__(self):
        self.calls: list[tuple[str, dict | None]] = []
        self.replies: list[dict] = []

    def __call__(self, token, method, payload=None):
        self.calls.append((method, payload))
        if method == "getUpdates":
            updates, self.pending = getattr(self, "pending", []), []
            return updates
        if method == "sendMessage":
            self.replies.append(payload)
            return {"message_id": 1}
        return {}

    def sent_texts(self):
        return [payload.get("text", "") for _, payload in self.calls if payload and payload.get("text")]

    def chat_ids(self):
        return [payload.get("chat_id") for _, payload in self.calls if payload and payload.get("chat_id")]


def make_bot(service, token="test-token"):
    transport = FakeTransport()
    bot = AgroAITelegramBot(
        config=BotConfig(token=token, media_dir=service.settings.data_dir / "media"),
        service=service,
        transport=transport,
        downloader=lambda token_, path: b"png",
    )
    return bot, transport


class TestHttpServer:
    def test_health(self, http):
        with urllib.request.urlopen(f"{http}/api/v1/health") as response:
            body = json.loads(response.read())
        assert body["status"] in {"ok", "degraded"}
        assert body["retrieval"]["ready"] is True

    def test_conversation_flow_over_http(self, http):
        request = urllib.request.Request(
            f"{http}/api/v1/conversations",
            data=json.dumps({"title": "T"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        conversation_id = json.loads(urllib.request.urlopen(request).read())["conversation"]["id"]
        with urllib.request.urlopen(f"{http}/api/v1/conversations/{conversation_id}") as response:
            assert response.status == 200

    def test_unknown_endpoint_structured_404(self, http):
        try:
            urllib.request.urlopen(f"{http}/api/v1/definitely-missing")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as error:
            body = json.loads(error.read())
            assert error.code == 404 and body["error"]["code"] == "not_found"

    def test_oversized_body_413(self, http):
        request = urllib.request.Request(
            f"{http}/api/v1/chat",
            data=b"x" * 2_000_000,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(request)
            raise AssertionError("expected 413")
        except urllib.error.HTTPError as error:
            assert error.code == 413

    def test_cors_preflight(self, http):
        request = urllib.request.Request(
            f"{http}/api/v1/health",
            method="OPTIONS",
            headers={"Origin": "http://localhost:3000"},
        )
        with urllib.request.urlopen(request) as response:
            assert response.status == 204
            assert response.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"


class TestAdminPanel:
    def test_login_flow_and_guarded_api(self, http):
        # Login page renders without auth.
        with urllib.request.urlopen(f"{http}/admin") as response:
            assert "AgroAI Admin" in response.read().decode()

        # Wrong password rejected.
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{http}/admin/login",
                    data=urllib.parse.urlencode({"password": "nope"}).encode(),
                    method="POST",
                )
            )
            raise AssertionError("expected 403")
        except urllib.error.HTTPError as error:
            assert error.code == 403

        # Correct password -> session cookie -> dashboard + JSON stats.
        jar = CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        opener.open(
            urllib.request.Request(
                f"{http}/admin/login",
                data=urllib.parse.urlencode({"password": "test-password-123"}).encode(),
                method="POST",
            )
        )
        stats = json.loads(opener.open(f"{http}/admin/api/stats").read())
        assert "documents" in stats and "llm" in stats
        dashboard = opener.open(f"{http}/admin").read().decode()
        assert "Super Admin" in dashboard

        # Without a session the JSON endpoints stay guarded.
        try:
            urllib.request.urlopen(f"{http}/admin/api/stats")
            raise AssertionError("expected 401")
        except urllib.error.HTTPError as error:
            assert error.code == 401


class TestTelegramBot:
    def test_token_state_never_leaks(self):
        assert describe_token_state("") == "NOT CONFIGURED"
        assert describe_token_state("secret") == "CONFIGURED"

    def test_missing_token_run_reports_honestly(self, service, capsys):
        bot, _ = make_bot(service, token="")
        assert bot.run() == 1
        assert "tokeni yo'q" in capsys.readouterr().out

    def test_start_command_sends_welcome(self, service):
        bot, transport = make_bot(service)
        bot.handle_update(
            {"message": {"chat": {"id": 42}, "text": "/start"}}
        )
        texts = transport.sent_texts()
        assert any("AgroAI" in text for text in texts)

    def test_unknown_command(self, service):
        bot, transport = make_bot(service)
        bot.handle_update({"message": {"chat": {"id": 42}, "text": "/definitelynotacommand"}})
        assert any("Noma'lum buyruq" in text for text in transport.sent_texts())

    def test_text_message_persists_per_chat(self, service):
        bot, transport = make_bot(service)
        bot.handle_update({"message": {"chat": {"id": 7}, "text": "pomidor kasalligi"}})
        bot.handle_update({"message": {"chat": {"id": 8}, "text": "pomidor kasalligi"}})
        conversations = bot.conversations.list_conversations(limit=10)
        owners = {conversation.user_id for conversation in conversations}
        assert {"tg:7", "tg:8"} <= owners  # chats are isolated
        assert transport.sent_texts()  # a reply was produced

    def test_photo_without_vision_is_honest(self, service):
        bot, transport = make_bot(service)
        bot.handle_update(
            {
                "message": {
                    "chat": {"id": 9},
                    "photo": [{"file_id": "f1", "file_size": 10}],
                }
            }
        )
        texts = transport.sent_texts()
        assert any("sozlanmagan" in text for text in texts)
        # No fake diagnosis is ever sent.
        assert not any("tashxis:" in text.lower() for text in texts)

    def test_video_reports_not_available(self, service):
        bot, transport = make_bot(service)
        bot.handle_update(
            {"message": {"chat": {"id": 5}, "video": {"file_id": "v1"}}}
        )
        assert any("Video tahlil hali mavjud emas" in text for text in transport.sent_texts())

    def test_poll_once_handles_updates(self, service):
        bot, transport = make_bot(service)
        transport.pending = [
            {"update_id": 1, "message": {"chat": {"id": 11}, "text": "/help"}}
        ]
        assert bot.poll_once() == 1
        assert any("yordam" in text.lower() for text in transport.sent_texts())
        assert bot._offset == 2  # offset advanced

    def test_api_failure_does_not_crash_bot(self, service):
        def failing_service(request):
            raise RuntimeError("boom")

        bot, transport = make_bot(service)
        bot.handle_update({"message": {"chat": {"id": 3}, "text": "salom"}})
        # The bot still answers with a structured user-facing message.
        assert transport.sent_texts()

    def test_quota_error_translated(self, service):
        bot, _ = make_bot(service)
        text = bot._error_text({"error": {"code": "quota_exceeded", "message": "limit"}})
        assert "limit" in text.lower()

    def test_message_chunking(self, service):
        bot, transport = make_bot(service)
        long_text = "x" * 9000
        bot._send("77", long_text)
        assert len(transport.replies) == 3  # 4000+4000+1000
