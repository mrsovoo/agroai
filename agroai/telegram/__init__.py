"""AgroAI Telegram bot: a thin client of the one shared AgroAI core.

```text
Telegram  ->  this bot  ->  AgroAIService (the same core Web uses)  ->  answer
```

Principles enforced here:

* **No AI logic in the bot.** Every question goes through the shared service;
  the bot only transports text/images and renders user-facing answers.
* **Token from the environment only** (``AGROAI_TELEGRAM_BOT_TOKEN``). It is
  never printed, never logged, never committed.
* **Honest media handling.** Images/videos are downloaded and passed to the real
  vision layer; with no vision model the bot says analysis is not configured and
  never pretends to diagnose.
* **Per-chat conversations** persist through the shared
  :class:`~agroai.product.conversations.ConversationStore`, so context survives
  restarts and different chats never mix memory.

Run locally (long polling, no public URL needed):

```bash
export AGROAI_TELEGRAM_BOT_TOKEN=123:abc        # from BotFather, in .env
python -m agroai.telegram
```

The Telegram HTTP API is called with ``urllib`` (standard library): no SDK
dependency, and tests can inject a fake transport function.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agroai import __version__
from agroai.api.contracts import ApiError, ApiRequest
from agroai.config.settings import Settings, get_settings
from agroai.product.conversations import ConversationStore

logger = logging.getLogger(__name__)

TELEGRAM_ENV_TOKEN = "AGROAI_TELEGRAM_BOT_TOKEN"
TELEGRAM_API_BASE = "https://api.telegram.org"

#: Telegram's own upload cap is 20 MB for photos via Bot API (bot can download
#: what Telegram already stored); this local cap protects the machine.
MAX_MEDIA_BYTES = 20_000_000
MAX_MESSAGE_LEN = 4000  # Telegram message hard limit is 4096; stay under.

WELCOME_TEXT = (
    "🌱 <b>AgroAI</b> — qishloq xo'jaligi va veterinariya yordamchisi.\n\n"
    "Savolingizni yozing: ekinlar, kasalliklar, tuproq, sug'orish, chorva, "
    "veterinariya.\n\n"
    "Buyruqlar:\n"
    "/new — yangi suhbat\n"
    "/clear — suhbatni tozalash\n"
    "/sources — so'nggi javob manbalari\n"
    "/tools — mavjud hisob vositalari\n"
    "/help — yordam"
)

HELP_TEXT = (
    "<b>AgroAI yordam</b>\n\n"
    "Savol misollari:\n"
    "• Pomidor barglarida sariq dog'lar — nima bo'lishi mumkin?\n"
    "• Tuproq EC 6.5 dS/m — sho'rlanishmi?\n"
    "• 5 gektar bug'doyga qancha suv kerak?\n\n"
    "Buyruqlar: /new /clear /sources /tools /help\n\n"
    "⚠️ AgroAI javoblari mahalliy bilimlar bazasiga asoslanadi; dorilar va "
    "dosajlar uchun mutaxannisga murojaat qiling."
)

MODEL_NOT_CONFIGURED_TEXT = (
    "🤖 AI model hozircha sozlanmagan, lekin mahalliy bilimlar bazasidan "
    "topilgan ma'lumotlar va manbalar bilan yordam beraman."
)

VISION_NOT_CONFIGURED_TEXT = (
    "🖼 Rasm qabul qilindi, ammo rasm tahlili uchun Vision modeli hali "
    "sozlanmagan. Hozircha savolingizni matn yozib berishingiz mumkin: "
    "belgilarni tasvirlab bering (qachon boshlangan, qaysi o'simlik/hayvon, "
    "qanday tarqalmoqda)."
)

VIDEO_NOT_CONFIGURED_TEXT = (
    "🎬 Video tahlil hali mavjud emas. Savolingizni matn yozib berishingiz mumkin."
)

QUOTA_TEXT = (
    "⏳ Kunlik limitga yetdingiz. Ertaga qayta urinib ko'ring yoki pro rejimga "
    "o'ting."
)

EMPTY_KB_TEXT = (
    "📭 Bilimlar bazasida bu savolga mos tasdiqlangan ma'lumot topilmadi. "
    "Aniqroq yozib ko'ring yoki savolingizni boshqacha bering."
)


# ---------------------------------------------------------------------- #
# Telegram transport (isolated for tests)
# ---------------------------------------------------------------------- #
def telegram_api_call(
    token: str, method: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    """One Bot API call. Never logs the token; raises TelegramError on failure."""
    url = f"{TELEGRAM_API_BASE}/bot{token}/{method}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=35) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = str(json.loads(exc.read().decode("utf-8")).get("description", ""))
        except Exception:  # noqa: BLE001 - error body is best-effort
            pass
        raise TelegramError(f"{method} failed: HTTP {exc.code} {detail}".strip()) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TelegramError(f"{method} failed: {type(exc).__name__}") from exc
    if not body.get("ok"):
        raise TelegramError(f"{method} failed: {body.get('description', 'unknown error')}")
    return body.get("result", {})


def download_file(token: str, file_path: str) -> bytes:
    """Download a file by its Telegram ``file_path`` (size-capped)."""
    url = f"{TELEGRAM_API_BASE}/file/bot{token}/{file_path}"
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(MAX_MEDIA_BYTES + 1)
    if len(data) > MAX_MEDIA_BYTES:
        raise TelegramError("File exceeds the local size limit.")
    return data


class TelegramError(RuntimeError):
    """A failure talking to the Telegram Bot API."""


# ---------------------------------------------------------------------- #
# Bot
# ---------------------------------------------------------------------- #
@dataclass
class BotConfig:
    """Everything the bot needs; built once from settings/environment."""

    token: str
    enabled: bool = True
    mode: str = "polling"
    poll_interval_seconds: float = 2.0
    media_dir: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "agroai-media")

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "BotConfig":
        settings = settings or get_settings()
        token = settings.telegram_bot_token or os.environ.get(TELEGRAM_ENV_TOKEN, "").strip()
        return cls(
            token=token,
            enabled=settings.telegram_enabled,
            mode=settings.telegram_mode,
        )


def describe_token_state(token: str) -> str:
    """Configured/missing status WITHOUT ever revealing the value."""
    return "CONFIGURED" if token and token.strip() else "NOT CONFIGURED"


class AgroAITelegramBot:
    """Polling Telegram client of the shared AgroAI service."""

    def __init__(
        self,
        *,
        config: BotConfig | None = None,
        service: Any = None,
        transport: Callable[[str, str, dict[str, Any] | None], dict[str, Any]] | None = None,
        downloader: Callable[[str, str], bytes] | None = None,
    ) -> None:
        self.config = config or BotConfig.from_settings()
        self.transport = transport or telegram_api_call
        self.downloader = downloader or download_file
        self._service = service
        self._offset = 0

    # ------------------------------------------------------------------ #
    @property
    def service(self) -> Any:
        if self._service is None:
            from agroai.api.service import build_service

            self._service = build_service()
        return self._service

    @property
    def conversations(self) -> ConversationStore:
        from agroai.api.conversations import conversations_service

        return conversations_service(self.service)

    def close(self) -> None:
        close = getattr(self.service, "close", None)
        if callable(close):
            close()

    # ------------------------------------------------------------------ #
    # Low-level helpers
    # ------------------------------------------------------------------ #
    def _call(self, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.transport(self.config.token, method, payload)

    # ------------------------------------------------------------------ #
    # Update handling
    # ------------------------------------------------------------------ #
    def handle_update(self, update: dict[str, Any]) -> None:
        """Route one Telegram update. Never raises to the polling loop."""
        try:
            message = update.get("message") or update.get("edited_message")
            if not message:
                return
            chat_id = str(message.get("chat", {}).get("id", "")).strip()
            if not chat_id:
                return
            text = str(message.get("text") or "").strip()
            if text.startswith("/"):
                self._handle_command(chat_id, text)
                return
            if text:
                self._handle_text(chat_id, text)
                return
            photo = message.get("photo")
            if photo:
                self._handle_photo(chat_id, photo)
                return
            video = message.get("video") or message.get("video_note")
            if video:
                self._send(chat_id, VIDEO_NOT_CONFIGURED_TEXT)
                return
            document = message.get("document")
            if document:
                self._send(chat_id, VISION_NOT_CONFIGURED_TEXT)
                return
        except TelegramError as exc:
            logger.warning("Telegram API failure: %s", exc)
            chat_id = str((update.get("message") or {}).get("chat", {}).get("id", ""))
            if chat_id:
                self._send(chat_id, "⌛ Telegram bilan aloqa muammosi. Keyinroq qayta urinib ko'ring.")
        except Exception:  # noqa: BLE001 - never leak stack traces to users
            logger.exception("Unhandled update error")

    # ------------------------------------------------------------------ #
    def _handle_command(self, chat_id: str, text: str) -> None:
        command = text.split("@", 1)[0].split(" ", 1)[0].lower()
        if command == "/start":
            self._send(chat_id, WELCOME_TEXT)
            return
        if command == "/help":
            self._send(chat_id, HELP_TEXT)
            return
        if command == "/new":
            conversation = self.conversations.create_conversation(
                user_id=f"tg:{chat_id}", title=""
            )
            self._send(chat_id, "🆕 Yangi suhbat boshlandi. Savolingizni yozing.")
            return
        if command == "/clear":
            count = self._clear_chat_conversations(chat_id)
            self._send(chat_id, f"🧹 {count} suhbat tarixi tozalandi.")
            return
        if command == "/sources":
            self._send_last_sources(chat_id)
            return
        if command == "/tools":
            self._send(chat_id, self._tools_text())
            return
        self._send(chat_id, "Noma'lum buyruq. /help ni bosing.")

    def _clear_chat_conversations(self, chat_id: str) -> int:
        user_id = f"tg:{chat_id}"
        found = self.conversations.list_conversations(user_id=user_id, limit=200)
        removed = 0
        for conversation in found:
            if self.conversations.delete_conversation(conversation.id, user_id=user_id):
                removed += 1
        return removed

    def _tools_text(self) -> str:
        registry = self.service.agent.tools
        lines = ["🧰 <b>Mavjud hisob vositalari:</b>"]
        for tool in registry.tools.values():
            lines.append(f"• <b>{tool.name}</b> — {tool.description}")
        lines.append("")
        lines.append("Hisob-kitob faqat siz bergan raqamlar bilan.")
        return "\n".join(lines)

    def _send_last_sources(self, chat_id: str) -> None:
        user_id = f"tg:{chat_id}"
        found = self.conversations.list_conversations(user_id=user_id, limit=1)
        if not found:
            self._send(chat_id, "Hozircha manbalar yo'q — avval savol bering.")
            return
        messages = self.conversations.list_messages(found[0].id, limit=50)
        sources: list[str] = []
        for message in reversed(messages):
            if message.role == "assistant":
                for source in message.metadata.get("sources", []):
                    line = source.get("title") or source.get("source") or ""
                    if line and line not in sources:
                        sources.append(line)
                if sources:
                    break
        if not sources:
            self._send(chat_id, "So'nggi javobda manba ko'rsatilmagan.")
            return
        self._send(chat_id, "📚 Manbalar:\n" + "\n".join(f"• {s}" for s in sources[:5]))

    # ------------------------------------------------------------------ #
    def _get_or_create_conversation(self, chat_id: str) -> Any:
        user_id = f"tg:{chat_id}"
        found = self.conversations.list_conversations(user_id=user_id, limit=1)
        return found[0] if found else self.conversations.create_conversation(user_id=user_id)

    def _handle_text(self, chat_id: str, text: str) -> None:
        conversation = self._get_or_create_conversation(chat_id)
        request = ApiRequest(
            method="POST",
            path="/api/v1/chat",
            body={
                "conversation_id": conversation.id,
                "message": text[:MAX_MESSAGE_LEN],
                "language": "uz",
            },
            user_id=f"tg:{chat_id}",
        )
        response = self.service.handle(request)
        payload = response.body
        if not response.ok:
            self._send(chat_id, self._error_text(payload))
            return
        answer = str(payload.get("answer") or "").strip()
        if payload.get("status") == "not_configured":
            answer = f"{MODEL_NOT_CONFIGURED_TEXT}\n\n{answer}"
        sources = payload.get("sources") or []
        if sources:
            lines = ["📚 Manbalar:"]
            for index, source in enumerate(sources[:5], start=1):
                title = source.get("title") or source.get("source") or "manba"
                url = source.get("url") or ""
                lines.append(f"{index}. {title}" + (f" ({url})" if url else ""))
            answer = f"{answer}\n\n" + "\n".join(lines)
        limitations = payload.get("limitations") or []
        if limitations:
            answer += "\n\nℹ️ " + limitations[0]
        self._send(chat_id, answer)
        self._store_assistant_metadata(conversation.id, payload)

    def _store_assistant_metadata(self, conversation_id: str, payload: dict[str, Any]) -> None:
        """Attach source titles to the stored assistant message for /sources."""
        messages = self.conversations.list_messages(conversation_id, limit=50)
        if not messages:
            return
        last = messages[-1]
        if last.role != "assistant":
            return
        sources = [
            {"title": s.get("title") or s.get("source") or "", "url": s.get("url") or ""}
            for s in (payload.get("sources") or [])[:5]
        ]
        metadata = dict(last.metadata)
        metadata["sources"] = sources
        self.conversations._database.connection.execute(
            "UPDATE messages SET metadata = ? WHERE id = ?",
            (json.dumps(metadata, ensure_ascii=False), last.id),
        )
        self.conversations._database.connection.commit()

    def _handle_photo(self, chat_id: str, photo_sizes: list[dict[str, Any]]) -> None:
        """Transport layer for images: download + pass to the real vision layer.

        With no vision model the service returns an honest not-configured state
        and the bot relays it — never a fake diagnosis.
        """
        largest = max(photo_sizes, key=lambda item: int(item.get("file_size") or 0))
        file_id = largest.get("file_id", "")
        try:
            file_info = self._call("getFile", {"file_id": file_id})
            file_path = str(file_info.get("file_path") or "")
            if not file_path:
                raise TelegramError("getFile returned no file_path")
            data = self.downloader(self.config.token, file_path)
            if len(data) > MAX_MEDIA_BYTES:
                self._send(chat_id, "Fayl hajmi juda katta (limit 20 MB).")
                return
            suffix = Path(file_path).suffix or ".jpg"
            self.config.media_dir.mkdir(parents=True, exist_ok=True)
            tmp = tempfile.NamedTemporaryFile(
                prefix="tg_", suffix=suffix, dir=self.config.media_dir, delete=False
            )
            tmp.write(data)
            tmp.close()
            try:
                request = ApiRequest(
                    method="POST",
                    path="/api/v1/vision",
                    body={"path": tmp.name, "question": "", "kind": "image"},
                    user_id=f"tg:{chat_id}",
                )
                response = self.service.handle(request)
                if response.ok:
                    configured = bool(response.body.get("vision", {}).get("configured"))
                    if not configured:
                        self._send(chat_id, VISION_NOT_CONFIGURED_TEXT)
                    else:
                        self._send(chat_id, str(response.body.get("answer") or "")[:MAX_MESSAGE_LEN])
                    return
                self._send(chat_id, VISION_NOT_CONFIGURED_TEXT)
            finally:
                # Delete the temp file; never leave user media on disk.
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
        except TelegramError as exc:
            logger.warning("Image transport failed: %s", exc)
            self._send(chat_id, VISION_NOT_CONFIGURED_TEXT)

    # ------------------------------------------------------------------ #
    def _error_text(self, payload: dict[str, Any]) -> str:
        error = payload.get("error") or {}
        code = error.get("code", "")
        if code == "quota_exceeded":
            return QUOTA_TEXT
        if code == "not_configured":
            return MODEL_NOT_CONFIGURED_TEXT
        if code == "not_found":
            return "Suhbat topilmadi. /new bilan yangi suhbat boshlang."
        logger.warning("API error for user: %s", code)
        return "Kechirasiz, xatolik yuz berdi. Keyinroq qayta urinib ko'ring."

    # ------------------------------------------------------------------ #
    def _send(self, chat_id: str, text: str, **kwargs: Any) -> None:
        """Send one message (chunked to Telegram's length limit)."""
        text = text or "(bo'sh javob)"
        while text:
            chunk, text = text[:MAX_MESSAGE_LEN], text[MAX_MESSAGE_LEN:]
            self._call(
                "sendMessage",
                {"chat_id": chat_id, "text": chunk, "parse_mode": "HTML", **kwargs},
            )

    # ------------------------------------------------------------------ #
    # Polling loop
    # ------------------------------------------------------------------ #
    def poll_once(self) -> int:
        """One getUpdates cycle; returns the number of updates processed."""
        result = self._call(
            "getUpdates",
            {"offset": self._offset, "timeout": 25, "allowed_updates": ["message", "edited_message"]},
        )
        updates = result if isinstance(result, list) else []
        for update in updates:
            self._offset = max(self._offset, int(update.get("update_id", 0)) + 1)
            self.handle_update(update)
        return len(updates)

    def run(self) -> int:
        """Long-polling loop (local development mode). Ctrl+C stops."""
        if not self.config.token:
            print(
                f"Telegram bot tokeni yo'q ({describe_token_state(self.config.token)}). "
                f"{TELEGRAM_ENV_TOKEN} ni .env faylida belgilang."
            )
            return 1
        me = self._call("getMe", None)
        username = me.get("username", "?")
        print(f"AgroAI Telegram boti ishga tushdi: @{username} (polling, Ctrl+C to stop)")
        try:
            while True:
                processed = self.poll_once()
                if not processed:
                    time.sleep(self.config.poll_interval_seconds)
        except KeyboardInterrupt:
            print("\nBot to'xtatildi.")
        return 0


def main(argv: list[str] | None = None) -> int:
    """``python -m agroai.telegram`` — run the local polling bot."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")
    bot = AgroAITelegramBot()
    return bot.run()


if __name__ == "__main__":
    raise SystemExit(main())
