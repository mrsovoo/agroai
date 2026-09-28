"""Super admin panel: stats + management over plain HTTP (stdlib only).

```text
GET  /admin                 login page / dashboard HTML
POST /admin/login           password check (AGROAI_ADMIN_PASSWORD), sets a cookie
GET  /admin/logout          clears the session cookie
GET  /admin/api/stats       JSON stats (auth required)
POST /admin/api/ingest      ingest a file path into the knowledge base (auth)
GET  /admin/api/health      full health report (auth)
```

Security:
* Password lives in ``AGROAI_ADMIN_PASSWORD`` (env/.env) — never in code.
* Session cookie is a random token stored in memory; HttpOnly; no secrets logged.
* If no password is configured the panel reports NOT CONFIGURED instead of
  silently allowing everyone (no fake security).
"""

from __future__ import annotations

import html
import json
import secrets
from http.cookies import SimpleCookie
from typing import Any

from agroai import __version__
from agroai.api.contracts import ApiRequest
from agroai.config.settings import Settings

SESSION_COOKIE = "agroai_admin_session"


class AdminPanel:
    """Session-guarded admin dashboard rendered as plain HTML."""

    def __init__(self, service: Any, settings: Settings | None = None) -> None:
        self.service = service
        self.settings = settings or service.settings
        self._sessions: set[str] = set()

    # ------------------------------------------------------------------ #
    @property
    def password(self) -> str:
        return getattr(self.settings, "admin_password", "") or ""

    def check_password(self, candidate: str) -> bool:
        """Constant-time compare; empty password never matches."""
        if not self.password:
            return False
        return secrets.compare_digest(candidate, self.password)

    def login(self, candidate: str) -> str | None:
        """Return a new session token on success, else None."""
        if not self.check_password(candidate):
            return None
        token = secrets.token_urlsafe(32)
        self._sessions.add(token)
        return token

    def _session_of(self, cookie_header: str) -> str | None:
        cookie = SimpleCookie()
        try:
            cookie.load(cookie_header or "")
        except Exception:  # noqa: BLE001 - malformed cookie = no session
            return None
        token = cookie[SESSION_COOKIE].value if SESSION_COOKIE in cookie else ""
        return token if token in self._sessions else None

    def logout(self, token: str | None) -> None:
        self._sessions.discard(token or "")

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, Any]:
        """Everything the dashboard shows, from real services only."""
        database = self.service.database
        knowledge = self.service.knowledge.stats()
        retrieval = self.service.retriever.health_check()
        llm = self.service.agent.llm.health_check()
        try:
            candidates = self.service.knowledge.list_candidates(limit=10)
        except Exception:  # noqa: BLE001
            candidates = []
        return {
            "version": __version__,
            "documents": database.count_documents(),
            "chunks": database.count_chunks(),
            "sources": len(database.list_sources()),
            "conversations": len(self.service._conversations.list_conversations(limit=200))
            if getattr(self.service, "_conversations", None)
            else 0,
            "knowledge": knowledge,
            "retrieval": {
                "backend": retrieval.backend,
                "ready": retrieval.ready,
                "mode": retrieval.search_mode,
            },
            "llm": {"provider": llm.provider, "configured": llm.configured, "detail": llm.detail},
            "candidates": [
                {
                    "id": c.id,
                    "status": getattr(getattr(c, "status", None), "value", c.status),
                    "statement": c.statement[:80],
                }
                for c in candidates
            ],
        }

    def ingest_path(self, path: str, overwrite: bool = False) -> dict[str, Any]:
        """Ingest one document by server-side path (admin action)."""
        from pathlib import Path

        report = self.service.knowledge.ingest_file(Path(path), overwrite=overwrite)
        return {
            "seen": report.documents_seen,
            "ingested": report.documents_ingested,
            "skipped": report.documents_skipped,
            "failed": report.documents_failed,
            "chunks": report.chunks_created,
            "errors": list(report.errors),
        }

    def health(self) -> dict[str, Any]:
        response = self.service.handle(ApiRequest("GET", "/api/v1/health"))
        return response.body

    # ------------------------------------------------------------------ #
    # HTML
    # ------------------------------------------------------------------ #
    def render_login(self, error: str = "") -> str:
        note = f'<p class="error">{html.escape(error)}</p>' if error else ""
        if not self.password:
            note = (
                '<p class="error">Admin paroli sozlanmagan: .env faylida '
                "AGROAI_ADMIN_PASSWORD ni belgilang va serverni qayta ishga tushiring.</p>"
            )
        return (
            "<!doctype html><html lang='uz'><head><meta charset='utf-8'>"
            "<title>AgroAI Admin</title><style>"
            "body{font-family:system-ui;background:#0f172a;color:#e2e8f0;display:flex;"
            "justify-content:center;align-items:center;height:100vh;margin:0}"
            "form{background:#1e293b;padding:2rem;border-radius:12px;width:320px}"
            "input{width:100%;padding:.6rem;margin:.4rem 0;border-radius:8px;border:1px solid #334155;"
            "background:#0f172a;color:#e2e8f0;box-sizing:border-box}"
            "button{width:100%;padding:.6rem;background:#16a34a;color:white;border:0;"
            "border-radius:8px;font-weight:600;cursor:pointer}"
            ".error{color:#f87171;font-size:.9rem}"
            "</style></head><body><form method='post' action='/admin/login'>"
            "<h2>🌱 AgroAI Admin</h2>" + note +
            "<input type='password' name='password' placeholder='Parol' autofocus>"
            "<button>Kirish</button></form></body></html>"
        )

    def render_dashboard(self, data: dict[str, Any]) -> str:
        rows = [
            ("Hujjatlar (documents)", data["documents"]),
            ("Bo'laklar (chunks)", data["chunks"]),
            ("Manbalar (sources)", data["sources"]),
            ("Suhbatlar (conversations)", data["conversations"]),
            ("Retrieval", f"{data['retrieval']['backend']} / {data['retrieval']['mode']} ({'ready' if data['retrieval']['ready'] else 'DOWN'})"),
            ("LLM", "sozlangan" if data["llm"]["configured"] else "SOZLANMAGAN"),
            ("Bilim bazasi", json.dumps(data["knowledge"], ensure_ascii=False)[:120]),
        ]
        rows_html = "".join(
            f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
            for k, v in rows
        )
        candidate_rows = "".join(
            f"<li>#{html.escape(c['id'])} [{html.escape(str(c['status']))}] {html.escape(c['statement'])}</li>"
            for c in data["candidates"]
        ) or "<li>yo'q</li>"
        return (
            "<!doctype html><html lang='uz'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>AgroAI Admin</title><style>"
            "body{font-family:system-ui;background:#0f172a;color:#e2e8f0;margin:0;padding:2rem}"
            "h1{color:#4ade80} table{border-collapse:collapse;width:100%;max-width:760px}"
            "td{border:1px solid #334155;padding:.55rem .8rem} tr:first-child td{background:#1e293b}"
            "a{color:#4ade80} .card{background:#1e293b;border-radius:12px;padding:1rem 1.5rem;"
            "margin-bottom:1rem;max-width:760px}"
            "</style></head><body>"
            "<h1>🌱 AgroAI Super Admin</h1>"
            f"<p>v{data['version']} · <a href='/admin/logout'>Chiqish</a></p>"
            "<div class='card'><h3>Umumiy holat</h3><table>" + rows_html + "</table></div>"
            "<div class='card'><h3>Bilim nomzodlari (so'nggi 10)</h3><ul>" + candidate_rows + "</ul></div>"
            "</body></html>"
        )
