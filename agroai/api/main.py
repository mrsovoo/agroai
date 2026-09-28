"""Local HTTP server for the AgroAI API (standard library only).

```bash
python -m agroai.api.main            # host/port from AGROAI_API_HOST/PORT
agroai serve                         # the same via the CLI (if present)
```

Design:
* The **only** logic is transport: parse HTTP -> build :class:`ApiRequest` ->
  ``service.handle`` -> serialize :class:`ApiResponse`. Every capability decision
  stays inside :mod:`agroai.api.service` (one core for Web, Telegram, CLI).
* CORS for the local Web client (``AGROAI_CORS_ORIGINS``) — browsers send
  ``OPTIONS`` preflights even against localhost.
* Body size is capped (``AGROAI_MAX_BODY_BYTES``); larger requests get a
  structured 413, never a stack trace.
* No secrets are ever logged; request ids make log lines traceable.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from agroai import __version__
from agroai.api.admin import SESSION_COOKIE, AdminPanel
from agroai.api.contracts import API_PREFIX, ApiRequest
from agroai.api.service import build_service
from agroai.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

DEFAULT_MAX_BODY_BYTES = 1_000_000  # 1 MB is generous for JSON chat payloads


class _Handler(BaseHTTPRequestHandler):
    """One HTTP request -> one ApiRequest. No logic beyond transport."""

    server_version = f"AgroAI/{__version__}"
    protocol_version = "HTTP/1.1"
    # injected via partial()/server attribute:
    service: Any = None
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    cors_origins: list[str] | None = None

    # ------------------------------------------------------------------ #
    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib naming
        self._send(204, {}, preflight=True)

    # ---- Admin panel (password-guarded, HTML + JSON) ------------------ #
    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def _admin(self) -> AdminPanel:
        panel = getattr(self.server, "admin_panel", None)
        if panel is None:
            panel = AdminPanel(self.service, self.service.settings)
            self.server.admin_panel = panel
        return panel

    def _admin_guarded(self) -> bool:
        """True if the request carries a valid admin session cookie."""
        return self._admin()._session_of(self.headers.get("Cookie", "")) is not None

    def _serve_admin(self, method: str) -> None:
        import html as html_mod

        panel = self._admin()
        parsed = urlparse(self.path)
        path = parsed.path
        form = lambda: parse_qs(self._read_body().decode("utf-8", "replace"))  # noqa: E731

        if method == "POST" and path == "/admin/login":
            fields = form()
            candidate = (fields.get("password") or [""])[0]
            token = panel.login(candidate)
            if token is None:
                body = panel.render_login(error="Parol xato." if panel.password else "")
                self._send_html(403, body)
                return
            self.send_response(303)
            self.send_header("Location", "/admin")
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE}={token}; HttpOnly; Path=/; SameSite=Lax; Max-Age=43200",
            )
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if path == "/admin/login":
            self._send_html(200, panel.render_login())
            return
        if path == "/admin/logout":
            panel.logout(panel._session_of(self.headers.get("Cookie", "")))
            self.send_response(303)
            self.send_header("Location", "/admin")
            self.send_header(
                "Set-Cookie", f"{SESSION_COOKIE}=; HttpOnly; Path=/; Max-Age=0"
            )
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if not self._admin_guarded():
            # Not authenticated: show the login form (200 so browsers render it;
            # protected JSON endpoints below still need the session).
            if path == "/admin" or path == "/admin/":
                self._send_html(200, panel.render_login())
                return
            self._send_html(401, panel.render_login())
            return

        if path == "/admin/api/stats":
            self._send_json(200, panel.stats())
            return
        if path == "/admin/api/health":
            self._send_json(200, panel.health())
            return
        if method == "POST" and path == "/admin/api/ingest":
            fields = form()
            path_arg = (fields.get("path") or [""])[0].strip()
            if not path_arg:
                self._send_json(400, {"error": {"code": "bad_request", "message": "'path' kerak."}})
                return
            self._send_json(200, panel.ingest_path(path_arg))
            return
        if path == "/admin":
            self._send_html(200, panel.render_dashboard(panel.stats()))
            return
        self._send_html(404, "<h1>404</h1>" + html_mod.escape(path))

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        return self.rfile.read(min(length, self.max_body_bytes))

    def _send_html(self, status: int, markup: str) -> None:
        raw = markup.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path.startswith("/admin"):
            self._serve_admin("GET")
            return
        self._route("GET")

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path.startswith("/admin"):
            self._serve_admin("POST")
            return
        self._route("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._route("DELETE")

    def do_PUT(self) -> None:  # noqa: N802
        self._route("PUT")

    def do_PATCH(self) -> None:  # noqa: N802
        self._route("PATCH")

    # ------------------------------------------------------------------ #
    def _route(self, method: str) -> None:
        parsed = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(parsed.query).items()}
        body: dict[str, Any] = {}
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            if length > self.max_body_bytes:
                self._send_json(
                    413,
                    {
                        "error": {
                            "code": "payload_too_large",
                            "message": f"Request body exceeds {self.max_body_bytes} bytes.",
                        }
                    },
                )
                return
            raw = self.rfile.read(length)
            try:
                parsed_body = json.loads(raw.decode("utf-8"))
                if isinstance(parsed_body, dict):
                    body = parsed_body
                else:
                    raise ValueError
            except (ValueError, UnicodeDecodeError):
                self._send_json(
                    400,
                    {
                        "error": {
                            "code": "bad_request",
                            "message": "The request body must be a JSON object.",
                        }
                    },
                )
                return

        request = ApiRequest(
            method=method,
            path=parsed.path,
            body=body,
            query=query,
            headers=dict(self.headers.items()),
        )
        response = self.service.handle(request)
        self._send_json(response.status, response.body, extra_headers=response.headers)

    # ------------------------------------------------------------------ #
    def _cors_headers(self) -> dict[str, str]:
        origin = self.headers.get("Origin", "")
        headers: dict[str, str] = {}
        if not origin:
            return headers
        allowed = self.cors_origins
        if allowed is None or "*" in allowed or origin in allowed:
            headers["Access-Control-Allow-Origin"] = origin
            headers["Vary"] = "Origin"
            headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key, X-User-Id, Authorization"
            headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
            headers["Access-Control-Max-Age"] = "600"
        return headers

    def _send(self, status: int, headers: dict[str, str], *, preflight: bool = False) -> None:
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        for key, value in self._cors_headers().items():
            self.send_header(key, value)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_json(
        self,
        status: int,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        for key, value in self._cors_headers().items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # One line per request, no headers (so no tokens can leak via logs).
        logger.info("%s - %s", self.address_string(), format % args)


def make_server(
    *,
    settings: Settings | None = None,
    service: Any = None,
    host: str | None = None,
    port: int | None = None,
) -> ThreadingHTTPServer:
    """Build (but do not start) the local API server."""
    settings = settings or get_settings()
    svc = service or build_service(settings)
    handler = type(
        "BoundHandler",
        (_Handler,),
        {
            "service": svc,
            "max_body_bytes": DEFAULT_MAX_BODY_BYTES,
            "cors_origins": [o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        },
    )
    # None-checks (not truthiness) so port=0 keeps its "pick an ephemeral port" meaning.
    address = (host if host is not None else settings.api_host, int(port) if port is not None else settings.api_port)
    server = ThreadingHTTPServer(address, handler, bind_and_activate=False)
    server.allow_reuse_address = True
    try:
        server.server_bind()
        server.server_activate()
    except OSError:
        server.server_close()
        raise
    return server


def main(argv: list[str] | None = None) -> int:
    """Run the local API server (blocking). ``python -m agroai.api.main``."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")
    server = make_server(settings=settings)
    host, port = server.server_address[:2]
    print(f"AgroAI API {__version__} listening on http://{host}:{port}{API_PREFIX} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
        service = server.RequestHandlerClass.service
        close = getattr(service, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
