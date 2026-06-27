"""
app/auth.py — Basic Auth middleware (v5).

Опциональная защита /chat, /ingest, /documents, /evaluation.
/health, /metrics, /docs — остаются публичными.

Включается через:
    AUTH_ENABLED=True
    AUTH_USER=admin
    AUTH_PASSWORD=secret

В .env. По умолчанию отключено (AUTH_ENABLED=False) для удобства разработки.
"""

import base64
import logging
import secrets
from typing import Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings

logger = logging.getLogger(__name__)


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """
    HTTP Basic Auth middleware.
    Использует secrets.compare_digest для защиты от timing-атак.
    """

    def __init__(self, app, public_paths: set = None):
        super().__init__(app)
        # Нормализуем public_paths
        self.public_paths = public_paths or set()
        logger.info("BasicAuth middleware enabled. Public paths: %s", self.public_paths)

    async def dispatch(self, request: Request, call_next):
        # Auth отключён — пропускаем всё
        if not settings.AUTH_ENABLED:
            return await call_next(request)

        # Публичный путь — пропускаем
        path = request.url.path
        if self._is_public(path):
            return await call_next(request)

        # Проверяем Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Basic "):
            return self._unauthorized_response()

        # Декодируем
        try:
            encoded = auth_header[6:]  # "Basic " → 6 символов
            decoded = base64.b64decode(encoded).decode("utf-8")
            username, password = decoded.split(":", 1)
        except Exception:
            return self._unauthorized_response()

        # Проверяем (timing-safe)
        user_ok = secrets.compare_digest(username, settings.AUTH_USER)
        pass_ok = secrets.compare_digest(password, settings.AUTH_PASSWORD)
        if not (user_ok and pass_ok):
            logger.warning("Auth failed for user='%s' from %s",
                           username, request.client.host if request.client else "?")
            return self._unauthorized_response()

        return await call_next(request)

    def _is_public(self, path: str) -> bool:
        """Проверяет, является ли путь публичным (точное совпадение или prefix)."""
        for public in self.public_paths:
            if path == public or path.startswith(public + "/"):
                return True
        return False

    def _unauthorized_response(self) -> Response:
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized. Provide valid Basic Auth credentials."},
            headers={"WWW-Authenticate": 'Basic realm="RAG Assistant"'},
        )


def get_public_paths() -> set:
    """Возвращает множество публичных путей из настроек."""
    paths_str = settings.AUTH_PUBLIC_PATHS or ""
    return {p.strip() for p in paths_str.split(",") if p.strip()}
