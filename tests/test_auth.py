"""Тесты auth middleware (v5) — Basic Auth."""
from __future__ import annotations

import base64

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from app.auth import BasicAuthMiddleware, get_public_paths
from app.config import settings


def _make_app() -> FastAPI:
    """Минимальное FastAPI приложение для теста middleware."""
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/chat")
    def chat():
        return {"answer": "test"}

    @app.post("/ingest")
    def ingest():
        return {"status": "ok"}

    return app


def _make_auth_header(user: str, password: str) -> dict:
    cred = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {cred}"}


def test_public_paths_parsing():
    """get_public_paths корректно парсит строку с запятыми."""
    # Меняем настройку через monkeypatch
    paths = get_public_paths()
    assert isinstance(paths, set)
    # По умолчанию health и metrics должны быть в списке
    assert "/health" in paths
    assert "/metrics" in paths


def test_auth_disabled_no_credentials_required():
    """Если AUTH_ENABLED=False — никакие заголовки не нужны."""
    app = _make_app()
    app.add_middleware(BasicAuthMiddleware, public_paths={"/health"})
    client = TestClient(app)

    # /chat должен быть доступен без auth (auth отключён)
    settings.AUTH_ENABLED = False
    r = client.get("/chat")
    assert r.status_code == 200


def test_public_path_accessible_without_auth(monkeypatch):
    """Публичные пути (/health) доступны без auth даже если auth включён."""
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "AUTH_USER", "admin")
    monkeypatch.setattr(settings, "AUTH_PASSWORD", "secret")

    app = _make_app()
    app.add_middleware(BasicAuthMiddleware, public_paths={"/health"})
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_protected_path_requires_auth(monkeypatch):
    """Защищённый путь (/chat) требует auth."""
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "AUTH_USER", "admin")
    monkeypatch.setattr(settings, "AUTH_PASSWORD", "secret")

    app = _make_app()
    app.add_middleware(BasicAuthMiddleware, public_paths={"/health"})
    client = TestClient(app)

    # Без auth заголовка
    r = client.get("/chat")
    assert r.status_code == 401
    assert "WWW-Authenticate" in r.headers


def test_protected_path_with_valid_credentials(monkeypatch):
    """С валидными кредами — доступ есть."""
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "AUTH_USER", "admin")
    monkeypatch.setattr(settings, "AUTH_PASSWORD", "secret")

    app = _make_app()
    app.add_middleware(BasicAuthMiddleware, public_paths={"/health"})
    client = TestClient(app)

    r = client.get("/chat", headers=_make_auth_header("admin", "secret"))
    assert r.status_code == 200
    assert r.json() == {"answer": "test"}


def test_protected_path_with_invalid_credentials(monkeypatch):
    """С неверным паролем — 401."""
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "AUTH_USER", "admin")
    monkeypatch.setattr(settings, "AUTH_PASSWORD", "secret")

    app = _make_app()
    app.add_middleware(BasicAuthMiddleware, public_paths={"/health"})
    client = TestClient(app)

    r = client.get("/chat", headers=_make_auth_header("admin", "wrong"))
    assert r.status_code == 401


def test_protected_path_with_wrong_user(monkeypatch):
    """С неверным юзером — 401."""
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "AUTH_USER", "admin")
    monkeypatch.setattr(settings, "AUTH_PASSWORD", "secret")

    app = _make_app()
    app.add_middleware(BasicAuthMiddleware, public_paths={"/health"})
    client = TestClient(app)

    r = client.get("/chat", headers=_make_auth_header("wronguser", "secret"))
    assert r.status_code == 401


def test_post_endpoint_protected(monkeypatch):
    """POST endpoints тоже защищены."""
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "AUTH_USER", "admin")
    monkeypatch.setattr(settings, "AUTH_PASSWORD", "secret")

    app = _make_app()
    app.add_middleware(BasicAuthMiddleware, public_paths={"/health"})
    client = TestClient(app)

    # Без auth
    r = client.post("/ingest")
    assert r.status_code == 401

    # С auth
    r = client.post("/ingest", headers=_make_auth_header("admin", "secret"))
    assert r.status_code == 200


def test_malformed_auth_header_rejected(monkeypatch):
    """Кривой Authorization header → 401."""
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "AUTH_USER", "admin")
    monkeypatch.setattr(settings, "AUTH_PASSWORD", "secret")

    app = _make_app()
    app.add_middleware(BasicAuthMiddleware, public_paths={"/health"})
    client = TestClient(app)

    # Не Basic
    r = client.get("/chat", headers={"Authorization": "Bearer abc"})
    assert r.status_code == 401

    # Base64 кривой
    r = client.get("/chat", headers={"Authorization": "Basic !!!"})
    assert r.status_code == 401


def test_public_path_prefix_match(monkeypatch):
    """Публичный путь с prefix-матчингом: /docs/... тоже публичный."""
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "AUTH_USER", "admin")
    monkeypatch.setattr(settings, "AUTH_PASSWORD", "secret")

    app = _make_app()

    @app.get("/docs/{path:path}")
    def docs(path: str):
        return {"path": path}

    app.add_middleware(BasicAuthMiddleware, public_paths={"/health", "/docs"})
    client = TestClient(app)

    # /docs/something должен быть доступен без auth
    r = client.get("/docs/something")
    assert r.status_code == 200
