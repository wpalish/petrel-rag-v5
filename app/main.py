"""
app/main.py — entrypoint для FastAPI backend (v4).

Изменения v4:
- lifespan context manager (вместо deprecated @app.on_event)
- async endpoints нативно
- TTFT метрика в /chat ответе (streaming)

Запуск:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import logging

import uvicorn
from fastapi import FastAPI

from app.api import build_app
from app.config import settings

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app: FastAPI = build_app()


if __name__ == "__main__":
    # reload=False — в production reload не нужен (утечки RAM, race conditions)
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
