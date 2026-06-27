"""
app/llm.py — клиент к локальному LLM (vLLM OpenAI-compatible) с async streaming.

Изменения v4 (перенесено из petrel-rag-assistant):
- async/await нативно (httpx.AsyncClient)
- Streaming ответа → измерение реального TTFT (time to first token)
- keep_alive / prefix caching для снижения TTFT на повторных запросах

Если vLLM не поддерживает streaming в каком-то режиме, можно отключить через
LLM_STREAM=False в .env (тогда ответ ждём целиком, TTFT = latency).

Также реализован локальный LLM-as-judge для groundedness метрики
(см. evaluation_runner.py) — критично для On-Premise: RAGAS по умолчанию
ходит в OpenAI, что нарушает ТЗ.
"""

import json
import logging
import time
from typing import Optional, Tuple

import httpx
from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async OpenAI-compatible client (singleton)
# ---------------------------------------------------------------------------
_client: Optional[AsyncOpenAI] = None


def get_llm_client() -> AsyncOpenAI:
    """Возвращает singleton AsyncOpenAI для общения с vLLM."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=settings.vllm_base_url,
            api_key="EMPTY",
            timeout=settings.LLM_TIMEOUT,
        )
    return _client


# ---------------------------------------------------------------------------
# Async streaming chat с измерением TTFT
# ---------------------------------------------------------------------------

async def chat_stream(
    system: str,
    user: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[str, int]:
    """
    Async chat со streaming-ответом.

    Возвращает (текст_ответа, ttft_ms).
    ttft_ms = время от отправки запроса до получения первого токена.

    Streaming критичен для UX: пользователь видит ответ по мере генерации,
    а не ждёт 5-10 секунд до полного завершения.
    """
    client = get_llm_client()
    start = time.perf_counter()
    ttft: Optional[int] = None
    parts: list = []

    try:
        stream = await client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
            top_p=settings.LLM_TOP_P,
            max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            token = getattr(delta, "content", None) or ""
            if token:
                if ttft is None:
                    ttft = int((time.perf_counter() - start) * 1000)
                parts.append(token)
    except Exception as exc:
        logger.exception("LLM streaming call failed")
        raise

    return "".join(parts).strip(), (ttft or 0)


async def chat_sync(
    system: str,
    user: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[str, int]:
    """
    Async chat без streaming (когда streaming недоступен или не нужен).
    TTFT ≈ время до полного ответа.
    """
    client = get_llm_client()
    start = time.perf_counter()

    try:
        completion = await client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
            top_p=settings.LLM_TOP_P,
            max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
            stream=False,
        )
    except Exception as exc:
        logger.exception("LLM sync call failed")
        raise

    answer = (completion.choices[0].message.content or "").strip()
    ttft = int((time.perf_counter() - start) * 1000)
    return answer, ttft


async def chat(system: str, user: str, **kwargs) -> Tuple[str, int]:
    """Унифицированный интерфейс: streaming или sync в зависимости от LLM_STREAM."""
    if settings.LLM_STREAM:
        return await chat_stream(system, user, **kwargs)
    return await chat_sync(system, user, **kwargs)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

async def health() -> dict:
    """Доступен ли vLLM и какая модель загружена."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as cli:
            r = await cli.get(f"{settings.vllm_base_url}/models")
            r.raise_for_status()
            data = r.json()
            models = [m.get("id", "") for m in data.get("data", [])]
        wanted = settings.LLM_MODEL
        ok = any(m == wanted or m.startswith(wanted.split("/")[0]) for m in models)
        return {"reachable": True, "model_loaded": ok, "models": models}
    except Exception as exc:
        return {"reachable": False, "model_loaded": False, "error": str(exc)}
