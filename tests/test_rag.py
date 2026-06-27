"""Тесты ядра RAG (v5): анти-галлюцинации, язык, цитаты, TTFT — без реального LLM.

Тяжёлые зависимости (LLM, embeddings) подменяются заглушками через monkeypatch,
поэтому тесты быстрые и детерминированные.
"""
from __future__ import annotations

import asyncio
from typing import List

from app.prompts import NO_ANSWER_KK, NO_ANSWER_RU
from app.rag_engine import RAGEngine


class FakeRAGEngine(RAGEngine):
    """Подкласс RAGEngine с подменённым retrieve() (без Qdrant/LLM в __init__)."""

    def __init__(self, hits: List[dict]):
        # Не вызываем __init__ родителя — он создаёт Qdrant-клиент и т.д.
        self._fake_hits = hits

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        return self._fake_hits[:top_k] if top_k else self._fake_hits


def _patch_llm(monkeypatch, response: str, ttft: int) -> None:
    """Подменяет app.rag_engine.llm_chat детерминированной заглушкой."""
    async def _fake(system: str, user: str, **kwargs):
        return response, ttft

    monkeypatch.setattr("app.rag_engine.llm_chat", _fake)


def _make_hit(text: str, score: float = 0.9, rerank: float = 0.9, doc_name: str = "reglament.pdf",
              page: int = 2, section: str = "Раздел 3") -> dict:
    return {
        "text": text, "doc_id": "test_doc", "doc_name": doc_name, "doc_version": "v1.0",
        "page_number": page, "section_title": section, "section_path": str(page),
        "paragraph_index": 1, "parent_id": None, "chunk_id": "test_doc_p0000_c0000",
        "language": "ru", "score": score, "rerank_score": rerank,
    }


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_relevant_context_returns_honest_refusal(monkeypatch):
    """Если retrieval пустой → честный отказ, БЕЗ вызова LLM."""
    engine = FakeRAGEngine(hits=[])

    async def _boom(*a, **k):
        raise AssertionError("LLM не должен вызываться при отсутствии контекста")

    monkeypatch.setattr("app.rag_engine.llm_chat", _boom)

    result = _run(engine.query("Чего нет в базе?"))
    assert result["no_answer"] is True
    assert result["answer"] == NO_ANSWER_RU
    assert result["sources"] == []
    assert result["ttft_ms"] == 0  # LLM не вызывался


def test_answer_attaches_citations(monkeypatch):
    """Если есть релевантный чанк → ответ с цитатой."""
    hit = _make_hit(text="Заявление на отпуск подаётся за 14 дней.", score=0.85, rerank=0.92)
    _patch_llm(monkeypatch, "Заявление на отпуск подаётся за 14 дней до начала. [1]", 900)
    engine = FakeRAGEngine(hits=[hit])

    result = _run(engine.query("За сколько дней подавать отпуск?"))
    assert result["no_answer"] is False
    assert result["language"] == "ru"
    assert len(result["sources"]) >= 1
    assert result["sources"][0]["doc_name"] == "reglament.pdf"
    assert result["ttft_ms"] == 900
    assert "14" in result["answer"]


def test_llm_no_answer_response_sets_flag(monkeypatch):
    """Если LLM сам вернул фразу-отказ → no_answer=True."""
    hit = _make_hit(text="нерелевантный текст", score=0.4, rerank=0.45)
    _patch_llm(monkeypatch, NO_ANSWER_RU, 500)
    engine = FakeRAGEngine(hits=[hit])

    result = _run(engine.query("вопрос"))
    assert result["no_answer"] is True
    assert result["ttft_ms"] == 500


def test_kazakh_question_gets_kazakh_refusal():
    """Казахский вопрос → казахский no-answer."""
    engine = FakeRAGEngine(hits=[])
    result = _run(engine.query("Демалыс рәсімі қандай?"))
    assert result["language"] == "kk"
    assert result["answer"] == NO_ANSWER_KK


def test_russian_question_gets_russian_refusal():
    """Русский вопрос → русский no-answer."""
    engine = FakeRAGEngine(hits=[])
    result = _run(engine.query("Какова процедура?"))
    assert result["language"] == "ru"
    assert result["answer"] == NO_ANSWER_RU


def test_empty_question_returns_error():
    engine = FakeRAGEngine(hits=[])
    result = _run(engine.query(""))
    assert result["no_answer"] is True
    assert "Пустой запрос" in result["answer"]


def test_citation_refs_extracted_correctly():
    """Парсинг [1], [2], [1, 3] работает."""
    from app.utils import extract_citation_refs
    assert extract_citation_refs("Ответ [1] и [2].") == [1, 2]
    assert extract_citation_refs("Комбинированный [1, 3] ответ.") == [1, 3]
    assert extract_citation_refs("Без ссылок.") == []


def test_sources_filtered_by_cited_refs(monkeypatch):
    """Если LLM упомянул только [1], sources должны содержать только ref=1."""
    hits = [
        _make_hit(text="первый источник", rerank=0.9),
        _make_hit(text="второй источник", rerank=0.85),
        _make_hit(text="третий источник", rerank=0.80),
    ]
    _patch_llm(monkeypatch, "Ответ основан только на [1].", 200)
    engine = FakeRAGEngine(hits=hits)

    result = _run(engine.query("вопрос"))
    assert len(result["sources"]) == 1
    assert result["sources"][0]["ref"] == 1


def test_latency_ms_present():
    """В ответе всегда есть latency_ms."""
    engine = FakeRAGEngine(hits=[])
    result = _run(engine.query("вопрос"))
    assert "latency_ms" in result
    assert isinstance(result["latency_ms"], int)
    assert result["latency_ms"] >= 0


def test_ttft_ms_present_when_llm_called(monkeypatch):
    """Если LLM вызывался, ttft_ms > 0."""
    hit = _make_hit(text="источник", score=0.9, rerank=0.9)
    _patch_llm(monkeypatch, "Ответ [1].", 350)
    engine = FakeRAGEngine(hits=[hit])

    result = _run(engine.query("вопрос"))
    assert result["ttft_ms"] == 350
