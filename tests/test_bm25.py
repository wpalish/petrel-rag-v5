"""Тесты BM25-индекса и Reciprocal Rank Fusion (v4).

Перенесено из petrel-rag-assistant с адаптацией под наш app.bm25_index.
"""
from __future__ import annotations

from app.bm25_index import BM25Index, reciprocal_rank_fusion, tokenize


def test_tokenize_handles_ru_kz_lat():
    """Токенизатор корректно обрабатывает рус/каз/лат/цифры."""
    toks = tokenize("Отпуск VPN демалыс 90 дней")
    assert "отпуск" in toks
    assert "vpn" in toks
    assert "демалыс" in toks
    assert "90" in toks


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize(None) == []


def test_tokenize_min_length_2():
    """Однобуквенные токены (< 2 символов) отбрасываются."""
    toks = tokenize("а я и о")
    # все токены здесь — одна буква, поэтому ничего не остаётся
    assert toks == []


def test_bm25_finds_exact_term():
    """BM25 лексический: сильная сторона — точные токены/числа."""
    idx = BM25Index()
    idx.rebuild([
        {"chunk_id": "c1", "text": "Минимальная длина пароля 12 символов обязательна", "meta": {}},
        {"chunk_id": "c2", "text": "Отпуск составляет 24 календарных дня", "meta": {}},
        {"chunk_id": "c3", "text": "Заявление подаётся за 14 дней", "meta": {}},
        {"chunk_id": "c4", "text": "Доступ только через корпоративный VPN", "meta": {}},
        {"chunk_id": "c5", "text": "Отзыв из отпуска по письменному согласию", "meta": {}},
    ])
    assert idx.is_ready()
    res = idx.search("длина пароля", top_k=3)
    assert res
    assert res[0][0] == "c1"  # chunk_id первого результата


def test_bm25_empty_when_not_built():
    """Без rebuild — search возвращает пустой список."""
    idx = BM25Index()
    assert idx.search("вопрос", top_k=5) == []
    assert idx.is_ready() is False


def test_bm25_stats():
    idx = BM25Index()
    assert idx.stats() == {"ready": False, "chunks": 0, "has_bm25_lib": True}

    idx.rebuild([
        {"chunk_id": "c1", "text": "тест", "meta": {}},
    ])
    stats = idx.stats()
    assert stats["ready"] is True
    assert stats["chunks"] == 1


def test_rrf_merges_and_dedups():
    """RRF объединяет dense + sparse, удаляет дубликаты."""
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("c", 5.0), ("a", 4.0), ("d", 3.0)]
    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    ids = [cid for cid, _ in fused]

    # все уникальные id присутствуют, без дублей
    assert set(ids) == {"a", "b", "c", "d"}
    assert len(ids) == len(set(ids))

    # 'a' и 'c' встречаются в обоих списках → выше, чем 'b'/'d' из одного
    assert ids.index("a") < ids.index("b")
    assert ids.index("c") < ids.index("d")


def test_rrf_top_n_limit():
    dense = [("a", 1.0), ("b", 0.5)]
    sparse = [("c", 1.0)]
    assert len(reciprocal_rank_fusion(dense, sparse, top_n=2)) == 2


def test_rrf_empty_inputs():
    """RRF с пустыми списками → пустой результат."""
    assert reciprocal_rank_fusion([], []) == []
    assert reciprocal_rank_fusion([], [("a", 1.0)]) == [("a", 1.0 / 61)]
