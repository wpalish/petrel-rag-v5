"""
app/bm25_index.py — BM25 sparse retrieval для hybrid search (НОВОЕ в v3).

Стратегия:
- BM25 строится поверх всех child-чанков из Qdrant.
- Индекс перестраивается после каждого ingest/delete (см. ingestion.py).
- Хранится в памяти процесса (singleton).
- На запрос: возвращает top-N chunk_id + score, которые объединяются с dense
  результатами через RRF в rag_engine.py.

Альтернатива — bge-m3 sparse через TEI, но:
+ BM25 не требует доп. ресурсов (процессор/память минимальны)
+ Не требует доп. HTTP-вызовов
+ Хорошо работает на русском/казахском через токенизацию по словам
- Менее точен на синонимах и парафразах, чем bge-m3 sparse

В будущем можно переключить через settings.SPARSE_METHOD = "bge_m3_sparse".
"""

import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient

from app.config import settings

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False
    logger.warning("rank_bm25 not installed — hybrid search will be disabled.")


# Простой токенизатор: слова в нижнем регистре, длина ≥ 2.
# Для русского/казахского этого достаточно — BM25 устойчив к морфологии.
_TOKEN_RE = re.compile(r"[а-яёәіңғүұқөһa-z0-9]{2,}", re.IGNORECASE | re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Токенизация для BM25: слова в нижнем регистре, длина ≥ 2."""
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    """
    In-memory BM25 индекс поверх всех child-чанков Qdrant.

    Хранит:
    - chunk_ids: список ID чанков (соответствует индексу в BM25)
    - chunk_payloads: dict chunk_id → payload (для быстрого доступа)
    - bm25: объект BM25Okapi
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._chunk_ids: List[str] = []
        self._payloads: Dict[str, Dict[str, Any]] = {}
        self._corpus_tokens: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._built = False

    def is_ready(self) -> bool:
        """Готов ли индекс к запросам."""
        return self._built and self._bm25 is not None and len(self._chunk_ids) > 0

    def rebuild_from_qdrant(self, client: QdrantClient, collection_name: str) -> None:
        """
        Полностью перестраивает индекс из всех child-чанков Qdrant.
        Вызывается после каждого ingest/delete.
        """
        if not _HAS_BM25:
            logger.warning("rank_bm25 not installed — BM25 index rebuild skipped.")
            return

        with self._lock:
            chunk_ids: List[str] = []
            payloads: Dict[str, Dict[str, Any]] = {}
            corpus_tokens: List[List[str]] = []
            offset: Optional[str] = None

            while True:
                result = client.scroll(
                    collection_name=collection_name,
                    limit=500,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                points, next_offset = result
                if not points:
                    break
                for p in points:
                    payload = p.payload or {}
                    if payload.get("is_parent"):
                        continue  # в BM25 только child-чанки
                    text = payload.get("text", "")
                    chunk_id = payload.get("chunk_id") or str(p.id)
                    chunk_ids.append(chunk_id)
                    payloads[chunk_id] = payload
                    corpus_tokens.append(tokenize(text))
                offset = next_offset
                if not next_offset:
                    break

            if not chunk_ids:
                self._chunk_ids = []
                self._payloads = {}
                self._corpus_tokens = []
                self._bm25 = None
                self._built = False
                logger.info("BM25 index empty (no child chunks in Qdrant).")
                return

            self._bm25 = BM25Okapi(corpus_tokens)
            self._chunk_ids = chunk_ids
            self._payloads = payloads
            self._corpus_tokens = corpus_tokens
            self._built = True
            logger.info("BM25 index rebuilt: %d child chunks.", len(chunk_ids))

    def rebuild(self, records: List[Dict[str, Any]]) -> None:
        """Перестроить индекс из списка чанков [{chunk_id, text, meta}].

        Удобно для unit-тестов и для источников данных, не привязанных к Qdrant.
        """
        if not _HAS_BM25:
            logger.warning("rank_bm25 not installed — BM25 rebuild skipped.")
            return
        with self._lock:
            chunk_ids: List[str] = []
            payloads: Dict[str, Dict[str, Any]] = {}
            corpus_tokens: List[List[str]] = []
            for r in records:
                cid = r.get("chunk_id") or str(len(chunk_ids))
                chunk_ids.append(cid)
                payloads[cid] = r.get("meta") or {}
                corpus_tokens.append(tokenize(r.get("text", "")))
            if not chunk_ids:
                self._chunk_ids, self._payloads, self._corpus_tokens = [], {}, []
                self._bm25, self._built = None, False
                return
            self._bm25 = BM25Okapi(corpus_tokens)
            self._chunk_ids = chunk_ids
            self._payloads = payloads
            self._corpus_tokens = corpus_tokens
            self._built = True

    def search(self, query: str, top_k: int = 16) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Возвращает [(chunk_id, score, payload), ...] отсортированный по убыванию score.
        """
        if not self.is_ready():
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        with self._lock:
            scores = self._bm25.get_scores(tokens)
            # top-k по score
            indexed = sorted(
                enumerate(scores),
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]
            return [
                (self._chunk_ids[i], float(s), self._payloads.get(self._chunk_ids[i], {}))
                for i, s in indexed
                if s > 0
            ]

    def stats(self) -> Dict[str, Any]:
        return {
            "ready": self.is_ready(),
            "chunks": len(self._chunk_ids),
            "has_bm25_lib": _HAS_BM25,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_bm25_index: Optional[BM25Index] = None
_bm25_lock = threading.Lock()


def get_bm25_index() -> BM25Index:
    """Возвращает singleton BM25Index."""
    global _bm25_index
    if _bm25_index is None:
        with _bm25_lock:
            if _bm25_index is None:
                _bm25_index = BM25Index()
    return _bm25_index


# ---------------------------------------------------------------------------
# RRF (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    dense_results: List[Tuple[str, float, Dict[str, Any]]],
    sparse_results: List[Tuple[str, float, Dict[str, Any]]],
    k: int = 60,
    top_n: Optional[int] = None,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Reciprocal Rank Fusion: объединяет dense и sparse результаты.

    Формула: rrf_score(d) = Σ 1 / (k + rank_i(d))

    где rank_i(d) — позиция документа d в i-м списке (1-индексация).

    Полиморфно по форме входа: принимает (chunk_id, score) ИЛИ
    (chunk_id, score, payload). Если хоть где-то есть payload — возвращает
    3-кортежи (chunk_id, rrf_score, payload), иначе 2-кортежи (chunk_id, rrf_score).
    """
    rrf_scores: Dict[str, float] = {}
    payloads: Dict[str, Dict[str, Any]] = {}
    has_payload = False

    for results in (dense_results, sparse_results):
        for rank, item in enumerate(results, start=1):
            chunk_id = item[0]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if len(item) >= 3:
                has_payload = True
                payloads.setdefault(chunk_id, item[2])

    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    if top_n is not None:
        sorted_ids = sorted_ids[:top_n]
    if has_payload:
        return [(cid, score, payloads.get(cid, {})) for cid, score in sorted_ids]
    return [(cid, score) for cid, score in sorted_ids]
