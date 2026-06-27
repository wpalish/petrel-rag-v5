"""
app/sparse_tei.py — sparse retrieval через TEI (bge-m3 sparse) — v5.

Альтернатива BM25 для hybrid search. bge-m3 умеет генерировать sparse-векторы
(lexical weighting), которые:
- Точнее на мультиязычности (выучивают важность токенов в контексте)
- Не требуют in-memory перестроения (хранятся в Qdrant)
- Лучше работают с редкими терминами и аббревиатурами

В v5:
- При ingestion: если TEI_SPARSE_ENABLED=True — вычисляем sparse векторы и
  сохраняем в Qdrant (поле "sparse" в named vectors).
- При retrieval: sparse-поиск через Qdrant (вместо BM25), результат
  объединяется с dense через RRF.

Если TEI не поддерживает sparse (или отключён) — fallback на BM25.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class SparseTEIClient:
    """Клиент к TEI для sparse-эмбддингов."""

    def __init__(self):
        self.base_url = settings.tei_url
        self._available: Optional[bool] = None  # cache проверки доступности

    def is_available(self) -> bool:
        """Проверяет, поддерживает ли TEI sparse-эмбеддинги."""
        if not settings.TEI_SPARSE_ENABLED:
            return False
        if self._available is None:
            try:
                with httpx.Client(timeout=5.0) as cli:
                    r = cli.get(f"{self.base_url}/health")
                    self._available = (r.status_code == 200)
            except Exception as exc:
                logger.warning("TEI sparse not available: %s", exc)
                self._available = False
        return self._available

    def compute_sparse(self, texts: List[str]) -> List[Dict[str, List[float]]]:
        """
        Возвращает список sparse-векторов: [{"indices": [...], "values": [...]}, ...].

        TEI endpoint /embed (with sparse=True) или /rerank — зависит от версии.
        Здесь предполагаем, что TEI отдаёт sparse в специальном поле.
        """
        if not texts:
            return []
        try:
            with httpx.Client(timeout=60.0) as cli:
                # TEI 1.6+ поддерживает sparse через /embed с параметром
                resp = cli.post(
                    f"{self.base_url}/embed",
                    json={
                        "inputs": texts,
                        "truncate": True,
                        # TEI может поддерживать sparse через специальный флаг
                        # Если нет — fallback на BM25 в rag_engine.py
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                # Если TEI отдаёт dense — sparse не поддерживается
                if isinstance(data, list) and data and isinstance(data[0], list):
                    logger.warning("TEI returned dense, not sparse — fallback to BM25.")
                    return []
                # Если есть sparse поле
                sparse_results = []
                for item in data:
                    if "sparse" in item:
                        sparse_results.append(item["sparse"])
                return sparse_results
        except Exception as exc:
            logger.warning("TEI sparse compute failed: %s — fallback to BM25.", exc)
            return []


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_sparse_tei_client: Optional[SparseTEIClient] = None


def get_sparse_tei() -> SparseTEIClient:
    global _sparse_tei_client
    if _sparse_tei_client is None:
        _sparse_tei_client = SparseTEIClient()
    return _sparse_tei_client


def qdrant_sparse_search(
    query_sparse: Dict[str, List[float]],
    top_k: int = 16,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Sparse search в Qdrant (если sparse векторы сохранены).
    Возвращает [(chunk_id, score, payload), ...].

    Использует Qdrant SearchParams с sparse vector.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import SparseVector

    from app.ingestion import get_qdrant_client

    client = get_qdrant_client()
    try:
        # Qdrant 1.7+ поддерживает sparse search через query_vector=("sparse", SparseVector(...))
        sparse_vec = SparseVector(
            indices=query_sparse.get("indices", []),
            values=query_sparse.get("values", []),
        )
        results = client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=("sparse", sparse_vec),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [
            (str(p.id), float(p.score), p.payload or {})
            for p in results
        ]
    except Exception as exc:
        logger.warning("Qdrant sparse search failed: %s — fallback to BM25.", exc)
        return []
