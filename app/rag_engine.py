"""
app/rag_engine.py — ядро RAG-пайплайна (v4, peer-reviewed + merge с petrel).

Поток ответа:
1. Определение языка вопроса (RU/KZ/other).
2. Dense search в Qdrant → top-K кандидатов (named vector "dense").
3. (v3) Sparse search (BM25) → top-K кандидатов.
4. (v3) RRF fusion объединяет dense + sparse.
5. Reranker (bge-reranker-v2-m3) с normalize=True → скоры в [0,1].
6. Фильтрация по MIN_RERANK_SCORE.
7. Если чанков нет → no-answer без вызова LLM.
8. (Опц.) Parent Document Retriever: заменяем child-текст на parent-текст.
9. Сборка пронумерованного контекста.
10. (v4) Async генерация через vLLM со streaming + TTFT метрикой.
11. Постобработка: парсинг [1], [2] → формирование источников со snippet.

v4 изменения:
- async query() и generate()
- streaming через app.llm.chat()
- TTFT метрика в ответе
- lazy loading reranker через threading.Lock

Lazy loading: тяжёлые модели (embedder, reranker) грузятся только при первом вызове.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient

from app.config import settings
from app.ingestion import embed_texts, fetch_parent_text, get_qdrant_client
from app.llm import chat as llm_chat
from app.prompts import get_no_answer, get_system_prompt
from app.utils import detect_language, extract_citation_refs, truncate

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Поток: query → embed → Qdrant search → rerank → threshold → PDR → LLM → citations.
    Singleton: создаётся один раз при старте API, переиспользуется.

    v4: query() и generate() — async, LLM-вызов через app.llm.chat() со streaming.
    """

    def __init__(self) -> None:
        # v4: LLM клиент лениво создаётся в app.llm (singleton AsyncOpenAI)
        self.llm_model = settings.LLM_MODEL

        # Qdrant клиент (переиспользуем из ingestion-модуля)
        self.qdrant: QdrantClient = get_qdrant_client()

        # Тяжёлые ресурсы — lazy loading через threading.Lock
        self._lock = threading.Lock()
        self._reranker = None
        self._reranker_loaded = False

        logger.info("RAGEngine v4 initialized (LLM=%s, Qdrant=%s, Reranker=lazy, Stream=%s)",
                    self.llm_model, settings.QDRANT_HOST, settings.LLM_STREAM)

    # =====================================================================
    # Lazy reranker
    # =====================================================================

    def _get_reranker(self):
        """Ленивая загрузка reranker (один раз, в отдельном потоке безопасно)."""
        if not settings.USE_RERANKER:
            return None
        if self._reranker_loaded:
            return self._reranker
        with self._lock:
            if self._reranker_loaded:
                return self._reranker
            try:
                from FlagEmbedding import FlagReranker
                logger.info("Loading reranker: %s (device=%s, normalize=%s)",
                            settings.RERANKER_MODEL, settings.RERANKER_DEVICE,
                            settings.RERANKER_NORMALIZE)
                self._reranker = FlagReranker(
                    settings.RERANKER_MODEL,
                    use_fp16=(settings.RERANKER_DEVICE != "cpu"),
                    cache_dir=settings.MODELS_DIR,
                )
                logger.info("Reranker loaded.")
            except Exception as exc:
                logger.error("Reranker load failed: %s — fallback to vector-only.", exc)
                self._reranker = None
            self._reranker_loaded = True
            return self._reranker

    # =====================================================================
    # Retrieval
    # =====================================================================

    def _dense_search(self, query_emb: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Dense search в Qdrant (named vector 'dense')."""
        try:
            search_result = self.qdrant.search(
                collection_name=settings.QDRANT_COLLECTION,
                query_vector=("dense", query_emb),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.exception("Qdrant dense search failed")
            return []

        candidates: List[Dict[str, Any]] = []
        for hit in search_result:
            payload = hit.payload or {}
            if payload.get("is_parent"):
                continue
            candidates.append({
                "text": payload.get("text", ""),
                "doc_id": payload.get("doc_id", ""),
                "doc_name": payload.get("doc_name", ""),
                "doc_version": payload.get("doc_version", "v1.0"),
                "page_number": payload.get("page_number", 0),
                "section_title": payload.get("section_title", ""),
                "section_path": payload.get("section_path", ""),
                "paragraph_index": payload.get("paragraph_index", 0),
                "parent_id": payload.get("parent_id"),
                "chunk_id": payload.get("chunk_id", ""),
                "language": payload.get("language", "ru"),
                "score": float(hit.score),
                "dense_score": float(hit.score),
            })
        return candidates

    def _sparse_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """BM25 sparse search (in-memory)."""
        if not settings.USE_HYBRID:
            return []
        try:
            from app.bm25_index import get_bm25_index
            bm25 = get_bm25_index()
            if not bm25.is_ready():
                return []
            results = bm25.search(query, top_k=top_k)
        except Exception as exc:
            logger.warning("BM25 search failed: %s", exc)
            return []

        candidates: List[Dict[str, Any]] = []
        for chunk_id, score, payload in results:
            candidates.append({
                "text": payload.get("text", ""),
                "doc_id": payload.get("doc_id", ""),
                "doc_name": payload.get("doc_name", ""),
                "doc_version": payload.get("doc_version", "v1.0"),
                "page_number": payload.get("page_number", 0),
                "section_title": payload.get("section_title", ""),
                "section_path": payload.get("section_path", ""),
                "paragraph_index": payload.get("paragraph_index", 0),
                "parent_id": payload.get("parent_id"),
                "chunk_id": chunk_id,
                "language": payload.get("language", "ru"),
                "score": score,
                "sparse_score": score,
            })
        return candidates

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: dense + sparse (BM25) → RRF → rerank → threshold → PDR.
        """
        top_k = top_k or settings.TOP_K

        # 1. Embedding запроса
        try:
            query_emb = embed_texts([query])[0]
        except Exception as exc:
            logger.exception("Embedding failed for query")
            return []

        # 2. Dense search в Qdrant
        dense_candidates = self._dense_search(query_emb, top_k=settings.DENSE_TOP_K)
        logger.info("Dense search: %d candidates.", len(dense_candidates))

        # 3. Sparse search (BM25) — если включён
        sparse_candidates: List[Dict[str, Any]] = []
        if settings.USE_HYBRID:
            sparse_candidates = self._sparse_search(query, top_k=settings.SPARSE_TOP_K)
            logger.info("Sparse (BM25) search: %d candidates.", len(sparse_candidates))

        # 4. Объединение через RRF или просто dense
        if settings.USE_HYBRID and sparse_candidates:
            from app.bm25_index import reciprocal_rank_fusion
            # Подготавливаем списки для RRF: [(chunk_id, score, payload)]
            dense_for_rrf = [(c["chunk_id"], c["dense_score"], c) for c in dense_candidates]
            sparse_for_rrf = [(c["chunk_id"], c["sparse_score"], c) for c in sparse_candidates]
            fused = reciprocal_rank_fusion(
                dense_for_rrf, sparse_for_rrf,
                k=settings.RRF_K,
                top_n=settings.DENSE_TOP_K + settings.SPARSE_TOP_K,
            )
            # Восстанавливаем кандидатов, сохраняя rrf_score
            candidates: List[Dict[str, Any]] = []
            for chunk_id, rrf_score, payload_dict in fused:
                # payload_dict — это сам кандидат (Dict[str, Any])
                c = dict(payload_dict)
                c["rrf_score"] = rrf_score
                candidates.append(c)
            logger.info("RRF fusion: %d unique candidates.", len(candidates))
        else:
            candidates = dense_candidates

        if not candidates:
            return []

        # 5. Reranker
        reranker = self._get_reranker()
        if reranker:
            pairs = [(query, c["text"]) for c in candidates]
            try:
                rerank_scores = reranker.compute_score(
                    pairs,
                    normalize=settings.RERANKER_NORMALIZE,
                )
                if isinstance(rerank_scores, float):
                    rerank_scores = [rerank_scores]
                for i, rs in enumerate(rerank_scores):
                    candidates[i]["rerank_score"] = float(rs)
            except Exception as exc:
                logger.warning("Reranker compute failed: %s — using vector scores.", exc)
                for c in candidates:
                    c["rerank_score"] = c.get("rrf_score", c.get("dense_score", c.get("score", 0)))
        else:
            for c in candidates:
                c["rerank_score"] = c.get("rrf_score", c.get("dense_score", c.get("score", 0)))

        # 6. Сортировка по rerank_score
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 7. Threshold
        filtered = [
            c for c in candidates
            if c["rerank_score"] >= settings.MIN_RERANK_SCORE
        ]
        if not filtered:
            logger.info("All candidates below threshold %.2f — no answer.",
                        settings.MIN_RERANK_SCORE)
            return []

        # 8. Top-N после reranker
        top_n_filtered = filtered[:settings.RERANK_TOP_K]

        # 9. PDR: заменяем child-текст на parent-текст
        if settings.USE_PDR:
            for c in top_n_filtered:
                parent_id = c.get("parent_id")
                if parent_id:
                    parent_text = fetch_parent_text(parent_id)
                    if parent_text:
                        c["original_text"] = c["text"]
                        c["text"] = parent_text

        logger.info(
            "Retrieval: query='%s...' → dense=%d sparse=%d fused=%d → %d after threshold → %d final",
            query[:50], len(dense_candidates), len(sparse_candidates),
            len(candidates), len(filtered), len(top_n_filtered),
        )
        return top_n_filtered

    # =====================================================================
    # Generation (v4: async + streaming + TTFT)
    # =====================================================================

    async def generate(self, query: str, retrieved: List[Dict[str, Any]],
                       variant: Optional[str] = None) -> Dict[str, Any]:
        """Сборка контекста + async вызов LLM со streaming. v5: variant для A/B."""
        language = detect_language(query)
        started = time.perf_counter()

        if not retrieved:
            return {
                "answer": get_no_answer(language),
                "sources": [],
                "language": language,
                "retrieval_count": 0,
                "no_answer": True,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "ttft_ms": 0,
            }

        # Пронумерованный контекст
        context_lines: List[str] = []
        sources: List[Dict[str, Any]] = []
        for i, item in enumerate(retrieved, start=1):
            text = item["text"]
            section = item.get("section_title") or ""
            section_part = f", раздел: \"{section}\"" if section else ""
            header = (
                f"[{i}] "
                f"(Документ: \"{item['doc_name']}\""
                f"{section_part}"
                f", стр. {item['page_number']}): "
            )
            context_lines.append(header + text)

            sources.append({
                "ref": i,
                "doc_id": item.get("doc_id", ""),
                "doc_name": item.get("doc_name", ""),
                "doc_version": item.get("doc_version", "v1.0"),
                "page_number": item.get("page_number", 0),
                "section_title": section,
                "paragraph_index": item.get("paragraph_index", 0),
                "snippet": truncate(item.get("original_text", text), 280),
                "score": round(item.get("score", 0), 4),
                "rerank_score": round(item.get("rerank_score", 0), 4),
            })

        context = "\n\n".join(context_lines)
        # v5: A/B prompt variant — берём из параметра или settings.PROMPT_VARIANT
        variant = variant or settings.PROMPT_VARIANT
        prompt = get_system_prompt(language, variant=variant).format(context=context, question=query)

        # v4: async LLM-вызов со streaming + TTFT
        try:
            answer_text, ttft_ms = await llm_chat(
                system=prompt,
                user=query,
            )
        except Exception as exc:
            logger.exception("vLLM call failed")
            error_msg = (
                "⚠️ Ошибка генерации ответа (LLM недоступна). "
                "Проверьте, что vLLM запущен и модель загружена."
                if language != "kk" else
                "⚠️ Жауап беру қатесі (LLM қолжетімсіз). "
                "vLLM іске қосылғанын және модель жүктелгенін тексеріңіз."
            )
            return {
                "answer": error_msg,
                "sources": sources,
                "language": language,
                "retrieval_count": len(retrieved),
                "no_answer": True,
                "error": str(exc),
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "ttft_ms": 0,
            }

        # Проверяем no-answer
        no_answer_flag = (
            answer_text.startswith(settings.NO_ANSWER_RU)
            or answer_text.startswith(settings.NO_ANSWER_KK)
            or answer_text.startswith("Берілген құжаттарда")
            or answer_text.startswith("Ұсынылған құжаттарда")
        )

        # Фильтруем источники: оставляем только упомянутые в ответе
        cited_refs = extract_citation_refs(answer_text)
        if cited_refs:
            sources = [s for s in sources if s["ref"] in cited_refs]

        return {
            "answer": answer_text,
            "sources": sources,
            "language": language,
            "retrieval_count": len(retrieved),
            "no_answer": no_answer_flag,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "ttft_ms": ttft_ms,
        }

    # =====================================================================
    # Public API (v4: async)
    # =====================================================================

    async def query(self, question: str, variant: Optional[str] = None) -> Dict[str, Any]:
        """Полный async RAG-конвейер: query → retrieve → generate. v5: variant для A/B."""
        if not question or not question.strip():
            return {
                "answer": "Пустой запрос.",
                "sources": [],
                "language": "ru",
                "retrieval_count": 0,
                "no_answer": True,
                "latency_ms": 0,
                "ttft_ms": 0,
                "variant": variant or settings.PROMPT_VARIANT,
            }

        logger.info("Query: '%s...' (variant=%s)", question[:70], variant or settings.PROMPT_VARIANT)
        retrieved = self.retrieve(question)
        if not retrieved:
            language = detect_language(question)
            return {
                "answer": get_no_answer(language),
                "sources": [],
                "language": language,
                "retrieval_count": 0,
                "no_answer": True,
                "latency_ms": 0,
                "ttft_ms": 0,
                "variant": variant or settings.PROMPT_VARIANT,
            }

        result = await self.generate(question, retrieved, variant=variant)
        # v5: записываем variant в ответ для /ab/test
        result["variant"] = variant or settings.PROMPT_VARIANT

        # v5: metrics
        try:
            from app.metrics import get_metrics
            m = get_metrics()
            m.observe_ttft("/chat", result.get("ttft_ms", 0))
            if result.get("no_answer"):
                m.record_no_answer(result.get("language", "ru"))
        except Exception as exc:
            logger.debug("metrics recording failed: %s", exc)

        logger.info("Answer: lang=%s, sources=%d, no_answer=%s, ttft=%dms, variant=%s",
                    result["language"], len(result["sources"]),
                    result["no_answer"], result.get("ttft_ms", 0),
                    result.get("variant"))
        return result

    def reset_context(self) -> None:
        """Заглушка для совместимости с UI — MVP stateless."""
        pass


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_engine: Optional[RAGEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> RAGEngine:
    """Возвращает singleton RAGEngine. Lazy init при первом вызове."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = RAGEngine()
    return _engine
