"""
app/metrics.py — Prometheus метрики (v5).

Метрики:
- rag_requests_total{endpoint, lang, status}      — счётчик запросов
- rag_query_latency_seconds{endpoint}              — гистограмма времени ответа
- rag_ttft_seconds{endpoint}                       — гистограмма TTFT
- rag_retrieval_candidates{phase}                  — dense / sparse / fused / final
- rag_no_answer_total                              — счётчик "нет информации" ответов
- rag_documents_total                              — gauge количества документов
- rag_chunks_total                                 — gauge количества чанков в Qdrant
- rag_llm_health{reachable, model_loaded}          — gauge статуса LLM (0/1)
- rag_bm25_chunks                                  — gauge количества чанков в BM25

Принципы:
- Все метрики регистрируются лениво (singleton)
- /metrics endpoint отдаёт текст в формате Prometheus exposition
- Не падает, если prometheus_client не установлен (graceful degradation)
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False
    logger.warning("prometheus_client not installed — metrics disabled.")


class MetricsRegistry:
    """Singleton с ленивой инициализацией Prometheus collectors."""

    def __init__(self):
        self.enabled = _HAS_PROMETHEUS
        if not self.enabled:
            return

        self.registry = CollectorRegistry()

        # Counters
        self.requests_total = Counter(
            "rag_requests_total",
            "Total HTTP requests by endpoint/language/status",
            ["endpoint", "lang", "status"],
            registry=self.registry,
        )

        self.no_answer_total = Counter(
            "rag_no_answer_total",
            "Total 'no information' answers",
            ["lang"],
            registry=self.registry,
        )

        # Histograms
        self.query_latency = Histogram(
            "rag_query_latency_seconds",
            "RAG query latency in seconds",
            ["endpoint"],
            buckets=(0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30, 60),
            registry=self.registry,
        )

        self.ttft_seconds = Histogram(
            "rag_ttft_seconds",
            "Time to first token in seconds",
            ["endpoint"],
            buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 10),
            registry=self.registry,
        )

        # Gauges
        self.documents_total = Gauge(
            "rag_documents_total",
            "Total documents in knowledge base",
            registry=self.registry,
        )

        self.chunks_total = Gauge(
            "rag_chunks_total",
            "Total chunks in Qdrant collection",
            registry=self.registry,
        )

        self.bm25_chunks = Gauge(
            "rag_bm25_chunks",
            "Chunks in BM25 in-memory index",
            registry=self.registry,
        )

        self.llm_reachable = Gauge(
            "rag_llm_reachable",
            "LLM service reachable (1=yes, 0=no)",
            registry=self.registry,
        )

        self.llm_model_loaded = Gauge(
            "rag_llm_model_loaded",
            "LLM model loaded in vLLM (1=yes, 0=no)",
            registry=self.registry,
        )

        # Retrieval candidates
        self.retrieval_candidates = Histogram(
            "rag_retrieval_candidates",
            "Number of candidates at each retrieval phase",
            ["phase"],
            buckets=(0, 1, 2, 4, 8, 16, 32, 64, 128),
            registry=self.registry,
        )

        logger.info("Prometheus metrics registry initialized.")

    # -----------------------------------------------------------
    # Recording helpers
    # -----------------------------------------------------------

    def record_request(self, endpoint: str, lang: str, status: str = "ok") -> None:
        if not self.enabled:
            return
        self.requests_total.labels(endpoint=endpoint, lang=lang, status=status).inc()

    def record_no_answer(self, lang: str) -> None:
        if not self.enabled:
            return
        self.no_answer_total.labels(lang=lang).inc()

    def observe_latency(self, endpoint: str, seconds: float) -> None:
        if not self.enabled:
            return
        self.query_latency.labels(endpoint=endpoint).observe(seconds)

    def observe_ttft(self, endpoint: str, ttft_ms: int) -> None:
        if not self.enabled:
            return
        self.ttft_seconds.labels(endpoint=endpoint).observe(ttft_ms / 1000.0)

    def observe_retrieval(self, phase: str, count: int) -> None:
        """phase: 'dense', 'sparse', 'fused', 'after_threshold', 'final'."""
        if not self.enabled:
            return
        self.retrieval_candidates.labels(phase=phase).observe(count)

    def set_documents_total(self, count: int) -> None:
        if not self.enabled:
            return
        self.documents_total.set(count)

    def set_chunks_total(self, count: int) -> None:
        if not self.enabled:
            return
        self.chunks_total.set(count)

    def set_bm25_chunks(self, count: int) -> None:
        if not self.enabled:
            return
        self.bm25_chunks.set(count)

    def set_llm_health(self, reachable: bool, model_loaded: bool) -> None:
        if not self.enabled:
            return
        self.llm_reachable.set(1 if reachable else 0)
        self.llm_model_loaded.set(1 if model_loaded else 0)

    # -----------------------------------------------------------
    # Exposition
    # -----------------------------------------------------------

    def export(self) -> bytes:
        """Возвращает метрики в формате Prometheus exposition."""
        if not self.enabled:
            return b"# prometheus_client not installed\n"
        return generate_latest(self.registry)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_metrics: Optional[MetricsRegistry] = None


def get_metrics() -> MetricsRegistry:
    global _metrics
    if _metrics is None:
        _metrics = MetricsRegistry()
    return _metrics


# ---------------------------------------------------------------------------
# Context manager для удобной записи latency
# ---------------------------------------------------------------------------

class RequestTimer:
    """Контекстный менеджер для измерения времени запроса."""

    def __init__(self, endpoint: str, lang: str = "ru"):
        self.endpoint = endpoint
        self.lang = lang
        self.start = None
        self.elapsed = 0.0
        self.metrics = get_metrics()

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start
        status = "error" if exc_type else "ok"
        self.metrics.record_request(self.endpoint, self.lang, status)
        self.metrics.observe_latency(self.endpoint, self.elapsed)
        return False  # не подавлять исключения
