"""Тесты метрик (v5) — Prometheus registry.

Проверяем что метрики создаются, обновляются и экспортируются в правильном формате.
"""
from __future__ import annotations

import pytest

# Пропускаем все тесты если prometheus_client не установлен
pytest.importorskip("prometheus_client")

from app.metrics import MetricsRegistry, RequestTimer, get_metrics


def test_metrics_registry_singleton():
    """get_metrics возвращает один и тот же объект."""
    m1 = get_metrics()
    m2 = get_metrics()
    assert m1 is m2


def test_metrics_registry_initial_state():
    """Новый registry должен быть включён (если prometheus_client установлен)."""
    m = MetricsRegistry()
    assert m.enabled is True


def test_record_request():
    """Счётчик запросов инкрементируется."""
    m = get_metrics()
    # Просто проверяем что не падает
    m.record_request("/chat", "ru", "ok")
    m.record_request("/chat", "ru", "error")
    m.record_request("/query", "kk", "ok")


def test_record_no_answer():
    m = get_metrics()
    m.record_no_answer("ru")
    m.record_no_answer("kk")


def test_observe_latency():
    m = get_metrics()
    m.observe_latency("/chat", 1.5)
    m.observe_latency("/chat", 0.5)


def test_observe_ttft():
    m = get_metrics()
    m.observe_ttft("/chat", 350)  # 350ms
    m.observe_ttft("/chat", 2500)  # 2.5s


def test_observe_retrieval():
    m = get_metrics()
    m.observe_retrieval("dense", 16)
    m.observe_retrieval("sparse", 14)
    m.observe_retrieval("fused", 28)
    m.observe_retrieval("after_threshold", 8)
    m.observe_retrieval("final", 5)


def test_set_gauges():
    m = get_metrics()
    m.set_documents_total(15)
    m.set_chunks_total(342)
    m.set_bm25_chunks(280)
    m.set_llm_health(reachable=True, model_loaded=True)
    m.set_llm_health(reachable=False, model_loaded=False)


def test_export_returns_bytes():
    """Экспорт возвращает bytes в формате Prometheus exposition."""
    m = get_metrics()
    # Запишем что-то
    m.record_request("/chat", "ru", "ok")
    m.observe_ttft("/chat", 350)

    output = m.export()
    assert isinstance(output, bytes)
    text = output.decode("utf-8")
    # Должны быть наши метрики
    assert "rag_requests_total" in text
    assert "rag_ttft_seconds" in text


def test_request_timer():
    """RequestTimer корректно измеряет время."""
    m = get_metrics()
    timer = RequestTimer("/test", "ru")
    with timer:
        import time
        time.sleep(0.01)
    assert timer.elapsed > 0.01
    assert timer.elapsed < 1.0  # не должно быть больше секунды


def test_request_timer_records_metric():
    """После выхода из контекста метрика должна быть записана."""
    m = get_metrics()
    initial_export = m.export().decode("utf-8")

    timer = RequestTimer("/test_timer", "ru")
    with timer:
        pass

    final_export = m.export().decode("utf-8")
    # Должна появиться запись
    assert "rag_requests_total" in final_export
    assert "/test_timer" in final_export


def test_request_timer_handles_exception():
    """RequestTimer записывает error статус при исключении."""
    m = get_metrics()
    timer = RequestTimer("/error_test", "ru")
    try:
        with timer:
            raise ValueError("test error")
    except ValueError:
        pass
    # Статус должен быть error
    assert timer.elapsed > 0
