# Monitoring (v5) — Prometheus + Grafana

## Архитектура

```
┌─────────────────┐    /metrics    ┌──────────────┐    query     ┌──────────┐
│  rag-backend    │ ─────────────► │ Prometheus   │ ───────────► │ Grafana  │
│ (FastAPI)       │                │ (port 9090)  │              │ (3000)   │
└─────────────────┘                └──────────────┘              └──────────┘
        │                                ▲
        │                                │ scrape
        │                                │
        └──── /metrics ──► ┌──────────────┐
                           │ qdrant, vllm │
                           └──────────────┘
```

## Запуск

```bash
docker compose up -d
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# RAG metrics: http://localhost:8000/metrics
```

## Дашборд Grafana

Преднастроенный дашборд "RAG Assistant v5 — Overview" с панелями:
- Requests per second
- TTFT (p50, p95, p99)
- Query latency
- No-answer rate
- Documents / chunks in knowledge base
- LLM health (reachable, model_loaded)
- BM25 chunks
- Requests by status

## Метрики

### Counters
- `rag_requests_total{endpoint, lang, status}` — всего HTTP-запросов
- `rag_no_answer_total{lang}` — всего "нет информации" ответов

### Histograms
- `rag_query_latency_seconds{endpoint}` — гистограмма времени ответа
- `rag_ttft_seconds{endpoint}` — гистограмма TTFT
- `rag_retrieval_candidates{phase}` — кандидатов на каждой фазе retrieval
  - phase: dense, sparse, fused, after_threshold, final

### Gauges
- `rag_documents_total` — документов в базе
- `rag_chunks_total` — чанков в Qdrant
- `rag_bm25_chunks` — чанков в BM25 индексе
- `rag_llm_reachable` — LLM доступен (0/1)
- `rag_llm_model_loaded` — модель загружена (0/1)

## Полезные PromQL запросы

```promql
# TTFT p95 за последние 5 минут
histogram_quantile(0.95, rate(rag_ttft_seconds_bucket[5m]))

# Средний latency
rate(rag_query_latency_seconds_sum[5m]) / rate(rag_query_latency_seconds_count[5m])

# Доля no-answer
sum(rate(rag_no_answer_total[5m])) / sum(rate(rag_requests_total[5m]))

# Request rate by endpoint
sum(rate(rag_requests_total[5m])) by (endpoint)

# Error rate
sum(rate(rag_requests_total{status="error"}[5m])) / sum(rate(rag_requests_total[5m]))
```

## Настройка алертов (опционально)

В `monitoring/prometheus.yml` можно добавить rules:

```yaml
rule_files:
  - alerts.yml
```

`alerts.yml`:
```yaml
groups:
  - name: rag_alerts
    rules:
      - alert: HighTTFT
        expr: histogram_quantile(0.95, rate(rag_ttft_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "TTFT > 5 seconds (p95)"
      
      - alert: LLMDown
        expr: rag_llm_reachable == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "LLM (vLLM) недоступен"
```

## Отключение метрик

В `.env`:
```bash
METRICS_ENABLED=False
```

Это уберёт `/metrics` endpoint. Prometheus соберёт пустые данные — можно отключить и его в `docker-compose.yml`.
