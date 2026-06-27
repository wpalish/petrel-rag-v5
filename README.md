# 🏢 Корпоративный ИИ-Ассистент (On-Premise RAG) — v5

Полностью локальная RAG-система поиска по корпоративной базе знаний с поддержкой русского и казахского языков. **Никаких внешних API** — все компоненты работают в защищённом контуре.

> **v5 — production-ready.** Финальная итерация с полным набором enterprise-функций.
>
> Сохранено из v4:
> - vLLM + bge-m3 + reranker + PDR + hybrid search
> - Async streaming + TTFT
> - Локальный LLM-judge (без RAGAS/OpenAI)
> - Pytest-тесты с заглушками
> - Демо-документы в `samples/`
> - Lifespan context manager
> - chunk_hash дедупликация
>
> Новое в v5:
> - 🔐 **Basic Auth middleware** (опционально)
> - 📊 **Prometheus + Grafana** monitoring
> - 🧪 **A/B prompt testing** (3 варианта: strict/balanced/concise)
> - 🎯 **Calibration endpoint** для подбора `MIN_RERANK_SCORE`
> - 🔍 **bge-m3 sparse через TEI** (опционально, точнее BM25)
> - 📈 **10+ Prometheus метрик** (requests, TTFT, latency, retrieval, no-answer)

---

## ✨ Полный список возможностей v5

### Архитектура RAG
- 🔒 **Полная локальность** — vLLM, TEI, Qdrant локально
- 📖 **Строгое цитирование** — документ + страница + раздел
- 🛡️ **Reranker** — bge-reranker-v2-m3, normalize=True
- 🌍 **Двуязычие** — RU/KZ + few-shot
- 📚 **PDR** — parent-child chunking
- 🔍 **Hybrid Search** — BM25 + dense + RRF (k=60)
- 🚫 **chunk_hash дедупликация** при повторной загрузке
- 📸 **Auto-snapshots Qdrant** после каждой индексации

### Performance & UX
- ⚡ **Async streaming + TTFT** — real-time выдача токенов
- 🧪 **Локальный LLM-judge** — groundedness без RAGAS/OpenAI
- 🏗️ **Lifespan context manager** — современный FastAPI паттерн

### Production (v5)
- 🔐 **Basic Auth** — опциональная защита всех endpoints
- 📊 **Prometheus + Grafana** — метрики, дашборды, алерты
- 🧪 **A/B prompt testing** — 3 варианта промптов (`strict`/`balanced`/`concise`)
- 🎯 **Calibration tool** — автоматический подбор `MIN_RERANK_SCORE`
- 🔍 **bge-m3 sparse через TEI** — опциональная замена BM25

### Качество кода
- 🧪 **50+ pytest тестов** с monkeypatch заглушками LLM
- 📦 **PEP 621** через `pyproject.toml`
- 📂 **3 демо-документа** в `samples/` (RU/KZ регламенты + IT policy)
- 📚 **Полная документация** — README, MONITORING.md, AUTH.md, qdrant_schema.md

---

## 🛠️ Технологический стек (v5)

| Компонент | Технология | Версия |
|-----------|------------|--------|
| LLM | Qwen2.5-7B-Instruct-AWQ через **vLLM** (async streaming) | v0.6.4.post1 |
| Embeddings | BAAI/bge-m3 (1024 dim, dense + sparse) | через TEI 1.6 или sentence-transformers |
| Reranker | BAAI/bge-reranker-v2-m3 (normalize=True) | FlagEmbedding 1.3.2 |
| Vector DB | Qdrant (named vectors: dense + sparse reserved) | v1.12.4 |
| Sparse retrieval | BM25 (rank-bm25) + RRF, опц. bge-m3 sparse через TEI | 0.2.2 |
| RAG Framework | LlamaIndex + кастомный async retrieval | 0.11.20 |
| Backend | FastAPI + Uvicorn (lifespan + auth middleware) | 0.115.6 |
| UI | Gradio (type="messages") | 5.9.1 |
| Monitoring | Prometheus + Grafana | v2.55.0 / v11.4.0 |
| Auth | Basic Auth (timing-safe) | — |
| Evaluation | Локальный LLM-judge (без RAGAS) | — |
| Testing | pytest + pytest-asyncio | 8.3.4 / 0.25.0 |
| Парсинг PDF | PyMuPDF + section title extraction | 1.24.14 |

---

## 📋 Системные требования

| Компонент | Минимум (MVP) | Рекомендуется |
|-----------|---------------|---------------|
| GPU | RTX 3060 12GB | RTX 4090 24GB или 2×3090 |
| RAM | 32 GB | 64 GB |
| Диск | 50 GB (модели + индекс) | 200 GB |

---

## 🚀 Быстрый старт

### 1. Подготовка окружения

```bash
git clone <repo-url> rag-system-v5
cd rag-system-v5
cp .env.example .env
```

### 2. Предзагрузка моделей (на машине с интернетом)

```bash
./scripts/preload_models.sh
```

### 3. Запуск всех сервисов

```bash
docker compose up -d --build
```

7 контейнеров:

| Сервис | URL | Назначение |
|--------|-----|------------|
| Qdrant | http://localhost:6333 | Векторная БД |
| vLLM | http://localhost:8001 | LLM-инференс (OpenAI-compatible, streaming) |
| TEI | http://localhost:8080 | Embeddings (bge-m3) |
| Backend | http://localhost:8000 | REST API + auth + metrics + A/B + calibrate |
| UI | http://localhost:7860 | Gradio-интерфейс |
| Prometheus | http://localhost:9090 | Сбор метрик |
| Grafana | http://localhost:3000 | Дашборды (admin/admin) |

### 4. Загрузить демо-документы (один клик)

В UI: вкладка «📁 Управление базой знаний» → **«📂 Загрузить демо-документы»**

Или через API:
```bash
curl -X POST http://localhost:8000/samples/ingest
```

### 5. Задать первый вопрос (с TTFT)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Сколько календарных дней длится ежегодный отпуск?"}'
```

### 6. Калибровка MIN_RERANK_SCORE

```bash
# Через скрипт (рекомендуется)
docker compose exec rag-backend python /app/scripts/calibrate_threshold.py

# Или через API
curl -X POST http://localhost:8000/calibrate \
  -H "Content-Type: application/json" \
  -d '{"limit": 10}'
```

### 7. A/B тестирование промптов

```bash
curl -X POST http://localhost:8000/ab/test \
  -H "Content-Type: application/json" \
  -d '{"question": "Сколько дней длится отпуск?"}'
# → ответы всех 3 вариантов (strict, balanced, concise)
```

### 8. Запуск оценки качества

```bash
./scripts/run_evaluation.sh 5  # smoke-test
./scripts/run_evaluation.sh    # полный прогон 30 вопросов
```

### 9. Запуск pytest

```bash
docker compose exec rag-backend pytest tests/ -v
```

### 10. Просмотр метрик

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Дашборд "RAG Assistant v5 — Overview" преднастроен

---

## 📊 Метрики Prometheus

### Counters
- `rag_requests_total{endpoint, lang, status}`
- `rag_no_answer_total{lang}`

### Histograms
- `rag_query_latency_seconds{endpoint}` — p50/p95/p99 latency
- `rag_ttft_seconds{endpoint}` — time to first token
- `rag_retrieval_candidates{phase}` — dense/sparse/fused/final

### Gauges
- `rag_documents_total`, `rag_chunks_total`, `rag_bm25_chunks`
- `rag_llm_reachable`, `rag_llm_model_loaded`

### Полезные PromQL

```promql
# TTFT p95
histogram_quantile(0.95, rate(rag_ttft_seconds_bucket[5m]))

# No-answer rate
sum(rate(rag_no_answer_total[5m])) / sum(rate(rag_requests_total[5m]))

# Error rate
sum(rate(rag_requests_total{status="error"}[5m])) / sum(rate(rag_requests_total[5m]))
```

Подробнее: `docs/MONITORING.md`

---

## 🔐 Auth (v5)

Включается через `.env`:
```bash
AUTH_ENABLED=True
AUTH_USER=admin
AUTH_PASSWORD=your_strong_password
```

Защищает: `/chat`, `/query`, `/ingest`, `/documents`, `/evaluation/*`, `/ab/test`, `/calibrate`.
Публичные: `/health`, `/metrics`, `/docs`.

```bash
curl -u admin:your_password http://localhost:8000/chat ...
```

Подробнее: `docs/AUTH.md`

---

## 🧪 A/B Prompt Testing (v5)

3 варианта промптов:

| Variant | Описание |
|---------|----------|
| `strict` | Максимально строгий, минимум галлюцинаций (по умолчанию) |
| `balanced` | Баланс между строгостью и полнотой |
| `concise` | Краткий (только факты + цитаты, 3-5 предложений) |

```bash
# Список вариантов
curl http://localhost:8000/variants

# A/B тест: один вопрос — все варианты
curl -X POST http://localhost:8000/ab/test \
  -H "Content-Type: application/json" \
  -d '{"question": "Сколько дней отпуск?"}'

# Явный выбор варианта в запросе
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "variant": "concise"}'
```

---

## 🎯 Calibration (v5)

Подбор `MIN_RERANK_SCORE` по тестовому набору:

```bash
docker compose exec rag-backend python /app/scripts/calibrate_threshold.py
```

Алгоритм:
1. Грузит `tests/testset.json`
2. Для каждого вопроса делает `retrieve()` (без LLM)
3. Делим rerank_scores на 2 группы: `expected_no_answer=True` vs `False`
4. Считаем p25 распределения "с ответом"
5. Рекомендует `MIN_RERANK_SCORE = p25`

После калибровки — обновить `.env` и `docker compose restart rag-backend`.

---

## 📁 Структура проекта

```
rag-system-v5/
├── app/                          (13 Python-модулей)
│   ├── __init__.py               (v5.0.0)
│   ├── main.py                   # FastAPI entrypoint
│   ├── config.py                 # +AUTH +METRICS +AB +TEI_SPARSE
│   ├── utils.py
│   ├── prompts.py                # v5: 3 variant × 2 lang = 6 промптов
│   ├── ingestion.py
│   ├── bm25_index.py             # BM25 + RRF
│   ├── sparse_tei.py             # v5: bge-m3 sparse через TEI
│   ├── llm.py                    # async streaming + TTFT
│   ├── rag_engine.py             # v5: variant param + metrics
│   ├── evaluation_runner.py      # локальный LLM-judge
│   ├── auth.py                   # v5: BasicAuthMiddleware
│   ├── metrics.py                # v5: Prometheus registry
│   ├── api.py                    # v5: /metrics /ab/test /calibrate /variants
│   └── ui.py
├── tests/                        (7 файлов, 50+ тестов)
│   ├── test_rag.py
│   ├── test_bm25.py
│   ├── test_lang.py
│   ├── test_ingest.py
│   ├── test_metrics.py           # v5: Prometheus
│   ├── test_auth.py              # v5: Basic Auth
│   ├── test_ab.py                # v5: A/B prompts
│   ├── testset.json
│   └── README.md
├── samples/                      (3 демо-документа)
│   ├── reglament_otpuska.txt     (RU)
│   ├── reglament_demalys_kk.txt  (KZ)
│   └── it_security_policy.txt    (RU)
├── monitoring/                   v5: Prometheus + Grafana
│   ├── prometheus.yml
│   └── grafana/
│       ├── dashboard.json
│       ├── dashboards-provider.yaml
│       └── datasources.yaml
├── evaluation/
├── data/{docs,index}/.gitkeep
├── docs/
│   ├── qdrant_schema.md
│   ├── OFFLINE_SECURITY_CHECKLIST.md
│   ├── MONITORING.md             # v5
│   └── AUTH.md                   # v5
├── scripts/
│   ├── preload_models.sh
│   ├── audit_telemetry.sh
│   ├── run_evaluation.sh
│   └── calibrate_threshold.py    # v5
├── .env.example                  # +AUTH +METRICS +AB +GRAFANA
├── .gitignore / .dockerignore
├── docker-compose.yml            # v5: +prometheus +grafana
├── Dockerfile                    # +COPY monitoring
├── requirements.txt              # +prometheus-client
├── pyproject.toml
└── README.md
```

---

## ⚙️ Конфигурация (основные параметры `.env`)

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct-AWQ` | LLM-модель |
| `LLM_STREAM` | `True` | Streaming для TTFT |
| `EMBED_MODEL` | `BAAI/bge-m3` | Модель эмбеддингов |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Реранкер |
| `USE_HYBRID` | `True` | Dense + BM25 + RRF |
| `TEI_SPARSE_ENABLED` | `False` | v5: bge-m3 sparse через TEI |
| `USE_CHUNK_HASH_DEDUP` | `True` | Дедупликация |
| `MIN_RERANK_SCORE` | `0.30` | Порог (КАЛИБРУЕМЫЙ через /calibrate) |
| `PROMPT_VARIANT` | `strict` | v5: strict/balanced/concise |
| `AUTH_ENABLED` | `False` | v5: Basic Auth |
| `AUTH_USER` / `AUTH_PASSWORD` | `admin` / `changeme` | v5: credentials |
| `METRICS_ENABLED` | `True` | v5: Prometheus |
| `GRAFANA_USER` / `GRAFANA_PASSWORD` | `admin` / `admin` | v5: Grafana UI |

Полный список — в `.env.example`.

---

## 🔒 Безопасность (On-Premise)

- ✅ **Нет внешних API.** vLLM, TEI, Qdrant — всё локально.
- ✅ **RAGAS удалён** — groundedness считается локальным LLM.
- ✅ **Телеметрия vLLM/Ollama отключена**.
- ✅ **HF_HUB_OFFLINE** для air-gap.
- ✅ **Auto-snapshots Qdrant**.
- ✅ **v5: Basic Auth** (опционально) с timing-safe сравнением.
- ⚠️ **HTTPS** — рекомендуется обратный прокси (nginx/caddy) с TLS в production.

---

## 🛣️ Что осталось после v5

v5 — финальная итерация. Дальнейшие улучшения требуют реальных данных:

| Что | Зачем |
|-----|-------|
| **Калибровка на реальных регламентах заказчика** | Точная подстройка `MIN_RERANK_SCORE` |
| **Fine-tuning Qwen** (QLoRA на корпоративных Q&A) | +5-10% faithfulness |
| **GraphRAG** для перекрёстных ссылок | Сложные multi-doc вопросы |
| **Мультимодальность** (таблицы как изображения) | Сохранение сложных таблиц |
| **SSO/OAuth** вместо Basic Auth | Enterprise-ready auth |

---

## 🆘 Поддержка

1. `docker compose ps` — все ли 7 сервисов `Up`?
2. `curl http://localhost:8000/health` — отвечает ли backend?
3. `curl http://localhost:8000/metrics` — отдаются ли метрики?
4. `curl http://localhost:8001/health` — отвечает ли vLLM?
5. `pytest tests/ -v` — проходят ли 50+ тестов?
6. `./scripts/run_evaluation.sh 5` — smoke-test оценки
7. `docker compose logs <service>` — логи

Подробнее: `docs/MONITORING.md`, `docs/AUTH.md`, `docs/OFFLINE_SECURITY_CHECKLIST.md`
