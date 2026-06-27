"""
app/config.py — конфигурация системы на Pydantic Settings.

Все параметры вынесены сюда. Большинство можно переопределить через .env.
Калибровочные параметры помечены комментарием `# CALIBRATABLE`.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # =====================================================================
    # Qdrant
    # =====================================================================
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "corporate_knowledge"
    # Embedded-режим: если задан путь — Qdrant работает локально из файла, без
    # сервера/Docker (удобно для запуска на ноутбуке). Имеет приоритет над host/port.
    QDRANT_PATH: str = ""

    # Размерность dense-вектора bge-m3 = 1024. Если сменить модель — изменить и это.
    EMBED_DIM: int = 1024

    # =====================================================================
    # vLLM (LLM-инференс, OpenAI-compatible API)
    # =====================================================================
    VLLM_HOST: str = "vllm"
    VLLM_PORT: int = 8000
    LLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    LLM_TEMPERATURE: float = 0.1          # CALIBRATABLE — ниже = меньше галлюцинаций
    LLM_TOP_P: float = 0.9
    LLM_MAX_TOKENS: int = 1024

    # Таймаут на один запрос к LLM (сек). Длинные ответы требуют больше времени.
    LLM_TIMEOUT: float = 120.0

    # Streaming ответа — v4 (из petrel). True = real-time выдача токенов с
    # измерением TTFT. False = ждём полный ответ.
    LLM_STREAM: bool = True

    # =====================================================================
    # Embeddings
    # =====================================================================
    # TEI даёт лучшую throughput, но локально через sentence-transformers проще для MVP.
    # Если USE_TEI=True — система ходит по HTTP в tei-embeddings сервис.
    USE_TEI: bool = False
    TEI_HOST: str = "tei-embeddings"
    TEI_PORT: int = 80
    EMBED_MODEL: str = "BAAI/bge-m3"
    EMBED_DEVICE: str = "cpu"   # "cuda" если доступен GPU и USE_TEI=False

    # =====================================================================
    # Reranker (ОБЯЗАТЕЛЕН для качества)
    # =====================================================================
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    USE_RERANKER: bool = True
    RERANKER_DEVICE: str = "cpu"   # "cuda" если доступен GPU
    # normalize=True → на выходе скоры в [0,1], иначе сырые логиты [-10..+10].
    # При normalize=True порог 0.30 осмысленный.
    RERANKER_NORMALIZE: bool = True

    # =====================================================================
    # Retrieval
    # =====================================================================
    # CALIBRATABLE — все пороги и размеры можно тюнить после тестов.
    CHUNK_SIZE: int = 512           # CALIBRATABLE — маленькие чанки для поиска
    CHUNK_OVERLAP: int = 80
    PARENT_CHUNK_SIZE: int = 1400   # CALIBRATABLE — большие parent для генерации
    PARENT_CHUNK_OVERLAP: int = 150
    TOP_K: int = 8                  # CALIBRATABLE — сколько кандидатов из Qdrant
    RERANK_TOP_K: int = 5           # CALIBRATABLE — сколько оставить после reranker
    MIN_RERANK_SCORE: float = 0.30  # CALIBRATABLE — стартовый порог. Замерить на 20-30 вопросах.

    USE_PDR: bool = True   # Parent Document Retriever

    # =====================================================================
    # Hybrid Search (dense + sparse + RRF) — НОВОЕ в v3
    # =====================================================================
    # Если True — параллельно с dense идёт sparse (BM25 или bge-m3 sparse),
    # результаты объединяются через Reciprocal Rank Fusion.
    USE_HYBRID: bool = True
    # Какой sparse использовать: 'bm25' (локальный rank_bm25) или 'bge_m3_sparse'.
    # BM25 проще и не требует доп. ресурсов. bge_m3_sparse точнее на мультиязычности.
    SPARSE_METHOD: str = "bm25"
    # RRF гиперпараметр k — типично 60 (стандарт из оригинальной статьи).
    RRF_K: int = 60
    # Сколько кандидатов брать из sparse retrieval
    SPARSE_TOP_K: int = 16
    # Сколько брать кандидатов из dense retrieval (до reranker)
    DENSE_TOP_K: int = 16

    # =====================================================================
    # Дедупликация по chunk_hash (из v2-fixed)
    # =====================================================================
    USE_CHUNK_HASH_DEDUP: bool = True   # пропускать дубликаты при повторном ingest

    # =====================================================================
    # Evaluation (RAGAS) — НОВОЕ в v3
    # =====================================================================
    # Путь к тестовому набору (JSON) внутри контейнера
    TESTSET_PATH: str = "/app/tests/testset.json"
    # Путь к результатам оценки
    EVAL_RESULTS_PATH: str = "/app/evaluation/results.json"
    # Пороги для прохождения smoke-тестов
    TARGET_FAITHFULNESS: float = 0.85
    TARGET_ANSWER_RELEVANCY: float = 0.80
    TARGET_LANGUAGE_PURITY: float = 0.95

    # =====================================================================
    # Пути (внутри контейнера)
    # =====================================================================
    DOCS_DIR: str = "/app/data/docs"
    INDEX_DIR: str = "/app/data/index"
    MODELS_DIR: str = "/app/models"

    # =====================================================================
    # Offline / Telemetry
    # =====================================================================
    HF_HUB_OFFLINE: bool = False    # True только ПОСЛЕ первичной загрузки моделей
    VLLM_NO_USAGE_STATS: bool = True
    OLLAMA_NO_TELEMETRY: int = 1    # на всякий случай, если вернёмся к Ollama

    # =====================================================================
    # Прочее
    # =====================================================================
    LOG_LEVEL: str = "INFO"
    # Максимальный размер ответа RRF (для безопасности)
    MAX_CONTEXT_TOKENS: int = 6000

    # =====================================================================
    # Защитные ответы (no-answer)
    # =====================================================================
    NO_ANSWER_RU: str = "В предоставленных документах нет информации по данному вопросу."
    NO_ANSWER_KK: str = "Ұсынылған құжаттарда бұл сұрақ бойынша ақпарат жоқ."

    # =====================================================================
    # Auth (v5 — Basic Auth для /chat, /ingest, /documents, /evaluation)
    # =====================================================================
    # Если AUTH_ENABLED=True и заданы user/pass — все write/read endpoints
    # требуют HTTP Basic Auth. /health и /metrics остаются открытыми.
    AUTH_ENABLED: bool = False
    AUTH_USER: str = "admin"
    AUTH_PASSWORD: str = "changeme"
    # Открытые пути (без auth): health, metrics, docs
    AUTH_PUBLIC_PATHS: str = "/health,/metrics,/docs,/openapi.json,/redoc"

    # =====================================================================
    # Prometheus Monitoring (v5)
    # =====================================================================
    METRICS_ENABLED: bool = True
    METRICS_PATH: str = "/metrics"

    # =====================================================================
    # A/B Prompt Testing (v5)
    # =====================================================================
    # Активный вариант промпта по умолчанию: 'strict', 'balanced', 'concise'.
    # См. app/prompts.py → PROMPT_VARIANTS.
    PROMPT_VARIANT: str = "strict"

    # =====================================================================
    # Sparse retrieval через TEI (v5)
    # =====================================================================
    # Если True и SPARSE_METHOD='bge_m3_sparse' — sparse-эмбеддинги вычисляются
    # через TEI HTTP endpoint (точнее на мультиязычности, чем BM25).
    # Требует TEI с поддержкой sparse (bge-m3 должен это уметь).
    TEI_SPARSE_ENABLED: bool = False

    # =====================================================================
    # Computed properties
    # =====================================================================
    @property
    def qdrant_url(self) -> str:
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    @property
    def vllm_base_url(self) -> str:
        return f"http://{self.VLLM_HOST}:{self.VLLM_PORT}/v1"

    @property
    def tei_url(self) -> str:
        return f"http://{self.TEI_HOST}:{self.TEI_PORT}"

    @property
    def docs_path(self) -> Path:
        p = Path(self.DOCS_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def models_path(self) -> Path:
        p = Path(self.MODELS_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p


# Singleton
settings = Settings()
