FROM python:3.11-slim

# ---------------------------------------------------------------------------
# Системные зависимости для PDF/DOCX парсинга и сборки пакетов
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Устанавливаем зависимости раньше копирования кода — для кэша Docker
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Копируем исходный код (без data/ — её монтируем через volume в compose)
# ---------------------------------------------------------------------------
COPY app/ ./app/
COPY tests/ ./tests/
COPY evaluation/ ./evaluation/
COPY samples/ ./samples/
COPY scripts/ ./scripts/
COPY monitoring/ ./monitoring/
COPY pyproject.toml ./

# Создаём директории для данных (мапятся в compose)
RUN mkdir -p /app/data/docs /app/data/index /app/models

# ---------------------------------------------------------------------------
# Переменные окружения по умолчанию
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/models \
    HF_HUB_OFFLINE=0 \
    OLLAMA_NO_TELEMETRY=1 \
    VLLM_NO_USAGE_STATS=1

EXPOSE 7860 8000

# По умолчанию запускаем backend. UI запускается отдельной командой в compose.
CMD ["python", "-m", "app.main"]
