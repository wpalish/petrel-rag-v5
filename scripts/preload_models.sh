#!/bin/bash
# =============================================================================
# preload_models.sh — предзагрузка моделей для offline-деплоя (v3)
# =============================================================================
# Запускать на машине с интернетом. Скрипт загружает все необходимые модели
# в стандартную структуру HuggingFace cache, совместимую с vLLM и TEI.
#
# После выполнения — скопировать папку ./models на целевой сервер
# (или ~/.cache/huggingface целиком) и смонтировать в контейнеры.
# =============================================================================

set -e

echo "=== Preloading models for Corporate RAG v3 ==="
echo "Date: $(date)"
echo "Models will be downloaded to standard HuggingFace cache structure."
echo

# Проверка наличия huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-cli + hf-transfer..."
    pip install --quiet huggingface_hub hf-transfer
fi

# Использовать быстрый transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "[1/3] Downloading LLM: Qwen/Qwen2.5-7B-Instruct-AWQ..."
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ
echo "  ✓ Done."

echo "[2/3] Downloading embeddings: BAAI/bge-m3..."
huggingface-cli download BAAI/bge-m3
echo "  ✓ Done."

echo "[3/3] Downloading reranker: BAAI/bge-reranker-v2-m3..."
huggingface-cli download BAAI/bge-reranker-v2-m3
echo "  ✓ Done."

echo
echo "=== All models preloaded to standard HuggingFace cache ==="
echo "Location: ~/.cache/huggingface/hub/"
du -sh ~/.cache/huggingface/hub/* 2>/dev/null || true

echo
echo "=== Next steps ==="
echo "1. Скопируйте папку ~/.cache/huggingface (или ./models) на целевой сервер."
echo "2. На целевом сервере: cp .env.example .env"
echo "3. Установите HF_HUB_OFFLINE=1 в .env (после первого успешного старта)."
echo "4. docker compose up -d --build"
echo "5. ./scripts/audit_telemetry.sh"
echo "6. ./scripts/run_evaluation.sh 5  (smoke-test на 5 вопросах)"
