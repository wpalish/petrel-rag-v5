#!/bin/bash
# =============================================================================
# audit_telemetry.sh — аудит исходящего трафика On-Premise системы
# =============================================================================
# Запускать на целевом сервере ПОСЛЕ старта всех контейнеров.
# Цель: убедиться, что нет утечки данных во внешние сети.
# =============================================================================

set -e

echo "=== On-Premise RAG Telemetry Audit (v2) ==="
echo "Date: $(date)"
echo

# 1. Проверка переменных окружения в backend
echo "=== 1. Environment variables (backend) ==="
docker compose exec -T rag-backend env 2>/dev/null | grep -E "(HF_HUB_OFFLINE|VLLM_NO_USAGE_STATS|OLLAMA_NO_TELEMETRY|RERANKER_NORMALIZE|USE_RERANKER)" | sort || \
    echo "  ⚠️ rag-backend container not running"
echo

# 2. Проверка, что vLLM запущен и отвечает
echo "=== 2. vLLM health check ==="
curl -sf http://localhost:8001/health > /dev/null 2>&1 && \
    echo "  ✓ vLLM is healthy" || \
    echo "  ✗ vLLM is NOT responding"
echo

# 3. Проверка Qdrant
echo "=== 3. Qdrant collections ==="
curl -s http://localhost:6333/collections | head -c 300
echo
echo

# 4. Audit outbound traffic — 30 секунд
echo "=== 4. Outbound traffic audit (30 sec) ==="
echo "Looking for ANY external connections (not to local docker network)..."
echo "If you see IPs below — there is telemetry leak. Investigate!"
echo

# Получаем IP-адреса всех контейнеров в сети compose
NETWORK_NAME="$(basename $(pwd))_default"
echo "Docker network: $NETWORK_NAME"
echo

# Получаем список всех IP-адресов контейнеров
CONTAINER_IPS=$(docker network inspect "$NETWORK_NAME" 2>/dev/null | \
    python3 -c "import json,sys; data=json.load(sys.stdin); ips=[v['IPv4Address'] for c in data[0]['Containers'].values() for k,v in [('ip',c)]; print(' '.join(ips))" 2>/dev/null || \
    echo "127.0.0.1")

echo "Local IPs to exclude: $CONTAINER_IPS"
echo

# tcpdump внутри контейнера backend
echo "Running tcpdump in rag-backend for 30 seconds..."
echo "---"

# Фильтр: исключаем локальные адреса (loopback, docker bridge, internal containers)
FILTER="not (host 127.0.0.1 or host 172.17.0.1 or host 172.18.0.1"
for ip in $CONTAINER_IPS; do
    [ "$ip" != "127.0.0.1" ] && FILTER="$FILTER or host $ip"
done
FILTER="$FILTER) and not (port 6333 or port 6334 or port 8000 or port 8001 or port 80 or port 7860)"

timeout 30s docker compose exec -T rag-backend tcpdump -i any -n -c 50 "$FILTER" 2>&1 | head -100 || \
    echo "  (no external traffic captured — that's GOOD)"

echo "---"
echo

# 5. Проверка логов на подозрительные ключевые слова
echo "=== 5. Backend log scan for suspicious outbound calls ==="
docker compose logs rag-backend 2>/dev/null | \
    grep -iE "(huggingface\.co|openai\.com|anthropic\.com|api\.openai|telemetry|sentry|analytics)" | \
    head -20 || \
    echo "  ✓ No suspicious outbound references in logs"
echo

# 6. Итоговый вердикт
echo "=== 6. Summary ==="
echo "  • If section 4 showed NO IPs — telemetry audit PASSED ✓"
echo "  • If section 5 showed no suspicious logs — log scan PASSED ✓"
echo "  • If HF_HUB_OFFLINE=1 — offline mode is active ✓"
echo
echo "Audit complete."
