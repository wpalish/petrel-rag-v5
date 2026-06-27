#!/bin/bash
# =============================================================================
# run_evaluation.sh — запуск RAGAS-оценки на тестовом наборе
# =============================================================================
# Запускать ПОСЛЕ того, как:
#   1. Все сервисы подняты: docker compose up -d
#   2. Документы загружены через UI или /ingest
#   3. BM25 индекс построен (последний ingest делает это автоматически)
#
# Использование:
#   ./scripts/run_evaluation.sh            # все 30 вопросов
#   ./scripts/run_evaluation.sh 5          # только первые 5 (smoke-test)
# =============================================================================

set -e

LIMIT="${1:-}"
URL="http://localhost:8000/evaluation/run"

if [ -n "$LIMIT" ]; then
    URL="${URL}?limit=${LIMIT}"
    echo "=== Running evaluation on first ${LIMIT} questions ==="
else
    echo "=== Running full evaluation (all questions) ==="
fi

echo "POST ${URL}"
echo

# Запускаем оценку
RESPONSE=$(curl -s -X POST "${URL}" --max-time 1800)

# Парсим summary через python
echo "${RESPONSE}" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
except Exception as e:
    print('ERROR parsing response:', e)
    print('Raw response (first 1000 chars):')
    print(sys.stdin.read()[:1000])
    sys.exit(1)

if 'summary' not in data:
    print('ERROR: no summary in response')
    print(json.dumps(data, ensure_ascii=False, indent=2)[:2000])
    sys.exit(1)

s = data['summary']
print('=' * 60)
print('EVALUATION SUMMARY')
print('=' * 60)
print(f'Total questions:        {s[\"total_questions\"]}')
print(f'RAGAS available:        {s[\"ragas_available\"]}')
print(f'RAGAS evaluated:        {s[\"ragas_evaluated\"]}')
print(f'No-answer count:        {s[\"no_answer_count\"]}')
print(f'Avg latency (sec):      {s[\"avg_latency_sec\"]}')
print()
print('-' * 60)
print('METRICS vs TARGETS')
print('-' * 60)
targets = s.get('targets', {})

def show(name, val, target=None):
    suffix = ''
    if target is not None:
        ok = '✓' if val >= target else '✗'
        suffix = f'  (target: {target}) {ok}'
    print(f'  {name:30s} {val:.3f}{suffix}')

show('avg_faithfulness',       s.get('avg_faithfulness', 0),       targets.get('faithfulness'))
show('avg_answer_relevancy',   s.get('avg_answer_relevancy', 0),   targets.get('answer_relevancy'))
show('avg_language_purity',    s.get('avg_language_purity', 0),    targets.get('language_purity'))
show('avg_citation_accuracy',  s.get('avg_citation_accuracy', 0))
show('avg_no_answer_accuracy', s.get('avg_no_answer_accuracy', 0))
show('avg_keywords_match',     s.get('avg_keywords_match', 0))
print()
print('=' * 60)
print('Detailed results saved to: /app/evaluation/results.json')
print('To view: docker compose exec rag-backend cat /app/evaluation/results.json')
print('=' * 60)
"
