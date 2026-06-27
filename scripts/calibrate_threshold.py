#!/usr/bin/env python3
"""
scripts/calibrate_threshold.py — подбор MIN_RERANK_SCORE по тестовому набору.

Запуск (внутри контейнера rag-backend):
    python /app/scripts/calibrate_threshold.py
    python /app/scripts/calibrate_threshold.py --limit 10

Логика:
1. Грузим tests/testset.json
2. Для каждого вопроса делаем retrieve (без LLM-генерации)
3. Сохраняем rerank_score всех кандидатов
4. Делим на 2 группы: expected_no_answer=True и False
5. Считаем распределение (p25, p50, p75) для группы "с ответом"
6. Предлагаем MIN_RERANK_SCORE = p25 группы "с ответом"
   (отсекает 25% релевантных, но повышает no_answer_accuracy на edge-cases)

После калибровки:
- Изменить MIN_RERANK_SCORE в .env
- Перезапустить: docker compose restart rag-backend
- Прогнать /evaluation/run для проверки
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

# Чтобы скрипт работал внутри контейнера
sys.path.insert(0, "/app")

from app.config import settings
from app.evaluation_runner import load_testset
from app.rag_engine import get_engine


def percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    if f == c:
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def main():
    parser = argparse.ArgumentParser(description="Calibrate MIN_RERANK_SCORE")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions")
    args = parser.parse_args()

    print(f"=== MIN_RERANK_SCORE Calibration ===")
    print(f"Current threshold: {settings.MIN_RERANK_SCORE}")
    print(f"Reranker normalize: {settings.RERANKER_NORMALIZE}")
    print()

    # Грузим testset
    try:
        testset = load_testset()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if args.limit:
        testset = testset[:args.limit]
    print(f"Testset size: {len(testset)} questions")

    engine = get_engine()

    scores_with_answer: list = []
    scores_no_answer: list = []
    questions_with_no_retrieval = 0

    for i, q in enumerate(testset, 1):
        expected_no = q.get("expected_no_answer", False)
        question = q["question"]
        qid = q.get("id", f"q{i}")

        try:
            retrieved = engine.retrieve(question)
        except Exception as exc:
            print(f"  [{qid}] retrieve failed: {exc}")
            continue

        if not retrieved:
            questions_with_no_retrieval += 1
            print(f"  [{qid}] no candidates retrieved (lang={q.get('language', '?')})")
            continue

        # Берём топ-1 rerank_score
        top_score = retrieved[0].get("rerank_score", 0)
        if expected_no:
            scores_no_answer.append(top_score)
            print(f"  [{qid}] expected_no=True, top_rerank={top_score:.3f}")
        else:
            scores_with_answer.append(top_score)
            print(f"  [{qid}] expected_no=False, top_rerank={top_score:.3f}")

    print()
    print("=" * 60)
    print("DISTRIBUTION ANALYSIS")
    print("=" * 60)

    print("\nQuestions EXPECTED WITH answer:")
    if scores_with_answer:
        print(f"  Count: {len(scores_with_answer)}")
        print(f"  Min:   {min(scores_with_answer):.4f}")
        print(f"  Max:   {max(scores_with_answer):.4f}")
        print(f"  Mean:  {statistics.mean(scores_with_answer):.4f}")
        print(f"  Median: {statistics.median(scores_with_answer):.4f}")
        print(f"  p25:   {percentile(scores_with_answer, 25):.4f}")
        print(f"  p50:   {percentile(scores_with_answer, 50):.4f}")
        print(f"  p75:   {percentile(scores_with_answer, 75):.4f}")
    else:
        print("  No data")

    print("\nQuestions EXPECTED NO answer (edge cases):")
    if scores_no_answer:
        print(f"  Count: {len(scores_no_answer)}")
        print(f"  Min:   {min(scores_no_answer):.4f}")
        print(f"  Max:   {max(scores_no_answer):.4f}")
        print(f"  Mean:  {statistics.mean(scores_no_answer):.4f}")
        print(f"  Median: {statistics.median(scores_no_answer):.4f}")
    else:
        print("  No data")

    print()
    print("=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    if scores_with_answer:
        # p25 = отсекает 25% правильных, но повышает no_answer_accuracy
        recommended_p25 = round(percentile(scores_with_answer, 25), 3)
        # p10 = более агрессивно, но может пропустить слабо-релевантные
        recommended_p10 = round(percentile(scores_with_answer, 10), 3)

        print(f"\nRecommended MIN_RERANK_SCORE:")
        print(f"  Conservative (p25): {recommended_p25}")
        print(f"    → отсекает 25% правильных, но ловит все edge-cases выше этого порога")
        print(f"  Aggressive (p10):   {recommended_p10}")
        print(f"    → отсекает 10% правильных, выше recall но ниже no_answer_accuracy")
        print()
        print(f"Current value: {settings.MIN_RERANK_SCORE}")

        # Если в no_answer группе все скоры выше p25 — значит порог не сработает
        if scores_no_answer:
            no_answer_max = max(scores_no_answer)
            print(f"\nNote: max rerank_score in 'no-answer' group = {no_answer_max:.4f}")
            if no_answer_max > recommended_p25:
                print(f"  ⚠ Some no-answer questions have rerank_score > {recommended_p25}")
                print(f"    These will pass through threshold — consider higher value.")
            else:
                print(f"  ✓ All no-answer questions below threshold — good separation.")

        print()
        print("To apply:")
        print(f"  1. Edit .env: MIN_RERANK_SCORE={recommended_p25}")
        print(f"  2. Restart: docker compose restart rag-backend")
        print(f"  3. Verify: curl -X POST http://localhost:8000/evaluation/run?limit=10")
    else:
        print("\nCannot recommend — no 'with-answer' data.")


if __name__ == "__main__":
    main()
