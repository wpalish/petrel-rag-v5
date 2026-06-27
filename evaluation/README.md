# evaluation/ — Результаты RAGAS-оценки

После запуска `POST /evaluation/run` здесь сохраняется файл `results.json` со всеми метриками и детальными результатами по каждому вопросу.

## Структура `results.json`

```json
{
  "summary": {
    "total_questions": 30,
    "ragas_available": true,
    "ragas_evaluated": 26,
    "avg_language_purity": 0.97,
    "avg_citation_accuracy": 0.92,
    "avg_no_answer_accuracy": 0.75,
    "avg_keywords_match": 0.68,
    "avg_faithfulness": 0.87,
    "avg_answer_relevancy": 0.83,
    "avg_latency_sec": 3.4,
    "no_answer_count": 4,
    "targets": {
      "faithfulness": 0.85,
      "answer_relevancy": 0.80,
      "language_purity": 0.95
    }
  },
  "details": [
    {
      "id": "q01",
      "question": "Сколько дней длится отпуск?",
      "answer": "Ежегодный отпуск составляет 24 календарных дня. [1]",
      "no_answer": false,
      "retrieval_count": 5,
      "elapsed_sec": 2.8,
      "metrics": {
        "language_purity": 1.0,
        "citation_accuracy": 1.0,
        "no_answer_accuracy": 1.0,
        "keywords_match": 1.0,
        "faithfulness": 0.95,
        "answer_relevancy": 0.92
      }
    }
  ]
}
```

## Использование для отчётности

`results.json` можно:
- Парсить для построения графиков (jupyter, pandas).
- Конвертировать в PDF/DOCX для заказчика.
- Диффить между запусками (A/B тестирование разных параметров).
