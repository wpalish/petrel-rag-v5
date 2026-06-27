"""
app/evaluation_runner.py — локальная RAG-оценка БЕЗ внешних API (v4).

ГЛАВНОЕ ИЗМЕНЕНИЕ v4 (перенесено из petrel-rag-assistant):
- groundedness считается ЛОКАЛЬНЫМ LLM через vLLM/Ollama как judge
- RAGAS удалён — он по умолчанию обращается к OpenAI, что нарушает On-Premise ТЗ
- Все метрики считаются локально, без единого внешнего вызова

Метрики:
1. groundedness — ответ подтверждается контекстом (ЛОКАЛЬНЫЙ LLM-as-judge)
2. language_purity — нет смешения языков (эвристика по буквам)
3. citation_accuracy — ответ сопровождается цитатами
4. no_answer_accuracy — корректность честного "нет информации"
5. keywords_match — доля ожидаемых ключевых слов в ответе
6. ttft_ms — время до первого токена (из streaming LLM)
7. latency_ms — полное время ответа

Алгоритм:
- Грузим tests/testset.json с вопросами и эталонами.
- Для каждого вопроса: async-вызов RAGEngine.query(), сохраняем ответ + TTFT.
- Если ответ не no-answer → запускаем локального LLM-judge для groundedness.
- Считаем эвристики по ответу.
- Возвращаем агрегированные метрики + детальные результаты.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings
from app.llm import chat as llm_chat
from app.rag_engine import get_engine
from app.utils import detect_language

logger = logging.getLogger(__name__)

# RAGAS ПОЛНОСТЬЮ УБРАН в v4 — нарушает On-Premise (ходит в OpenAI по умолчанию).


# ---------------------------------------------------------------------------
# Тестовый набор
# ---------------------------------------------------------------------------

def load_testset() -> List[Dict[str, Any]]:
    """Грузит tests/testset.json."""
    path = Path(settings.TESTSET_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Testset not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Invalid testset format in {path}")


# ---------------------------------------------------------------------------
# Эвристики (не требуют LLM)
# ---------------------------------------------------------------------------

_KZ_MARKERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
_RU_MARKERS = set("ыэъьёЫЭЪЬЁ")


def language_purity_score(question: str, answer: str) -> float:
    """
    1.0 если язык ответа совпадает с языком вопроса без смешения.
    """
    if not answer or not answer.strip():
        return 0.0
    q_lang = detect_language(question)
    a_lang = detect_language(answer)

    if q_lang == a_lang and q_lang != "other":
        if q_lang == "ru":
            kz_in_answer = sum(1 for c in answer if c in _KZ_MARKERS)
            return 1.0 if kz_in_answer == 0 else max(0.0, 1.0 - kz_in_answer * 0.05)
        if q_lang == "kk":
            ru_in_answer = sum(1 for c in answer if c in _RU_MARKERS)
            return 1.0 if ru_in_answer == 0 else max(0.0, 1.0 - ru_in_answer * 0.05)
        return 1.0
    return 0.3


def citation_accuracy_score(answer: str, sources: List[Dict[str, Any]]) -> float:
    """1.0 если есть [N] ссылки и sources непустые."""
    if not sources:
        return 1.0  # no-answer — без цитат ОК
    import re
    has_citation = bool(re.search(r"\[\d+(?:\s*,\s*\d+)*\]", answer))
    return 1.0 if has_citation else 0.5


def no_answer_accuracy_score(expected_no_answer: bool, actual_no_answer: bool) -> float:
    return 1.0 if expected_no_answer == actual_no_answer else 0.0


def keywords_match_score(answer: str, expected_keywords: List[str]) -> float:
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    matched = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return matched / len(expected_keywords)


# ---------------------------------------------------------------------------
# Локальный LLM-as-judge для groundedness (KEY CHANGE v4)
# ---------------------------------------------------------------------------

async def groundedness_score(answer: str, context_snippets: List[str]) -> Optional[float]:
    """
    Локальный LLM-as-judge: следует ли ответ из контекста.
    Возвращает 1.0 (ДА) / 0.0 (НЕТ) / None (не удалось определить).

    ИЗ petrel-rag-assistant: критически важно для On-Premise — RAGAS ходит в
    OpenAI, что нарушает ТЗ. Здесь judge = тот же локальный Qwen/vLLM.
    """
    if not context_snippets:
        return None
    ctx = "\n---\n".join(context_snippets)
    system = (
        "Ты — строгий проверяющий фактов. Определи, полностью ли УТВЕРЖДЕНИЕ "
        "следует из приведённого КОНТЕКСТА. Ответь ровно одним словом: ДА или НЕТ."
    )
    user = (
        f"КОНТЕКСТ:\n{ctx}\n\n"
        f"УТВЕРЖДЕНИЕ:\n{answer}\n\n"
        f"Следует ли утверждение из контекста?"
    )
    try:
        text, _ = await llm_chat(system=system, user=user, temperature=0.0, max_tokens=8)
    except Exception as exc:
        logger.warning("LLM judge failed: %s", exc)
        return None

    verdict = text.strip().lower().lstrip("«\"'*-•. ")
    if verdict.startswith(("нет", "no", "жоқ")):
        return 0.0
    if verdict.startswith(("да", "yes", "иә")):
        return 1.0
    return 0.5  # неоднозначный ответ


# ---------------------------------------------------------------------------
# Главный runner
# ---------------------------------------------------------------------------

async def run_evaluation(limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Полный цикл оценки:
    1. Грузит testset.
    2. async-прогон каждого вопроса через RAGEngine.query().
    3. Эвристики (language_purity, citation, no_answer, keywords).
    4. Локальный LLM-judge для groundedness (если ответ не no-answer).
    5. Агрегаты + детальные результаты.
    """
    testset = load_testset()
    if limit:
        testset = testset[:limit]

    logger.info("Starting v4 evaluation: %d questions (local LLM judge, NO RAGAS).",
                len(testset))

    engine = get_engine()
    detailed: List[Dict[str, Any]] = []

    for i, q in enumerate(testset):
        question = q["question"]
        t0 = time.perf_counter()
        try:
            result = await engine.query(question)
        except Exception as exc:
            logger.exception("Query failed for qid=%s", q.get("id"))
            result = {
                "answer": f"ERROR: {exc}",
                "sources": [],
                "language": "ru",
                "retrieval_count": 0,
                "no_answer": True,
                "ttft_ms": 0,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
            }
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        answer_text = result.get("answer", "")
        sources = result.get("sources", [])
        actual_no_answer = result.get("no_answer", False)
        ttft_ms = result.get("ttft_ms", 0)

        # Эвристики
        lang_purity = language_purity_score(question, answer_text)
        cit_acc = citation_accuracy_score(answer_text, sources)
        no_ans_acc = no_answer_accuracy_score(
            q.get("expected_no_answer", False),
            actual_no_answer,
        )
        kw_match = keywords_match_score(answer_text, q.get("expected_keywords", []))

        entry = {
            "id": q.get("id", f"q{i+1:02d}"),
            "question": question,
            "language": q.get("language", "ru"),
            "answer": answer_text,
            "no_answer": actual_no_answer,
            "expected_no_answer": q.get("expected_no_answer", False),
            "retrieval_count": result.get("retrieval_count", 0),
            "sources_count": len(sources),
            "ttft_ms": ttft_ms,
            "elapsed_ms": elapsed_ms,
            "metrics": {
                "language_purity": round(lang_purity, 3),
                "citation_accuracy": round(cit_acc, 3),
                "no_answer_accuracy": round(no_ans_acc, 3),
                "keywords_match": round(kw_match, 3),
            },
        }

        # Локальный LLM-judge для groundedness (только если есть ответ и контекст)
        if not actual_no_answer and sources:
            # Берём полные snippet'ы источников как контекст для judge
            snippets = [s.get("snippet", "") for s in sources if s.get("snippet")]
            g_score = await groundedness_score(answer_text, snippets)
            if g_score is not None:
                entry["metrics"]["groundedness"] = round(g_score, 3)

        detailed.append(entry)

    # Агрегаты
    n = len(detailed)
    if n == 0:
        return {"error": "Пустой testset.", "details": []}

    def avg(metric_key: str) -> float:
        vals = [d["metrics"].get(metric_key) for d in detailed if d["metrics"].get(metric_key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    summary = {
        "total_questions": n,
        "judge": "local_llm",  # v4: явно указываем, что judge локальный
        "ragas_used": False,   # v4: RAGAS удалён
        "avg_language_purity": avg("language_purity"),
        "avg_citation_accuracy": avg("citation_accuracy"),
        "avg_no_answer_accuracy": avg("no_answer_accuracy"),
        "avg_keywords_match": avg("keywords_match"),
        "avg_groundedness": avg("groundedness"),
        "avg_ttft_ms": round(sum(d.get("ttft_ms", 0) for d in detailed) / n),
        "avg_latency_ms": round(sum(d["elapsed_ms"] for d in detailed) / n),
        "no_answer_count": sum(1 for d in detailed if d["no_answer"]),
        "targets": {
            "groundedness": settings.TARGET_FAITHFULNESS,
            "language_purity": settings.TARGET_LANGUAGE_PURITY,
        },
        "note": "Все метрики вычислены локально (groundedness — через локальный vLLM). Внешние API не использовались.",
    }

    # Сохраняем результаты
    try:
        out_path = Path(settings.EVAL_RESULTS_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "details": detailed}, f, ensure_ascii=False, indent=2)
        logger.info("Evaluation results saved to %s", out_path)
    except Exception as exc:
        logger.warning("Could not save eval results: %s", exc)

    return {"summary": summary, "details": detailed}
