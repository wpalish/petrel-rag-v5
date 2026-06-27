"""Тесты A/B prompt testing (v5)."""
from __future__ import annotations

from app.prompts import (
    PROMPT_VARIANTS,
    get_system_prompt,
    get_no_answer,
    list_variants,
)


def test_three_variants_available():
    """Должно быть 3 варианта промптов."""
    assert set(PROMPT_VARIANTS) == {"strict", "balanced", "concise"}


def test_list_variants_endpoint_format():
    """list_variants возвращает правильную структуру."""
    result = list_variants()
    assert "variants" in result
    assert "default" in result
    assert "description" in result
    assert result["default"] == "strict"
    assert "strict" in result["description"]
    assert "balanced" in result["description"]
    assert "concise" in result["description"]


def test_get_prompt_strict_ru():
    """strict variant для русского содержит ключевые правила."""
    prompt = get_system_prompt("ru", "strict")
    assert "СТРОГО" in prompt or "строго" in prompt
    assert "{context}" in prompt
    assert "{question}" in prompt
    assert "ОТВЕТ:" in prompt


def test_get_prompt_balanced_ru():
    """balanced variant более мягкий."""
    prompt = get_system_prompt("ru", "balanced")
    # В balanced нет слова "СТРОГО"
    assert "СТРОГО" not in prompt
    assert "{context}" in prompt
    assert "{question}" in prompt


def test_get_prompt_concise_ru():
    """concise variant короткий."""
    prompt = get_system_prompt("ru", "concise")
    # concise короче strict
    strict_prompt = get_system_prompt("ru", "strict")
    assert len(prompt) < len(strict_prompt)
    assert "Максимум 3-5" in prompt or "1-3" in prompt


def test_get_prompt_strict_kk():
    """strict variant для казахского."""
    prompt = get_system_prompt("kk", "strict")
    assert "СТРОГО" in prompt
    assert "{context}" in prompt
    assert "{question}" in prompt


def test_get_prompt_balanced_kk():
    """balanced для казахского."""
    prompt = get_system_prompt("kk", "balanced")
    assert "СТРОГО" not in prompt
    assert "{context}" in prompt


def test_get_prompt_concise_kk():
    """concise для казахского."""
    prompt = get_system_prompt("kk", "concise")
    strict_prompt = get_system_prompt("kk", "strict")
    assert len(prompt) < len(strict_prompt)


def test_unknown_variant_falls_back_to_strict():
    """Неизвестный variant → fallback на strict."""
    prompt = get_system_prompt("ru", "unknown")
    strict_prompt = get_system_prompt("ru", "strict")
    assert prompt == strict_prompt


def test_get_no_answer_ru():
    assert get_no_answer("ru") == "В предоставленных документах нет информации по данному вопросу."


def test_get_no_answer_kk():
    assert "ақпарат жоқ" in get_no_answer("kk")


def test_kk_prompt_has_few_shot():
    """Казахский промпт содержит few-shot примеры."""
    prompt = get_system_prompt("kk", "strict")
    # Few-shot примеры должны быть
    assert "Мысал" in prompt or "мысал" in prompt.lower()


def test_ru_prompt_no_few_shot():
    """Русский промпт не имеет few-shot (только для казахского)."""
    prompt = get_system_prompt("ru", "strict")
    assert "Мысал" not in prompt


def test_all_variants_format_correctly():
    """Все варианты должны корректно форматироваться с context и question."""
    test_context = "Тестовый контекст документа."
    test_question = "Тестовый вопрос?"

    for lang in ("ru", "kk"):
        for variant in PROMPT_VARIANTS:
            prompt = get_system_prompt(lang, variant)
            formatted = prompt.format(context=test_context, question=test_question)
            assert test_context in formatted
            assert test_question in formatted
            # Не должно остаться незаполненных плейсхолдеров
            assert "{context}" not in formatted
            assert "{question}" not in formatted
