"""Тесты определения языка (v4) — без смешения KZ/RU (требование ТЗ)."""
from __future__ import annotations

from app.utils import detect_language


def test_russian_detected():
    assert detect_language("Какова процедура согласования отпуска?") == "ru"


def test_kazakh_detected_by_specific_letters():
    # ә, қ, ң — буквы, которых нет в русском алфавите
    assert detect_language("Демалыс рәсімі қандай?") == "kk"


def test_plain_cyrillic_defaults_to_russian():
    """Чистая кириллица без KZ-специфики → ru (наиболее вероятный язык)."""
    assert detect_language("отпуск") == "ru"


def test_mixed_but_kazakh_letters_win():
    """Если в тексте есть казахские буквы — определяем как kk."""
    assert detect_language("VPN арқылы қосылу") == "kk"


def test_short_text_russian():
    assert detect_language("Привет") == "ru"


def test_empty_text_default_russian():
    """Пустой текст → дефолт 'ru' (безопасный выбор для корпоративной базы)."""
    assert detect_language("") == "ru"


def test_kazakh_with_many_specific_letters():
    assert detect_language("Әр қызметкер өз демалысын жоспарлайды") == "kk"


def test_pure_latin_detected_as_other():
    """Чисто латинский текст без кириллицы → 'other' (мало кириллицы)."""
    result = detect_language("Hello world this is english text")
    assert result == "other"
