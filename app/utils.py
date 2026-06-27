"""
app/utils.py — вспомогательные функции.

Детектор языка (RU/KZ/other), стабильный doc_id, chunk_hash, sanitize_text,
извлечение заголовка раздела из PDF по bold-эвристике.
"""

import hashlib
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Казахские специфические буквы (нет в русском алфавите)
_KZ_MARKERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
# Русские специфические буквы (нет в казахском)
_RU_MARKERS = set("ыэъьёЫЭЪЬЁ")


# ---------------------------------------------------------------------------
# Язык
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """
    Возвращает 'kk', 'ru' или 'other'.

    Алгоритм:
    1. Если в тексте есть казахские специфические буквы в значимом количестве → 'kk'.
    2. Иначе если есть русские специфические буквы → 'ru'.
    3. Иначе если кириллицы мало → 'other'.
    4. Дефолт: 'ru' (наиболее вероятный язык в корпоративной базе).
    """
    if not text or not text.strip():
        return "ru"

    sample = text[:1500]
    chars = set(sample)

    kz_count = sum(1 for c in sample if c in _KZ_MARKERS)
    ru_count = sum(1 for c in sample if c in _RU_MARKERS)
    cyrillic_total = sum(1 for c in sample if "а" <= c.lower() <= "я" or c in _KZ_MARKERS)

    if cyrillic_total < 5:
        return "other"

    # Если казахских букв > 15% от всех кириллических → kk
    if kz_count > 0 and cyrillic_total > 0:
        if kz_count / max(cyrillic_total, 1) > 0.05:
            return "kk"

    if ru_count > 0 or cyrillic_total > 10:
        return "ru"

    return "ru"


# ---------------------------------------------------------------------------
# Идентификаторы
# ---------------------------------------------------------------------------

def generate_doc_id(file_path: str | Path) -> str:
    """
    Стабильный doc_id: имя_файла + md5(содержимое)[:12].
    Один и тот же файл всегда даёт один и тот же doc_id.
    """
    path = Path(file_path)
    try:
        content_hash = hashlib.md5(path.read_bytes()).hexdigest()[:12]
    except OSError:
        content_hash = hashlib.md5(str(path).encode()).hexdigest()[:12]
    name = re.sub(r"[^\w]+", "_", path.stem.lower()).strip("_")[:60]
    return f"{name}_{content_hash}"


def compute_chunk_hash(text: str) -> str:
    """MD5 от текста чанка — для дедупликации при повторной индексации."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def parent_id_for(doc_id: str, parent_idx: int) -> str:
    """Детерминированный parent_id. Без uuid — иначе PDR не найдёт родителя."""
    return f"{doc_id}_p{parent_idx:04d}"


def child_id_for(doc_id: str, parent_idx: int, child_idx: int) -> str:
    """Детерминированный chunk_id."""
    return f"{doc_id}_p{parent_idx:04d}_c{child_idx:04d}"


# ---------------------------------------------------------------------------
# Текст
# ---------------------------------------------------------------------------

def sanitize_text(text: str) -> str:
    """
    Очистка артефактов после PDF/DOCX экстракторов:
    - нормализация Unicode → NFC
    - удаление мягких переносов (\u00ad)
    - удаление управляющих символов (кроме \n \t)
    - схлопывание множественных пробелов
    - не более 2 переводов строк подряд
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00ad", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^[ \t]+", "", text, flags=re.MULTILINE)
    return text.strip()


def safe_filename(name: str) -> str:
    """Делает имя файла безопасным для файловой системы."""
    name = Path(name).name
    name = re.sub(r"[^\w.\-]+", "_", name)
    return name[:200]


def truncate(text: str, max_len: int = 280) -> str:
    """Безопасное обрезание для сниппетов."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "…"


def utc_now_iso() -> str:
    """ISO timestamp UTC для created_at."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Извлечение заголовков разделов из PDF
# ---------------------------------------------------------------------------

def extract_section_title_from_pdf_page(page) -> str:
    """
    Эвристическое извлечение заголовка раздела со страницы PDF.
    Ищем первый span с bold-флагом и разумной длиной (5..120 символов).

    page: объект fitz.Page
    """
    try:
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    flags = span.get("flags", 0)
                    text = span.get("text", "").strip()
                    # Бит 4 (16) = bold в PyMuPDF
                    if (flags & 16) and 5 <= len(text) <= 120:
                        return text[:120]
    except Exception as exc:
        logger.debug("Section title extraction failed: %s", exc)
    return ""


def extract_headings_from_docx(paragraphs: List[Any]) -> str:
    """
    Возвращает заголовок последнего Heading-параграфа (для текущего места).
    Используется при итерации по DOCX.
    """
    last_heading = ""
    for para in paragraphs:
        style = para.style.name if para.style else ""
        text = para.text.strip() if para.text else ""
        if style.startswith("Heading") and text:
            last_heading = text[:120]
    return last_heading


# ---------------------------------------------------------------------------
# Цитирование
# ---------------------------------------------------------------------------

def extract_citation_refs(text: str) -> List[int]:
    """Парсит [1], [2], [1,3] → [1, 2, 3]."""
    refs: set = set()
    for match in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", text or ""):
        for part in match.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                refs.add(int(part))
    return sorted(refs)
