"""Тесты ingestion (v4): чанкинг с перекрытием, привязка к источнику, hash-дедуп.

Идея перенесена из petrel-rag-assistant, адаптирована под наш app.ingestion
(parent-child chunking, section title extraction, chunk_hash).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.utils import (
    compute_chunk_hash,
    generate_doc_id,
    sanitize_text,
    extract_citation_refs,
    parent_id_for,
    child_id_for,
    truncate,
)


# ---------------------------------------------------------------------------
# compute_chunk_hash
# ---------------------------------------------------------------------------

def test_chunk_hash_stable():
    """Один и тот же текст → один и тот же хэш."""
    h1 = compute_chunk_hash("тестовый чанк")
    h2 = compute_chunk_hash("тестовый чанк")
    assert h1 == h2


def test_chunk_hash_different_for_different_text():
    h1 = compute_chunk_hash("первый")
    h2 = compute_chunk_hash("второй")
    assert h1 != h2


def test_chunk_hash_empty():
    """Пустой текст тоже хэшируется (не падает)."""
    h = compute_chunk_hash("")
    assert isinstance(h, str)
    assert len(h) > 0


# ---------------------------------------------------------------------------
# generate_doc_id
# ---------------------------------------------------------------------------

def test_generate_doc_id_stable(tmp_path: Path):
    """Один и тот же файл → один и тот же doc_id."""
    f = tmp_path / "reglament.pdf"
    f.write_bytes("содержимое регламента".encode("utf-8"))
    id1 = generate_doc_id(str(f))
    id2 = generate_doc_id(str(f))
    assert id1 == id2


def test_generate_doc_id_different_content(tmp_path: Path):
    """Разное содержимое → разные doc_id."""
    f1 = tmp_path / "v1.pdf"
    f1.write_bytes("версия 1".encode("utf-8"))
    f2 = tmp_path / "v2.pdf"  # другое имя, другое содержимое
    f2.write_bytes("версия 2".encode("utf-8"))
    assert generate_doc_id(str(f1)) != generate_doc_id(str(f2))


def test_generate_doc_id_contains_name(tmp_path: Path):
    """doc_id должен содержать имя файла (для читаемости)."""
    f = tmp_path / "Регламент_отпусков.pdf"
    f.write_bytes(b"x")
    doc_id = generate_doc_id(str(f))
    assert "регламент_отпусков" in doc_id.lower()


# ---------------------------------------------------------------------------
# parent_id_for / child_id_for
# ---------------------------------------------------------------------------

def test_parent_id_deterministic():
    """v4 фикс: parent_id должен быть детерминированным (без uuid)."""
    pid1 = parent_id_for("doc123", 0)
    pid2 = parent_id_for("doc123", 0)
    assert pid1 == pid2
    assert pid1 == "doc123_p0000"


def test_child_id_deterministic():
    cid1 = child_id_for("doc123", 0, 1)
    cid2 = child_id_for("doc123", 0, 1)
    assert cid1 == cid2
    assert cid1 == "doc123_p0000_c0001"


def test_parent_id_different_indices():
    assert parent_id_for("d", 0) != parent_id_for("d", 1)
    assert parent_id_for("d", 0) == "d_p0000"
    assert parent_id_for("d", 1) == "d_p0001"


# ---------------------------------------------------------------------------
# sanitize_text
# ---------------------------------------------------------------------------

def test_sanitize_removes_control_chars():
    text = "hello\x00world\x07end"
    cleaned = sanitize_text(text)
    assert "\x00" not in cleaned
    assert "\x07" not in cleaned
    assert "hello" in cleaned and "world" in cleaned and "end" in cleaned


def test_sanitize_collapses_spaces():
    text = "много    пробелов   тут"
    cleaned = sanitize_text(text)
    assert "    " not in cleaned
    assert "много пробелов тут" == cleaned


def test_sanitize_collapses_newlines():
    text = "строка1\n\n\n\n\nстрока2"
    cleaned = sanitize_text(text)
    assert "\n\n\n" not in cleaned


def test_sanitize_removes_soft_hyphens():
    """Мягкие переносы (\u00ad) удаляются — артефакт PDF."""
    text = "корот\u00adкий"
    cleaned = sanitize_text(text)
    assert "\u00ad" not in cleaned
    assert "короткий" in cleaned


def test_sanitize_empty():
    assert sanitize_text("") == ""
    assert sanitize_text(None) == ""


# ---------------------------------------------------------------------------
# extract_citation_refs
# ---------------------------------------------------------------------------

def test_citation_refs_single():
    assert extract_citation_refs("ответ [1] здесь") == [1]


def test_citation_refs_multiple():
    assert extract_citation_refs("ответ [1] и [2] здесь") == [1, 2]


def test_citation_refs_combined():
    assert extract_citation_refs("ответ [1, 3] здесь") == [1, 3]


def test_citation_refs_no_refs():
    assert extract_citation_refs("ответ без ссылок") == []


def test_citation_refs_sorted_dedup():
    assert extract_citation_refs("[3] [1] [3] [2]") == [1, 2, 3]


# ---------------------------------------------------------------------------
# truncate
# ---------------------------------------------------------------------------

def test_truncate_short_text_unchanged():
    assert truncate("короткий", 100) == "короткий"


def test_truncate_long_text_cut():
    text = "слово " * 100
    result = truncate(text, 50)
    assert len(result) <= 51  # 50 + "…"
    assert result.endswith("…")


def test_truncate_empty():
    assert truncate("", 100) == ""
    assert truncate(None, 100) == ""
