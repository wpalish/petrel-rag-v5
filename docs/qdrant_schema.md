# Qdrant Collection Schema — Corporate Knowledge Bot v2

**Collection name:** `corporate_knowledge`
**Формат:** Named vectors (dense + sparse reserved)
**Цель:** Поддержка dense retrieval сейчас + готовность к hybrid (dense + sparse + RRF) без миграции коллекции.

## Создание коллекции

```python
client.create_collection(
    collection_name="corporate_knowledge",
    vectors_config={
        "dense": VectorParams(size=1024, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
    }
)
```

- **`dense`**: 1024-dim float vector (bge-m3 dense)
- **`sparse`**: зарезервирован. В MVP не заполняется. Добавляется в будущем для hybrid.

## Payload-поля

| Field | Type | Обязательно | Назначение |
|-------|------|-------------|------------|
| `text` | string | Да | Текст чанка |
| `doc_id` | string | Да | ID документа (имя + md5 контента) |
| `doc_name` | string | Да | Имя файла |
| `source_path` | string | Рекомендуется | Полный путь к исходному файлу |
| `doc_version` | string | Рекомендуется | Версия документа (для versioning) |
| `page_number` | integer | Да (для PDF) | Номер страницы |
| `paragraph_index` | integer | Рекомендуется | Номер абзаца |
| `section_title` | string | Рекомендуется | Заголовок раздела |
| `section_path` | string | Рекомендуется | Иерархический путь (например, "3 > 3.2") |
| `parent_id` | string | Да (для PDR) | ID parent-чанка |
| `is_parent` | boolean | Да | Является ли это parent-чанк |
| `chunk_id` | string | Да | Уникальный ID чанка |
| `chunk_hash` | string | Рекомендуется | MD5 чанка (для дедупликации) |
| `language` | string | Да | "ru" / "kk" / "other" |
| `source_type` | string | Да | "pdf" / "docx" / "txt" |
| `created_at` | string | Да | ISO timestamp UTC |

## Payload Indexes (создаются автоматически в `ensure_collection`)

```python
indexes = [
    ("doc_id", PayloadSchemaType.KEYWORD),
    ("language", PayloadSchemaType.KEYWORD),
    ("source_type", PayloadSchemaType.KEYWORD),
    ("page_number", PayloadSchemaType.INTEGER),
    ("doc_version", PayloadSchemaType.KEYWORD),
    ("parent_id", PayloadSchemaType.KEYWORD),
    ("chunk_hash", PayloadSchemaType.KEYWORD),
]
```

Эти индексы ускоряют фильтрацию и delete-by-filter по `doc_id` (идемпотентная переиндексация).

## Поиск

```python
# Dense search через named vector
results = client.search(
    collection_name="corporate_knowledge",
    query_vector=("dense", query_embedding),
    limit=16,
    with_payload=True,
    with_vectors=False,
)
```

## Удаление документа

```python
# По doc_id (опционально с фильтром по версии)
client.delete(
    collection_name="corporate_knowledge",
    points_selector=Filter(
        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    )
)
```

## Backup (snapshots)

```python
snapshot = client.create_snapshot(collection_name="corporate_knowledge")
```

Создаётся автоматически после каждой индексации в `ingest_document()`.

## Переход к Hybrid Search (после MVP)

1. При индексации: вычислить sparse вектор (bge-m3 sparse или BM25).
2. Сохранить в `vector["sparse"]`.
3. При поиске: dense + sparse query → RRF (Reciprocal Rank Fusion).

Текущая схема позволяет добавить sparse **без миграции коллекции** — поле уже зарезервировано.

---

**Версия документа:** 2.0
**Дата:** 2026-06-26
