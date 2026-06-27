"""
app/ui.py — Gradio UI (v4).

Изменения v4:
- httpx с async-обёрткой через anyio (не блокирует event loop)
- Отображение TTFT и latency_ms под каждым ответом
- Кнопка "Загрузить демо-документы" (samples/)
- 3 вкладки: Чат, Управление базой, О системе
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

import gradio as gr
import httpx

from app.config import settings

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BACKEND_URL = "http://rag-backend:8000"
HTTP_TIMEOUT = 180.0


# ---------------------------------------------------------------------------
# HTTP-обёртки (sync, но через httpx с таймаутом)
# ---------------------------------------------------------------------------

def _post_json(endpoint: str, payload: dict, timeout: float = HTTP_TIMEOUT) -> dict:
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{BACKEND_URL}{endpoint}", json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        logger.exception("POST %s failed", endpoint)
        return {"error": str(exc)}


def _get_json(endpoint: str, timeout: float = 30.0) -> Any:
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{BACKEND_URL}{endpoint}")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        logger.exception("GET %s failed", endpoint)
        return {"error": str(exc)}


def _post_file(endpoint: str, file_path: str, timeout: float = 300.0) -> dict:
    try:
        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f, "application/octet-stream")}
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(f"{BACKEND_URL}{endpoint}", files=files)
                resp.raise_for_status()
                return resp.json()
    except httpx.HTTPError as exc:
        logger.exception("POST %s (file) failed", endpoint)
        return {"error": str(exc)}


def _delete(endpoint: str, timeout: float = 30.0) -> dict:
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.delete(f"{BACKEND_URL}{endpoint}")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        logger.exception("DELETE %s failed", endpoint)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Чат
# ---------------------------------------------------------------------------

def _format_sources(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return ""
    lines = ["\n\n---\n**📚 Источники ответа:**\n"]
    for s in sources:
        section = f" · *{s.get('section_title', '')}*" if s.get("section_title") else ""
        lines.append(
            f"**[{s['ref']}]** `{s.get('doc_name', '?')}`"
            f" — стр. {s.get('page_number', '?')}{section}  \n"
            f"   ▸ rerank: `{s.get('rerank_score', 0):.3f}` · vector: `{s.get('score', 0):.3f}`  \n"
            f"   ▸ фрагмент: {s.get('snippet', '')}\n"
        )
    return "".join(lines)


def _format_latency(result: dict) -> str:
    """v4: метрика TTFT под ответом."""
    ttft = result.get("ttft_ms", 0)
    latency = result.get("latency_ms", 0)
    if ttft == 0 and latency == 0:
        return ""
    return (
        f"\n\n⏱️ *TTFT: {ttft} мс · Полное время: {latency} мс*"
    )


def chat_respond(message: str, history: List[dict]) -> tuple:
    """Обработчик ввода: Gradio 5 с type='messages' (role/content)."""
    if not message or not message.strip():
        return "", history

    result = _post_json("/chat", {"question": message})
    if "error" in result:
        full_response = f"⚠️ Ошибка: {result['error']}"
    else:
        answer_text = result.get("answer", "")
        sources_md = _format_sources(result.get("sources", []))
        latency_md = _format_latency(result)
        full_response = answer_text + sources_md + latency_md

    history = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": full_response},
    ]
    return "", history


def reset_chat() -> tuple:
    """Сброс контекста."""
    _post_json("/reset", {}, timeout=10.0)
    return [], ""


# ---------------------------------------------------------------------------
# Управление документами
# ---------------------------------------------------------------------------

def upload_and_ingest(file) -> tuple:
    if file is None:
        return "⚠️ Файл не выбран.", _render_docs_table()

    file_path = file.name if hasattr(file, "name") else str(file)
    result = _post_file("/ingest", file_path)
    if "error" in result:
        return f"❌ Ошибка: {result['error']}", _render_docs_table()

    skipped = result.get("skipped_duplicates", 0)
    skipped_msg = f" (пропущено дубликатов: {skipped})" if skipped else ""

    msg = (
        f"✅ **{result['doc_name']}** проиндексирован:{skipped_msg}\n"
        f"• страниц: **{result['pages']}**\n"
        f"• parent-чанков: **{result['parents']}**\n"
        f"• child-чанков: **{result['chunks']}**\n"
        f"• всего точек в Qdrant: **{result['total_nodes']}**\n"
        f"• snapshot: `{result.get('snapshot', '—')}`\n"
        f"• `doc_id`: `{result['doc_id']}`"
    )
    return msg, _render_docs_table()


def ingest_samples() -> tuple:
    """v4: загрузка демо-документов из samples/ одним кликом."""
    result = _post_json("/samples/ingest", {})
    if isinstance(result, dict) and "error" in result:
        return f"❌ Ошибка: {result['error']}", _render_docs_table()
    return (
        f"✅ Загружено из samples/: **{result.get('ingested', 0)}** файлов. "
        f"Ошибок: **{result.get('errors', 0)}**",
        _render_docs_table(),
    )


def _render_docs_table() -> str:
    docs = _get_json("/documents")
    if isinstance(docs, dict) and "error" in docs:
        return f"⚠️ Не удалось получить список: {docs['error']}"

    if not docs:
        return (
            "ℹ️ **База знаний пуста.**\n\n"
            "Загрузите документ выше или нажмите «📥 Загрузить демо-документы»."
        )

    lines = [
        "| # | Документ | Версия | Страниц | Чанков | Разделы | doc_id |",
        "|---|----------|--------|---------|--------|---------|--------|",
    ]
    for i, d in enumerate(docs, start=1):
        sections = ", ".join(d.get("sections", []))[:60] or "—"
        lines.append(
            f"| {i} | {d['doc_name']} | {d.get('doc_version', 'v1.0')} | "
            f"{d.get('pages', '?')} | {d.get('chunks', '?')} | "
            f"{sections} | `{d['doc_id']}` |"
        )
    return "\n".join(lines)


def remove_doc(doc_id: str) -> tuple:
    doc_id = (doc_id or "").strip()
    if not doc_id:
        return "⚠️ Введите doc_id документа для удаления.", _render_docs_table()

    result = _delete(f"/documents/{doc_id}")
    if isinstance(result, dict) and "error" in result:
        return f"❌ Ошибка: {result['error']}", _render_docs_table()

    return f"✅ Документ `{doc_id}` удалён.", _render_docs_table()


def refresh_docs() -> str:
    return _render_docs_table()


def reindex_all() -> str:
    result = _post_json("/documents/reindex", {})
    if isinstance(result, dict) and "error" in result:
        return f"❌ Ошибка: {result['error']}"
    return (
        f"✅ Переиндексировано: **{result.get('reindexed', 0)}** файлов.\n"
        f"❌ Ошибок: **{result.get('errors', 0)}**"
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_health() -> dict:
    return _get_json("/health", timeout=5.0)


def get_stats() -> dict:
    return _get_json("/stats", timeout=10.0)


# ---------------------------------------------------------------------------
# Сборка UI
# ---------------------------------------------------------------------------

CSS = """
footer {display: none !important;}
.gradio-container {max-width: 1200px !important; margin: auto !important;}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Корпоративный ИИ-Ассистент",
        theme=gr.themes.Soft(),
        css=CSS,
    ) as demo:

        gr.Markdown(
            "# 🏢 Корпоративный ИИ-Ассистент (On-Premise RAG) — v4\n"
            "Полностью локальная система поиска по корпоративной базе знаний. "
            "Поддержка русского и казахского. vLLM + bge-m3 + reranker + PDR + "
            "hybrid search + **async streaming + TTFT** + **локальный LLM-judge**."
        )

        # -------------------------------------------------------------------
        # Вкладка 1: Чат
        # -------------------------------------------------------------------
        with gr.Tab("💬 Чат"):
            chatbot = gr.Chatbot(
                height=520,
                type="messages",
                show_label=False,
                avatar_images=(None, "🤖"),
                show_copy_button=True,
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Задайте вопрос на русском или казахском...",
                    scale=8,
                    show_label=False,
                    lines=2,
                    max_lines=4,
                )
                send_btn = gr.Button("Отправить", variant="primary", scale=1)
            with gr.Row():
                clear_btn = gr.Button("🗑️ Сбросить контекст", variant="stop", size="sm")
                hint_md = gr.Markdown(
                    "💡 *Чем точнее вопрос — тем релевантнее ответ. "
                    "Цитаты [1], [2] в ответе ссылаются на источники ниже. "
                    "Под ответом показывается TTFT (time to first token).",
                    scale=8,
                )

            gr.Examples(
                examples=[
                    "Сколько календарных дней длится ежегодный отпуск?",
                    "За сколько дней нужно подавать заявление на отпуск?",
                    "Какая минимальная длина пароля?",
                    "Можно ли использовать публичный Wi-Fi для работы?",
                    "Жылдық демалыс қанша күнге беріледі?",
                    "Демалысқа өтінішті қанша күн бұрын беру керек?",
                ],
                inputs=msg_input,
            )

            send_btn.click(chat_respond, [msg_input, chatbot], [msg_input, chatbot])
            msg_input.submit(chat_respond, [msg_input, chatbot], [msg_input, chatbot])
            clear_btn.click(reset_chat, None, [chatbot, msg_input])

        # -------------------------------------------------------------------
        # Вкладка 2: Управление базой знаний
        # -------------------------------------------------------------------
        with gr.Tab("📁 Управление базой знаний"):
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="Загрузить документ",
                        file_types=[".pdf", ".docx", ".txt"],
                        file_count="single",
                    )
                with gr.Column(scale=1):
                    ingest_btn = gr.Button("📥 Индексировать", variant="primary")
                    samples_btn = gr.Button("📂 Загрузить демо-документы", size="sm")
                    reindex_btn = gr.Button("🔄 Переиндексировать все", size="sm")
                    ingest_status = gr.Markdown("")

            gr.Markdown("### Текущая база знаний")
            docs_table = gr.Markdown(_render_docs_table())
            refresh_btn = gr.Button("🔄 Обновить список", size="sm")
            refresh_btn.click(refresh_docs, None, docs_table)

            gr.Markdown("---\n### Удаление документа")
            with gr.Row():
                doc_id_input = gr.Textbox(
                    label="doc_id",
                    placeholder="вставьте doc_id из таблицы выше",
                    scale=4,
                )
                del_btn = gr.Button("❌ Удалить", variant="stop", scale=1)
            del_status = gr.Markdown("")

            ingest_btn.click(upload_and_ingest, [file_input], [ingest_status, docs_table])
            samples_btn.click(ingest_samples, None, [ingest_status, docs_table])
            reindex_btn.click(reindex_all, None, [ingest_status])
            reindex_btn.click(refresh_docs, None, [docs_table])
            del_btn.click(remove_doc, [doc_id_input], [del_status, docs_table])

        # -------------------------------------------------------------------
        # Вкладка 3: О системе
        # -------------------------------------------------------------------
        with gr.Tab("ℹ️ О системе"):
            gr.Markdown(
                f"""
## Технологический стек (v4)

| Компонент | Технология |
|-----------|------------|
| LLM | `{settings.LLM_MODEL}` через **vLLM** (OpenAI-compatible, **async streaming**) |
| Embeddings | `{settings.EMBED_MODEL}` через TEI / sentence-transformers |
| Reranker | `{settings.RERANKER_MODEL}` (обязательно, normalize=True) |
| Vector DB | Qdrant (named vectors: dense + sparse reserved) |
| Sparse retrieval | BM25 (rank-bm25) + RRF (Reciprocal Rank Fusion) |
| Framework | LlamaIndex + кастомный async retrieval layer |
| Backend | FastAPI + Uvicorn (lifespan context manager) |
| UI | Gradio 5.x (type="messages") |
| Парсинг | PyMuPDF + python-docx (+ опц. Docling) |
| Evaluation | **Локальный LLM-as-judge** (без RAGAS/OpenAI) |

## Ключевые принципы v4

- **🔒 Полная локальность.** Никаких внешних API. vLLM, TEI, Qdrant — всё локально.
- **📖 Строгое цитирование.** Каждый ответ сопровождается ссылками на документ, страницу и раздел.
- **🛡️ Защита от галлюцинаций.** Reranker threshold `{settings.MIN_RERANK_SCORE}` + строгий промпт.
- **🌍 Двуязычие.** Автоопределение RU/KZ + few-shot для казахского.
- **🔍 Hybrid Search.** Dense (bge-m3) + Sparse (BM25) → RRF (k=60).
- **📚 Parent Document Retriever (PDR).** Маленькие чанки для поиска, большие parent — для генерации.
- **⚡ Async Streaming + TTFT (НОВОЕ v4).** Streaming-ответ с измерением time to first token.
- **🧪 Локальный LLM-judge (НОВОЕ v4).** Groundedness считается локальным LLM — без RAGAS/OpenAI.
- **🚫 Дедупликация по chunk_hash.** При повторной загрузке того же файла дубликаты пропускаются.

## Параметры (калибруемые)

| Параметр | Значение | Назначение |
|----------|----------|------------|
| `CHUNK_SIZE` | {settings.CHUNK_SIZE} | Размер child-чанка |
| `PARENT_CHUNK_SIZE` | {settings.PARENT_CHUNK_SIZE} | Размер parent-чанка |
| `TOP_K` | {settings.TOP_K} | Кандидатов из Qdrant |
| `RERANK_TOP_K` | {settings.RERANK_TOP_K} | Финальное число после reranker |
| `MIN_RERANK_SCORE` | {settings.MIN_RERANK_SCORE} | Порог reranker |
| `LLM_TEMPERATURE` | {settings.LLM_TEMPERATURE} | Температура LLM |
| `LLM_STREAM` | {settings.LLM_STREAM} | Streaming ответа (для TTFT) |

## REST API

Доступен на `http://localhost:8000`:
- `POST /query` / `POST /chat` — async вопрос → ответ с цитатами + TTFT
- `POST /ingest` — загрузка документа
- `POST /samples/ingest` — загрузка демо-документов (v4)
- `GET /documents` — список документов
- `DELETE /documents/{{doc_id}}` — удаление
- `POST /documents/reindex` — переиндексация всех файлов
- `GET /health` — проверка здоровья
- `GET /stats` — статистика индекса
- `POST /reset` — сброс состояния чата
- `GET /evaluation/testset` — получить тестовый набор
- `POST /evaluation/run` — async оценка (локальный LLM-judge)
- `POST /bm25/rebuild` — перестроение BM25 индекса
                """
            )

            with gr.Row():
                health_btn = gr.Button("🩺 Проверить здоровье")
                stats_btn = gr.Button("📊 Статистика")

            health_output = gr.JSON(label="Health")
            stats_output = gr.JSON(label="Stats")

            health_btn.click(check_health, None, health_output)
            stats_btn.click(get_stats, None, stats_output)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
