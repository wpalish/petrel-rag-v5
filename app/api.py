"""
app/api.py — FastAPI backend (v5: auth + metrics + A/B + calibrate).

Изменения v5:
- BasicAuthMiddleware (опционально через AUTH_ENABLED)
- /metrics endpoint (Prometheus)
- /ab/test endpoint — A/B тестирование промптов на одном вопросе
- /calibrate endpoint — подбор MIN_RERANK_SCORE по testset
- /variants endpoint — список доступных prompt variants
- /query и /chat принимают optional variant параметр
- Все endpoints оборачиваются в RequestTimer для метрик
"""

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field

from app.config import settings
from app.ingestion import (
    delete_document,
    ensure_collection,
    ingest_document,
    list_documents,
)
from app.rag_engine import get_engine
from app.utils import generate_doc_id, safe_filename

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic-схемы
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: Optional[int] = Field(None, ge=1, le=20)
    variant: Optional[str] = Field(None, description="A/B variant: strict/balanced/concise")


class SourceItem(BaseModel):
    ref: int
    doc_id: str = ""
    doc_name: str
    doc_version: str = "v1.0"
    page_number: int = 0
    section_title: str = ""
    paragraph_index: int = 0
    snippet: str = ""
    score: float = 0.0
    rerank_score: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    citations: List[SourceItem] = Field(default_factory=list, alias="sources")
    sources: List[SourceItem] = Field(default_factory=list)
    language: str
    retrieval_count: int = 0
    has_answer: bool = True
    no_answer: bool = False
    latency_ms: int = 0
    ttft_ms: int = 0
    variant: str = "strict"

    model_config = {"populate_by_name": True}


class IngestResponse(BaseModel):
    doc_id: str
    doc_name: str
    pages: int
    parents: int
    chunks: int
    total_nodes: int
    snapshot: Optional[str] = None
    skipped_duplicates: int = 0


class DocumentInfo(BaseModel):
    doc_id: str
    doc_name: str
    doc_version: str
    chunks: int
    pages: int
    sections: List[str] = []


class DeleteResponse(BaseModel):
    deleted: str
    success: bool = True


class ABTestRequest(BaseModel):
    """Запрос на A/B тест: один вопрос, все варианты промптов."""
    question: str = Field(..., min_length=1, max_length=4000)


class ABTestResponse(BaseModel):
    question: str
    results: dict  # {variant: QueryResponse-like dict}


class CalibrateRequest(BaseModel):
    """Запрос на калибровку MIN_RERANK_SCORE."""
    limit: Optional[int] = Field(None, ge=1, le=50, description="Сколько вопросов из testset использовать")


# ---------------------------------------------------------------------------
# Factory с lifespan (v4) + auth + metrics (v5)
# ---------------------------------------------------------------------------
def build_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """v4: lifespan. v5: + init metrics + bm25."""
        try:
            ensure_collection()
            logger.info("Qdrant collection ensured: OK")
        except Exception as exc:
            logger.warning("Qdrant not ready on startup: %s", exc)

        # Готовим BM25 индекс
        try:
            from app.bm25_index import get_bm25_index
            from app.ingestion import get_qdrant_client
            bm25 = get_bm25_index()
            bm25.rebuild_from_qdrant(get_qdrant_client(), settings.QDRANT_COLLECTION)
            logger.info("BM25 index ready on startup.")
            # v5: обновляем gauge
            from app.metrics import get_metrics
            get_metrics().set_bm25_chunks(len(bm25._records))
        except Exception as exc:
            logger.warning("BM25 index not built on startup: %s", exc)

        yield

        logger.info("Shutting down RAG backend.")

    app = FastAPI(
        title="Корпоративный ИИ-Ассистент (On-Premise RAG) — v5",
        description="Локальная RAG-система. RU/KZ. "
                    "Hybrid (dense + BM25 + RRF) + Reranker + PDR + async streaming + "
                    "локальный LLM-judge + A/B prompts + Auth + Prometheus monitoring.",
        version="5.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # v5: BasicAuthMiddleware (если AUTH_ENABLED=True)
    if settings.AUTH_ENABLED:
        from app.auth import BasicAuthMiddleware, get_public_paths
        app.add_middleware(BasicAuthMiddleware, public_paths=get_public_paths())
        logger.info("Auth middleware enabled (user=%s)", settings.AUTH_USER)

    # -------------------------------------------------------------------
    # Endpoints
    # -------------------------------------------------------------------

    @app.get("/health")
    async def health():
        """Базовая проверка: API жив, модель, коллекция, BM25, version."""
        bm25_status = {"ready": False, "chunks": 0}
        try:
            from app.bm25_index import get_bm25_index
            bm25 = get_bm25_index()
            bm25_status = bm25.stats()
        except Exception as exc:
            logger.warning("BM25 stats failed: %s", exc)

        llm_status = {"reachable": False}
        try:
            from app.llm import health as llm_health
            llm_status = await llm_health()
        except Exception as exc:
            logger.warning("LLM health failed: %s", exc)

        # v5: обновляем Prometheus gauges
        try:
            from app.metrics import get_metrics
            m = get_metrics()
            m.set_llm_health(
                reachable=llm_status.get("reachable", False),
                model_loaded=llm_status.get("model_loaded", False),
            )
            m.set_bm25_chunks(bm25_status.get("chunks", 0))
        except Exception as exc:
            logger.debug("metrics update failed: %s", exc)

        return {
            "status": "ok",
            "version": "5.0.0",
            "collection": settings.QDRANT_COLLECTION,
            "llm": llm_status,
            "llm_model": settings.LLM_MODEL,
            "llm_stream": settings.LLM_STREAM,
            "embed_model": settings.EMBED_MODEL,
            "reranker_model": settings.RERANKER_MODEL,
            "use_reranker": settings.USE_RERANKER,
            "use_pdr": settings.USE_PDR,
            "use_hybrid": settings.USE_HYBRID,
            "use_chunk_hash_dedup": settings.USE_CHUNK_HASH_DEDUP,
            "bm25": bm25_status,
            "min_rerank_score": settings.MIN_RERANK_SCORE,
            "auth_enabled": settings.AUTH_ENABLED,
            "metrics_enabled": settings.METRICS_ENABLED,
            "prompt_variant": settings.PROMPT_VARIANT,
        }

    @app.get("/stats")
    async def stats():
        """Статистика по индексу."""
        try:
            docs = list_documents()
            total_chunks = sum(d["chunks"] for d in docs)
            # v5: обновляем gauge
            try:
                from app.metrics import get_metrics
                get_metrics().set_documents_total(len(docs))
                get_metrics().set_chunks_total(total_chunks)
            except Exception:
                pass
            return {
                "documents": len(docs),
                "total_chunks": total_chunks,
                "items": docs,
            }
        except Exception as exc:
            logger.exception("stats failed")
            raise HTTPException(status_code=500, detail=str(exc))

    async def _do_query(req: QueryRequest, endpoint: str = "/query") -> QueryResponse:
        """Общий обработчик для /query и /chat. v5: RequestTimer + variant."""
        from app.metrics import RequestTimer
        from app.utils import detect_language

        # Определяем язык для метрик
        lang = detect_language(req.question)
        timer = RequestTimer(endpoint, lang)
        try:
            with timer:
                engine = get_engine()
                result = await engine.query(req.question, variant=req.variant)
                return QueryResponse(**result)
        except Exception as exc:
            logger.exception("query failed")
            raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {exc}")

    @app.post("/query", response_model=QueryResponse)
    async def query_endpoint(req: QueryRequest):
        """v5: async вопрос → ответ + TTFT + variant."""
        return await _do_query(req, "/query")

    @app.post("/chat", response_model=QueryResponse)
    async def chat_endpoint(req: QueryRequest):
        """Alias для /query (petrel naming convention)."""
        return await _do_query(req, "/chat")

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest_endpoint(file: UploadFile = File(...)):
        """Загрузка и индексация документа (PDF/DOCX/TXT)."""
        if not file.filename:
            raise HTTPException(status_code=400, detail="Имя файла не указано")

        safe_name = safe_filename(file.filename)
        ext = Path(safe_name).suffix.lower()
        if ext not in {".pdf", ".docx", ".txt"}:
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый формат: {ext}. Допустимы: .pdf, .docx, .txt",
            )

        dest = settings.docs_path / safe_name
        try:
            with dest.open("wb") as f:
                shutil.copyfileobj(file.file, f)
        finally:
            await file.close()

        logger.info("Saved uploaded file: %s", dest)

        try:
            result = ingest_document(dest)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.exception("ingest failed")
            raise HTTPException(status_code=500, detail=f"Ошибка индексации: {exc}")

        return IngestResponse(**result)

    @app.get("/documents", response_model=List[DocumentInfo])
    async def documents_endpoint():
        """Список всех проиндексированных документов."""
        try:
            docs = list_documents()
            return [DocumentInfo(**d) for d in docs]
        except Exception as exc:
            logger.exception("documents failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.delete("/documents/{doc_id}", response_model=DeleteResponse)
    async def delete_endpoint(
        doc_id: str,
        doc_version: Optional[str] = Query(default=None),
    ):
        """Удаление документа из индекса."""
        try:
            delete_document(doc_id, doc_version)
        except Exception as exc:
            logger.exception("delete failed")
            raise HTTPException(status_code=500, detail=f"Ошибка удаления: {exc}")

        for f in settings.docs_path.iterdir():
            if f.is_file():
                try:
                    if generate_doc_id(f) == doc_id:
                        f.unlink()
                        logger.info("Removed local file: %s", f.name)
                except OSError as exc:
                    logger.warning("Could not remove file %s: %s", f, exc)

        return DeleteResponse(deleted=doc_id)

    @app.post("/documents/reindex")
    async def reindex_endpoint():
        """Переиндексация всех файлов в data/docs."""
        try:
            ensure_collection()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Qdrant: {exc}")

        files = [
            f for f in settings.docs_path.iterdir()
            if f.is_file() and f.suffix.lower() in {".pdf", ".docx", ".txt"}
        ]
        if not files:
            return {"reindexed": 0, "message": "Нет файлов для индексации в data/docs"}

        results = []
        errors = []
        for f in files:
            try:
                r = ingest_document(f)
                results.append(r)
            except Exception as exc:
                errors.append({"file": f.name, "error": str(exc)})

        return {
            "reindexed": len(results),
            "errors": len(errors),
            "details": results,
            "error_details": errors,
        }

    @app.post("/reset")
    async def reset_endpoint():
        """Сброс состояния чата (stateless — заглушка)."""
        engine = get_engine()
        engine.reset_context()
        return {"status": "context reset"}

    # -------------------------------------------------------------------
    # Evaluation (v4: локальный LLM-judge, без RAGAS)
    # -------------------------------------------------------------------

    @app.get("/evaluation/testset")
    async def get_testset():
        from app.evaluation_runner import load_testset
        try:
            return load_testset()
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Тестовый набор не найден.")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/evaluation/run")
    async def run_evaluation_endpoint(limit: Optional[int] = Query(default=None)):
        """v4: async RAGAS-free оценка."""
        from app.evaluation_runner import run_evaluation
        try:
            results = await run_evaluation(limit=limit)
            return results
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            logger.exception("evaluation failed")
            raise HTTPException(status_code=500, detail=f"Ошибка оценки: {exc}")

    @app.post("/bm25/rebuild")
    async def rebuild_bm25_endpoint():
        try:
            from app.bm25_index import get_bm25_index
            from app.ingestion import get_qdrant_client
            bm25 = get_bm25_index()
            bm25.rebuild_from_qdrant(get_qdrant_client(), settings.QDRANT_COLLECTION)
            try:
                from app.metrics import get_metrics
                get_metrics().set_bm25_chunks(len(bm25._records))
            except Exception:
                pass
            return bm25.stats()
        except Exception as exc:
            logger.exception("BM25 rebuild failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/samples/ingest")
    async def ingest_samples():
        """v4: Загружает все файлы из samples/ в базу знаний."""
        samples_dir = Path(__file__).resolve().parent.parent / "samples"
        if not samples_dir.exists():
            raise HTTPException(status_code=404, detail=f"Папка samples/ не найдена: {samples_dir}")

        files = [
            f for f in samples_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".pdf", ".docx", ".txt"}
        ]
        if not files:
            return {"ingested": 0, "message": "В samples/ нет подходящих файлов"}

        results = []
        errors = []
        for src in files:
            dest = settings.docs_path / src.name
            try:
                shutil.copy2(src, dest)
                r = ingest_document(dest)
                results.append(r)
            except Exception as exc:
                errors.append({"file": src.name, "error": str(exc)})

        return {
            "ingested": len(results),
            "errors": len(errors),
            "details": results,
            "error_details": errors,
        }

    # -------------------------------------------------------------------
    # v5: A/B Prompt Testing
    # -------------------------------------------------------------------

    @app.get("/variants")
    async def list_variants():
        """v5: список доступных вариантов промптов для A/B тестирования."""
        from app.prompts import list_variants as _lv
        return _lv()

    @app.post("/ab/test", response_model=ABTestResponse)
    async def ab_test_endpoint(req: ABTestRequest):
        """
        v5: A/B тест — один вопрос прогоняется через все 3 варианта промпта.
        Возвращает ответы всех вариантов для сравнения.
        """
        from app.prompts import PROMPT_VARIANTS
        engine = get_engine()
        results = {}
        for variant in PROMPT_VARIANTS:
            try:
                r = await engine.query(req.question, variant=variant)
                results[variant] = r
            except Exception as exc:
                results[variant] = {"error": str(exc)}
        return ABTestResponse(question=req.question, results=results)

    # -------------------------------------------------------------------
    # v5: Calibration
    # -------------------------------------------------------------------

    @app.post("/calibrate")
    async def calibrate_endpoint(req: CalibrateRequest):
        """
        v5: Подбор MIN_RERANK_SCORE по тестовому набору.
        Запускает retrieval на N вопросах, замеряет распределение rerank_score
        и предлагает оптимальный порог (по квантилю).
        """
        from app.evaluation_runner import load_testset
        from app.ingestion import embed_texts, get_qdrant_client
        from app.rag_engine import get_engine

        try:
            testset = load_testset()
            if req.limit:
                testset = testset[:req.limit]
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Тестовый набор не найден.")

        engine = get_engine()

        # Собираем rerank_scores для всех вопросов
        scores_with_answer: list = []
        scores_no_answer: list = []

        for q in testset:
            expected_no = q.get("expected_no_answer", False)
            try:
                # Прямой retrieve без генерации
                retrieved = engine.retrieve(q["question"])
                for r in retrieved:
                    rs = r.get("rerank_score", 0)
                    if expected_no:
                        scores_no_answer.append(rs)
                    else:
                        scores_with_answer.append(rs)
            except Exception as exc:
                logger.warning("Calibration retrieve failed for %s: %s", q.get("id"), exc)

        # Анализ распределения
        import statistics
        analysis = {
            "questions_total": len(testset),
            "expected_with_answer": sum(1 for q in testset if not q.get("expected_no_answer", False)),
            "expected_no_answer": sum(1 for q in testset if q.get("expected_no_answer", False)),
            "scores_with_answer": {
                "count": len(scores_with_answer),
                "min": min(scores_with_answer) if scores_with_answer else 0,
                "max": max(scores_with_answer) if scores_with_answer else 0,
                "mean": round(statistics.mean(scores_with_answer), 4) if scores_with_answer else 0,
                "median": round(statistics.median(scores_with_answer), 4) if scores_with_answer else 0,
                "p25": round(_percentile(scores_with_answer, 25), 4) if scores_with_answer else 0,
                "p75": round(_percentile(scores_with_answer, 75), 4) if scores_with_answer else 0,
            },
            "scores_no_answer": {
                "count": len(scores_no_answer),
                "min": min(scores_no_answer) if scores_no_answer else 0,
                "max": max(scores_no_answer) if scores_no_answer else 0,
                "mean": round(statistics.mean(scores_no_answer), 4) if scores_no_answer else 0,
                "median": round(statistics.median(scores_no_answer), 4) if scores_no_answer else 0,
            },
        }

        # Рекомендация: 25-й перцентиль распределения with_answer
        # (отсекает 25% правильных ответов, но ловит большинство)
        if scores_with_answer:
            recommended = round(_percentile(scores_with_answer, 25), 4)
            analysis["recommended_threshold"] = recommended
            analysis["current_threshold"] = settings.MIN_RERANK_SCORE
            analysis["recommendation"] = (
                f"Рекомендуется MIN_RERANK_SCORE={recommended} "
                f"(p25 распределения релевантных). Текущее значение: {settings.MIN_RERANK_SCORE}."
            )
        else:
            analysis["recommended_threshold"] = settings.MIN_RERANK_SCORE
            analysis["recommendation"] = "Недостаточно данных для рекомендации."

        return analysis

    # -------------------------------------------------------------------
    # v5: Prometheus metrics
    # -------------------------------------------------------------------

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics_endpoint():
        """v5: Prometheus exposition endpoint."""
        if not settings.METRICS_ENABLED:
            return PlainTextResponse("# metrics disabled\n")
        try:
            from app.metrics import get_metrics
            return PlainTextResponse(
                get_metrics().export().decode("utf-8"),
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )
        except Exception as exc:
            logger.exception("metrics endpoint failed")
            raise HTTPException(status_code=500, detail=str(exc))

    return app


def _percentile(data: list, p: float) -> float:
    """Простой расчёт перцентиля (linear interpolation)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    if f == c:
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
