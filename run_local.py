"""Локальный прогон v5 на Ollama-профиле: индексация samples + smoke-запросы."""
import asyncio
from pathlib import Path

from app.config import settings
from app.ingestion import ensure_collection, ingest_document, list_documents
from app.rag_engine import get_engine


async def main() -> None:
    print(f"LLM base_url = {settings.vllm_base_url} · model = {settings.LLM_MODEL}")
    print(f"Embed = {settings.EMBED_MODEL} (dim {settings.EMBED_DIM}) · Reranker = {settings.RERANKER_MODEL}")
    print(f"Qdrant embedded = {settings.QDRANT_PATH or '(server)'}")
    print("-" * 60)

    ensure_collection()
    for f in sorted(Path("samples").glob("*.txt")):
        r = ingest_document(f)
        print(f"  indexed {f.name}: {r.get('chunks_indexed', r)}")
    print(f"документов в базе: {len(list_documents())}")
    print("-" * 60)

    eng = get_engine()
    for q in [
        "За сколько дней нужно подавать заявление на отпуск?",
        "Демалысқа өтінішті қанша күн бұрын беру керек?",
        "Сколько стоит парковка для сотрудников?",
    ]:
        res = await eng.query(q)
        print(f"\nВ: {q}")
        print(f"О: {res['answer'][:180]}")
        srcs = res.get("sources", [])
        if srcs:
            print("   источники:", [f"{s['doc_name']} стр.{s.get('page_number')}" for s in srcs[:3]])


if __name__ == "__main__":
    asyncio.run(main())
