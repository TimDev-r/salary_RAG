"""FastAPI service.

Request flow, and the order is deliberate:

    guard -> filter -> retrieve -> generate -> cite

The guard runs FIRST. An advice-seeking question is refused before any index
is touched or any model is loaded: refusing is cheap, and a refusal that took
two seconds of retrieval is two seconds wasted on an answer we were never
going to give.

Generation is routed. If a model server is reachable it writes the answer; if
not, the extractive provider returns the top source passage verbatim with its
citation. That degradation is deliberate. A worse-worded answer that is
provably from a source document beats both an error page and a fluent
invention.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from atkv import guard, logging as jlog
from atkv.generate.base import GenerationProvider
from atkv.generate.extractive import ExtractiveProvider
from atkv.generate.ollama import OllamaProvider
from atkv.models import QueryRequest, QueryResponse
from atkv.retrieve.pipeline import RetrievalPipeline

ROOT = Path(os.environ.get("ATKV_ROOT", Path(__file__).resolve().parents[2]))
INDEX_DIR = ROOT / "data/index/serving"
ENABLE_RERANK = os.environ.get("ATKV_RERANK", "0") == "1"

log = logging.getLogger("atkv.api")
STATE: dict = {}


def _load_pipeline() -> RetrievalPipeline:
    reranker = translator = None
    if ENABLE_RERANK:
        from atkv.retrieve.rerank import Reranker
        reranker = Reranker()
    try:
        from atkv.retrieve.translate import QueryTranslator
        translator = QueryTranslator()
    except Exception as e:  # translation is an optimisation, not a dependency
        jlog.log(log, logging.WARNING, "translator unavailable", error=str(e))

    if (INDEX_DIR / "dense.faiss").exists():
        return RetrievalPipeline.load(INDEX_DIR, reranker=reranker, translator=translator)

    from atkv.ingest.build import build_corpus
    chunks = build_corpus(ROOT, fetch=True)
    p = RetrievalPipeline.build(chunks, reranker=reranker, translator=translator)
    p.save(INDEX_DIR)
    return p


@asynccontextmanager
async def lifespan(app: FastAPI):
    jlog.configure()
    t0 = time.perf_counter()
    STATE["pipeline"] = _load_pipeline()
    STATE["ollama"] = OllamaProvider()
    STATE["extractive"] = ExtractiveProvider()
    jlog.log(log, logging.INFO, "ready",
             chunks=len(STATE["pipeline"].chunks),
             startup_s=round(time.perf_counter() - t0, 1),
             rerank_enabled=ENABLE_RERANK)
    yield
    STATE.clear()


app = FastAPI(
    title="AT-KV Assistant",
    version="0.1.0",
    description="Retrieval over the Austrian IT collective agreement (IT-KV) and related labour law.",
    lifespan=lifespan,
)


def _provider() -> GenerationProvider:
    """Route to a model if one is reachable, else to the extractive provider.

    Checked per request, not at startup: a local model server can be running
    when the process starts and gone ten minutes later, and we would rather
    find out before committing an answer to a user than after.
    """
    ollama: OllamaProvider = STATE["ollama"]
    return ollama if ollama.available() else STATE["extractive"]


@app.get("/healthz")
def healthz() -> dict:
    p: RetrievalPipeline = STATE.get("pipeline")
    return {
        "status": "ok" if p else "starting",
        "chunks": len(p.chunks) if p else 0,
        "embedding_model": p.dense.model_name if p else None,
        "rerank_enabled": ENABLE_RERANK,
        "generation_provider": _provider().name if p else None,
    }


@app.post("/ingest")
def ingest(fetch: bool = True) -> dict:
    """Rebuild the index from the manifest."""
    from atkv.ingest.build import build_corpus

    trace_id = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    chunks = build_corpus(ROOT, fetch=fetch)
    old = STATE.get("pipeline")
    p = RetrievalPipeline.build(
        chunks,
        reranker=getattr(old, "reranker", None),
        translator=getattr(old, "translator", None),
    )
    p.save(INDEX_DIR)
    STATE["pipeline"] = p
    took = round(time.perf_counter() - t0, 1)
    jlog.log(log, logging.INFO, "ingest complete", trace_id=trace_id,
             chunks=len(chunks), seconds=took)
    return {"trace_id": trace_id, "chunks": len(chunks), "seconds": took,
            "documents": sorted({c.doc_id for c in chunks})}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    trace_id = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()

    verdict = guard.check(req.question)
    if verdict.refuse:
        jlog.log(log, logging.INFO, "refused", trace_id=trace_id,
                 rule=verdict.rule, reason=verdict.reason, lang=req.lang)
        return QueryResponse(answer=verdict.message(req.lang), citations=[],
                             trace_id=trace_id, refused=True,
                             refusal_reason=verdict.reason)

    p: RetrievalPipeline = STATE.get("pipeline")
    if p is None:
        raise HTTPException(503, "index not ready")

    result = p.search(req.question, as_of=req.as_of(), k=req.k, lang=req.lang,
                      tenant_id=req.tenant_id, rerank=req.rerank)
    chunks = [r.chunk for r in result.chunks]

    provider = _provider()
    gen = provider.generate(req.question, chunks, req.lang)

    # Everything needed to replay this answer, with the component scores kept
    # apart so "which retriever dragged that in" is answerable from the log.
    jlog.log(
        log, logging.INFO, "query", trace_id=trace_id,
        question=req.question, lang=req.lang, valid_year=req.valid_year,
        as_of=str(req.as_of()), tenant_id=req.tenant_id,
        rerank=result.used_rerank, translated=result.used_translation,
        pool_size=result.pool_size, provider=gen.provider, model=gen.model,
        retrieved=[
            {"chunk_id": r.chunk.chunk_id, "doc_id": r.chunk.doc_id,
             "section_ref": r.chunk.section_ref, "lang": r.chunk.lang,
             "dense": r.dense_score, "lexical": r.lexical_score, "fused": r.fused_score}
            for r in result.chunks
        ],
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )

    return QueryResponse(
        answer=gen.text,
        citations=[c.to_citation() for c in chunks],
        trace_id=trace_id,
        refused=False,
    )
