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

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from atkv import guard, logging as jlog
from atkv.generate.base import GenerationProvider
from atkv.generate.extractive import ExtractiveProvider
from atkv.generate.ollama import OllamaProvider
from atkv.models import QueryRequest, QueryResponse
from atkv.retrieve.pipeline import Coverage, RetrievalPipeline

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


def _coverage_notice(gaps: list[Coverage], as_of, lang: str) -> str | None:
    """Say plainly when the index cannot cover the date that was asked about.

    The failure this prevents: ask about 2027 and every collective-agreement
    chunk falls out of the filter while all the law chunks survive, because
    statutes have no end date. The service then answers from labour law alone
    and looks entirely normal. A user asking for a 2027 salary gets no figure
    and no reason -- which reads as "the system is bad", not "the index stops
    at 2026".

    Nothing out-of-window is ever substituted. Quoting the 2026 table in 2027
    as though it were current would be a wrong number with a real citation
    attached, which is the one outcome this project exists to avoid.
    """
    if not gaps:
        return None
    names = {"kv": ("Kollektivvertrag", "collective agreement"),
             "law": ("Gesetzestexte", "statutes")}
    parts = []
    for g in gaps:
        de, en = names.get(g.source_type, (g.source_type, g.source_type))
        until = g.latest.isoformat() if g.latest else "-"
        parts.append((de, en, g.earliest.isoformat(), until))
    if lang == "de":
        return (
            "Hinweis: Der Index deckt den Stichtag " + as_of.isoformat() + " nicht "
            "vollständig ab. " + "; ".join(
                f"{de}: nur {frm} bis {to}" for de, _, frm, to in parts) + ". "
            "Die Antwort stützt sich daher nur auf die übrigen Quellen. Es werden "
            "bewusst KEINE Werte aus einem abgelaufenen Dokument als aktuell ausgegeben."
        )
    return (
        "Note: the index does not fully cover " + as_of.isoformat() + ". " +
        "; ".join(f"{en}: only {frm} to {to}" for _, en, frm, to in parts) + ". "
        "The answer therefore draws only on the remaining sources. Figures from an "
        "expired document are deliberately NOT presented as current."
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
    if p is None:
        return {"status": "starting", "chunks": 0}
    from datetime import date as _date
    today = _date.today()
    cov = p.coverage()
    stale = [g.source_type for g in p.gaps_on(today)]
    return {
        # Degraded, not ok: the service answers, but a source type cannot cover
        # today, so an operator should see it here rather than learn it from a
        # user's puzzling answer.
        "status": "degraded" if stale else "ok",
        "chunks": len(p.chunks),
        "embedding_model": p.dense.model_name,
        "rerank_enabled": ENABLE_RERANK,
        "generation_provider": _provider().name,
        "today": today.isoformat(),
        "coverage": {st: {"from": c.earliest.isoformat(),
                          "to": c.latest.isoformat() if c.latest else None,
                          "chunks": c.n_chunks}
                     for st, c in sorted(cov.items())},
        "stale_sources": stale,
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

    as_of = req.as_of()
    notice = _coverage_notice(p.gaps_on(as_of), as_of, req.lang)
    result = p.search(req.question, as_of=as_of, k=req.k, lang=req.lang,
                      tenant_id=req.tenant_id, rerank=req.rerank)
    chunks = [r.chunk for r in result.chunks]

    provider = _provider()
    gen = provider.generate(req.question, chunks, req.lang)

    # Everything needed to replay this answer, with the component scores kept
    # apart so "which retriever dragged that in" is answerable from the log.
    jlog.log(
        log, logging.INFO, "query", trace_id=trace_id,
        question=req.question, lang=req.lang, valid_year=req.valid_year,
        as_of=str(as_of), tenant_id=req.tenant_id, coverage_gap=bool(notice),
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
        as_of=as_of,
        notice=notice,
    )


@app.post("/query/stream")
def query_stream(req: QueryRequest) -> StreamingResponse:
    """Server-sent events: citations first, then the answer token by token.

    CITATIONS ARE SENT BEFORE THE ANSWER, deliberately. They are known the
    moment retrieval finishes, seconds before the first generated token, so
    the reader can see WHICH paragraphs are being answered from while the
    answer is still being written. On a system whose whole claim is provenance,
    the sources should not be the last thing to arrive.

    Event types:
        refusal   the guard stopped it; no retrieval happened
        meta      trace_id and citations, emitted as soon as retrieval is done
        token     one piece of the answer
        done      final latency, and the provider that served it
        error     generation failed mid-stream (see below)

    Note on errors: once a 200 and the first byte have gone out, HTTP offers no
    way to retract them. A failure halfway through cannot become a 500, so it
    is sent as an `error` event and the client must handle it. A stream that
    simply stops looks identical to a short answer.
    """
    trace_id = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

    def events():
        verdict = guard.check(req.question)
        if verdict.refuse:
            jlog.log(log, logging.INFO, "refused", trace_id=trace_id,
                     rule=verdict.rule, reason=verdict.reason, lang=req.lang, stream=True)
            yield sse("refusal", {"trace_id": trace_id, "answer": verdict.message(req.lang),
                                  "reason": verdict.reason})
            return

        p: RetrievalPipeline = STATE.get("pipeline")
        if p is None:
            yield sse("error", {"trace_id": trace_id, "error": "index not ready"})
            return

        as_of = req.as_of()
        notice = _coverage_notice(p.gaps_on(as_of), as_of, req.lang)
        result = p.search(req.question, as_of=as_of, k=req.k, lang=req.lang,
                          tenant_id=req.tenant_id, rerank=req.rerank)
        chunks = [r.chunk for r in result.chunks]
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)

        # The notice rides in meta, which arrives BEFORE any answer text, so a
        # caveat about coverage is on screen before the figure it qualifies.
        yield sse("meta", {
            "trace_id": trace_id,
            "retrieval_ms": retrieval_ms,
            "as_of": as_of.isoformat(),
            "notice": notice,
            "citations": [c.to_citation().model_dump() for c in chunks],
        })

        provider = _provider()
        pieces: list[str] = []
        try:
            for piece in provider.stream(req.question, chunks, req.lang):
                pieces.append(piece)
                yield sse("token", {"text": piece})
        except Exception as e:
            jlog.log(log, logging.ERROR, "stream failed", trace_id=trace_id, error=str(e))
            yield sse("error", {"trace_id": trace_id, "error": str(e)})
            return

        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        jlog.log(
            log, logging.INFO, "query", trace_id=trace_id, stream=True,
            question=req.question, lang=req.lang, valid_year=req.valid_year,
            as_of=str(as_of), tenant_id=req.tenant_id, coverage_gap=bool(notice),
            rerank=result.used_rerank, translated=result.used_translation,
            pool_size=result.pool_size, provider=provider.name,
            retrieved=[
                {"chunk_id": r.chunk.chunk_id, "doc_id": r.chunk.doc_id,
                 "section_ref": r.chunk.section_ref, "lang": r.chunk.lang,
                 "dense": r.dense_score, "lexical": r.lexical_score, "fused": r.fused_score}
                for r in result.chunks
            ],
            retrieval_ms=retrieval_ms, latency_ms=total_ms,
            answer_chars=len("".join(pieces)),
        )
        yield sse("done", {"trace_id": trace_id, "latency_ms": total_ms,
                           "provider": provider.name})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Without this a reverse proxy will happily buffer the whole response
        # and deliver it in one lump, which is exactly what streaming exists to
        # avoid -- and it fails silently, looking like a slow server.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )
