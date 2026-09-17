"""End-to-end retrieval: filter -> dense + lexical -> fuse -> rerank -> cap.

Everything here was chosen by measurement, and the numbers are recorded beside
each choice so a future reader can tell a decision from a default.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from atkv.models import Chunk, RetrievedChunk
from atkv.retrieve.dense import DenseIndex
from atkv.retrieve.embed import SMALL, Embedder
from atkv.retrieve.fuse import cap_sections, rrf, to_retrieved
from atkv.retrieve.lexical import LexicalIndex

# Measured in step 7: pool 50 beat pool 20 (ceiling 0.92 vs 0.96) and pool 100
# (R@10 0.96 vs 0.92 -- more candidates give the reranker more chances to be
# wrong). Not a round number picked for comfort.
POOL = 50


@dataclass
class Coverage:
    """What date range the index can actually answer for, per source type.

    The laws have no end date -- a paragraph in force stays in force until
    repealed -- so they cover any future date. The collective agreements do
    not: each is valid for one calendar year. Ask about 2027 today and every
    KV chunk drops out of the filter while all the law chunks remain, so the
    system answers from labour law alone and nothing says the agreement is
    missing. That is the gap this type exists to make visible.
    """

    source_type: str
    earliest: date
    latest: date | None   # None means "no end date" -- in force indefinitely
    n_chunks: int

    def covers(self, day: date) -> bool:
        return day >= self.earliest and (self.latest is None or day <= self.latest)


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    used_rerank: bool
    used_translation: bool
    pool_size: int


class RetrievalPipeline:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, dense: DenseIndex,
                 lexical: LexicalIndex, reranker=None, translator=None) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.dense = dense
        self.lexical = lexical
        self.reranker = reranker
        self.translator = translator

    @classmethod
    def build(cls, chunks: list[Chunk], model: str = SMALL,
              reranker=None, translator=None) -> "RetrievalPipeline":
        emb = Embedder(model)
        dense = DenseIndex(chunks, emb.encode_passages([c.text for c in chunks]), model)
        return cls(chunks, emb, dense, LexicalIndex(chunks), reranker, translator)

    def coverage(self) -> dict[str, Coverage]:
        out: dict[str, Coverage] = {}
        for st in {c.source_type for c in self.chunks}:
            group = [c for c in self.chunks if c.source_type == st]
            latest = None if any(c.valid_to is None for c in group) else max(
                c.valid_to for c in group if c.valid_to is not None)
            out[st] = Coverage(st, min(c.valid_from for c in group), latest, len(group))
        return out

    def gaps_on(self, day: date) -> list[Coverage]:
        """Source types the index cannot answer for on `day`."""
        return [c for c in self.coverage().values() if not c.covers(day)]

    def search(self, question: str, *, as_of: date | None = None, k: int = 8,
               lang: str = "de", tenant_id: str = "public",
               rerank: bool = False) -> RetrievalResult:
        # Query translation feeds the LEXICAL leg only. BM25 scores an English
        # query against a German passage at exactly zero -- no shared token --
        # so without this the correct chunk is not a weak candidate, it is not
        # a candidate. Measured: pool ceiling 0.96 -> 1.00.
        lex_query = question
        de_query = None
        translated = False
        if self.translator is not None and lang != "de":
            de_query = self.translator.translate(question)
            lex_query = f"{question} {de_query}"
            translated = True

        pool_k = POOL if (rerank and self.reranker) else max(k, 10)
        d = self.dense.search(self.embedder.encode_query(question), k=pool_k,
                              as_of=as_of, tenant_id=tenant_id)
        d_de = None
        if de_query:
            d_de = self.dense.search(self.embedder.encode_query(de_query), k=pool_k,
                                     as_of=as_of, tenant_id=tenant_id)
        l = self.lexical.search(lex_query, k=pool_k, as_of=as_of, tenant_id=tenant_id)

        # NOTE: retrieval is deliberately NOT restricted to the question's
        # language. AZG and UrlG exist only in German, so a language filter
        # would make every English question about them unanswerable.
        fused = rrf(d, l, k=pool_k, extra_dense=d_de)

        if rerank and self.reranker:
            reranked = self.reranker.rerank(question, [r.chunk for r in fused], k=pool_k)
            by_id = {r.chunk.chunk_id: r for r in fused}
            ordered = [by_id[h.chunk.chunk_id] for h in reranked if h.chunk.chunk_id in by_id]
        else:
            ordered = fused

        # Diversity cap LAST. Applied earlier it removes candidates the
        # reranker never sees, which cost a question at R@5 when measured.
        final = cap_sections(ordered, k)
        return RetrievalResult(to_retrieved(final), bool(rerank and self.reranker),
                               translated, len(fused))

    # NOTE: the English and German dense searches are fused by RANK in rrf(),
    # not merged here by score. See rrf()'s docstring for why the score merge
    # was wrong and what it cost.

    # -- persistence -------------------------------------------------------

    def save(self, dirpath: Path) -> None:
        self.dense.save(Path(dirpath))

    @classmethod
    def load(cls, dirpath: Path, reranker=None, translator=None) -> "RetrievalPipeline":
        """Load a prebuilt index.

        Stage 2 bakes the index into the container image at BUILD time, so a
        cold start only has to load a model for query embedding rather than
        re-embed the corpus. This is the method that makes that possible.
        """
        d = Path(dirpath)
        dense = DenseIndex.load(d)
        chunks = pickle.loads((d / "chunks.pkl").read_bytes())
        emb = Embedder(dense.model_name)
        return cls(chunks, emb, dense, LexicalIndex(chunks), reranker, translator)
