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

    def search(self, question: str, *, as_of: date | None = None, k: int = 8,
               lang: str = "de", tenant_id: str = "public",
               rerank: bool = False) -> RetrievalResult:
        # Query translation feeds the LEXICAL leg only. BM25 scores an English
        # query against a German passage at exactly zero -- no shared token --
        # so without this the correct chunk is not a weak candidate, it is not
        # a candidate. Measured: pool ceiling 0.96 -> 1.00.
        lex_query = question
        translated = False
        if self.translator is not None and lang != "de":
            lex_query = self.translator.expand(question, lang)
            translated = lex_query != question

        pool_k = POOL if (rerank and self.reranker) else max(k, 10)
        d = self.dense.search(self.embedder.encode_query(question), k=pool_k,
                              as_of=as_of, tenant_id=tenant_id)
        l = self.lexical.search(lex_query, k=pool_k, as_of=as_of, tenant_id=tenant_id)

        # NOTE: retrieval is deliberately NOT restricted to the question's
        # language. AZG and UrlG exist only in German, so a language filter
        # would make every English question about them unanswerable.
        fused = rrf(d, l, k=pool_k)

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
