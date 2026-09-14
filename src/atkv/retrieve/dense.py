"""Dense vector index over the corpus, with a PRE-RANKING metadata filter.

THE POINT OF THIS FILE
----------------------
The brief demands the version filter be applied "before ranking, not after",
and that distinction is the whole project. It is worth being precise about why,
because post-filtering looks correct and is not.

POST-filtering (the wrong way):
    search the whole index for top-k, then drop results from the wrong year.

    Ask for k=5 about 2026. The index holds 2025 and 2026 copies of near-
    identical text, so the top 5 may be four 2025 chunks and one 2026 chunk.
    Filter afterwards and you are left with ONE result -- or, when the 2025
    copies happen to score marginally higher across the board, with NONE.
    Retrieval silently returns less than you asked for, or nothing, and the
    generator answers from whatever scraps remain. Widening k papers over it
    without fixing it: the ratio of wrong-year to right-year candidates does
    not improve, you just burn compute.

PRE-filtering (what this does):
    restrict the searchable set to chunks that pass the filter, THEN rank.

    Ask for k=5 and you get the 5 best chunks that are actually in force in
    2026. The 2025 copies are not competitors; they are not in the contest.

FAISS implements this through IDSelector, which is a genuine restriction of the
search, not a post-hoc mask. The same mechanism serves tenant_id, which is why
Chunk carries a tenant field nothing uses yet: access control and version
scoping are one problem wearing two hats.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import faiss
import numpy as np

from atkv.models import Chunk


@dataclass
class DenseHit:
    chunk: Chunk
    score: float


class DenseIndex:
    """Flat inner-product index over L2-normalised vectors (so IP == cosine).

    Flat, not IVF/HNSW: the corpus is ~400 chunks. An approximate index would
    add tuning parameters and recall loss to save microseconds on a search that
    is already exact and instant. Approximate indexes earn their keep at 10^6+.
    """

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, model_name: str) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"{len(chunks)} chunks but {vectors.shape[0]} vectors")
        self.chunks = chunks
        self.model_name = model_name
        self.dim = int(vectors.shape[1])
        base = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIDMap2(base)
        self.index.add_with_ids(np.ascontiguousarray(vectors), np.arange(len(chunks), dtype=np.int64))

    # -- the filter --------------------------------------------------------

    def _allowed_ids(self, as_of: date, tenant_id: str, lang: str | None) -> np.ndarray:
        return np.array(
            [i for i, c in enumerate(self.chunks)
             if c.in_force_on(as_of)
             and c.tenant_id == tenant_id
             and (lang is None or c.lang == lang)],
            dtype=np.int64,
        )

    def search(self, query_vec: np.ndarray, k: int = 8, *, as_of: date | None = None,
               tenant_id: str = "public", lang: str | None = None) -> list[DenseHit]:
        q = np.ascontiguousarray(query_vec.reshape(1, -1).astype(np.float32))

        params = None
        if as_of is not None:
            ids = self._allowed_ids(as_of, tenant_id, lang)
            if ids.size == 0:
                return []
            # IDSelectorBatch restricts what the scan may consider. The excluded
            # vectors are never scored, so they cannot crowd out the survivors.
            sel = faiss.IDSelectorBatch(ids.size, faiss.swig_ptr(ids))
            params = faiss.SearchParametersIVF() if hasattr(faiss, "SearchParametersIVF") else faiss.SearchParameters()
            params.sel = sel

        scores, idx = self.index.search(q, min(k, len(self.chunks)), params=params)
        return [DenseHit(self.chunks[i], float(s))
                for s, i in zip(scores[0], idx[0]) if i != -1]

    def search_unfiltered(self, query_vec: np.ndarray, k: int = 8) -> list[DenseHit]:
        """Deliberately no filter. Exists ONLY so the eval can measure the
        before/after the brief asks for -- never call it from the API."""
        return self.search(query_vec, k=k, as_of=None)

    # -- persistence -------------------------------------------------------

    def save(self, dirpath: Path) -> None:
        d = Path(dirpath)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "dense.faiss"))
        (d / "chunks.pkl").write_bytes(pickle.dumps(self.chunks))
        (d / "meta.json").write_text(json.dumps(
            {"model": self.model_name, "dim": self.dim, "n": len(self.chunks)}, indent=2))

    @classmethod
    def load(cls, dirpath: Path) -> "DenseIndex":
        d = Path(dirpath)
        meta = json.loads((d / "meta.json").read_text())
        obj = cls.__new__(cls)
        obj.index = faiss.read_index(str(d / "dense.faiss"))
        obj.chunks = pickle.loads((d / "chunks.pkl").read_bytes())
        obj.model_name = meta["model"]
        obj.dim = meta["dim"]
        return obj
