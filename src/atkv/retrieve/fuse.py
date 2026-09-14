"""Reciprocal Rank Fusion of the dense and lexical result lists.

WHY RRF AND NOT A WEIGHTED SCORE SUM
------------------------------------
The two retrievers produce incomparable numbers. Cosine similarity from e5 sits
in a narrow band -- step 6 measured five candidates between 0.895 and 0.902 --
while BM25 is unbounded and varies with query length and corpus statistics.
Normalising them onto a common scale means choosing a normalisation, and any
choice is a tuning knob fitted to 25 questions, which is overfitting with extra
steps.

RRF discards the scores and uses only RANK:

    score(d) = sum over retrievers of  1 / (K + rank(d))

Rank is the one thing both retrievers agree on the meaning of. It has a single
constant, K, whose conventional value of 60 we keep rather than tune -- tuning
it on 25 questions would fit noise.

THE DUPLICATE PROBLEM THIS ALSO HAS TO SOLVE
--------------------------------------------
Step 4 indexed every salary figure twice: once as a verbalised cell chunk and
once inside the whole-table chunk. Both are legitimate, and both match the same
query, so a naive top-k spends slots on restatements of one fact. § 15 alone
has 18 table chunks that could crowd out every other section.

The fix is a per-section CAP rather than deduplication. Deduplicating on
(doc_id, section_ref) would collapse all 18 cells of the salary table into one
result and make "compare ST1 and ST2" unanswerable. A cap keeps the best few
and leaves room for other sections.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from atkv.models import Chunk, RetrievedChunk

RRF_K = 60
MAX_PER_SECTION = 3


@dataclass
class Ranked:
    chunk: Chunk
    fused: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    dense_score: float | None = None
    lexical_score: float | None = None


def rrf(dense_hits, lexical_hits, k: int = 10, *, rrf_k: int = RRF_K,
        max_per_section: int | None = None) -> list[Ranked]:
    by_id: dict[str, Ranked] = {}

    def note(hits, which: str) -> None:
        for rank, h in enumerate(hits, 1):
            r = by_id.get(h.chunk.chunk_id)
            if r is None:
                r = Ranked(chunk=h.chunk, fused=0.0)
                by_id[h.chunk.chunk_id] = r
            r.fused += 1.0 / (rrf_k + rank)
            setattr(r, f"{which}_rank", rank)
            setattr(r, f"{which}_score", h.score)

    note(dense_hits, "dense")
    note(lexical_hits, "lexical")

    ordered = sorted(by_id.values(), key=lambda r: -r.fused)

    if max_per_section is None:
        return ordered[:k]
    return cap_sections(ordered, k, max_per_section)


def cap_sections(items, k: int, max_per_section: int = MAX_PER_SECTION, key=lambda x: x.chunk):
    """Keep at most `max_per_section` results from any one (doc_id, section_ref).

    APPLY THIS LAST. An earlier version capped inside rrf(), before reranking,
    which removed candidates the cross-encoder never got to see and cost a
    question at R@5. Diversity is a property of what the user is shown, not of
    the candidate pool a reranker works from.
    """
    out = []
    seen: dict[tuple[str, str | None], int] = defaultdict(int)
    for it in items:
        c = key(it)
        sec = (c.doc_id, c.section_ref)
        if seen[sec] >= max_per_section:
            continue
        seen[sec] += 1
        out.append(it)
        if len(out) >= k:
            break
    return out


def to_retrieved(ranked: list[Ranked]) -> list[RetrievedChunk]:
    """Convert to the logged form, keeping component scores separate so an odd
    answer can be traced to the retriever that caused it."""
    return [
        RetrievedChunk(
            chunk=r.chunk,
            dense_score=r.dense_score,
            lexical_score=r.lexical_score,
            fused_score=r.fused,
        )
        for r in ranked
    ]
