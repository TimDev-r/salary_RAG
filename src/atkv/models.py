"""Core data contract.

Every layer -- ingest, retrieval, API, evals -- agrees on the shapes in this
file. That is why it is written first: changing a field here ripples outward,
so it is cheap now and expensive later.

Two ideas drive the whole design:

1. A Chunk must carry enough metadata to be FILTERED BEFORE RANKING.
   The 2025 and 2026 collective agreements are ~95% identical text; only
   numbers move. An embedding model cannot tell them apart, and will return
   the wrong year confidently. The only thing that can separate them is
   metadata we stamp at ingest time.

2. A Chunk must be CITABLE ON ITS OWN, without consulting another object.
   If rendering "[IT-KV 2026, Gehaltstabelle, § 15]" requires a second lookup,
   some code path will eventually skip the lookup and emit a bare answer.
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------
# Controlled vocabularies. Literal rather than str so a typo is a validation
# error at ingest time, not a silently empty filter result at query time.
# --------------------------------------------------------------------------

SourceType = Literal["kv", "law"]
Lang = Literal["de", "en"]
ContentKind = Literal["prose", "table"]


class Chunk(BaseModel):
    """One retrievable unit of text, with everything needed to filter and cite it."""

    # --- identity ---------------------------------------------------------
    chunk_id: str
    doc_id: str = Field(description="Stable id of the source document, e.g. 'it-kv-2026-de'")

    # --- version scoping: the reason this project is interesting ----------
    # An interval, not a year. The documents themselves say "gilt ab 1.1.2026",
    # and a paragraph unchanged since 2024 should not be duplicated once per
    # year just so a `year` field can match it.
    valid_from: date
    valid_to: date | None = Field(
        default=None,
        description="None means 'still in force'. Set when a successor document supersedes this one.",
    )

    # --- access scoping ---------------------------------------------------
    # Not used by any feature yet, and deliberately so: it exists to show that
    # version filtering and tenant filtering are THE SAME MECHANISM. Whatever
    # makes `valid_from` safe to filter on makes access control safe too.
    tenant_id: str = "public"

    # --- provenance, sufficient for a citation ---------------------------
    source_type: SourceType
    short_title: str = Field(description="Citation form: 'IT-KV', 'AZG', 'UrlG'")
    source_title: str = Field(description="Full title as printed on the document")
    source_url: str = Field(description="Where this was fetched from, for replay and audit")
    section_ref: str | None = Field(
        default=None, description="'§ 15', 'Art. 3', 'Anlage 1' -- the citable anchor"
    )
    section_title: str | None = Field(default=None, description="'Gehaltstabelle', 'Urlaubsausmaß'")
    heading_path: list[str] = Field(
        default_factory=list,
        description="Breadcrumb of enclosing headings, e.g. ['IV. Entgelt', '§ 15 Mindestgrundgehälter']",
    )
    page: int | None = Field(default=None, description="1-based page in the source PDF, if any")

    # --- content ----------------------------------------------------------
    lang: Lang
    content_kind: ContentKind = "prose"
    text: str

    @model_validator(mode="after")
    def _check_interval(self) -> "Chunk":
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError(f"valid_to {self.valid_to} precedes valid_from {self.valid_from}")
        if not self.text.strip():
            raise ValueError(f"chunk {self.chunk_id} has empty text")
        return self

    # --- helpers ----------------------------------------------------------

    @staticmethod
    def make_id(doc_id: str, ordinal: int) -> str:
        """Deterministic id, so re-ingesting the same document twice is idempotent.

        NOTE: this is derived from position, so it CHANGES if we change the
        chunking strategy -- and we will, in step 4. That is why eval ground
        truth must reference (doc_id, section_ref), which is a property of the
        source document, and never a chunk_id, which is a property of our code.
        """
        return hashlib.sha256(f"{doc_id}:{ordinal}".encode()).hexdigest()[:16]

    def in_force_on(self, day: date) -> bool:
        """The predicate the pre-ranking filter is built from."""
        if day < self.valid_from:
            return False
        return self.valid_to is None or day <= self.valid_to

    def to_citation(self) -> "Citation":
        return Citation(
            chunk_id=self.chunk_id,
            short_title=self.short_title,
            year=self.valid_from.year,
            section_ref=self.section_ref,
            section_title=self.section_title,
            source_url=self.source_url,
            page=self.page,
        )


class Citation(BaseModel):
    """A reference the user can verify by opening the source document."""

    chunk_id: str
    short_title: str
    year: int | None = None
    section_ref: str | None = None
    section_title: str | None = None
    source_url: str
    page: int | None = None

    def render(self) -> str:
        """'[IT-KV 2026, Gehaltstabelle, § 15]'"""
        head = f"{self.short_title} {self.year}" if self.year else self.short_title
        parts = [head, *(p for p in (self.section_title, self.section_ref) if p)]
        return "[" + ", ".join(parts) + "]"


class RetrievedChunk(BaseModel):
    """A chunk plus the scores that selected it.

    This is the object we write to the structured log. Keeping every component
    score -- not just the final one -- is what makes an answer replayable:
    when a query returns something odd we can see whether dense or lexical
    retrieval dragged it in, without re-running anything.
    """

    chunk: Chunk
    dense_score: float | None = None
    lexical_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None


# --------------------------------------------------------------------------
# API surface
# --------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    valid_year: int = Field(ge=2025, le=2030, description="Scopes retrieval to documents in force in this year")
    lang: Lang = Field(default="de", description="Language of the ANSWER. Sources may be in the other language.")
    tenant_id: str = "public"
    k: int = Field(default=8, ge=1, le=50, description="Chunks to retrieve before generation")
    rerank: bool = Field(default=False, description="Run the cross-encoder. Measured with and without.")

    def as_of(self) -> date:
        """The date the version filter tests against.

        Mid-year is deliberate: 1 January would sit exactly on the boundary of
        every 'gilt ab 1.1.' document, and boundary conditions are where
        off-by-one bugs live. 1 July is unambiguously inside the year.
        """
        return date(self.valid_year, 7, 1)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    trace_id: str
    refused: bool = False
    refusal_reason: str | None = None
