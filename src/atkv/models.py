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
        # The year is rendered only for collective agreements, where "IT-KV 2026"
        # names a distinct document a reader can go and open. For consolidated
        # law, valid_from is the paragraph's Inkrafttretensdatum -- correct for
        # filtering, but rendering "[AZG 2022]" would claim a "2022 version of
        # the AZG" that does not exist. Austrian practice is simply "§ 1 AZG".
        return Citation(
            chunk_id=self.chunk_id,
            short_title=self.short_title,
            year=self.valid_from.year if self.source_type == "kv" else None,
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

    def render(self, short: bool = False) -> str:
        """'[IT-KV 2026, Gehaltstabelle, § 15]', or '[IT-KV 2026, § 15]' if short.

        The short form exists for text a language model has to reproduce. Some
        section titles run to sixty characters ("Tätigkeitsfamilien,
        Vorrückungsstufen und Mindestgrundgehälter"), and every token of that
        is a token generated at 8-10 tok/s -- about a fifth of the whole
        answer, spent on a heading the reader can look up from the paragraph
        number alone. Nothing is lost: the full citation, title included, is
        still returned in the structured `citations` field of the response.
        """
        head = f"{self.short_title} {self.year}" if self.year else self.short_title
        title = None if short else self.section_title
        parts = [head, *(p for p in (title, self.section_ref) if p)]
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
    valid_year: int | None = Field(
        default=None, ge=2025, le=2030,
        description=(
            "Year to scope retrieval to. Omit it for 'whatever is in force today', "
            "which is what a user who does not name a year means."
        ),
    )
    lang: Lang = Field(default="de", description="Language of the ANSWER. Sources may be in the other language.")
    tenant_id: str = "public"
    k: int = Field(
        default=3, ge=1, le=50,
        description=(
            "Chunks passed to generation. Measured: k=3 scores 19/25 against k=5's "
            "17/25 with the same retrieval recall, because a smaller context gives "
            "the model fewer candidates to choose wrongly among. It is also faster, "
            "since prompt evaluation dominates latency."
        ),
    )
    rerank: bool = Field(default=False, description="Run the cross-encoder. Measured with and without.")

    def as_of(self) -> date:
        """The date the version filter tests against.

        With no year given this is TODAY, which is what someone asking "what is
        the ST1 minimum?" means. Requiring a year rejected that question
        outright with a 422, which is a poor answer to the commonest question
        the system will get.

        With a year given, 1 July. Mid-year is deliberate: 1 January sits
        exactly on the boundary of every "gilt ab 1.1." document, and boundary
        conditions are where off-by-one bugs live. 1 July is unambiguously
        inside the year.
        """
        if self.valid_year is None:
            return date.today()
        return date(self.valid_year, 7, 1)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    trace_id: str
    refused: bool = False
    refusal_reason: str | None = None
    as_of: date | None = Field(
        default=None,
        description="The date the version filter used. Echoed back so a caller who "
                    "omitted the year can see which one they actually got.",
    )
    notice: str | None = Field(
        default=None,
        description=(
            "Set when the answer is limited in a way the user cannot see -- above all "
            "when the requested date falls outside the collective agreement's coverage. "
            "Without it, asking about 2027 returns an answer built only from labour law, "
            "with no indication that the agreement is missing entirely."
        ),
    )
