"""Turn parsed sections into indexable Chunks.

STRATEGY: prose chunks + one whole-table chunk + one chunk per table cell.

Why cells get their own chunks
------------------------------
The question "Wie hoch ist das Mindestgrundgehalt für ST1 Erfahrungsstufe
2026?" has to retrieve one cell of a 5x4 grid. Embedding a grid of bare
numbers and hoping it matches a natural-language question is a bad bet: the
grid contains almost no words, and the words it does contain ("ST1", "2026")
are equally present in the other year's identical grid.

Verbalising each cell turns a geometry problem into a text problem, which is
what the retriever is actually good at. The chunk then literally contains the
question's terms AND the answer:

    IT-KV 2026, § 15 Tätigkeitsfamilien, Vorrückungsstufen und Mindestgrundgehälter
    Die Mindestgrundgehälter betragen ab 1.1.2026:
    ST1 / Erfahrungsstufe: 4.476 EUR pro Monat

Why the whole table is ALSO kept
--------------------------------
Cell chunks cannot answer "compare ST1 and ST2" or "which family pays most" --
each one has been severed from its neighbours. The full table costs one extra
chunk per document and covers those.

The price, and it is real: every figure is now indexed twice, so a naive top-k
can return the cell chunk and the table chunk as two "different" results,
burning context on a duplicate. Fusion must dedupe on (doc_id, section_ref).
That is work owed in step 7, not something this file solves.

WHY THE VERBALISATION SAYS "brutto pro Monat"
---------------------------------------------
The salary table itself prints bare numbers: no currency, no period, no
gross/net marker. Both qualifiers are nonetheless grounded in the document:

  monthly  § 13 sets the annual entitlement at "das Vierzehnfache des ...
           Mindestgrundgehaltes" -- fourteen monthly payments.
  gross    the agreement's unit for a monthly salary is a
           "Bruttomonatsgehalt" (§ 14, the aliquot rule divides it by 30),
           and "netto" appears nowhere in the document at all.
  EUR      the same agreement prices night work at "€ 7,55".

Stating "brutto" is not decoration. A reader who takes 4.476 for take-home
pay has been materially misled, and silence is no defence when silence is
predictably misread. The qualifier is asserted because it is sourced -- had
it not been, the right fix would have been to find the source, not to guess.
"""

from __future__ import annotations

import re

from atkv.ingest.parse import Section, Table, parse_amount
from atkv.ingest.text import MAX_PROSE_CHARS, split_text
from atkv.ingest.wko import CorpusDoc
from atkv.models import Chunk





def _fmt(amount: int, lang: str) -> str:
    """4476 -> '4.476' (de) / '4,476' (en).

    Rendered in the reader's own convention, so a query typed as "4,476"
    matches the English chunk lexically and "4.476" matches the German one.
    """
    return f"{amount:,}".replace(",", ".") if lang == "de" else f"{amount:,}"


def _is_real_table(t: Table) -> bool:
    """A single-column 'table' is a list that happens to have box borders.

    § 16 Lehrlingseinkommen produces header=['im 1. Lehrjahr: 960 ,-'] with one
    column. Its content is already in the section prose, so indexing it again
    as a table would duplicate it and verbalise nothing useful.
    """
    return len(t.header) >= 2 and len(t.rows) >= 1


def _verbalise_cell(doc: CorpusDoc, sec: Section, t: Table, row_label: str,
                    col_label: str, amount: int) -> str:
    unit = "EUR brutto pro Monat" if doc.lang == "de" else "EUR gross per month"
    head = f"{doc.short_title} {doc.valid_from.year}, {sec.ref} {sec.title}"
    caption = (t.caption or "").strip()
    return "\n".join(x for x in [head, caption, f"{col_label} / {row_label}: {_fmt(amount, doc.lang)} {unit}"] if x)


def _render_table(doc: CorpusDoc, sec: Section, t: Table) -> str:
    head = f"{doc.short_title} {doc.valid_from.year}, {sec.ref} {sec.title}"
    lines = [head]
    if t.caption:
        lines.append(t.caption.strip())
    lines.append(" | ".join(t.header))
    for r in t.rows:
        lines.append(" | ".join(c or "-" for c in r))
    return "\n".join(lines)


def chunk_document(doc: CorpusDoc, sections: list[Section]) -> list[Chunk]:
    chunks: list[Chunk] = []

    def add(text: str, sec: Section, kind: str, page: int | None) -> None:
        chunks.append(
            Chunk(
                chunk_id=Chunk.make_id(doc.doc_id, len(chunks)),
                doc_id=doc.doc_id,
                valid_from=doc.valid_from,
                valid_to=doc.valid_to,
                source_type="kv",
                short_title=doc.short_title,
                source_title=doc.source_title,
                source_url=doc.url,
                section_ref=sec.ref,
                section_title=sec.title,
                heading_path=[sec.title],
                page=page,
                lang=doc.lang,          # type: ignore[arg-type]
                content_kind=kind,      # type: ignore[arg-type]
                text=text,
            )
        )

    for sec in sections:
        header = f"{doc.short_title} {doc.valid_from.year}, {sec.ref} {sec.title}"

        for part in split_text(sec.text):
            # Every chunk carries its § and year. Without this a retrieved
            # fragment reads as anonymous text and the generator has no way to
            # attribute it, even though the metadata is on the object.
            add(f"{header}\n{part}", sec, "prose", sec.page_start)

        for t in sec.tables:
            if not _is_real_table(t):
                continue
            add(_render_table(doc, sec, t), sec, "table", t.page)

            before = len(chunks)
            for row in t.rows:
                row_label = (row[0] or "").strip()
                if not row_label:
                    continue
                for col_idx, col_label in enumerate(t.header[1:], start=1):
                    if col_idx >= len(row) or not col_label.strip():
                        continue
                    amount = parse_amount(row[col_idx], doc.lang)
                    if amount is None:
                        continue  # blank or '0' -- no value defined, not a salary of zero
                    add(_verbalise_cell(doc, sec, t, row_label, col_label.strip(), amount),
                        sec, "table", t.page)

            # A grid with columns and rows that produces no parseable amounts
            # means the number format was not understood. This exact bug --
            # English "2,420" against a German-only parser -- silently emptied
            # every English salary cell while raising nothing at all.
            if len(chunks) == before:
                raise RuntimeError(
                    f"{doc.doc_id} {sec.ref} p{t.page}: table has "
                    f"{len(t.header)} cols x {len(t.rows)} rows but no cell parsed as an "
                    f"amount (lang={doc.lang}). Sample row: {t.rows[0]!r}"
                )

    return chunks
