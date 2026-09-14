"""PDF parsing for the WKO collective agreements.

THE ONE THING THAT MATTERS HERE
-------------------------------
Most eval questions resolve to a single cell of the Gehaltstabelle in § 15. If
a row label detaches from its number, retrieval can be perfect, the citation
can point at the right paragraph, and the answer is still wrong -- with a
correct-looking citation attached. That is the worst failure mode available to
this project, so table handling gets the care, and prose gets the leftovers.

WHY extract_tables() AND NOT TEXT PARSING
-----------------------------------------
extract_text() flattens the salary grid to:

    Einstiegsstufe 2 236 2 547 3 267 4061 5 301

The thousands separator is a plain space (U+0020), so those five values are
nine whitespace tokens -- and the spacing is not even internally consistent
("4061" in one cell, "4 611" in the next). Column alignment is unrecoverable.

Worse, the Berufseinsteiger row is ragged: it has values only in AT and ST1,
with "gemäß § 15 I. (11)" spanning the rest. Reading numbers in text order
assigns them to ZT and AT, shifting every value one column left.

extract_tables() gets it right because the PDF draws cell rectangles (120 of
them on that page), giving lattice detection real geometry to work from.

TITLES COME FROM THE TABLE OF CONTENTS
--------------------------------------
Section headings wrap across lines in the body ("§ 15 Tätigkeitsfamilien,
Vorrückungsstufen und" / "Mindestgrundgehälter"). Rather than guess where a
heading ends, we read the publisher's own TOC and use it as the authority,
then cross-check that the body contains every section the TOC promises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

# A TOC line ends in dotted leaders and/or a page number. Body headings do not.
TOC_LEADER = re.compile(r"\.{4,}")

# German writes "§ 15". The English translation writes BOTH "Art. 1" and
# "Article 4" -- inconsistently, within the same table of contents.
#
# All three canonicalise to "§ N". The provisions are the same provisions, and
# the brief requires that an English question cite the German paragraph, so a
# language-native section_ref would make cross-lingual ground truth impossible
# to express. The English document's own wording is preserved in the chunk text.
SECTION_START = re.compile(r"^\s*(?:§|Art\.|Article)\s*(\d+[a-z]?)\b")


@dataclass
class Table:
    page: int
    header: list[str]
    rows: list[list[str]]
    caption: str | None = None  # the line just above, e.g. "Die Mindestgrundgehälter betragen ab 1.1.2026:"


@dataclass
class Section:
    ref: str                   # "§ 15"
    title: str                 # from the TOC
    page_start: int
    text: str = ""
    tables: list[Table] = field(default_factory=list)


def parse_toc(pdf: pdfplumber.PDF, max_pages: int = 4) -> dict[str, str]:
    """{'§ 15': 'Tätigkeitsfamilien, Vorrückungsstufen und Mindestgrundgehälter'}.

    TOC entries wrap: "§ 17 Ermittlung der kollektivvertraglichen
    Mindestgrundgehälter für" continues on the next line. An entry is complete
    when we hit the dotted leader / page number that terminates it.
    """
    toc: dict[str, str] = {}
    ref: str | None = None
    parts: list[str] = []

    def flush() -> None:
        nonlocal ref, parts
        if ref and parts:
            title = " ".join(" ".join(parts).split())
            title = TOC_LEADER.sub(" ", title)
            title = re.sub(r"[\s.]*\d*\s*$", "", title).strip(" .")
            if title:
                toc.setdefault(ref, title)
        ref, parts = None, []

    for pg in pdf.pages[:max_pages]:
        for line in (pg.extract_text() or "").splitlines():
            m = SECTION_START.match(line)
            if m:
                flush()
                ref = f"§ {m.group(1)}"
                parts = [line[m.end():]]
            elif ref and line.strip() and not SECTION_START.match(line):
                parts.append(line)
            if ref and (TOC_LEADER.search(line) or re.search(r"\d+\s*$", line)):
                flush()
    flush()
    return toc


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _starts_title(rest: str, title: str) -> bool:
    """Does the text after "§ N" begin that section's TOC title?

    This is what separates a heading from a cross-reference. Both can start a
    line, because prose wraps:

        "Article 15 Task groups, advancement levels and minimum basic"  <- heading
        "Art. 15 III. is the same or higher."                           <- reference
        "Art. 17 [1])."                                                 <- reference

    Only the first continues into the title the publisher's own TOC gives for
    that section. An empty remainder is accepted: some headings put the whole
    title on the following line.
    """
    r = _norm(rest)
    return not r or _norm(title).startswith(r[: len(_norm(title))])


def canonical_ref(number: str) -> str:
    """'15' -> '§ 15'. One spelling for both languages."""
    return f"§ {number}"


def _page_stream(pg: pdfplumber.page.Page) -> list[tuple[float, str, object]]:
    """Text lines and tables on one page, interleaved in reading order.

    Tables are placed by their bbox top so that a table lands inside the
    section it visually belongs to, rather than whichever section happens to
    own the page.
    """
    items: list[tuple[float, str, object]] = []
    for ln in pg.extract_text_lines():
        items.append((ln["top"], "line", ln["text"]))

    for tb in pg.find_tables():
        data = tb.extract()
        if not data or len(data) < 2:
            continue
        items.append((tb.bbox[1], "table", data))

    items.sort(key=lambda x: x[0])
    return items


def parse_pdf(path: Path) -> list[Section]:
    with pdfplumber.open(path) as pdf:
        toc = parse_toc(pdf)
        if not toc:
            raise RuntimeError(f"{path.name}: no table of contents found -- cannot title sections")

        sections: list[Section] = []
        cur: Section | None = None
        buf: list[str] = []
        last_line: str | None = None
        toc_pages = {1, 2}  # front matter; TOC entries must not become sections

        for pno, pg in enumerate(pdf.pages, 1):
            for _top, kind, payload in _page_stream(pg):
                if kind == "line":
                    line = str(payload)
                    m = SECTION_START.match(line)
                    ref_candidate = f"§ {m.group(1)}" if m else None
                    is_heading = (
                        m is not None
                        and pno not in toc_pages
                        and not TOC_LEADER.search(line)
                        and ref_candidate in toc
                        and _starts_title(line[m.end():], toc[ref_candidate])
                    )
                    if is_heading:
                        if cur:
                            cur.text = _clean_body(buf, cur.title)
                            sections.append(cur)
                        ref = ref_candidate
                        cur = Section(ref=ref, title=toc[ref], page_start=pno)
                        buf = []
                    elif cur:
                        buf.append(line)
                    last_line = line

                elif kind == "table" and cur:
                    data = [[(c or "").replace("\n", " ").strip() for c in row] for row in payload]
                    cur.tables.append(
                        Table(page=pno, header=data[0], rows=data[1:], caption=last_line)
                    )

        if cur:
            cur.text = _clean_body(buf, cur.title)
            sections.append(cur)

    _verify_against_toc(path, toc, sections)
    return sections


def _clean_body(lines: list[str], title: str) -> str:
    """Drop heading-continuation lines that repeat the (wrapped) section title."""
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    t = norm(title)
    i = 0
    acc = ""
    while i < len(lines) and acc != t:
        nxt = norm(acc + " " + lines[i])
        if not t.startswith(nxt):
            break
        acc = nxt
        i += 1
    return "\n".join(lines[i:]).strip()


def _verify_against_toc(path: Path, toc: dict[str, str], sections: list[Section]) -> None:
    """The publisher tells us what sections exist. Check we found them.

    Silent under-extraction is the failure to fear: a § that never becomes a
    section is simply absent from the index, and absence produces no error --
    only a question that mysteriously cannot be answered.
    """
    found = {s.ref for s in sections}
    missing = [r for r in toc if r not in found]
    if missing:
        raise RuntimeError(
            f"{path.name}: TOC lists {len(toc)} sections, body parse found {len(found)}. "
            f"Missing: {missing}. Refusing to build a silently incomplete index."
        )
    dupes = [s.ref for s in sections if [x.ref for x in sections].count(s.ref) > 1]
    if dupes:
        raise RuntimeError(f"{path.name}: duplicate sections {sorted(set(dupes))}")


# --------------------------------------------------------------------------
# Money
# --------------------------------------------------------------------------

def parse_amount(cell: str, lang: str = "de") -> int | None:
    """Parse a money cell. MUST be told the language -- the conventions conflict.

        German  : "2 420" "4061" "1.210,50" "960 ,-"   space/. = thousands, , = decimal
        English : "2,420" "1,210.00" "960.00"          , = thousands, . = decimal

    "1.210" is 1210 in the German document and 1.21 in the English one. The
    same character means opposite things, so a language-agnostic parser is not
    merely imprecise, it is wrong by a factor of 1000.

    Returns None for "no value defined". The 2026 table leaves the empty
    Berufseinsteiger cells blank; the 2025 table puts '0' in them. Same
    meaning, two spellings -- and left alone, '0' would be reported as a salary
    of zero. That is not a missing answer, it is a wrong one, and zero is
    plausible enough to survive review.
    """
    s = (cell or "").strip().replace("\u00a0", " ")
    if not s:
        return None

    # German writes whole euros as "960 ,-" / "1 210 ,-".
    s = re.sub(r"[,.]\s*-\s*$", "", s).strip()

    if lang == "de":
        s = s.replace(" ", "").replace(".", "").split(",")[0]
    else:
        s = s.replace(" ", "").replace(",", "").split(".")[0]

    if not s.isdigit():
        return None
    v = int(s)
    if v == 0:
        return None
    # A monthly minimum in this KV is 3-5 figures (Lehrlinge from 960).
    # Anything outside that is an extraction fault, not a salary -- raise
    # rather than return a number nobody will question.
    if not (500 <= v <= 99_999):
        raise ValueError(f"implausible amount {v!r} parsed from cell {cell!r} (lang={lang})")
    return v
