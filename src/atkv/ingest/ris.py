"""Client for the RIS OGD API (Austrian federal law, consolidated).

Endpoint verified against the official handbook V2.6, not from memory:
    https://data.bka.gv.at/ris/ogd/v2.6/Documents/Dokumentation_OGD-RIS_API.pdf

TWO THINGS THIS API DOES THAT WILL RUIN YOUR DAY
------------------------------------------------
1. It ignores unknown query parameters SILENTLY. Misspelling `Titel` as `Title`
   returns HTTP 200 with 441,158 results -- the entire consolidated federal
   corpus -- instead of an error. Every guard in this module exists because of
   that one observation.

2. Missing documents are served as an HTML 404 page with status 200. So a
   successful-looking fetch can hand you a webpage saying "Seite nicht
   gefunden". We check the payload, not the status code.

WHY WE FETCH ONE PARAGRAPH AT A TIME
------------------------------------
In `Bundesrecht konsolidiert` one documentation unit is exactly one
Paragraf/Artikel/Anlage. That means structural chunking is already solved by
the publisher -- we would have to work to make it worse. Fetching the whole law
as a single HTML blob would save ~100 requests and force us to re-derive
paragraph boundaries by parsing headings, which is where chunk-boundary bugs
come from. We pay the requests once and cache to disk forever.
"""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

from atkv.ingest.text import split_text
from atkv.models import Chunk

BASE = "https://data.bka.gv.at/ris/api/v2.6"
NS = "{http://www.bka.gv.at}"

# A single named law has tens of units, not tens of thousands. If a query comes
# back with more than this, a parameter was ignored and we are looking at the
# whole corpus. Fail loudly rather than indexing 441k documents.
MAX_PLAUSIBLE_UNITS = 2_000


@dataclass(frozen=True)
class Law:
    """A law pinned by its RIS Gesetzesnummer.

    Pinned by number, not title: `Titel=Arbeitszeitgesetz` is a full-text match
    and returns 103 units, 24 of which belong to *other* laws that merely
    mention it. `Gesetzesnummer=10008238` returns exactly the 79 that are AZG.
    """

    abbrev: str            # "AZG"       -> Chunk.short_title
    gesetzesnummer: str    # "10008238"
    expected_kurztitel: str  # "Arbeitszeitgesetz" -- asserted on every unit


LAWS: dict[str, Law] = {
    "AZG": Law("AZG", "10008238", "Arbeitszeitgesetz"),
    "UrlG": Law("UrlG", "10008376", "Urlaubsgesetz"),
}


# --------------------------------------------------------------------------
# XML text extraction
# --------------------------------------------------------------------------

# Block-level elements. Text is collected per block; a block never absorbs the
# text of a nested block, because lists nest (aufzaehlung > listelem >
# aufzaehlung) and a naive walk emits the same sentence three times.
BLOCK = {"absatz", "ueberschrift", "listelem", "schlussteil"}

# List markers ("1.", "a)", "aa)") sit in their own element with no trailing
# space, so without this the text reads "1.Arbeitnehmer".
MARKER = {"symbol", "gldsym"}

# Page header/footer furniture. Not stripping this puts "Bundesrecht
# konsolidiert www.ris.bka.gv.at Seite 1 von 1" into every chunk we index.
FURNITURE = {"kzinhalt", "fzinhalt"}


def _tag(el: ET.Element) -> str:
    return el.tag.replace(NS, "")


def _block_text(el: ET.Element) -> str:
    """Text owned by this block, excluding any nested block's text."""
    parts: list[str] = []

    def rec(node: ET.Element, is_root: bool) -> None:
        if not is_root and _tag(node) in BLOCK:
            return  # emitted separately as its own block
        if node.text:
            parts.append(node.text)
        for child in node:
            if _tag(child) in MARKER:
                parts.append("".join(child.itertext()) + " ")
            else:
                rec(child, False)
            if child.tail:
                parts.append(child.tail)

    rec(el, True)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def extract_text(xml_bytes: bytes) -> tuple[str, list[str]]:
    """Return (legal text, heading path) for one documentation unit.

    The unit's XML is a metadata block, then the legal text, then more
    metadata. The boundary is a literal marker: an <ueberschrift typ="titel">
    whose text is exactly "Text". Everything before it is Kurztitel /
    Kundmachungsorgan / Inkrafttretensdatum etc. (which we already have, in
    better form, from the search JSON); everything after the *next* typ="titel"
    is Schlagworte / Dokumentnummer.
    """
    root = ET.fromstring(xml_bytes)
    nutzdaten = root.find(f"{NS}nutzdaten")
    if nutzdaten is None:
        raise ValueError("no <nutzdaten> element -- not a RIS document")

    lines: list[str] = []
    headings: list[str] = []
    in_body = False

    def walk(node: ET.Element) -> None:
        nonlocal in_body
        for el in node:
            tag = _tag(el)
            if tag in FURNITURE:
                continue
            if tag == "ueberschrift":
                typ = el.get("typ")
                text = _block_text(el)
                if typ == "titel":
                    # "Text" opens the body; any later titel closes it.
                    in_body = text.strip().lower() == "text"
                    continue
                if in_body and typ and typ.startswith("g"):
                    headings.append(text)   # ABSCHNITT 1 / Geltungsbereich
                    lines.append(text)
                continue
            if in_body and tag in BLOCK:
                text = _block_text(el)
                if text:
                    lines.append(text)
            walk(el)  # descend for nested blocks

    walk(nutzdaten)
    return "\n".join(lines).strip(), headings


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class RisClient:
    def __init__(self, cache_dir: Path, throttle_s: float = 0.25) -> None:
        self.cache = Path(cache_dir)
        (self.cache / "units").mkdir(parents=True, exist_ok=True)
        self.throttle_s = throttle_s
        self._http = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "atkv-research/0.1 (educational RAG project)"},
        )

    # -- search ------------------------------------------------------------

    def search_law(self, law: Law, fassung_vom: date) -> list[dict]:
        """All documentation units of one law, as in force on `fassung_vom`.

        `Fassung.FassungVom` makes RIS resolve amendments server-side: we ask
        for the law *as it stood* on a date and it applies the right version.
        Note the contrast with the collective agreements, where we hold several
        PDFs and must do that filtering ourselves. Same `valid_from` field on
        the Chunk, two entirely different mechanisms behind it.
        """
        cache_file = self.cache / f"search_{law.gesetzesnummer}_{fassung_vom}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())

        units: list[dict] = []
        page = 1
        while True:
            r = self._http.get(
                f"{BASE}/Bundesrecht",
                params={
                    "Applikation": "BrKons",
                    "Gesetzesnummer": law.gesetzesnummer,
                    "Fassung.FassungVom": fassung_vom.isoformat(),
                    "DokumenteProSeite": "OneHundred",
                    "Seitennummer": page,
                },
            )
            r.raise_for_status()
            result = r.json()["OgdSearchResult"]["OgdDocumentResults"]
            total = int(result["Hits"]["#text"])

            # GUARD 1: the silent-ignored-parameter check.
            if total > MAX_PLAUSIBLE_UNITS:
                raise RuntimeError(
                    f"{law.abbrev}: {total} hits. A single law has tens of units, not "
                    f"thousands -- RIS almost certainly ignored a misspelled parameter "
                    f"and returned the whole corpus. Refusing to index this."
                )
            if total == 0:
                raise RuntimeError(f"{law.abbrev}: 0 hits for Gesetzesnummer={law.gesetzesnummer}")

            refs = result.get("OgdDocumentReference") or []
            if isinstance(refs, dict):  # the API unwraps single-element lists
                refs = [refs]
            units.extend(refs)

            if len(units) >= total:
                break
            page += 1
            time.sleep(self.throttle_s)

        # GUARD 2: pinning by number should make every unit belong to this law.
        wrong = {
            u["Data"]["Metadaten"]["Bundesrecht"].get("Kurztitel")
            for u in units
        } - {law.expected_kurztitel}
        if wrong:
            raise RuntimeError(f"{law.abbrev}: unexpected Kurztitel in results: {wrong}")

        cache_file.write_text(json.dumps(units, ensure_ascii=False))
        return units

    # -- content -----------------------------------------------------------

    def fetch_unit_xml(self, unit: dict) -> bytes:
        """Fetch one unit's XML.

        The URL is READ FROM the search response, never constructed. The real
        pattern is /Dokumente/Bundesnormen/{ID}/{ID}.xml -- the id repeats --
        and a hand-built /Dokumente/Bundesnormen/{ID} returns an HTML 404 page
        with status 200.
        """
        doc_id = unit["Data"]["Metadaten"]["Technisch"]["ID"]
        cached = self.cache / "units" / f"{doc_id}.xml"
        if cached.exists():
            return cached.read_bytes()

        urls = unit["Data"]["Dokumentliste"]["ContentReference"]["Urls"]["ContentUrl"]
        urls = urls if isinstance(urls, list) else [urls]
        xml_url = next((u["Url"] for u in urls if u["DataType"] == "Xml"), None)
        if xml_url is None:
            raise RuntimeError(f"{doc_id}: no XML content URL in {[u['DataType'] for u in urls]}")

        r = self._http.get(xml_url)
        r.raise_for_status()

        # GUARD 3: 200-but-actually-a-404-page.
        if not r.content.lstrip().startswith(b"<?xml"):
            raise RuntimeError(f"{doc_id}: expected XML, got {r.content[:80]!r} (RIS serves 404s as HTML/200)")

        cached.write_bytes(r.content)
        time.sleep(self.throttle_s)
        return r.content

    # -- assembly ----------------------------------------------------------

    def chunks_for_law(self, law: Law, fassung_vom: date) -> list[Chunk]:
        units = self.search_law(law, fassung_vom)
        chunks: list[Chunk] = []

        for unit in units:
            meta = unit["Data"]["Metadaten"]
            br = meta["Bundesrecht"]
            kons = br["BrKons"]
            section_ref = kons.get("ArtikelParagraphAnlage")

            # "§ 0" is the promulgation header (Kundmachungstitel): amendment
            # history, Schlagworte, Anmerkungen. It is not substantive law and
            # must never be retrievable as though it were.
            if section_ref in (None, "§ 0"):
                continue

            text, headings = extract_text(self.fetch_unit_xml(unit))
            if not text:
                continue  # repealed paragraphs exist and are legitimately empty

            doc_id = f"{law.abbrev.lower()}-{law.gesetzesnummer}"
            abbrev = kons.get("Abkuerzung") or law.abbrev
            kurztitel = br.get("Kurztitel") or law.abbrev
            section_title = headings[-1] if headings else None

            # A topical header, exactly as the KV chunks get one.
            #
            # Without it, law chunks began with raw statute text while KV chunks
            # began "IT-KV 2026, § 4 Arbeitszeit". A question about working time
            # then matched the KV's literal heading far better than the actual
            # Arbeitszeitgesetz, and law retrieval measured 0.33 while cross-
            # lingual measured 0.00. The asymmetry was ours, not the model's.
            header = f"{kurztitel} ({abbrev}), {section_ref}"
            if section_title:
                header += f" {section_title}"

            # Split like everything else. 24% of law units exceeded the
            # embedder's 512-token window and were silently truncated.
            for part in split_text(text):
                chunks.append(
                    Chunk(
                        chunk_id=Chunk.make_id(doc_id, len(chunks)),
                        doc_id=doc_id,
                        valid_from=date.fromisoformat(kons["Inkrafttretensdatum"]),
                        valid_to=None,
                        source_type="law",
                        short_title=abbrev,
                        source_title=br.get("Titel") or kurztitel,
                        source_url=meta["Allgemein"]["DokumentUrl"],
                        section_ref=section_ref,
                        section_title=section_title,
                        heading_path=headings,
                        page=None,
                        lang="de",
                        content_kind="prose",
                        text=f"{header}\n{part}",
                    )
                )
        return chunks

    def close(self) -> None:
        self._http.close()
