"""Fetcher for the WKO collective agreement documents.

WHY THIS FILE EXISTS SEPARATELY FROM THE PARSER
-----------------------------------------------
Downloading and trusting are different jobs. This module's only responsibility
is to turn manifest entries into verified local files. Nothing here knows what
a Chunk is.

THE COPYRIGHT RULE
------------------
These PDFs belong to the social partners (WKO and GPA). They are fetched at
ingest time from manifests/sources.yaml and cached under data/raw/, which is
gitignored. They are never committed. Only URLs live in the repo.

THE VERIFICATION RULE
---------------------
Every download is checked against `expect_on_page_1` from the manifest. This is
not defensive padding -- it is the single most important guard in the project.

The 2025 PDF URL is not linked from WKO's own site (their 2025 page is archived
and renders HTML instead); we derived it by analogy and verified it once, by
hand. If WKO ever repoints that path at a different year's document, every
guard we have EXCEPT this one would pass: valid PDF, right page count, German
text about collective agreements, plausible salary tables. Retrieval would look
excellent. The citations would say 2025. The numbers would be wrong.

A system whose whole premise is "return the right year's figures" cannot detect
that failure downstream. It has to be caught here, at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import pdfplumber
import yaml


@dataclass(frozen=True)
class CorpusDoc:
    """A document that goes INTO the retrieval index."""

    doc_id: str
    url: str
    landing_page: str
    short_title: str
    source_title: str
    lang: str
    valid_from: date
    valid_to: date | None
    expect_on_page_1: str


@dataclass(frozen=True)
class EvalSource:
    """A document the eval set is built FROM. Never indexed."""

    doc_id: str
    url: str
    title: str
    lang: str
    about_year: int


def load_manifest(path: Path) -> tuple[list[CorpusDoc], list[EvalSource]]:
    """Load the manifest as two separate lists that share no code path.

    The separation is the safety property. If these were one list with an
    `index: bool` flag, a single wrong boolean would put the answer key into
    the retrieval index -- and every metric would improve, which is precisely
    what makes that bug so hard to notice.
    """
    raw = yaml.safe_load(Path(path).read_text())
    corpus = [CorpusDoc(**d) for d in raw["corpus"]]
    evals = [EvalSource(**d) for d in raw["eval_sources"]]

    ids = [d.doc_id for d in corpus] + [d.doc_id for d in evals]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate doc_id in manifest: {ids}")
    return corpus, evals


class WkoFetcher:
    def __init__(self, cache_dir: Path) -> None:
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._http = httpx.Client(
            timeout=60.0,
            follow_redirects=True,
            # WKO's CDN serves a bot-challenge page to unrecognised agents.
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"},
        )

    def fetch(self, doc: CorpusDoc) -> Path:
        """Download (or reuse) one PDF, verified. Returns its local path."""
        dest = self.cache / f"{doc.doc_id}.pdf"
        if dest.exists():
            self._verify(dest, doc)  # re-verify cached files: cheap, catches a poisoned cache
            return dest

        r = self._http.get(doc.url)
        r.raise_for_status()

        # A bot-challenge or error page is HTML served with status 200.
        if not r.content.startswith(b"%PDF"):
            raise RuntimeError(
                f"{doc.doc_id}: expected a PDF, got {r.headers.get('content-type')} "
                f"starting {r.content[:60]!r}"
            )

        tmp = dest.with_suffix(".part")
        tmp.write_bytes(r.content)
        try:
            self._verify(tmp, doc)
        except Exception:
            tmp.unlink(missing_ok=True)  # never leave an unverified file in the cache
            raise
        tmp.rename(dest)
        return dest

    @staticmethod
    def _verify(path: Path, doc: CorpusDoc) -> None:
        with pdfplumber.open(path) as pdf:
            if not pdf.pages:
                raise RuntimeError(f"{doc.doc_id}: PDF has no pages")
            page1 = " ".join((pdf.pages[0].extract_text() or "").split())

        if doc.expect_on_page_1.lower() not in page1.lower():
            raise RuntimeError(
                f"{doc.doc_id}: page 1 does not contain {doc.expect_on_page_1!r}.\n"
                f"  URL: {doc.url}\n"
                f"  page 1 begins: {page1[:180]!r}\n"
                f"  This means the URL now serves a different document. Do NOT index it: "
                f"a wrong-year KV is indistinguishable from a right one downstream."
            )

    def fetch_all(self, docs: list[CorpusDoc]) -> dict[str, Path]:
        return {d.doc_id: self.fetch(d) for d in docs}

    def close(self) -> None:
        self._http.close()
