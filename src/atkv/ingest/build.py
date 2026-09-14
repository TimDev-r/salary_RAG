"""Build the full corpus: manifest -> fetch -> parse -> chunk -> Chunk list.

One function, used by /ingest, by the eval suite and by the Docker build, so
all three index exactly the same thing. When they diverge, the eval stops
measuring what the service serves.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from atkv.ingest.chunk import chunk_document
from atkv.ingest.parse import parse_pdf
from atkv.ingest.ris import LAWS, RisClient
from atkv.ingest.wko import WkoFetcher, load_manifest
from atkv.models import Chunk

LAW_FASSUNG = date(2026, 7, 1)


def build_corpus(root: Path, fetch: bool = True) -> list[Chunk]:
    root = Path(root)
    docs, _evals = load_manifest(root / "manifests/sources.yaml")

    wko_dir = root / "data/raw/wko"
    if fetch:
        f = WkoFetcher(wko_dir)
        try:
            f.fetch_all(docs)
        finally:
            f.close()

    chunks: list[Chunk] = []
    for d in docs:
        chunks += chunk_document(d, parse_pdf(wko_dir / f"{d.doc_id}.pdf"))

    rc = RisClient(root / "data/raw/ris")
    try:
        for law in LAWS.values():
            chunks += rc.chunks_for_law(law, LAW_FASSUNG)
    finally:
        rc.close()
    return chunks
