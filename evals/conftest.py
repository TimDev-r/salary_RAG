"""Shared fixtures: build the corpus once for the whole eval session."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from atkv.ingest.chunk import chunk_document
from atkv.ingest.parse import parse_pdf
from atkv.ingest.ris import LAWS, RisClient
from atkv.ingest.wko import load_manifest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def corpus_chunks():
    """Every indexed chunk. Requires `make ingest` to have run once (cached, offline after)."""
    docs, _ = load_manifest(ROOT / "manifests/sources.yaml")
    chunks = []
    for d in docs:
        pdf = ROOT / "data/raw/wko" / f"{d.doc_id}.pdf"
        if not pdf.exists():
            pytest.skip(f"{pdf} missing -- run the ingest first")
        chunks += chunk_document(d, parse_pdf(pdf))

    rc = RisClient(ROOT / "data/raw/ris")
    try:
        for law in LAWS.values():
            chunks += rc.chunks_for_law(law, date(2026, 7, 1))
    finally:
        rc.close()
    return chunks


@pytest.fixture(scope="session")
def questions():
    import yaml
    return yaml.safe_load((ROOT / "evals/questions.yaml").read_text())["questions"]
