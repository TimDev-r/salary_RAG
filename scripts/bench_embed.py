"""Measure embedding throughput: model size x device, on the real corpus.

Not a synthetic benchmark. The texts are the actual chunks, whose length
distribution is what will be embedded in practice.
"""
from __future__ import annotations

import gc
import time
from datetime import date
from pathlib import Path

import torch

from atkv.ingest.chunk import chunk_document
from atkv.ingest.parse import parse_pdf
from atkv.ingest.ris import LAWS, RisClient
from atkv.ingest.wko import load_manifest
from atkv.retrieve.embed import LARGE, SMALL, Embedder

ROOT = Path(__file__).resolve().parent.parent


def build_corpus() -> list[str]:
    docs, _ = load_manifest(ROOT / "manifests/sources.yaml")
    chunks = [c for d in docs
              for c in chunk_document(d, parse_pdf(ROOT / "data/raw/wko" / f"{d.doc_id}.pdf"))]
    rc = RisClient(ROOT / "data/raw/ris")
    try:
        for law in LAWS.values():
            chunks += rc.chunks_for_law(law, date(2026, 7, 1))
    finally:
        rc.close()
    return [c.text for c in chunks]


def bench(texts: list[str], model: str, device: str) -> tuple[float, int]:
    emb = Embedder(model, device=device, batch_size=16)
    emb.encode_passages(texts[:8])           # warm up: first call pays lazy init + kernel compile
    t0 = time.perf_counter()
    vecs = emb.encode_passages(texts)
    dt = time.perf_counter() - t0
    dim = vecs.shape[1]
    del emb, vecs
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return dt, dim


if __name__ == "__main__":
    texts = build_corpus()
    chars = sum(len(t) for t in texts)
    print(f"corpus: {len(texts)} chunks, {chars:,} chars, mean {chars//len(texts)} chars/chunk\n")
    print(f"{'model':<8}{'device':<8}{'dim':>6}{'seconds':>10}{'chunks/s':>11}")
    print("-" * 43)
    for name, model in [("small", SMALL), ("large", LARGE)]:
        for device in ["cpu", "mps"]:
            try:
                dt, dim = bench(texts, model, device)
                print(f"{name:<8}{device:<8}{dim:>6}{dt:>10.1f}{len(texts)/dt:>11.1f}")
            except Exception as e:
                print(f"{name:<8}{device:<8}{'-':>6}{'FAILED':>10}  {type(e).__name__}: {str(e)[:60]}")
