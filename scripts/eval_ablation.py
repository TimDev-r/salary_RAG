"""Ablation: source-stratified pooling and query translation.

Both fixes target the SAME measured failure -- KV chunks crowding out law
chunks -- from different directions:

  stratified   builds the candidate pool from each source_type separately, so
               law chunks cannot be crowded out of it by the KV at all. Only
               pool COMPOSITION changes; the reranker still decides the order.

  translate    gives the lexical retriever a German form of an English query.
               Without it BM25 scores the correct German chunk at exactly
               zero, because it shares no token with the English question.

Measured separately and together, so a gain can be attributed.
"""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import yaml

from atkv.ingest.chunk import chunk_document
from atkv.ingest.parse import parse_pdf
from atkv.ingest.ris import LAWS, RisClient
from atkv.ingest.wko import load_manifest
from atkv.retrieve.dense import DenseIndex
from atkv.retrieve.embed import SMALL, Embedder
from atkv.retrieve.fuse import cap_sections, rrf
from atkv.retrieve.lexical import LexicalIndex
from atkv.retrieve.rerank import Reranker
from atkv.retrieve.translate import QueryTranslator

ROOT = Path(__file__).resolve().parent.parent
KS = (1, 3, 5, 10)
POOL = 50
SOURCES = ("kv", "law")


def build_chunks():
    docs, _ = load_manifest(ROOT / "manifests/sources.yaml")
    chunks = [c for d in docs
              for c in chunk_document(d, parse_pdf(ROOT / "data/raw/wko" / f"{d.doc_id}.pdf"))]
    rc = RisClient(ROOT / "data/raw/ris")
    try:
        for law in LAWS.values():
            chunks += rc.chunks_for_law(law, date(2026, 7, 1))
    finally:
        rc.close()
    return chunks


def is_hit(c, es):
    return (c.short_title == es["short_title"] and c.section_ref == es["section_ref"]
            and (not es.get("doc_id") or c.doc_id == es["doc_id"]))


def make_pool(dense, lex, qv, qtext, as_of, *, stratified: bool):
    if not stratified:
        d = dense.search(qv, k=POOL, as_of=as_of)
        l = lex.search(qtext, k=POOL, as_of=as_of)
        return [r.chunk for r in rrf(d, l, k=POOL)]

    # Equal budget per source. The KV is 4 near-duplicate documents and the
    # laws are 2; on raw similarity the KV simply has more chances to score
    # well, which is a property of the corpus mix, not of relevance.
    per = POOL // len(SOURCES)
    out, seen = [], set()
    for st in SOURCES:
        d = dense.search(qv, k=per, as_of=as_of, source_type=st)
        l = lex.search(qtext, k=per, as_of=as_of, source_type=st)
        for r in rrf(d, l, k=per):
            if r.chunk.chunk_id not in seen:
                seen.add(r.chunk.chunk_id)
                out.append(r.chunk)
    return out


def run(dense, lex, emb, rr, tr, questions, *, stratified: bool, translate: bool):
    ans = [q for q in questions if q["type"] == "answerable"]
    qv = emb.encode_queries([q["question"] for q in ans])
    hits = {k: 0 for k in KS}
    tags: dict[str, dict] = {}
    ceil = 0
    misses = []
    t0 = time.perf_counter()

    for q, v in zip(ans, qv):
        as_of = date(q["valid_year"], 7, 1)
        qtext = tr.expand(q["question"], q["lang"]) if translate else q["question"]
        pool = make_pool(dense, lex, v, qtext, as_of, stratified=stratified)
        ceil += any(is_hit(c, q["expect_source"]) for c in pool)
        res = [h.chunk for h in cap_sections(rr.rerank(q["question"], pool, k=POOL), max(KS))]
        ranks = [i for i, c in enumerate(res, 1) if is_hit(c, q["expect_source"])]
        best = ranks[0] if ranks else None
        for k in KS:
            ok = best is not None and best <= k
            hits[k] += ok
            for t in q["tests"]:
                tags.setdefault(t, {kk: 0 for kk in KS} | {"n": 0})
                if k == KS[0]:
                    tags[t]["n"] += 1
                tags[t][k] += ok
        if best is None or best > 5:
            misses.append((q["id"], best))
    n = len(ans)
    return ({k: hits[k] / n for k in KS}, tags, ceil / n,
            (time.perf_counter() - t0) / n * 1000, misses)


def main():
    chunks = build_chunks()
    questions = yaml.safe_load((ROOT / "evals/questions.yaml").read_text())["questions"]
    emb = Embedder(SMALL)
    dense = DenseIndex(chunks, emb.encode_passages([c.text for c in chunks]), SMALL)
    lex = LexicalIndex(chunks)
    # CPU deliberately. Holding the bi-encoder, the cross-encoder and the
    # translation model on MPS at once segfaulted on this 8 GB machine with
    # 11 GB of swap already in use. Slower and reproducible beats fast and
    # crashing -- and this is an offline measurement, not the serving path.
    import os
    dev = os.environ.get("ATKV_DEVICE", "cpu")
    rr = Reranker(device=dev)
    tr = QueryTranslator(device="cpu")
    print(f"corpus {len(chunks)} chunks | pool {POOL} | reranker on {rr.device}\n")

    configs = [("baseline", False, False), ("+stratified", True, False),
               ("+translate", False, True), ("+both", True, True)]
    out = {}
    print(f"{'config':<14}" + "".join(f"{'R@'+str(k):>8}" for k in KS) + f"{'ceiling':>9}{'ms/query':>11}")
    print("-" * 64)
    for name, st, tl in configs:
        r, tg, ceil, lat, ms = run(dense, lex, emb, rr, tr, questions, stratified=st, translate=tl)
        out[name] = (r, tg, ms)
        print(f"{name:<14}" + "".join(f"{r[k]:>8.2f}" for k in KS) + f"{ceil:>9.2f}{lat:>11.0f}")

    print(f"\n=== by category (R@5) ===")
    print(f"{'tag':<16}{'n':>4}" + "".join(f"{c[0]:>13}" for c in configs))
    print("-" * 72)
    for tag in sorted(out["baseline"][1]):
        n = out["baseline"][1][tag]["n"]
        print(f"{tag:<16}{n:>4}" + "".join(f"{out[c[0]][1][tag][5]/n:>13.2f}" for c in configs))

    for name, _, _ in configs:
        ms = out[name][2]
        print(f"\n{name} misses: {[(q, r) for q, r in ms] or 'none'}")


if __name__ == "__main__":
    main()
