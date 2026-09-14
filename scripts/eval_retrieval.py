"""Compare dense, lexical and hybrid retrieval on the frozen eval set.

Each retriever is measured ALONE before the combination, so a gain can be
attributed rather than guessed at. The prediction from step 6 was specific:
BM25 should fix the German sibling-paragraph case, where the query shares exact
tokens with the target, and should do nothing for the cross-lingual cases,
where an English query shares no tokens with "Urlaubsausmaß".
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

ROOT = Path(__file__).resolve().parent.parent
KS = (1, 3, 5, 10)
POOL = 50  # measured: 50 beats both 20 (lower ceiling) and 100 (more distractors)


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


def is_hit(chunk, es) -> bool:
    return (chunk.short_title == es["short_title"]
            and chunk.section_ref == es["section_ref"]
            and (not es.get("doc_id") or chunk.doc_id == es["doc_id"]))


def run(mode, dense, lex, emb, questions, filtered=True, reranker=None):
    answerable = [q for q in questions if q["type"] == "answerable"]
    qvecs = emb.encode_queries([q["question"] for q in answerable])
    hits = {k: 0 for k in KS}
    tags: dict[str, dict] = {}
    misses = []
    t0 = time.perf_counter()

    for q, qv in zip(answerable, qvecs):
        as_of = date(q["valid_year"], 7, 1) if filtered else None
        if mode == "dense":
            res = [h.chunk for h in dense.search(qv, k=max(KS), as_of=as_of)]
        elif mode == "lexical":
            res = [h.chunk for h in lex.search(q["question"], k=max(KS), as_of=as_of)]
        else:
            d = dense.search(qv, k=POOL, as_of=as_of)
            l = lex.search(q["question"], k=POOL, as_of=as_of)
            if reranker:
                # Rerank the UNCAPPED fused pool: a candidate the retrievers
                # ranked 40th is exactly the kind the cross-encoder exists to
                # rescue, and it cannot rescue what it never sees. The
                # diversity cap is applied afterwards, to what is shown.
                fused = rrf(d, l, k=POOL)
                rer = reranker.rerank(q["question"], [r.chunk for r in fused], k=POOL)
                res = [h.chunk for h in cap_sections(rer, max(KS))]
            else:
                res = [r.chunk for r in cap_sections(rrf(d, l, k=POOL), max(KS))]

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
            misses.append((q["id"], best,
                           [f"{c.short_title}/{c.section_ref}/{c.lang}" for c in res[:3]]))

    n = len(answerable)
    return {k: hits[k] / n for k in KS}, tags, misses, (time.perf_counter() - t0) / n * 1000


def main():
    chunks = build_chunks()
    questions = yaml.safe_load((ROOT / "evals/questions.yaml").read_text())["questions"]
    emb = Embedder(SMALL)
    dense = DenseIndex(chunks, emb.encode_passages([c.text for c in chunks]), SMALL)
    t0 = time.perf_counter()
    lex = LexicalIndex(chunks)
    lex_build = time.perf_counter() - t0
    print(f"corpus {len(chunks)} chunks | BM25 index built in {lex_build:.1f}s "
          f"| vocab {len(lex.analyzer.vocab)}, nouns {len(lex.analyzer.nouns)}\n")

    out = {}
    print(f"{'retriever':<12}{'filter':<9}" + "".join(f"{'R@'+str(k):>8}" for k in KS) + f"{'ms/query':>11}")
    print("-" * 62)
    for mode in ("dense", "lexical", "hybrid"):
        for filt in ((False, True) if mode != "lexical" else (True,)):
            r, tg, ms, lat = run(mode, dense, lex, emb, questions, filtered=filt)
            out[(mode, filt)] = (r, tg, ms)
            print(f"{mode:<12}{('ON' if filt else 'OFF'):<9}"
                  + "".join(f"{r[k]:>8.2f}" for k in KS) + f"{lat:>11.1f}")

    rr = Reranker()
    r, tg, ms, lat = run("hybrid", dense, lex, emb, questions, filtered=True, reranker=rr)
    out[("hybrid+rerank", True)] = (r, tg, ms)
    print(f"{'hyb+rerank':<12}{'ON':<9}" + "".join(f"{r[k]:>8.2f}" for k in KS) + f"{lat:>11.1f}")

    print(f"\n=== by category (filter ON, R@5) ===")
    print(f"{'tag':<16}{'n':>4}{'dense':>9}{'lexical':>10}{'hybrid':>9}{'+rerank':>11}")
    print("-" * 59)
    d, l, h = out[("dense", True)][1], out[("lexical", True)][1], out[("hybrid", True)][1]
    rk = out[("hybrid+rerank", True)][1]
    for tag in sorted(d):
        n = d[tag]["n"]
        print(f"{tag:<16}{n:>4}{d[tag][5]/n:>9.2f}{l[tag][5]/n:>10.2f}{h[tag][5]/n:>9.2f}{rk[tag][5]/n:>11.2f}")

    for mode in ("hybrid", "hybrid+rerank"):
        ms = out[(mode, True)][2]
        if ms:
            print(f"\n=== {mode} misses (filter ON, not in top-5) ===")
            for qid, rank, got in ms:
                print(f"  {qid:<32} rank={rank}  top3={got}")


if __name__ == "__main__":
    main()
