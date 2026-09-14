"""Dense retrieval quality on the frozen eval set.

Reports the two comparisons the brief asks for, plus query latency:
  * retrieval@k with the version filter vs without it
  * e5-large vs e5-small

A "hit" means a retrieved chunk whose (short_title, section_ref) matches the
question's expect_source -- and its doc_id too, where the question pins one.
Ground truth never references chunk_id, which changes whenever chunking does.

NOTE ON LANGUAGE: retrieval is NOT restricted to the question's language. The
cross-lingual cases ask in English about AZG/UrlG, which exist only in German;
a language filter would make them unanswerable by construction.
"""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from atkv.ingest.chunk import chunk_document
from atkv.ingest.parse import parse_pdf
from atkv.ingest.ris import LAWS, RisClient
from atkv.ingest.wko import load_manifest
from atkv.retrieve.dense import DenseIndex
from atkv.retrieve.embed import LARGE, SMALL, Embedder

ROOT = Path(__file__).resolve().parent.parent
KS = (1, 3, 5, 10)


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


def evaluate(index, embedder, questions, *, filtered: bool):
    answerable = [q for q in questions if q["type"] == "answerable"]
    qvecs = embedder.encode_queries([q["question"] for q in answerable])

    hits = {k: 0 for k in KS}
    per_tag: dict[str, dict[str, int]] = {}
    misses = []

    for q, qv in zip(answerable, qvecs):
        as_of = date(q["valid_year"], 7, 1) if filtered else None
        res = index.search(qv, k=max(KS), as_of=as_of)
        ranks = [i for i, h in enumerate(res, 1) if is_hit(h.chunk, q["expect_source"])]
        best = ranks[0] if ranks else None
        for k in KS:
            ok = best is not None and best <= k
            hits[k] += ok
            for tag in q["tests"]:
                per_tag.setdefault(tag, {kk: 0 for kk in KS} | {"n": 0})
                if k == KS[0]:
                    per_tag[tag]["n"] += 1
                per_tag[tag][k] += ok
        if best is None or best > 5:
            # Show short_title AND lang, not just section_ref. Both the KV and
            # the AZG have a "§ 4", so a bare ref cannot tell you whether
            # retrieval found the wrong paragraph of the right law or the
            # wrong law entirely -- which are completely different bugs.
            misses.append((q["id"], best,
                           [f"{h.chunk.short_title}/{h.chunk.section_ref}/{h.chunk.lang}" for h in res[:3]]))

    n = len(answerable)
    return {k: hits[k] / n for k in KS}, per_tag, misses, n


def main():
    chunks = build_chunks()
    questions = yaml.safe_load((ROOT / "evals/questions.yaml").read_text())["questions"]
    print(f"corpus {len(chunks)} chunks | {sum(1 for q in questions if q['type']=='answerable')} answerable questions\n")

    results = {}
    for label, model in [("e5-small", SMALL), ("e5-large", LARGE)]:
        emb = Embedder(model, batch_size=16)
        t0 = time.perf_counter()
        vecs = emb.encode_passages([c.text for c in chunks])
        build_s = time.perf_counter() - t0
        index = DenseIndex(chunks, vecs, model)

        # single-query latency = what the API pays per request
        emb.encode_query("warmup")
        t0 = time.perf_counter()
        for _ in range(10):
            emb.encode_query("Wie hoch ist das Mindestgrundgehalt für ST1?")
        q_ms = (time.perf_counter() - t0) / 10 * 1000

        for filt in (False, True):
            r, tags, misses, n = evaluate(index, emb, questions, filtered=filt)
            results[(label, filt)] = (r, tags, misses)
        results[(label, "meta")] = (build_s, q_ms, emb.dim, index)
        index.save(ROOT / "data/index" / label)

    print(f"{'model':<10}{'filter':<9}" + "".join(f"{'R@'+str(k):>8}" for k in KS))
    print("-" * 51)
    for label in ("e5-small", "e5-large"):
        for filt in (False, True):
            r = results[(label, filt)][0]
            print(f"{label:<10}{('ON' if filt else 'OFF'):<9}" + "".join(f"{r[k]:>8.2f}" for k in KS))

    print(f"\n{'model':<10}{'dim':>6}{'index build':>14}{'query latency':>16}")
    print("-" * 46)
    for label in ("e5-small", "e5-large"):
        b, q, d, _ = results[(label, "meta")]
        print(f"{label:<10}{d:>6}{b:>12.1f}s{q:>14.1f}ms")

    print("\n=== by test category (filter ON, R@5) ===")
    print(f"{'tag':<16}{'n':>4}{'e5-small':>11}{'e5-large':>11}")
    print("-" * 42)
    tags_s = results[("e5-small", True)][1]
    tags_l = results[("e5-large", True)][1]
    for tag in sorted(tags_s):
        n = tags_s[tag]["n"]
        print(f"{tag:<16}{n:>4}{tags_s[tag][5]/n:>11.2f}{tags_l[tag][5]/n:>11.2f}")

    for label in ("e5-small", "e5-large"):
        misses = results[(label, True)][2]
        if misses:
            print(f"\n=== {label} misses (not in top-5, filter ON) ===")
            for qid, rank, got in misses:
                print(f"  {qid:<32} rank={rank}  top3_refs={got}")


if __name__ == "__main__":
    main()
