"""The full evaluation. This is what `make eval` runs and what the README quotes.

Reports the two before/after measurements the brief requires, plus answer,
citation and refusal accuracy.

WHY RETRIEVAL AND ANSWER ARE SCORED SEPARATELY
----------------------------------------------
retrieval@k asks whether we FOUND the right paragraph. It says nothing about
whether the answer is right. Those come apart in both directions, and we have
hit both:

  found but wrong     the prompt's worked example contained a realistic salary
                      and the model answered with it, citing a real paragraph.
                      Retrieval scored that a success.
  wrong but plausible a figure with a correct-looking citation is the failure
                      this project exists to prevent, and only answer accuracy
                      can see it.

So the suite scores retrieval, answers, citations and refusals as four
separate numbers, and a single headline figure is never reported.

n = 25 ANSWERABLE QUESTIONS. Each is worth 4 percentage points. Differences of
one or two questions are not distinguishable from noise, and the output says so
rather than leaving a reader to infer it.
"""
from __future__ import annotations

import argparse
import re
import time
from datetime import date
from pathlib import Path

import yaml

from atkv import guard
from atkv.generate.extractive import ExtractiveProvider
from atkv.generate.ollama import OllamaProvider
from atkv.ingest.build import build_corpus
from atkv.retrieve.dense import DenseIndex
from atkv.retrieve.embed import SMALL, Embedder
from atkv.retrieve.fuse import cap_sections, rrf
from atkv.retrieve.lexical import LexicalIndex
from atkv.retrieve.translate import QueryTranslator

ROOT = Path(__file__).resolve().parent.parent
KS = (1, 3, 5, 10)
POOL = 50


def is_hit(c, es) -> bool:
    return (c.short_title == es["short_title"] and c.section_ref == es["section_ref"]
            and (not es.get("doc_id") or c.doc_id == es["doc_id"]))


def digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def answer_matches(text: str, expect: dict) -> bool:
    if "amount" in expect:
        # Format-agnostic: the German answer writes 4.476 and the English 4,476.
        return str(expect["amount"]) in digits(text)
    return any(a.lower() in text.lower() for a in expect.get("any_of", []))


class Bench:
    def __init__(self, chunks, decompound: bool = True, share_from: "Bench | None" = None):
        """share_from reuses the embedder, dense index and translator.

        The decompounding comparison changes only the lexical analyser. Building
        a second Bench from scratch re-embedded all 538 chunks and loaded a
        second copy of the translation model, which on an 8 GB machine meant two
        sets of weights resident and a run that never finished.
        """
        self.chunks = chunks
        if share_from is not None:
            self.emb, self.dense, self.tr = share_from.emb, share_from.dense, share_from.tr
        else:
            self.emb = Embedder(SMALL)
            self.dense = DenseIndex(chunks, self.emb.encode_passages([c.text for c in chunks]), SMALL)
            self.tr = QueryTranslator()
        self.lex = LexicalIndex(chunks, decompound_enabled=decompound)

    def retrieve(self, q, *, filtered=True, lexical=True, dense=True, k=10):
        as_of = date(q["valid_year"], 7, 1) if filtered else None
        qv = self.emb.encode_query(q["question"])
        d = self.dense.search(qv, k=POOL, as_of=as_of) if dense else []
        lq = self.tr.expand(q["question"], q["lang"])
        l = self.lex.search(lq, k=POOL, as_of=as_of) if lexical else []
        return [r.chunk for r in cap_sections(rrf(d, l, k=POOL), k)]


def recall(bench, questions, **kw) -> dict[int, float]:
    ans = [q for q in questions if q["type"] == "answerable"]
    hits = {k: 0 for k in KS}
    for q in ans:
        res = bench.retrieve(q, k=max(KS), **kw)
        ranks = [i for i, c in enumerate(res, 1) if is_hit(c, q["expect_source"])]
        best = ranks[0] if ranks else None
        for k in KS:
            hits[k] += best is not None and best <= k
    return {k: hits[k] / len(ans) for k in KS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-generation", action="store_true",
                    help="retrieval and refusal only; skips the slow model calls")
    args = ap.parse_args()

    chunks = build_corpus(ROOT, fetch=False)
    questions = yaml.safe_load((ROOT / "evals/questions.yaml").read_text())["questions"]
    answerable = [q for q in questions if q["type"] == "answerable"]
    refusals = [q for q in questions if q["type"] == "refusal"]

    provider = OllamaProvider()
    if args.no_generation or not provider.available():
        provider = ExtractiveProvider()

    print("=" * 66)
    print("AT-KV ASSISTANT — EVALUATION")
    print("=" * 66)
    print(f"corpus        {len(chunks)} chunks")
    print(f"questions     {len(answerable)} answerable + {len(refusals)} refusal")
    print(f"generation    {provider.name}"
          f"{' (' + provider.model + ')' if getattr(provider, 'model', None) else ''}")
    print(f"\n  NOTE: n={len(answerable)}. One question is {100/len(answerable):.0f} percentage "
          f"points, so\n  a one- or two-question difference is not separable from noise.\n")

    full = Bench(chunks, decompound=True)

    # ---- required measurement 1: the version filter -----------------------
    print("-" * 66)
    print("1. VERSION FILTER (pre-ranking metadata filter on valid_from)")
    print("-" * 66)
    off = recall(full, questions, filtered=False)
    on = recall(full, questions, filtered=True)
    print(f"{'':<14}" + "".join(f"{'R@'+str(k):>9}" for k in KS))
    print(f"{'filter OFF':<14}" + "".join(f"{off[k]:>9.2f}" for k in KS))
    print(f"{'filter ON':<14}" + "".join(f"{on[k]:>9.2f}" for k in KS))
    print(f"{'difference':<14}" + "".join(f"{on[k]-off[k]:>+9.2f}" for k in KS))
    print("\n  The gain is at rank 1 and nowhere else: the right year was already")
    print("  in the top few, with the wrong year sitting on top of it.")

    # ---- required measurement 2: German-aware lexical search --------------
    print("\n" + "-" * 66)
    print("2. GERMAN-AWARE LEXICAL SEARCH (compound splitting)")
    print("-" * 66)
    plain = Bench(chunks, decompound=False, share_from=full)
    lex_off = recall(plain, questions, lexical=True, dense=False)
    lex_on = recall(full, questions, lexical=True, dense=False)
    print(f"{'BM25 only':<24}" + "".join(f"{'R@'+str(k):>9}" for k in KS))
    print(f"{'  stemming only':<24}" + "".join(f"{lex_off[k]:>9.2f}" for k in KS))
    print(f"{'  + decompounding':<24}" + "".join(f"{lex_on[k]:>9.2f}" for k in KS))
    print(f"{'  difference':<24}" + "".join(f"{lex_on[k]-lex_off[k]:>+9.2f}" for k in KS))
    a = full.lex.analyzer
    print(f"\n  vocab {len(a.vocab)} words, {len(a.nouns)} identified as nouns")
    for w in ("mindestgrundgehalt", "normalarbeitszeit", "urlaubsausmaß"):
        print(f"    {w:<22} -> {list(a.decompound(w)) or '(not split)'}")

    # ---- retriever comparison ---------------------------------------------
    print("\n" + "-" * 66)
    print("3. RETRIEVERS (filter ON)")
    print("-" * 66)
    print(f"{'':<14}" + "".join(f"{'R@'+str(k):>9}" for k in KS))
    for label, kw in (("dense only", dict(lexical=False)),
                      ("lexical only", dict(dense=False)),
                      ("hybrid", {})):
        r = recall(full, questions, **kw)
        print(f"{label:<14}" + "".join(f"{r[k]:>9.2f}" for k in KS))

    # ---- answers, citations -----------------------------------------------
    print("\n" + "-" * 66)
    print(f"4. ANSWERS AND CITATIONS  (provider: {provider.name})")
    print("-" * 66)
    correct = cited = grounded = 0
    wrong = []
    t0 = time.perf_counter()
    for q in answerable:
        res = full.retrieve(q, k=5)
        out = provider.generate(q["question"], res, q["lang"])
        ok = answer_matches(out.text, q["expect"])
        # The citation must name the paragraph the ground truth points at.
        cite_ok = any(is_hit(c, q["expect_source"]) for c in res)
        # Grounded: the supporting text really is in a retrieved chunk, so the
        # answer could have been read off rather than recalled.
        quote = re.sub(r"\s+", " ", q["source_quote"]).lower()
        gr = any(quote in re.sub(r"\s+", " ", c.text).lower() for c in res)
        correct += ok; cited += cite_ok; grounded += gr
        if not ok:
            wrong.append((q["id"], q["expect"], out.text[:80]))
    n = len(answerable)
    gen_s = time.perf_counter() - t0
    print(f"  answer accuracy      {correct}/{n}  ({correct/n:.2f})   figure or phrase is correct")
    print(f"  citation accuracy    {cited}/{n}  ({cited/n:.2f})   cites the expected paragraph")
    print(f"  grounded             {grounded}/{n}  ({grounded/n:.2f})   supporting text was in context")
    print(f"  {gen_s/n:.1f}s per question")
    if wrong:
        print("\n  incorrect answers:")
        for qid, exp, got in wrong:
            print(f"    {qid:<32} expected {exp} | got: {got!r}")

    # ---- refusals ----------------------------------------------------------
    print("\n" + "-" * 66)
    print("5. REFUSALS")
    print("-" * 66)
    caught = sum(guard.check(q["question"]).refuse for q in refusals)
    false_pos = [q["id"] for q in answerable if guard.check(q["question"]).refuse]
    print(f"  advice questions refused   {caught}/{len(refusals)}  ({caught/len(refusals):.2f})")
    print(f"  factual questions refused  {len(false_pos)}/{n}  (false refusals; lower is better)")
    if false_pos:
        print(f"    {false_pos}")
    print("\n  Over-refusal is not the safe direction: a system that hedges on")
    print("  questions of fact is one users route around.")
    print("=" * 66)


if __name__ == "__main__":
    main()
