"""BM25 with German-aware analysis.

WHY LEXICAL SEARCH AT ALL, WHEN WE HAVE EMBEDDINGS
--------------------------------------------------
Step 6 measured a specific dense failure: asked "Wie lang darf die tägliche
Normalarbeitszeit nach dem Arbeitszeitgesetz sein?", the index returned
AZG §§ 4a, 4, 4, 4, 5 -- the right law, the wrong paragraph -- with all five
candidates inside a 0.007 score band. The embedding space does not separate
sibling paragraphs that are all about Normalarbeitszeit.

But AZG § 3 contains the literal string "Die tägliche Normalarbeitszeit darf
acht Stunden" and its siblings do not. Exact lexical overlap is precisely the
discrimination the dense retriever is failing at. This is not a fallback for
when embeddings are unavailable; it is a different signal that is strong
exactly where the other is weak.

THE COMPOUND PROBLEM
--------------------
German glues words together. A user asks about "Gehalt"; the document says
"Mindestgrundgehalt". A user asks about "Arbeitszeit"; the statute says
"Normalarbeitszeit". Plain BM25 sees no shared token at all and scores zero --
the failure is total, not partial, which is what makes it worth fixing.

Snowball stemming alone does NOT solve this. It normalises inflection
(Gehälter -> Gehalt, Werktage -> Werktag) but leaves compounds whole:
Normalarbeitszeit stems to Normalarbeitszeit.

WHY THE SPLITTER IS CORPUS-DRIVEN
---------------------------------
A compound is split only into parts that occur as standalone words ELSEWHERE
IN THIS CORPUS. No dictionary is downloaded (the project must cost nothing and
run offline), and more usefully, the split is domain-adapted: it produces the
parts this corpus actually uses, which is what a query can match. The original
token is always kept as well, so an exact-match query never loses.

The risk is oversplitting into parts that happen to exist but are not
morphemes. Guarded by a minimum part length and a preference for the fewest,
longest parts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import numpy as np
import snowballstemmer
from rank_bm25 import BM25Okapi

from atkv.models import Chunk

WORD = re.compile(r"[A-Za-zÄÖÜäöüß0-9]+(?:[.,]\d+)?")

# Linking morphemes ("Fugenelemente"): Urlaub+s+ausmaß, Arbeit+s+zeit.
LINKERS = ("", "s", "es", "n", "en", "er")

MIN_PART = 4        # shorter "parts" are usually coincidence, not morphemes
MIN_COMPOUND = 11   # below this, splitting costs more precision than it buys
MAX_PARTS = 3


@dataclass
class LexicalHit:
    chunk: Chunk
    score: float


class GermanAnalyzer:
    def __init__(self, corpus_texts: list[str]) -> None:
        self._stem_de = snowballstemmer.stemmer("german").stemWord
        self._stem_en = snowballstemmer.stemmer("english").stemWord
        # Vocabulary of plausible compound PARTS, drawn from the corpus itself.
        counts: dict[str, int] = {}
        upper: dict[str, int] = {}
        for t in corpus_texts:
            for w in WORD.findall(t):
                lw = w.lower()
                if len(lw) >= MIN_PART and lw.isalpha():
                    counts[lw] = counts.get(lw, 0) + 1
                    if w[0].isupper():
                        upper[lw] = upper.get(lw, 0) + 1
        # A part must be a reasonably common standalone word; a hapax is more
        # likely to be another compound than a morpheme.
        self.vocab = {w for w, n in counts.items() if n >= 2}

        # German nouns are ALWAYS capitalised; verbs and adjectives only at the
        # start of a sentence. So a word capitalised in most of its occurrences
        # is a noun, and that is a usable part-of-speech signal for free.
        #
        # It matters because compounds are right-headed and the head is a noun.
        # Without this check "lehrlingseinkommen" fell back to the suffix
        # "kommen" -- the verb "to come" -- because "einkommen" never occurs
        # standalone here. Matching queries on "kommen" is worse than not
        # splitting at all.
        self.nouns = {w for w, n in counts.items() if upper.get(w, 0) * 2 > n}

    @lru_cache(maxsize=50_000)
    def decompound(self, word: str) -> tuple[str, ...]:
        """'normalarbeitszeit' -> ('normal', 'arbeitszeit'). () if not splittable."""
        if len(word) < MIN_COMPOUND or not word.isalpha():
            return ()

        def split(rest: str, depth: int, allow_whole: bool = True) -> list[str] | None:
            if depth > MAX_PARTS:
                return None
            # The whole-word check must NOT fire on the entry call. Every
            # compound in the corpus is itself a corpus word, so accepting
            # `rest` whole at depth 1 returns a single part and the recursion
            # never runs -- which silently disabled decompounding entirely.
            if allow_whole and rest in self.vocab:
                return [rest]
            # Longest head first: prefer ('normal','arbeitszeit') over
            # ('normal','arbeit','zeit') -- fewer, longer, more meaningful parts.
            for cut in range(len(rest) - MIN_PART, MIN_PART - 1, -1):
                head = rest[:cut]
                if head not in self.vocab:
                    continue
                for link in LINKERS:
                    tail = rest[cut:]
                    if link and tail.startswith(link):
                        tail = tail[len(link):]
                    elif link:
                        continue
                    if len(tail) < MIN_PART:
                        continue
                    got = split(tail, depth + 1)
                    if got:
                        return [head, *got]
            return None

        parts = split(word, 1, allow_whole=False)
        if parts and len(parts) > 1:
            return tuple(parts)

        # Fallback: the longest known SUFFIX.
        #
        # German compounds are right-headed -- a Mindestgrundgehalt is a kind
        # of Gehalt, a Lehrlingseinkommen is a kind of Einkommen. So the final
        # element carries the meaning a query is most likely to name. Several
        # modifiers here ("mindest", "lehrling") never occur standalone in this
        # corpus, so full decomposition is impossible, but the head still is.
        for cut in range(MIN_PART, len(word) - MIN_PART + 1):
            suffix = word[cut:]
            if suffix in self.nouns and len(suffix) >= MIN_PART:
                return (suffix,)
        return ()

    def analyze(self, text: str, lang: str = "de") -> list[str]:
        stem = self._stem_de if lang == "de" else self._stem_en
        out: list[str] = []
        for raw in WORD.findall(text.lower()):
            # Hyphenated forms first: "Ist-Gehaltserhöhung" -> both halves.
            pieces = [raw]
            if "-" in raw:
                pieces += [p for p in raw.split("-") if len(p) >= MIN_PART]
            for p in pieces:
                out.append(stem(p))          # always keep the whole token
                for part in self.decompound(p):
                    out.append(stem(part))   # plus its morphemes
        return out


class LexicalIndex:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.analyzer = GermanAnalyzer([c.text for c in chunks])
        # Analyse each chunk in ITS OWN language, so German chunks get German
        # stemming and English chunks get English. Using one stemmer for both
        # mangles the other language's morphology.
        self.bm25 = BM25Okapi([self.analyzer.analyze(c.text, c.lang) for c in chunks])

    def search(self, query: str, k: int = 8, *, as_of: date | None = None,
               tenant_id: str = "public", lang: str | None = None) -> list[LexicalHit]:
        # A query is analysed under BOTH stemmers and the results unioned. We
        # cannot know the language of the chunk we are looking for -- the whole
        # point of the cross-lingual cases is that an English question must
        # reach a German document.
        terms = list(dict.fromkeys(
            self.analyzer.analyze(query, "de") + self.analyzer.analyze(query, "en")
        ))
        scores = np.asarray(self.bm25.get_scores(terms), dtype=np.float64)

        # PRE-ranking, same discipline as the dense index: disallowed chunks are
        # removed from contention BEFORE top-k is taken, not filtered out of the
        # results afterwards.
        if as_of is not None:
            mask = np.array([
                c.in_force_on(as_of) and c.tenant_id == tenant_id
                and (lang is None or c.lang == lang)
                for c in self.chunks
            ])
            scores = np.where(mask, scores, -np.inf)

        order = np.argsort(-scores)[:k]
        return [LexicalHit(self.chunks[i], float(scores[i]))
                for i in order if np.isfinite(scores[i]) and scores[i] > 0]
