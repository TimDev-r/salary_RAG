"""A provider that does no generation at all.

It returns the top retrieved chunk verbatim with its citation. That is a real
answer to a large share of the questions this system gets -- the verbalised
salary cells already read as sentences ("ST1 / Erfahrungsstufe: 4.476 EUR
brutto pro Monat") because step 4 built them that way.

Three jobs:

  1. The eval baseline. Retrieval and citation accuracy can be measured with
     no language model in the loop, so a retrieval regression cannot be hidden
     by a fluent model papering over it.
  2. The fallback when no model server is reachable. Returning the source text
     with a correct citation is a worse answer than a fluent summary and a far
     better one than an error page or an invented figure.
  3. The second implementation the Stage 2 routing rule needs.

It cannot hallucinate, because it cannot write. Every word it emits came from
a source document.
"""

from __future__ import annotations

from atkv.generate.base import Generated, GenerationProvider
from atkv.models import Chunk


class ExtractiveProvider(GenerationProvider):
    name = "extractive"

    def available(self) -> bool:
        return True

    def generate(self, question: str, chunks: list[Chunk], lang: str) -> Generated:
        if not chunks:
            msg = ("Dazu findet sich in den vorliegenden Dokumenten keine Antwort."
                   if lang == "de" else
                   "The available documents do not answer this question.")
            return Generated(text=msg, used_chunk_ids=[], provider=self.name)

        top = chunks[0]
        return Generated(
            text=f"{top.text.strip()}\n\n{top.to_citation().render()}",
            used_chunk_ids=[top.chunk_id],
            provider=self.name,
        )
