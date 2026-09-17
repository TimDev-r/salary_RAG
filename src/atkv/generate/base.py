"""Generation provider interface.

WHY AN INTERFACE BEFORE THERE ARE TWO IMPLEMENTATIONS
-----------------------------------------------------
Stage 2 requires a pluggable provider with at least two implementations and a
routing rule. Retrofitting that later means finding every place a model got
called and every assumption that leaked out of it. Defining it now costs one
small file.

It also makes the eval honest. With a provider that does no generation at all,
the suite measures RETRIEVAL and CITATION accuracy without a language model in
the loop -- so a regression in retrieval cannot be masked by a fluent model
covering for it, and a change of model cannot silently move retrieval numbers.

WHAT A PROVIDER MAY AND MAY NOT DO
----------------------------------
It is given a question and a list of retrieved chunks, and must answer ONLY
from those chunks. It does not retrieve, it does not decide refusals (that is
guard.py, which runs first and can stop the pipeline before a model is
loaded), and it must not supply facts of its own. A provider that answers from
its own knowledge produces text that looks identical to a cited answer and is
not one -- which is the failure this whole project is built to avoid.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

from atkv.models import Chunk


@dataclass
class Generated:
    text: str
    used_chunk_ids: list[str]
    provider: str
    model: str | None = None


class GenerationProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, question: str, chunks: list[Chunk], lang: str) -> Generated:
        ...

    def stream(self, question: str, chunks: list[Chunk], lang: str) -> Iterator[str]:
        """Yield the answer in pieces as it is produced.

        The default implementation yields the whole thing at once, so a
        provider that cannot stream still satisfies the interface and the
        endpoint does not need to know which kind it is talking to.

        Streaming does not make generation faster -- the same tokens are
        produced at the same rate. It changes WHEN the user sees the first
        one: about a second, instead of a blank screen for nine.
        """
        yield self.generate(question, chunks, lang).text

    @abstractmethod
    def available(self) -> bool:
        """Whether this provider can actually serve right now.

        Checked at request time, not construction time. A local model server
        can be running when the process starts and gone ten minutes later, and
        the routing rule needs to find that out before it commits an answer to
        a user rather than after.
        """


def build_context(chunks: list[Chunk], max_chars: int = 6000, lang: str = "de") -> str:
    """Render retrieved chunks for a prompt, each tagged with its citation.

    EXACTLY ONE BRACKETED FORM APPEARS IN THE CONTEXT.
    An earlier version numbered the passages "[1] [IT-KV 2026, ..., § 15]",
    and the model dutifully copied the nearest thing in brackets -- answering
    "4.476 EUR brutto pro Monat [1] [3]". The figure was right and the citation
    was useless. When two things look like citations, the model cannot be
    blamed for picking the wrong one; the context has to offer only the form
    we actually want back.

    chunk_id is deliberately omitted too: it is an internal identifier, it
    means nothing to a reader, and any token in the context is a token the
    model might emit.
    """
    label = "QUELLE" if lang == "de" else "SOURCE"
    out, total = [], 0
    for c in chunks:
        block = f"{label}: {c.to_citation().render(short=True)}\n{c.text}"
        if total + len(block) > max_chars:
            break
        out.append(block)
        total += len(block)
    return "\n\n---\n\n".join(out)
