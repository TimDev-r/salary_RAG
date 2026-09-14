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

    @abstractmethod
    def available(self) -> bool:
        """Whether this provider can actually serve right now.

        Checked at request time, not construction time. A local model server
        can be running when the process starts and gone ten minutes later, and
        the routing rule needs to find that out before it commits an answer to
        a user rather than after.
        """


def build_context(chunks: list[Chunk], max_chars: int = 6000) -> str:
    """Render retrieved chunks for a prompt, each tagged with its citation.

    The tag is inside the context, not appended afterwards, so the model can
    attribute a specific sentence to a specific source instead of being handed
    a wall of text and a list of citations to pair up by guesswork.
    """
    out, total = [], 0
    for i, c in enumerate(chunks, 1):
        block = f"[{i}] {c.to_citation().render()} (chunk_id={c.chunk_id})\n{c.text}"
        if total + len(block) > max_chars:
            break
        out.append(block)
        total += len(block)
    return "\n\n".join(out)
