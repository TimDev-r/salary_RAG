"""Cross-encoder reranking of a candidate pool.

WHY A CROSS-ENCODER CAN DO WHAT NEITHER RETRIEVER COULD
-------------------------------------------------------
Both earlier retrievers score a query and a document SEPARATELY and compare the
results. A bi-encoder embeds them into one space; BM25 counts shared terms.
Neither ever looks at the pair together, and that limitation produced both
measured failures:

  cross-lingual   An English query and a German passage share no tokens, so
                  BM25 scores zero, and the bi-encoder's same-language
                  preference put English KV chunks above the German statute
                  that actually answers the question.

  sibling paras   AZG § 3 states the rule once -- "Die tägliche
                  Normalarbeitszeit darf acht Stunden nicht überschreiten" --
                  while §§ 4, 4a, 4b elaborate on derogations and mention
                  "Normalarbeitszeit" three times as often. BM25's term
                  frequency therefore ranks the correct answer LAST, and the
                  embedding space puts all five within 0.007 of each other.
                  In legal text the defining provision is routinely the
                  shortest and least repetitive one.

A cross-encoder reads query and passage jointly and scores relevance directly,
so it can recognise that a terse sentence ANSWERS a question while a long
passage merely discusses the topic. That is exactly the distinction both
failures turn on.

THE COST, WHICH IS NOT SMALL
----------------------------
It runs the model once per candidate instead of once per query. Reranking 20
candidates is 20 forward passes at query time, where the bi-encoder does one.
That is why it sits behind a flag and why the eval reports latency next to
quality: on an 8 GB laptop this is the difference between a fast answer and a
slow one, and the brief asks for the trade-off to be measured, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from atkv.models import Chunk
from atkv.retrieve.embed import pick_device

# Multilingual, trained on mMARCO in 14 languages including German. ~470 MB,
# chosen to fit alongside the bi-encoder and a generation model on 8 GB.
DEFAULT_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


@dataclass
class RerankedHit:
    chunk: Chunk
    score: float
    prior_rank: int


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER, device: str | None = None,
                 batch_size: int = 16) -> None:
        self.model_name = model_name
        self.device = pick_device(device)
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=self.device, max_length=512)

    def rerank(self, query: str, chunks: list[Chunk], k: int = 10) -> list[RerankedHit]:
        if not chunks:
            return []
        scores = self.model.predict(
            [(query, c.text) for c in chunks],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        order = sorted(range(len(chunks)), key=lambda i: -float(scores[i]))
        return [RerankedHit(chunks[i], float(scores[i]), prior_rank=i + 1) for i in order[:k]]
