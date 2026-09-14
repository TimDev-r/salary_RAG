"""Dense embedding provider.

THE E5 PREFIX RULE -- the detail that silently halves quality
-------------------------------------------------------------
The multilingual-e5 models were trained with asymmetric prefixes:

    "query: <the user's question>"
    "passage: <the indexed text>"

They are not decoration. The model learned a query space and a passage space
and the prefix is what selects between them. Drop them, or use the same prefix
for both, and retrieval still WORKS -- it returns plausible neighbours, no
error, no warning -- it is just materially worse. This is the single easiest
way to lose quality in an e5 pipeline and one of the hardest to notice, because
nothing about the output looks broken.

So the prefix is applied here, in one place, and the two directions have
separate methods that cannot be confused for one another.

MODEL CHOICE IS A MEASUREMENT, NOT A PREFERENCE
-----------------------------------------------
Two models from the same family, differing essentially only in size, so a
quality difference is attributable to size and not to training recipe:

    intfloat/multilingual-e5-small   118M params,  384 dim
    intfloat/multilingual-e5-large   560M params, 1024 dim

Anything from a different family (bge-m3 is the obvious candidate and is
excellent at German) would differ in recipe AND size AND architecture, and a
measured difference could not be attributed to any of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

SMALL = "intfloat/multilingual-e5-small"
LARGE = "intfloat/multilingual-e5-large"


def pick_device(requested: str | None = None) -> str:
    """'mps' on Apple silicon when available, else 'cpu'.

    MPS is not free memory: it shares the machine's unified memory with
    everything else. On an 8 GB laptop already swapping, the GPU can lose to
    the CPU. That is why this is measurable rather than assumed -- see
    scripts/bench_embed.py.
    """
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class EmbedStats:
    model: str
    device: str
    dim: int
    n_texts: int
    seconds: float

    @property
    def per_second(self) -> float:
        return self.n_texts / self.seconds if self.seconds else float("nan")


class Embedder:
    def __init__(self, model_name: str = SMALL, device: str | None = None,
                 batch_size: int = 16, normalize: bool = True) -> None:
        self.model_name = model_name
        self.device = pick_device(device)
        self.batch_size = batch_size
        # Normalised vectors mean inner product IS cosine similarity, so FAISS
        # can use its fastest index (IndexFlatIP) with no accuracy loss.
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=self.device)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.model.encode(
            [f"{prefix}{t}" for t in texts],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        ).astype(np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """For text going INTO the index."""
        return self._encode(texts, "passage: ")

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """For text coming FROM a user. Never use this on corpus text."""
        return self._encode(texts, "query: ")

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_queries([text])[0]
