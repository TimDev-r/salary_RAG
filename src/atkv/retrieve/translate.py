"""Query-side EN->DE translation, for the lexical retriever only.

THE PROBLEM THIS EXISTS FOR
---------------------------
BM25 matches tokens. An English question -- "How many working days of paid
annual leave am I entitled to?" -- shares no token at all with the German
statute that answers it, which says "Urlaubsausmaß" and "Werktage". The score
is not merely low, it is zero: the correct chunk is not a weak candidate, it is
not a candidate.

Decompounding cannot help, because the compound parts are German too. The only
way a lexical retriever can reach a German document from an English query is if
one side is translated.

WHY THE QUERY AND NOT THE CORPUS
--------------------------------
Translating the corpus would mean embedding and indexing a machine translation
of every chunk, doubling the index, and -- worse -- creating citable text that
no source document contains. This project cites paragraphs; a citation must
point at words the publisher actually wrote. Translating the query leaves the
corpus untouched: the translation is a retrieval aid that never reaches the
user or the citation.

WHY A LOCAL MARIAN MODEL
------------------------
Helsinki-NLP/opus-mt-en-de is ~300 MB, runs offline and costs nothing, which
the brief requires. A hosted translation API would be faster and better and is
not available to us at EUR 0.

The translated query is used ALONGSIDE the original, never instead of it. A
mistranslation should cost us nothing: the original terms are still searched.
"""

from __future__ import annotations

from functools import lru_cache

from atkv.retrieve.embed import pick_device

MODEL = "Helsinki-NLP/opus-mt-en-de"


class QueryTranslator:
    def __init__(self, model_name: str = MODEL, device: str | None = None) -> None:
        from transformers import MarianMTModel, MarianTokenizer

        self.model_name = model_name
        # Marian is tiny and MPS setup cost per call exceeds the compute;
        # CPU is measurably the right choice for single short queries.
        self.device = device or "cpu"
        self.tok = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @lru_cache(maxsize=1024)
    def translate(self, text: str) -> str:
        import torch

        batch = self.tok([text], return_tensors="pt", padding=True, truncation=True,
                         max_length=256).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**batch, max_new_tokens=128, num_beams=1)
        return self.tok.decode(out[0], skip_special_tokens=True)

    def expand(self, query: str, lang: str) -> str:
        """Original + German translation, concatenated.

        Concatenated rather than substituted so a bad translation cannot lose
        us a match the original would have found. BM25 simply sees more terms.
        """
        if lang == "de":
            return query
        return f"{query} {self.translate(query)}"
