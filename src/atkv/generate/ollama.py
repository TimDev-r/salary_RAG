"""Ollama-backed generation. Local, free, offline.

MODEL CHOICE
------------
qwen2.5:3b-instruct at 4-bit, ~2 GB. The reasoning from step 0 stands: the
generator is the LEAST important component here. Its job is to read a figure
off a paragraph we already handed it and attribute it. It does not need to be
clever; it needs to not invent. The scarce RAM on an 8 GB machine is better
spent on retrieval, which is what the evals actually measure.

THE PROMPT'S ONE JOB
--------------------
Make abstention preferable to invention. A model that answers "I don't know"
when the context is thin costs the user one failed query. A model that fills
the gap with a plausible salary figure and a real-looking citation costs them
a wrong number they have no reason to doubt -- and this system's entire value
proposition is that its numbers can be trusted.

temperature is 0. There is no creative latitude in reading a number out of a
table, and sampling would make the same question answerable differently on
consecutive requests, which would make the eval suite non-reproducible.
"""

from __future__ import annotations

import httpx

from atkv.generate.base import Generated, GenerationProvider, build_context
from atkv.models import Chunk

DEFAULT_MODEL = "qwen2.5:3b-instruct"
DEFAULT_HOST = "http://localhost:11434"

# The citation string ALREADY contains square brackets. An earlier prompt said
# "give the source in square brackets", which double-bracketed it -- the English
# model emitted "[SOURCE: [IT-KV 2026, ..., § 15]]" and the German one
# generalised the bracket rule onto the refusal sentence, answering
# "[Dazu findet sich ... keine Antwort.]" for questions it had the answer to.
#
# A 3B model follows a worked example far more reliably than a description of
# a format, so each prompt ends with one.
#
# THE EXAMPLE USES A FICTIONAL DOCUMENT AND A SPELLED-OUT NUMBER, DELIBERATELY.
# A first version demonstrated the format with a real-looking salary
# ("ZT / Regelstufe: 2.459 EUR"). Asked for the ST1 Erfahrungsstufe figure, the
# model answered "2.459 EUR brutto pro Monat [IT-KV 2026, § 15]" -- it copied
# the number out of the EXAMPLE and attached a genuine citation to it. Correct
# format, real source, wrong figure: exactly the failure this system exists to
# prevent, and undetectable downstream because it looks perfect.
#
# So the example must not contain anything that could be mistaken for an
# answer: a made-up agreement, a made-up paragraph, and a duration written as
# a word rather than a figure. It carries NO four-digit number at all -- not
# even a year -- because any salary-shaped token in the example is one the
# model can emit as a figure. tests/test_generation.py enforces this
# statically, and caught a stray "1999" the first time round.
SYSTEM = {
    "de": (
        "Du bist ein Auskunftssystem für den österreichischen IT-Kollektivvertrag "
        "und das zugehörige Arbeitsrecht.\n"
        "REGELN:\n"
        "1. Antworte AUSSCHLIESSLICH auf Basis der Auszüge. Verwende kein eigenes Wissen.\n"
        "2. Nenne zuerst die Zahl oder die Regel, dann die Fundstelle. Kopiere als "
        "Fundstelle den Text nach 'QUELLE:' exakt so, wie er dort steht.\n"
        "3. Steht die Antwort nicht in den Auszügen, antworte mit genau diesem Satz "
        "und sonst nichts: Dazu findet sich in den vorliegenden Dokumenten keine Antwort.\n"
        "4. Beträge sind brutto pro Monat, sofern der Auszug nichts anderes sagt.\n"
        "5. Kurz, höchstens zwei Sätze, ohne die Frage zu wiederholen. Die Fundstelle "
        "gehört IMMER dazu. Rate niemals.\n\n"
        "BEISPIEL (erfundenes Dokument, nur zur Veranschaulichung des Formats)\n"
        "QUELLE: [MUSTER-KV, § 99]\n"
        "Die Musterfrist beträgt sieben Werktage.\n"
        "FRAGE: Wie lang ist die Musterfrist?\n"
        "ANTWORT: Die Musterfrist beträgt sieben Werktage "
        "[MUSTER-KV, § 99]."
    ),
    "en": (
        "You are a reference system for the Austrian IT collective agreement "
        "and related labour law.\n"
        "RULES:\n"
        "1. Answer ONLY from the excerpts. Do not use your own knowledge.\n"
        "2. Give the figure or rule first, then the citation. For the citation, "
        "copy the text after 'SOURCE:' exactly as it appears there.\n"
        "3. If the excerpts do not contain the answer, reply with exactly this "
        "sentence and nothing else: The available documents do not answer this question.\n"
        "4. Amounts are gross per month unless the excerpt says otherwise.\n"
        "5. Brief, at most two sentences, without restating the question. The citation "
        "is ALWAYS required. Never guess.\n\n"
        "EXAMPLE (fictional document, shown only to illustrate the format)\n"
        "SOURCE: [SAMPLE-CA, § 99]\n"
        "The sample period is seven working days.\n"
        "QUESTION: How long is the sample period?\n"
        "ANSWER: The sample period is seven working days "
        "[SAMPLE-CA, § 99]."
    ),
}


class OllamaProvider(GenerationProvider):
    name = "ollama"

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                 timeout: float = 120.0, keep_alive: str = "30m") -> None:
        # Ollama evicts an idle model after 5 minutes by default, and reloading
        # this one costs 4.5s -- measured. On a service answering a handful of
        # questions an hour, that eviction lands on a real user almost every
        # time. Holding it resident trades ~2 GB of RAM for that latency.
        self.keep_alive = keep_alive
        self.model = model
        self.host = host.rstrip("/")
        self._http = httpx.Client(timeout=timeout)

    def available(self) -> bool:
        """Is the server up AND does it have our model?

        Both, because a running server missing the model fails at generation
        time with a 404 the caller cannot distinguish from a real outage.
        """
        try:
            r = self._http.get(f"{self.host}/api/tags", timeout=2.0)
            r.raise_for_status()
            names = {m.get("name", "") for m in r.json().get("models", [])}
            return any(n == self.model or n.startswith(f"{self.model}:") for n in names)
        except Exception:
            return False

    def generate(self, question: str, chunks: list[Chunk], lang: str) -> Generated:
        context = build_context(chunks, lang=lang)
        user = (
            f"{'AUSZÜGE' if lang == 'de' else 'EXCERPTS'}:\n{context}\n\n"
            f"{'FRAGE' if lang == 'de' else 'QUESTION'}: {question}"
        )
        r = self._http.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM[lang]},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": {"temperature": 0, "num_predict": 160},
            },
        )
        r.raise_for_status()
        text = r.json()["message"]["content"].strip()
        return Generated(
            text=text,
            # Every chunk we PUT IN the prompt, not what the model claims to
            # have used. The trace has to record what the model was actually
            # shown, or replaying a bad answer tells you nothing.
            used_chunk_ids=[c.chunk_id for c in chunks],
            provider=self.name,
            model=self.model,
        )

    def close(self) -> None:
        self._http.close()
