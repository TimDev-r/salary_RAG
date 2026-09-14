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

SYSTEM = {
    "de": (
        "Du bist ein Auskunftssystem für den österreichischen IT-Kollektivvertrag "
        "und das zugehörige Arbeitsrecht.\n"
        "REGELN:\n"
        "1. Antworte AUSSCHLIESSLICH auf Basis der bereitgestellten Auszüge. "
        "Verwende kein eigenes Wissen.\n"
        "2. Nenne die Zahl oder Regel und danach die Fundstelle in eckigen Klammern, "
        "genau so wie sie im Auszug steht.\n"
        "3. Steht die Antwort nicht in den Auszügen, schreibe genau: "
        "'Dazu findet sich in den vorliegenden Dokumenten keine Antwort.' "
        "Rate niemals und ergänze nichts.\n"
        "4. Beträge sind Bruttobeträge pro Monat, sofern der Auszug nichts anderes sagt.\n"
        "5. Fasse dich kurz: zwei bis drei Sätze."
    ),
    "en": (
        "You are a reference system for the Austrian IT collective agreement "
        "and related labour law.\n"
        "RULES:\n"
        "1. Answer ONLY from the provided excerpts. Do not use your own knowledge.\n"
        "2. State the figure or rule, then the citation in square brackets, "
        "exactly as it appears in the excerpt.\n"
        "3. If the excerpts do not contain the answer, reply exactly: "
        "'The available documents do not answer this question.' "
        "Never guess and never add anything.\n"
        "4. Amounts are gross per month unless the excerpt says otherwise.\n"
        "5. Be brief: two or three sentences."
    ),
}


class OllamaProvider(GenerationProvider):
    name = "ollama"

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                 timeout: float = 120.0) -> None:
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
        context = build_context(chunks)
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
                "options": {"temperature": 0, "num_predict": 300},
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
