"""Generation tests. Skipped when no model server is reachable.

The centrepiece is test_few_shot_example_does_not_leak. The prompt's worked
example once contained a realistic salary ("ZT / Regelstufe: 2.459 EUR"), and
asked for a DIFFERENT figure the model answered with the example's number and
attached a real citation to it:

    "Das Mindestgrundgehalt beträgt 2.459 EUR brutto pro Monat [IT-KV 2026, § 15]"

Correct format, genuine source, wrong figure. Nothing downstream can catch
that -- it is indistinguishable from a right answer -- so it has to be caught
here, and it has to stay caught.
"""
from __future__ import annotations

import pytest

from atkv.generate.extractive import ExtractiveProvider
from atkv.generate.ollama import SYSTEM, OllamaProvider


def _ollama():
    p = OllamaProvider()
    if not p.available():
        pytest.skip("no ollama server with the configured model")
    return p


def test_prompt_examples_contain_no_real_figures():
    """Static check: runs with no server, so it guards every CI run.

    The example must be visibly fictional. If someone edits it back to a
    realistic salary, this fails before anyone notices a wrong answer.
    """
    for lang, prompt in SYSTEM.items():
        body = prompt.split("BEISPIEL" if lang == "de" else "EXAMPLE", 1)[-1]
        assert "MUSTER" in body or "SAMPLE" in body, f"{lang} example is not marked fictional"
        # Any 3-4 digit group is a plausible monthly salary and must not appear.
        import re
        nums = re.findall(r"\b\d[.,]?\d{3}\b", body)
        assert not nums, f"{lang} example contains salary-shaped numbers {nums}"


@pytest.mark.parametrize("lang,question,expected", [
    ("de", "Wie hoch ist das Mindestgrundgehalt für ST1 Erfahrungsstufe?", "4.476"),
    ("en", "What is the minimum basic salary for ST1 at the experienced level?", "4,476"),
])
def test_answers_from_context_only(lang, question, expected, corpus_chunks):
    p = _ollama()
    hits = [c for c in corpus_chunks
            if c.doc_id == f"it-kv-2026-{lang}" and "ST1 / " in c.text
            and ("Erfahrungsstufe" in c.text or "Experienced" in c.text)]
    assert hits, "test fixture missing the expected cell chunk"
    out = p.generate(question, hits[:3], lang)
    assert expected in out.text, f"expected {expected} in: {out.text!r}"


def test_few_shot_example_does_not_leak(corpus_chunks):
    """The regression test for the bug described in this module's docstring."""
    p = _ollama()
    hits = [c for c in corpus_chunks
            if c.doc_id == "it-kv-2026-de" and "ST1 / Erfahrungsstufe" in c.text]
    out = p.generate("Wie hoch ist das Mindestgrundgehalt für ST1 Erfahrungsstufe?",
                     hits[:3], "de")
    for leaked in ("2.459", "2,459", "Muster", "MUSTER", "§ 99"):
        assert leaked not in out.text, f"prompt example leaked into the answer: {out.text!r}"
    assert "4.476" in out.text


def test_refuses_when_context_lacks_the_answer(corpus_chunks):
    """Abstention must be preferred to invention."""
    p = _ollama()
    unrelated = [c for c in corpus_chunks if c.section_ref == "§ 1"][:2]
    out = p.generate("Wie viele Urlaubstage bekomme ich, wenn ich auf dem Mars arbeite?",
                     unrelated, "de")
    assert "keine Antwort" in out.text, out.text


def test_extractive_provider_cannot_invent(corpus_chunks):
    """It can only echo, so every word it emits is in a source document."""
    hits = [c for c in corpus_chunks if "ST1 / Erfahrungsstufe" in c.text][:1]
    out = ExtractiveProvider().generate("egal", hits, "de")
    assert out.text.split("\n")[0].strip() in hits[0].text
    assert out.used_chunk_ids == [hits[0].chunk_id]
