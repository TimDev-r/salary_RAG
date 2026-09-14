"""The eval set must be correct before it can measure anything.

These tests do NOT exercise retrieval. They check that the questions are
well-formed and that every claimed fact is genuinely in the corpus. A question
whose supporting text is absent would score as a retrieval failure forever,
blaming the retriever for our own transcription error -- and it would look
exactly like a real weakness, so nobody would think to check the question.

Run before trusting any number the eval suite produces.
"""
from __future__ import annotations

import re
from datetime import date

import pytest

norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
digits = lambda s: re.sub(r"\D", "", s)


def test_set_shape(questions):
    """25 answerable + 5 refusal, as the brief specifies."""
    answerable = [q for q in questions if q["type"] == "answerable"]
    refusal = [q for q in questions if q["type"] == "refusal"]
    assert len(answerable) == 25, f"{len(answerable)} answerable cases"
    assert len(refusal) == 5, f"{len(refusal)} refusal cases"


def test_ids_unique(questions):
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))


def test_every_case_records_provenance(questions):
    """No invented numbers: each case says where its fact came from."""
    for q in questions:
        assert q.get("source_note", "").strip(), f"{q['id']} has no source_note"
        if q["type"] == "answerable":
            assert q.get("source_quote", "").strip(), f"{q['id']} has no source_quote"


def test_controlled_pairs_are_actually_paired(questions):
    """A version pair must differ ONLY in the year, or it measures two things."""
    by_q: dict[str, list[dict]] = {}
    for q in questions:
        if "version_filter" in q.get("tests", []):
            by_q.setdefault(norm(q["question"]), []).append(q)
    assert by_q, "no version pairs found"
    for text, group in by_q.items():
        years = sorted(q["valid_year"] for q in group)
        assert years == [2025, 2026], f"{text[:50]!r} is not a 2025/2026 pair: {years}"
        assert len({q["expect"].get("amount") for q in group}) == 2, (
            f"{text[:50]!r}: both years expect the same value, so the filter is untestable"
        )


@pytest.mark.parametrize("field", ["short_title", "section_ref"])
def test_expect_source_is_complete(questions, field):
    for q in questions:
        if q["type"] == "answerable":
            assert q["expect_source"].get(field), f"{q['id']} missing expect_source.{field}"


def test_ground_truth_resolves_to_a_real_chunk(questions, corpus_chunks):
    """The heart of it: for each case, a chunk exists that is in force in the
    asked-about year, matches the cited section, and literally contains the
    quoted supporting text."""
    failures = []
    for q in questions:
        if q["type"] != "answerable":
            continue
        es = q["expect_source"]
        as_of = date(q["valid_year"], 7, 1)
        pool = [
            c for c in corpus_chunks
            if c.short_title == es["short_title"]
            and c.section_ref == es["section_ref"]
            and (not es.get("doc_id") or c.doc_id == es["doc_id"])
            and c.in_force_on(as_of)
        ]
        if not pool:
            failures.append(f"{q['id']}: no chunk matches {es} in force on {as_of}")
            continue
        quote = norm(q["source_quote"])
        if not any(quote in norm(c.text) for c in pool):
            failures.append(f"{q['id']}: source_quote {q['source_quote']!r} not in any of {len(pool)} chunks")
    assert not failures, "\n".join(failures)


def test_version_pairs_are_separable(questions, corpus_chunks):
    """The wrong year's answer must NOT survive the filter.

    This is the experiment's premise, asserted directly: if the 2025 figure is
    still reachable when asking about 2026, no amount of ranking can save us.
    """
    failures = []
    for q in questions:
        if "version_filter" not in q.get("tests", []):
            continue
        amount = q["expect"].get("amount")
        if amount is None:
            continue
        as_of = date(q["valid_year"], 7, 1)
        surviving = [c for c in corpus_chunks if c.in_force_on(as_of) and c.source_type == "kv"]
        other_year = 2025 if q["valid_year"] == 2026 else 2026
        twin = next((x for x in questions
                     if norm(x["question"]) == norm(q["question"]) and x["valid_year"] == other_year), None)
        if twin is None:
            continue
        wrong = str(twin["expect"]["amount"])
        hits = [c for c in surviving if c.content_kind == "table" and wrong in digits(c.text)
                and c.section_ref == q["expect_source"]["section_ref"]]
        if hits:
            failures.append(
                f"{q['id']}: the {other_year} figure {wrong} survives the {q['valid_year']} filter "
                f"in {len(hits)} chunk(s), e.g. {hits[0].doc_id}"
            )
    assert not failures, "\n".join(failures)
