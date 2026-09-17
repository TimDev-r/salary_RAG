"""Version filtering when no date is asked for, and when the index runs out.

Two failures motivated these:

  * valid_year was required, so "Wie hoch ist das Mindestgrundgehalt für ST1?"
    -- the commonest question the system will ever get -- was answered with a
    422. Omitting the year now means "whatever is in force today".

  * Statutes have no end date and collective agreements expire. Asking about
    2027 therefore dropped every KV chunk from the filter while all 194 law
    chunks survived, and the service answered from labour law alone, looking
    entirely normal. The user sees no figure and no reason.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from atkv.models import QueryRequest
from atkv.retrieve.pipeline import Coverage


def test_no_year_means_today():
    r = QueryRequest(question="Wie hoch ist das Mindestgrundgehalt für ST1?")
    assert r.valid_year is None
    assert r.as_of() == date.today()


def test_explicit_year_uses_midyear_not_the_boundary():
    """1 January sits on the edge of every 'gilt ab 1.1.' document."""
    assert QueryRequest(question="x", valid_year=2026).as_of() == date(2026, 7, 1)


def test_todays_question_reaches_the_current_agreement(corpus_chunks):
    today = date.today()
    kv = [c for c in corpus_chunks if c.source_type == "kv" and c.in_force_on(today)]
    assert kv, f"no collective agreement is in force on {today} -- the index is stale"
    years = {c.valid_from.year for c in kv}
    assert years == {today.year}, f"expected only {today.year} chunks, got {years}"


def test_only_one_year_of_the_agreement_survives_the_filter(corpus_chunks):
    """The premise of the whole project, asserted directly."""
    for year in (2025, 2026):
        kv = [c for c in corpus_chunks
              if c.source_type == "kv" and c.in_force_on(date(year, 7, 1))]
        assert {c.valid_from.year for c in kv} == {year}


def test_coverage_reports_the_real_window(corpus_chunks):
    from atkv.retrieve.pipeline import RetrievalPipeline
    p = RetrievalPipeline.__new__(RetrievalPipeline)
    p.chunks = corpus_chunks
    cov = p.coverage()
    assert cov["kv"].latest is not None, "agreements must have an end date"
    # Statutes stay in force until repealed, so they have none.
    assert cov["law"].latest is None


def test_a_year_past_the_agreement_is_reported_as_a_gap(corpus_chunks):
    from atkv.retrieve.pipeline import RetrievalPipeline
    p = RetrievalPipeline.__new__(RetrievalPipeline)
    p.chunks = corpus_chunks
    beyond = p.coverage()["kv"].latest + timedelta(days=1)
    gaps = p.gaps_on(beyond)
    assert [g.source_type for g in gaps] == ["kv"], (
        "the agreement must be reported as missing; the statutes legitimately survive"
    )


def test_notice_names_the_window_and_refuses_to_substitute():
    from atkv.api import _coverage_notice
    gap = [Coverage("kv", date(2025, 1, 1), date(2026, 12, 31), 344)]
    for lang, must in (("de", ["2026-12-31", "2027-07-01"]), ("en", ["2026-12-31", "2027-07-01"])):
        msg = _coverage_notice(gap, date(2027, 7, 1), lang)
        assert msg
        for m in must:
            assert m in msg, f"{lang} notice does not name {m}: {msg}"
    assert _coverage_notice([], date(2026, 7, 1), "de") is None


@pytest.mark.parametrize("year", [2025, 2026])
def test_in_window_dates_produce_no_notice(year, corpus_chunks):
    from atkv.api import _coverage_notice
    from atkv.retrieve.pipeline import RetrievalPipeline
    p = RetrievalPipeline.__new__(RetrievalPipeline)
    p.chunks = corpus_chunks
    d = date(year, 7, 1)
    assert _coverage_notice(p.gaps_on(d), d, "de") is None
