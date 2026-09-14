"""Unit tests for the refusal rules.

Separate from evals/. The frozen eval set's five refusal cases all happen to
contain "soll ich" / "should I", so they exercise exactly ONE of the four
rules. The rest would be untested code shipping in a safety path.

The hard cases here are the NEGATIVES. Over-refusal is not the safe direction:
a system that hedges on ordinary factual questions is useless, and users route
around it to somewhere with no citations at all.
"""
import pytest

from atkv.guard import check

REFUSE = [
    ("soll ich",        "Soll ich meinen Arbeitsvertrag kündigen?"),
    ("should i",        "Should I accept this offer?"),
    ("recommend",       "Können Sie mir einen guten Arbeitsrechtsanwalt empfehlen?"),
    ("recommend en",    "What do you recommend I do about my underpayment?"),
    ("lawyer",          "Welchen Anwalt brauche ich für meinen Fall?"),
    ("litigation",      "Kann ich meinen Arbeitgeber wegen Unterentlohnung verklagen?"),
    ("litigation en",   "Can I take my employer to court over this?"),
    ("strategy",        "Wie verhandle ich am besten mein Gehalt?"),
    ("worth it",        "Lohnt es sich, zur Arbeiterkammer zu gehen?"),
]

ALLOW = [
    # Legal vocabulary, pure questions of fact.
    ("termination",     "Mit welcher Frist kann der Kollektivvertrag gekündigt werden?"),
    ("arbitration",     "Wie läuft die Schlichtung von Gesamtstreitigkeiten nach § 20 ab?"),
    ("dismissal law",   "Was regelt § 19 zur Abfertigung?"),
    # First person about one's own entitlement -- what the system is FOR.
    ("entitlement de",  "Habe ich Anspruch auf ein 13. und 14. Monatsgehalt?"),
    ("entitlement en",  "Am I entitled to a 13th and 14th monthly salary?"),
    ("my leave",        "Wie viele Werktage Urlaub stehen mir nach 25 Dienstjahren zu?"),
    ("my salary",       "Wie hoch ist mein Mindestgrundgehalt als ST1 Erfahrungsstufe?"),
    # Facts that merely sound adjacent to advice.
    ("negotiable?",     "Was ist der Weiterqualifizierungsbonus?"),
    ("percentage",      "Um wie viel Prozent ist die Gehaltssumme zu erhöhen?"),
]


@pytest.mark.parametrize("label,q", REFUSE, ids=[x[0] for x in REFUSE])
def test_refuses_advice(label, q):
    v = check(q)
    assert v.refuse, f"should have refused: {q!r}"
    assert v.rule and v.reason


@pytest.mark.parametrize("label,q", ALLOW, ids=[x[0] for x in ALLOW])
def test_allows_questions_of_fact(label, q):
    v = check(q)
    assert not v.refuse, f"false refusal by rule {v.rule!r}: {q!r}"


def test_all_four_rules_are_reachable():
    """Every rule must fire on something, or it is dead code in a safety path."""
    fired = {check(q).rule for _, q in REFUSE}
    assert {"advice", "recommend", "litigation", "strategy"} <= fired, f"unreached: {fired}"


def test_refusal_message_is_useful_in_both_languages():
    v = check("Soll ich klagen?")
    for lang, must in (("de", "Arbeiterkammer"), ("en", "Arbeiterkammer")):
        msg = v.message(lang)
        assert must in msg
        # It must say where to go instead, not merely decline.
        assert len(msg) > 200
