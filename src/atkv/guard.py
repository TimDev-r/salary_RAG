"""Refusal: the system documents the law, it does not advise on it.

WHY RULES AND NOT A MODEL
-------------------------
A refusal decision must be deterministic and inspectable. If the system
declines to answer, the user deserves to know exactly why, and the operator
deserves to reproduce it. A classifier that refuses 1 in 20 questions on a
whim is worse than a blunt rule, because nobody can tell the whim from a
policy. It also runs before anything is loaded, so an advice-seeking question
never costs a retrieval pass or a model.

THE DISCRIMINATOR IS "SHOULD I", NOT THE TOPIC
----------------------------------------------
The naive version refuses anything mentioning lawsuits or salaries. That fails
in both directions on this eval set:

  "Mit welcher Frist kann der Kollektivvertrag gekündigt werden?"
       -- contains "gekündigt", is a pure question of fact about § 3.
  "Wie viele Werktage Urlaub stehen mir nach 25 Dienstjahren zu?"
       -- first person, about the user's own entitlement, perfectly answerable.
  "Habe ich Anspruch auf ein 13. und 14. Monatsgehalt?"
       -- "do I have a right to", which is what this system is FOR.

First person is not the signal. Legal vocabulary is not the signal. What
separates the two classes is whether the user is asking what the documents
SAY or asking what they should DO. "Soll ich ...?" / "Should I ...?" is that
distinction expressed in grammar, and it is the primary rule here.

DELIBERATELY NARROW
-------------------
Over-refusal is not the safe direction. A system that hedges on ordinary
factual questions is useless, and users route around it -- to a forum, or to a
chatbot with no citations at all. Refusal covers advice, recommendations and
predictions about the user's own situation. It does not cover questions that
merely sound legal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 1. Asking what to DO. The core rule; catches the great majority of cases.
ADVICE = re.compile(
    r"\b("
    r"soll(?:te)?\s+ich|soll(?:te)?\s+man|was\s+soll\s+ich|"
    r"should\s+i|should\s+we|what\s+should\s+i|"
    r"muss\s+ich\s+(?:klagen|kündigen)|"
    r"lohnt\s+(?:es\s+)?sich|is\s+it\s+worth"
    r")\b",
    re.I,
)

# 2. Asking for a recommendation, of a person or a course of action.
RECOMMEND = re.compile(
    r"\b("
    r"empfehl\w*|rat(?:en|schlag)\w*|"
    r"recommend\w*|advise|advice|"
    r"welchen\s+anwalt|which\s+lawyer|besten\s+anwalt"
    r")\b",
    re.I,
)

# 3. Litigation framed around the user's own case. Note this requires a
#    first-person marker: "Wie läuft ein Verfahren nach § 20 ab?" is a
#    documentable question about the agreement's arbitration clause.
LITIGATION = re.compile(
    r"\b(klage|klagen|verklagen|sue|suing|lawsuit|court|gericht)\w*\b", re.I)
FIRST_PERSON = re.compile(r"\b(ich|mein\w*|mir|mich|i|my|me|we|our)\b", re.I)

# 4. Negotiation and career strategy: adjacent to documented facts, not in them.
STRATEGY = re.compile(
    # verhandl\w* not verhandeln: German conjugates, and "Wie verhandle ich"
    # is exactly the form a user types. Matching the infinitive only missed it.
    r"\b(verhandl\w*|negotiat\w*|"
    r"job\s?angebot|jobangebot|job\s+offer|kündigen\s+oder|quit\s+or)\b", re.I)

REFUSAL_TEXT = {
    "de": ("Diese Frage betrifft eine persönliche Empfehlung oder Rechtsberatung. "
           "Dieses System gibt ausschließlich den Inhalt des Kollektivvertrags und "
           "der zitierten Gesetze wieder und beantwortet keine Fragen dazu, was Sie "
           "tun sollten. Für eine Einschätzung Ihres konkreten Falls wenden Sie sich "
           "bitte an die Arbeiterkammer, Ihre Gewerkschaft oder eine Rechtsanwältin "
           "bzw. einen Rechtsanwalt.\n\n"
           "Gerne nenne ich Ihnen die einschlägigen Bestimmungen und Beträge – "
           "fragen Sie z. B. nach dem Mindestgrundgehalt einer Tätigkeitsfamilie."),
    "en": ("This question asks for a personal recommendation or legal advice. This "
           "system only reports what the collective agreement and the cited statutes "
           "say; it does not answer questions about what you should do. For an "
           "assessment of your own situation, please contact the Arbeiterkammer, your "
           "trade union, or a lawyer.\n\n"
           "It can tell you the relevant provisions and figures – ask, for example, "
           "for the minimum salary of a given task group."),
}


@dataclass
class GuardVerdict:
    refuse: bool
    reason: str | None = None
    rule: str | None = None

    def message(self, lang: str) -> str:
        return REFUSAL_TEXT.get(lang, REFUSAL_TEXT["en"])


def check(question: str) -> GuardVerdict:
    q = question.strip()

    if ADVICE.search(q):
        return GuardVerdict(True, "asks what the user should do", "advice")
    if RECOMMEND.search(q):
        return GuardVerdict(True, "asks for a recommendation", "recommend")
    if LITIGATION.search(q) and FIRST_PERSON.search(q):
        return GuardVerdict(True, "asks about litigating the user's own case", "litigation")
    if STRATEGY.search(q):
        return GuardVerdict(True, "asks for negotiation or career strategy", "strategy")
    return GuardVerdict(False)
