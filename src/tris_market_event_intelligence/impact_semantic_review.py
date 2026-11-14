"""Review-only triage for evidence/rationale context loss.

This intentionally does NOT establish entailment, factuality or trading direction.
No production validation, storage, or automatically approved classification uses it.
"""
from __future__ import annotations

import re

_NEGATION = re.compile(
    r"\b(?:not|never|no|without|denied|denies|false)\b|"
    r"않았|않는|아니|없었|없다|부인|부정|철회",
    re.IGNORECASE,
)
_CONDITIONAL = re.compile(
    r"\b(?:if|may|might|could|possibly|reportedly|allegedly|"
    r"subject to|pending|unconfirmed)\b|"
    r"가능성|경우|조건|예상|추정|보도에 따르면|미확인",
    re.IGNORECASE,
)
_TEMPORAL = re.compile(
    r"\b(?:previously|formerly|historically|last year|next year|"
    r"planned|proposed|forecast|expected|as of)\b|"
    r"지난해|작년|내년|과거|당시|예정|전망|계획",
    re.IGNORECASE,
)
_CERTAINTY = re.compile(
    r"\b(?:definitely|certainly|guarantees?|will inevitably|"
    r"has been confirmed)\b|반드시|확실히|무조건|확정됐다",
    re.IGNORECASE,
)


def semantic_review_packet(
    *, source_text: str, evidence_excerpt: str, rationale: str,
) -> dict[str, object]:
    """Generate bounded human-review prompts, never a semantic pass verdict.

    A missing cue is a *triage signal*, not proof of a contradiction. Looking
    only at the quoted excerpt can miss negation/conditions just outside it.
    """
    if not all(isinstance(value, str) for value in (
        source_text, evidence_excerpt, rationale,
    )):
        raise TypeError("Source, excerpt and rationale must be strings")
    if not source_text.strip() or not evidence_excerpt.strip() or not rationale.strip():
        raise ValueError("Source, excerpt and rationale must be nonempty")
    if any(len(value) > limit for value, limit in (
        (source_text, 100_000), (evidence_excerpt, 1_000), (rationale, 2_000),
    )):
        raise ValueError("Review inputs exceed bounded limits")

    at = source_text.find(evidence_excerpt)
    codes: list[str] = []
    if at < 0:
        codes.append("EXCERPT_NOT_IN_SOURCE")
        context = source_text[:300]
    else:
        context = source_text[max(0, at - 200):min(
            len(source_text), at + len(evidence_excerpt) + 200,
        )]
        surrounding = context.replace(evidence_excerpt, "", 1)
        if _NEGATION.search(surrounding):
            codes.append("NEGATION_OUTSIDE_EXCERPT_REVIEW")
        if _CONDITIONAL.search(surrounding):
            codes.append("CONDITION_OUTSIDE_EXCERPT_REVIEW")
        if _TEMPORAL.search(surrounding):
            codes.append("TIME_OUTSIDE_EXCERPT_REVIEW")

    for code, pattern in (
        ("SOURCE_NEGATION_OMITTED_REVIEW", _NEGATION),
        ("SOURCE_CONDITION_OMITTED_REVIEW", _CONDITIONAL),
        ("SOURCE_TIME_OMITTED_REVIEW", _TEMPORAL),
    ):
        if pattern.search(context) and not pattern.search(rationale):
            codes.append(code)
    if _CERTAINTY.search(rationale):
        codes.append("RATIONALE_CERTAINTY_REVIEW")

    return {
        "status": "REQUIRES_HUMAN_REVIEW",
        "independentlyVerified": False,
        "semanticallyVerified": False,
        "sourceExcerptMatches": at >= 0,
        "reviewCodes": codes,
        "reviewQuestions": [
            "Does the quoted text support the same claim, actor, polarity and time?",
            "Are negations, conditions or corrections outside the quotation relevant?",
            "Is the alleged market transmission conditional rather than an observed fact?",
            "Are market expectations and realized returns unknown?",
        ],
    }
