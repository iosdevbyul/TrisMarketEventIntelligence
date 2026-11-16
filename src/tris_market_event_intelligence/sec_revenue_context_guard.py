"""Deterministic revenue-share context status, never a semantic truth label.

An isolated percentage cannot establish management strategy, margins, industry
causality or even what entity the numerator refers to. Fail closed rather
than turning generic speculative prose into mandatory causal hypotheses.
"""
from __future__ import annotations

import re

REFERENT_UNRESOLVED = "REFERENT_UNRESOLVED"
DESCRIPTIVE_ONLY = "DESCRIPTIVE_REVENUE_SHARE_ONLY"
STOP_STATUSES = frozenset({
    "INSUFFICIENT_SUBSTANTIVE_EVIDENCE",
    REFERENT_UNRESOLVED,
    DESCRIPTIVE_ONLY,
})
SUMMARIES = {
    REFERENT_UNRESOLVED: (
        "A historical revenue share is quoted from the original SEC filing, "
        "but the selected statement does not establish what contributed "
        "that share. No issuer or sector causal hypothesis is supported."
    ),
    DESCRIPTIVE_ONLY: (
        "The original SEC filing contains a historical revenue share with "
        "a directly stated referent. This descriptive percentage alone "
        "does not establish a management decision, margins, or sector causality."
    ),
}
REVENUE_SHARE = re.compile(
    r"\baccounted\s+for\s+\d+(?:\.\d+)?\s*%\s+of\s+"
    r"(?:our|the company's|total|nvidia's)\s+(?:total\s+)?revenue\b",
    re.IGNORECASE,
)
# Only direct, local grammatical subjects; an earlier paragraph's noun is
# NOT silently assigned to this verb. Conservative false-negative -> abstain.
DIRECT_REFERENT = re.compile(
    r"\b(?:a|an|one|the|our)\s+(?:single\s+|major\s+|largest\s+)?"
    r"(?:customer|distributor|partner|segment|product|region|business|"
    r"licensee|supplier|geographic\s+region)"
    r"(?:\s+[a-z]+){0,2}\s+accounted\s+for\b",
    re.IGNORECASE,
)
# Distinct from case-insensitive mention of "growth"; these are positive
# corporate/sector narrative assertions that cannot be derived from ratios.
UNSUPPORTED_CAUSES = re.compile(
    r"\b(?:deliberate|strategic\s+decision|management\s+decid|"
    r"management\s+prioriti|high.margin|resource\s+reallocat|"
    r"allocat(?:ed|ion)\s+of\s+resources|shift\s+in\s+operational\s+focus|"
    r"targeted\s+growth\s+strategy|macro(?:economic)?\s+(?:shift|pressure)|"
    r"sector.wide\s+(?:shift|trend)|market\s+contraction|"
    r"industry\s+dynamics|unexpected\s+expansion)\b",
    re.IGNORECASE,
)


def revenue_context_status(focus: str) -> str | None:
    """Classify only revenue-share statements, never all financial filings."""
    hit = REVENUE_SHARE.search(focus)
    if hit is None:
        return None
    # The subject must precede this verb in the same selected sentence.
    sentence_start = max(
        focus.rfind(". ", 0, hit.start()),
        focus.rfind("! ", 0, hit.start()),
        focus.rfind("? ", 0, hit.start()),
    )
    sentence = focus[sentence_start + 2 if sentence_start >= 0 else 0:hit.end()]
    if DIRECT_REFERENT.search(sentence):
        return DESCRIPTIVE_ONLY
    return REFERENT_UNRESOLVED


def validated_stopped_interpretation(status: str) -> dict:
    if status not in SUMMARIES:
        raise ValueError("SEC_CONTEXT_UNEXPECTED_STATUS")
    return {"observations": [], "hypotheses": [], "summary": SUMMARIES[status]}


def reject_unsupported_revenue_causes(excerpt: str, interpretation: dict) -> None:
    """Conservative fail-closed check for unsupported strategy/sector prose.

    It never declares a claim true. Positive causes require explicit original
    supporting statements, not a ratio or hypothetical wording alone.
    """
    if REVENUE_SHARE.search(excerpt) is None:
        return
    statements = [interpretation["summary"]]
    statements.extend(obs["interpretation"] for obs in interpretation["observations"])
    for row in interpretation["hypotheses"]:
        statements.extend((row["mechanism"], row["limitation"]))
    if any(UNSUPPORTED_CAUSES.search(statement) for statement in statements):
        # A revenue percentage alone cannot substantiate those claims even
        # when introduced with might/could/potentially.
        raise ValueError("SEC_REVENUE_UNSUPPORTED_CAUSAL_NARRATIVE")
