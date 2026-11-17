"""Explainable, non-corrective audit of official-source economic mechanisms.

Mechanical quality checks are deliberately *not* semantic entailment. An
incorrect channel stays as originally proposed in the rejection ledger;
Python never reassigns economic meaning or fabricates an alternative.
"""
from __future__ import annotations

import re

_NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?(?![A-Za-z])")
_REGULATORY_FACT = re.compile(
    r"\b(?:regulat(?:ion|or|ory|ed|ing)|compliance|"
    r"legislation|statut(?:e|ory)|law(?:s)?|tariff(?:s)?|"
    r"sanction(?:s)?|export\s+control(?:s)?|"
    r"government\s+(?:ban|order|restriction))\b",
    re.IGNORECASE,
)
_REGULATORY_MECHANISM = re.compile(
    r"\b(?:regulat(?:ion|or|ory|ed|ing)|compliance|"
    r"tariff(?:s)?|sanction(?:s)?|export\s+control(?:s)?|"
    r"law(?:s)?|legislation|government\s+restriction)\b",
    re.IGNORECASE,
)
_UNGROUNDED_EVALUATION = re.compile(
    r"\b(?:high[\s-]margin|industry[\s-]leading|"
    r"market[\s-]dominant|guaranteed\s+growth|"
    r"best[\s-]in[\s-]class)\b",
    re.IGNORECASE,
)


def _numbers(text: str) -> set[str]:
    return {match.group(0) for match in _NUMBER.finditer(text)}


def inspect_interpretation(explanation: str, source_fact: str) -> list[str]:
    """Identify especially strong unquoted interpretations for human review."""
    return (
        ["UNSUPPORTED_EVALUATIVE_QUALIFIER"]
        if any(
            term.group(0).casefold() not in source_fact.casefold()
            for term in _UNGROUNDED_EVALUATION.finditer(explanation)
        )
        else []
    )


def inspect_mechanism(proposal: dict, fact_by_id: dict[str, dict]) -> dict:
    ids = proposal["sourceFactIds"]
    if any(fid not in fact_by_id for fid in ids):
        return {
            "acceptedForQualitativeReview": False,
            "blockers": ["UNKNOWN_SOURCE_FACT"],
            "warnings": [],
            "semanticEntailmentVerified": False,
            "economicChannelVerified": False,
        }
    statements = [fact_by_id[fid]["statement"] for fid in ids]
    cited_text = " ".join(statements)
    narrative = " ".join(
        str(proposal[key]) for key in (
            "triggerCondition", "conditionalEffect", "counterHypothesis",
        )
    )
    blockers = []
    warnings = []
    if proposal["transmission"] == "REGULATION" and not (
        _REGULATORY_FACT.search(cited_text)
        and _REGULATORY_MECHANISM.search(narrative)
    ):
        blockers.append("REGULATION_CHANNEL_NOT_GROUNDED_IN_SOURCE_OR_MECHANISM")
    # Thresholds invented by the model may be perfectly useful exploratory
    # assumptions, but they cannot be represented as a fact-derived scenario.
    # Preserve them in the audit trail without pretending source support.
    stripped_narrative = narrative
    for fid in ids:
        stripped_narrative = stripped_narrative.replace(fid, "")
    extra_values = sorted(_numbers(stripped_narrative) - _numbers(cited_text))
    if extra_values:
        blockers.append("UNSOURCED_NUMERIC_THRESHOLD")
    if proposal["transmission"] != "REGULATION":
        warnings.append("TRANSMISSION_SEMANTICS_REQUIRES_HUMAN_REVIEW")
    return {
        "acceptedForQualitativeReview": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "unsupportedNumericValues": extra_values,
        "semanticEntailmentVerified": False,
        "economicChannelVerified": False,
        "humanReviewRequired": True,
    }
