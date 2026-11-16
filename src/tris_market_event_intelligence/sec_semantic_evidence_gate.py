"""Fail-closed SEC excerpt materiality and concrete-claim source grounding.

This is a provenance gate, not semantic entailment or a gold-standard judgment.
It only inspects SHA-verified retrospective source excerpts, never prices.
"""
from __future__ import annotations

import re

INSUFFICIENT = "INSUFFICIENT_SUBSTANTIVE_EVIDENCE"
SUBSTANTIVE = "SUBSTANTIVE_ISSUER_EVIDENCE_PRESENT"
ABSTENTION_SUMMARY = (
    "The SHA-verified SEC excerpt provides filing identity or administrative "
    "context only. Issuer-event and sector hypotheses are withheld because "
    "the available excerpt does not substantiate a material business event."
)

# SEC cover, index, legal and table-of-contents boilerplate never count as
# company-event observations even if they contain dates, item numbers or forms.
BOILERPLATE = re.compile(
    r"(?i)\b(?:form\s+10-[kq]|form\s+8-k|annual report|"
    r"securities and exchange commission|united states|washington\s*,?\s*d\.?c\.?|"
    r"table of contents|registrant|pursuant to section|"
    r"securities exchange act|commission file number|"
    r"fiscal year ended|exhibits? index|check (?:the )?appropriate box|"
    r"for the transition period|incorporated by reference|"
    r"documents incorporated|corporate governance|"
    r"results of operations and financial condition)\b"
)
# A quantitative company statement or a specifically described issuer event is
# a prerequisite for generating causal research hypotheses.
FINANCIAL = re.compile(
    r"(?i)\b(?:revenue|net income|gross margin|operating income|"
    r"earnings|cash flow|guidance|sales|operating expenses|"
    r"capital expenditures|net loss|backlog)\b"
)
MAGNITUDE = re.compile(
    r"(?i)(?:\$\s*\d|\d[\d,.]*\s*(?:%|percent|million|billion|trillion))"
)
EVENT = re.compile(
    r"(?i)\b(?:acquired|acquisition of|sold its|divested|"
    r"restructured|restructuring of|terminated|appointed|"
    r"resigned|launched|released|entered into an agreement|"
    r"announced a merger|settled a lawsuit|"
    r"raised capital|issued debt|repurchased shares)\b"
)
RISK_DISCLOSURE = re.compile(
    r"(?i)\b(?:depends? on|relies? on|reliance on|exposed to|"
    r"subject to|may adversely affect|risk of)\b"
)
RISK_SUBJECT = re.compile(
    r"(?i)\b(?:suppliers?|manufacturing|litigation|"
    r"export controls?|regulations?|customers?|cybersecurity|"
    r"intellectual property|production)\b"
)
ISSUER = re.compile(r"(?i)\b(?:nvidia|the company|we|our|issuer)\b")
STRUCTURE = re.compile(r"(?i)\b(?:restructur|accounting|oversight|governance|supply.chain|financ|trade|export|subsid|lawsuit|launch|production|capacity|valuation|r\s*&\s*d)\w*")
CHANGE = re.compile(
    r"(?i)\b(?:change[ds]?|shift(?:ed|s)?|restructur\w*|"
    r"revis(?:ed|ion)|pivot(?:ed)?|alter(?:ed)?|"
    r"depend(?:s|ent|ency)?|driven|caus(?:ed|es)|"
    r"trigger(?:ed|s)?|necessitat(?:e|ed|es))\b"
)
MATERIAL_EVENT = re.compile(
    r"(?i)\b(?:product launch|lawsuit settlement|major acquisition|"
    r"merger|internal restructuring|accounting practice changes?|"
    r"executive oversight changes?|semiconductor supply.chain shifts?)\b"
)
FILING_MOTIVE = re.compile(
    r"(?i)\b(?:decision to file|filing (?:was|is) (?:driven|influenced|caused)|"
    r"(?:pressure|dynamics) (?:necessitat|require)\w* (?:the )?filing)\b"
)


def source_evidence_status(excerpt: str) -> str:
    """Conservatively require business detail beyond front-matter metadata.

    False negatives are acceptable (abstention), false positives only allow
    model research and never imply automatic source entailment.
    """
    if not isinstance(excerpt, str):
        raise TypeError("SEC_SUBSTANTIVE_SOURCE_INVALID")
    # Split physical lines first: a cover contains years and legal headings
    # that must not be combined with unrelated financial keywords.
    pieces = re.split(r"(?<=[.!?])\s+|[\r\n]+", excerpt)
    for piece in pieces:
        line = piece.strip()
        if len(line) < 36 or BOILERPLATE.search(line):
            continue
        if not ISSUER.search(line):
            continue
        if (
            (FINANCIAL.search(line) and MAGNITUDE.search(line))
            or EVENT.search(line)
            or (RISK_DISCLOSURE.search(line) and RISK_SUBJECT.search(line))
        ):
            return SUBSTANTIVE
    return INSUFFICIENT


def _source_fact_spans(excerpt: str) -> list[str]:
    return [
        piece.strip() for piece in re.split(r"(?<=[.!?])\s+|[\r\n]+", excerpt)
        if len(piece.strip()) >= 36 and not BOILERPLATE.search(piece)
        and ISSUER.search(piece)
    ]


def validate_concrete_claims(case: dict, interpretation: dict) -> None:
    """Stop ungrounded concrete event narratives and unsupported sector causes.

    Require explicit issuer fact/action in ONE original-source sentence; do not
    treat mention of an isolated keyword anywhere as evidence. Even a passage
    meeting this gate remains HUMAN-UNVERIFIED (no automatic truth label).
    """
    excerpt = case["evidence"][0]["text"]
    facts = _source_fact_spans(excerpt)
    statements = [interpretation["summary"]]
    statements.extend(x["interpretation"] for x in interpretation["observations"])
    for hypothesis in interpretation["hypotheses"]:
        category = hypothesis["category"]
        for field in ("mechanism", "limitation"):
            text = hypothesis[field]
            # Concrete sector acts cannot be grounded by an issuer-only filing.
            if category == "SECTOR_ALTERNATIVE" and (
                (STRUCTURE.search(text) and CHANGE.search(text))
                or FILING_MOTIVE.search(text)
            ):
                raise ValueError("SEC_SEMANTIC_UNVERIFIED_SECTOR_CAUSE")
            statements.append(text)
    for statement in statements:
        # Ordinary mention of a filing or company is not an event assertion.
        if not (
            (STRUCTURE.search(statement) and CHANGE.search(statement))
            or MATERIAL_EVENT.search(statement)
            or FILING_MOTIVE.search(statement)
        ):
            continue
        # Require matching fact+change terms in a single issuer-sourced passage,
        # rather than accepting the presence of a keyword in the full excerpt.
        subject = set(re.findall(r"[a-z]{5,}", statement.lower())) - {
            "change", "changes", "company", "could", "might", "would",
            "filing", "because", "report", "their", "these", "there",
            "about", "which", "should", "underlying", "possible",
        }
        grounded = any(
            (CHANGE.search(fact) or MATERIAL_EVENT.search(fact))
            and STRUCTURE.search(fact)
            and len(subject & set(re.findall(r"[a-z]{5,}", fact.lower()))) >= 2
            for fact in facts
        )
        if not grounded:
            raise ValueError("SEC_SEMANTIC_CONCRETE_CLAIM_NOT_SOURCE_GROUNDED")
