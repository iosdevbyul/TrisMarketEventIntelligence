"""Deterministic source excerpt IDs for small-model semantic analysis.

The LLM chooses evidence IDs rather than recreating arbitrary quotations.
The code inserts canonical exact-source quotations and validates every span.
No future prices, model feedback, or synthesized facts enter the evidence.
"""
from __future__ import annotations

import re
from typing import Any

from tris_market_event_intelligence.source_grounded_semantic_analysis import (
    SEMANTIC_SCHEMA,
)

EXCERPT_SCHEMA: dict[str, Any] = {
    **SEMANTIC_SCHEMA,
    "properties": {
        **SEMANTIC_SCHEMA["properties"],
        "sourceStatements": {
            "type": "array", "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "evidenceId": {"type": "string"},
                    "interpretation": {"type": "string"},
                    "factualStatus": {
                        "type": "string",
                        "enum": ["REPORTED", "OFFICIAL", "ALLEGED", "DENIED"],
                    },
                },
                "required": ["evidenceId", "interpretation", "factualStatus"],
                "additionalProperties": False,
            },
        },
        "economicMechanisms": {
            "type": "array", "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "affectedScope": {
                        "type": "string",
                        "enum": ["ISSUER", "INDUSTRY", "BROAD_MARKET", "UNKNOWN"],
                    },
                    "economicDirection": {
                        "type": "string",
                        "enum": ["POSITIVE", "NEGATIVE", "MIXED", "UNKNOWN"],
                    },
                    "reasoning": {"type": "string"},
                    "evidenceId": {"type": "string"},
                    "uncertainty": {"type": "string"},
                },
                "required": [
                    "affectedScope", "economicDirection", "reasoning",
                    "evidenceId", "uncertainty",
                ],
                "additionalProperties": False,
            },
        },
    },
}

SYSTEM = """Interpret a historical economic document, not a future stock price.
Only the numbered exact SOURCE EXCERPTS are factual evidence; the headline
is identity metadata and may not supply unstated facts.
Do not create verbatim quotes. Each source statement MUST reference the
correct evidenceId from sourceExcerpts and explain its meaning separately.
Each economic mechanism is a HYPOTHESIS and must reference the evidenceId
that supports its premise. An excerpt ID only proves quoted source presence,
NOT economic direction or a causal effect. No invented statistics, figures,
asset prices, consensus, market reaction or subsequent knowledge.
Classify by the event that actually occurred, not by words appearing in the
source organization's name. An M2 change is ECONOMIC_INDICATOR_RELEASE.
Observed bank lending/deposit rates are ECONOMIC_INDICATOR_RELEASE.
A central-bank financial-stability inspection without a policy-rate action
is FINANCIAL_STABILITY_REVIEW. MONETARY_POLICY_DECISION requires a specific
adopted central-bank policy action, not a report or committee meeting. Official issuer Form
8-K Item 1.01 is a corporate agreement, never monetary policy. Item 2.02
is an earnings disclosure, but its cover does not give revenue/EPS figures;
if the referenced Exhibit 99.1 is not available, record missingEvidence.
Explicitly separate facts from economic hypothesis and unknowns.
Do not label a factual statement describing an expectation as a realized fact.
UNKNOWN is permitted and necessary when economic consequence cannot be
supported; no effect is NOT the same as insufficient evidence.
Do not invent an affected security: a verified SEC issuer can be exposed to
its own filing, while an industry-level mechanism does not identify a ticker.
When text contains a contract, distinguish potential revenue benefit from
financial obligation, payment, and counterparty risks.
Return only a JSON object conforming to the supplied schema, using evidence
IDs exactly as given, without any embedded source quotes or further text."""


def _chunks(body: str, max_chars: int = 260) -> list[str]:
    """Split into source-identical readable excerpts without paraphrasing."""
    # The regex preserves original spacing in each source slice.
    text = body.strip()
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    position = 0
    while position < len(text):
        limit = min(len(text), position + max_chars)
        if limit < len(text):
            # Do not split mid-word if a reasonable word boundary exists.
            segment = text[position:limit]
            cut = max(segment.rfind(" "), segment.rfind(". "), segment.rfind("。"))
            if cut >= max_chars // 2:
                limit = position + cut + 1
        chunk = text[position:limit].strip()
        if len(chunk) >= 10:
            spans.append((position, limit))
        position = limit
    return [text[start:end].strip() for start, end in spans]


def source_excerpts(
    body: str, *, max_excerpts: int = 26,
) -> list[dict[str, str]]:
    """Stable evidence roster biased to substantive figures and contract terms.

    Scores are fixed and text-only. Chunks are sorted back into original
    document order. Source index order and excerpt text are immutable.
    """
    # The caller passes a canonical text snapshot. Re-parsing it as HTML here
    # silently deletes angle-bracketed financial text and breaks quote offsets.
    # Evidence must be a literal slice of exactly that snapshot.
    chunks = _chunks(body)
    if len(chunks) <= max_excerpts:
        return [
            {"evidenceId": f"E{i:03d}", "text": chunk}
            for i, chunk in enumerate(chunks, start=1)
        ]
    keys = (
        "item 1.01", "item 2.02", "exhibit 99", "revenue", "net income",
        "gross margin", "earnings", "guidance", "forecast", "agreement",
        "guarant", "payment", "billion", "million", "financial result",
        "영업이익", "매출", "증가", "감소", "금리", "물가", "전망", "계약", "통화",
    )
    scored = []
    for i, chunk in enumerate(chunks):
        lower = chunk.casefold()
        score = sum(2 for key in keys if key in lower)
        score += min(3, len(re.findall(r"\d+(?:[.,]\d+)?\s*[%$조억]?", chunk)))
        if i == 0:
            score += 3
        scored.append((score, i))
    # Reserve the first excerpt for identity/critical opening context.
    selected = {0}
    selected.update(
        i for _, i in sorted(
            scored[1:], key=lambda row: (-row[0], row[1]),
        )[:max_excerpts - 1]
    )
    return [
        {"evidenceId": f"E{i + 1:03d}", "text": chunks[i]}
        for i in sorted(selected)
    ]


def schema_for_excerpts(excerpts: list[dict[str, str]]) -> dict[str, Any]:
    """Constrain both model evidence ID fields to this article's actual IDs."""
    import copy

    ids = [item["evidenceId"] for item in excerpts]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("NO_UNIQUE_SOURCE_EVIDENCE_IDS")
    schema = copy.deepcopy(EXCERPT_SCHEMA)
    for field in ("sourceStatements", "economicMechanisms"):
        schema["properties"][field]["items"]["properties"]["evidenceId"]["enum"] = ids
    return schema


def expand_evidence_ids(
    raw: dict[str, Any], excerpts: list[dict[str, str]],
) -> dict[str, Any]:
    """Resolve only known source IDs to exact quotes; fail on fabricated IDs."""
    lookup = {row["evidenceId"]: row["text"] for row in excerpts}
    if len(lookup) != len(excerpts):
        raise ValueError("DUPLICATED_SOURCE_EVIDENCE_IDS")
    answer = dict(raw)
    facts = answer.get("sourceStatements")
    mechanisms = answer.get("economicMechanisms")
    if not isinstance(facts, list) or not isinstance(mechanisms, list):
        raise TypeError("SEMANTIC_EVIDENCE_SCHEMA_INVALID")
    expanded_facts = []
    for row in facts:
        if not isinstance(row, dict) or set(row) != {
            "evidenceId", "interpretation", "factualStatus",
        }:
            raise ValueError("SEMANTIC_EVIDENCE_SCHEMA_INVALID")
        evidence = lookup.get(row["evidenceId"])
        if evidence is None:
            raise ValueError("SEMANTIC_EVIDENCE_ID_NOT_IN_SOURCE")
        expanded_facts.append({
            "quote": evidence, "interpretation": row["interpretation"],
            "factualStatus": row["factualStatus"],
        })
    expanded_mechanisms = []
    for row in mechanisms:
        if not isinstance(row, dict) or set(row) != {
            "affectedScope", "economicDirection", "reasoning",
            "evidenceId", "uncertainty",
        }:
            raise ValueError("SEMANTIC_EVIDENCE_SCHEMA_INVALID")
        evidence = lookup.get(row["evidenceId"])
        if evidence is None:
            raise ValueError("SEMANTIC_EVIDENCE_ID_NOT_IN_SOURCE")
        expanded_mechanisms.append({
            **{k: v for k, v in row.items() if k != "evidenceId"},
            "groundingQuote": evidence,
        })
    answer["sourceStatements"] = expanded_facts
    answer["economicMechanisms"] = expanded_mechanisms
    return answer
