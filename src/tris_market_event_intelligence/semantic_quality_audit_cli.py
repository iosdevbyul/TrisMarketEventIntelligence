"""Independent read-only quality audit for source-grounded semantic reports.

A valid exact-source quote is not a verified interpretation. This module
surfaces higher-risk category, unit, market-scope, factual-status and
transmission-mechanism problems without rewriting model conclusions or
declaring automatic factual accuracy. The price-study labels are never read.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

_STATUS = ("REQUIRES_FACTUAL_REVIEW", "REQUIRES_SEMANTIC_REVIEW",
           "STRUCTURE_ONLY_NO_GOLD_VALIDATION")
_XBRL = re.compile(
    r"(?:us-gaap:|[a-z]+:[A-Za-z0-9]+Member|"
    r"\b(?:CommonClassAMember|SeniorNotesDue\d{4}Member)\b)",
    re.IGNORECASE,
)
_NO_MEANINGFUL_UNCERTAINTY = {"", "unknown", "n/a", "none", "not applicable"}
_PIT_ATTACHMENT_MARKERS = ("<붙임>", "<첨부파일>", "자세한 내용은 붙임")


def _flag(
    findings: list[dict[str, str]], code: str, *, detail: str,
    priority: str = "REVIEW",
) -> None:
    findings.append({
        "code": code, "priority": priority, "detail": detail,
        "automaticCorrectionApplied": "false",
    })


def audit_article(article: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the original model's claims, never modify the original JSON."""
    semantic = article.get("semanticAnalysis")
    if not isinstance(semantic, dict):
        return {
            "articleId": article.get("articleId"),
            "qualityStatus": "REQUIRES_FACTUAL_REVIEW",
            "findings": [{
                "code": "MODEL_SEMANTIC_ANALYSIS_MISSING",
                "priority": "CRITICAL",
                "detail": "No structured analysis is available",
                "automaticCorrectionApplied": "false",
            }],
        }
    model = semantic["modelAnalysis"]
    facts = model["sourceStatements"]
    mechanisms = model["economicMechanisms"]
    source = str(article.get("source") or "")
    title = str(article.get("title") or "").casefold()
    official_sec = source.startswith("sec_8k_")
    official_bok = source.startswith("bok_")
    findings: list[dict[str, str]] = []
    primary_category = semantic["classification"]["eventCategory"]
    predicted_category = model["eventCategory"]

    if official_sec and "item 2.02" in title and (
        predicted_category != "EARNINGS_OR_GUIDANCE"
    ):
        _flag(
            findings, "SEC_EARNINGS_ITEM_CATEGORY_CONFLICT",
            detail=(
                f"Official Form 8-K Item 2.02 denotes an earnings disclosure; "
                f"model selected {predicted_category}. Verify all filed items."
            ), priority="CRITICAL",
        )
    if official_sec and "item 1.01" in title and "item 2.02" not in title and (
        predicted_category != "CORPORATE_ACTION_OR_CONTRACT"
    ):
        _flag(
            findings, "SEC_AGREEMENT_ITEM_CATEGORY_CONFLICT",
            detail=f"Item 1.01 agreement classified {predicted_category}.",
            priority="CRITICAL",
        )
    if article.get("status") == "NEEDS_CATEGORY_REVIEW":
        _flag(
            findings, "MODEL_POLICY_CLASSIFICATION_DISAGREEMENT",
            detail=f"Independent source policy: {primary_category}; "
                   f"model: {predicted_category}.",
        )
    # SEC provenance verifies an issuer's identity, but model must still
    # explain whose balance sheet or operations the event affects.
    if official_sec and mechanisms and not any(
        m.get("affectedScope") == "ISSUER" for m in mechanisms
    ):
        _flag(
            findings, "SEC_ISSUER_TRANSMISSION_NOT_IDENTIFIED",
            detail="Model inferred only industry/market implications for a "
                   "filing about a specific company's own activity.",
        )
    if official_bok and any(
        m.get("affectedScope") == "ISSUER" for m in mechanisms
    ):
        _flag(
            findings, "MACRO_REPORT_UNGROUNDED_SINGLE_ISSUER",
            detail="Official central-bank statistics do not themselves "
                   "identify a specific listed issuer.",
        )
    if official_sec and any(
        _XBRL.search(str(f.get("quote") or "")) for f in facts
    ):
        _flag(
            findings, "SEC_XBRL_IDENTIFIER_MISTAKEN_FOR_EVENT_EVIDENCE",
            detail="Source quotations include machine-readable taxonomy "
                   "identifiers; they do not prove debt was newly issued.",
        )
    if (official_sec or official_bok) and facts and all(
        f.get("factualStatus") == "ALLEGED" for f in facts
    ):
        _flag(
            findings, "OFFICIAL_SOURCE_ALL_STATEMENTS_ALLEGED",
            detail="All official-source statements are labeled ALLEGED. "
                   "Review actual quoted allegation vs official disclosure.",
        )
    if mechanisms and all(
        str(m.get("uncertainty") or "").strip().casefold()
        in _NO_MEANINGFUL_UNCERTAINTY for m in mechanisms
    ):
        _flag(
            findings, "UNCERTAINTY_NOT_EXPLAINED",
            detail="Economic hypotheses contain no case-specific uncertainty.",
        )
    sign = model["overallEconomicDirection"]
    local_directions = {
        str(m.get("economicDirection")) for m in mechanisms
    }
    if sign == "UNKNOWN" and local_directions.intersection(
        {"POSITIVE", "NEGATIVE"}
    ):
        _flag(
            findings, "OVERALL_VS_MECHANISM_DIRECTION_UNRECONCILED",
            detail="Directional mechanism but UNKNOWN overall impact; "
                   "the uncertainty and aggregation need an explanation.",
        )
    if official_bok and (
        "금융안정" in title or "산업연관표" in title
    ) and sign in {"POSITIVE", "NEGATIVE", "MIXED"}:
        _flag(
            findings, "METHODOLOGY_OR_REVIEW_MARKET_EFFECT_UNPROVEN",
            detail="Statistical methodology or stability review is not by "
                   "itself proof of a directional market shock.",
        )
    if official_bok and "북한" in title and (
        sign in {"POSITIVE", "NEGATIVE"}
    ):
        _flag(
            findings, "CROSS_BORDER_MARKET_MAPPING_UNPROVEN",
            detail="North Korean GDP growth is not direct evidence for "
                   "a Korean or U.S. investable-market direction.",
        )
    if official_bok and any(
        marker in str(f.get("quote") or "") for f in facts
        for marker in _PIT_ATTACHMENT_MARKERS
    ) and not model.get("missingEvidence"):
        _flag(
            findings, "SOURCE_ATTACHMENT_EVIDENCE_MISSING",
            detail="Original release references an attachment, but no "
                   "missing source material is acknowledged.",
        )
    # Facts in a reasoning paragraph must not invent specific financial
    # figures absent from all *selected* source quotations. This flags
    # discrepancies for review, not an automatic statement of falsehood:
    # the figure may exist elsewhere in an omitted source excerpt.
    cited_text = " ".join(
        str(row.get("quote") or "") for row in facts
    ) + " " + " ".join(
        str(row.get("groundingQuote") or "") for row in mechanisms
    )
    uncited_numbers: set[str] = set()
    for mechanism in mechanisms:
        reasoning = str(mechanism.get("reasoning") or "")
        for value in re.findall(r"(?<![\\w])\\d[\\d,]*(?:\\.\\d+)?%?", reasoning):
            compact = value.rstrip(",").replace(",", "")
            if (len(compact.replace(".", "").replace("%", "")) >= 3
                    and compact not in {"2024", "2025", "2026", "2027"}
                    and compact not in cited_text):
                uncited_numbers.add(compact)
    if uncited_numbers:
        _flag(
            findings, "NUMERIC_MECHANISM_NOT_IN_SELECTED_EVIDENCE",
            detail="Numbers in reasoning absent from the selected source "
                   "quotes (not necessarily absent from the full document): "
                   + ", ".join(sorted(uncited_numbers)[:8]),
        )
    # Conservative deterministic numeric/unit traps observed in the heldout
    # study. These trigger an audit, never a synthesized corrected value.
    for fact in facts:
        interpretation = str(fact.get("interpretation") or "")
        quote = str(fact.get("quote") or "")
        if (re.search(r"\d+억\s*[\d,]+만원", quote)
                and re.search(r"(?:million|billion)\s*(?:KRW|won)", interpretation, re.I)):
            _flag(
                findings, "KOREAN_LARGE_NUMBER_UNIT_CONVERSION_REVIEW",
                detail="Interpretation converts 억/만원 into English monetary "
                       "units; verify the scale independently.",
                priority="CRITICAL",
            )
            break
    for fact in facts:
        interpretation = str(fact.get("interpretation") or "")
        quote = str(fact.get("quote") or "")
        if ("전기대비" in quote and re.search(
            r"(year.over.year|yoy)", interpretation, re.I,
        ) and not re.search(r"전년동기대비|전년대비", quote)):
            _flag(
                findings, "PERIOD_COMPARISON_QOQ_VS_YOY_REVIEW",
                detail="Source describes preceding-period change while "
                       "interpretation describes year-over-year change.",
                priority="CRITICAL",
            )
            break
    return {
        "articleId": article.get("articleId"),
        "title": article.get("title"),
        "source": source,
        "originalModelStatus": article.get("status"),
        "originalModelCategory": predicted_category,
        "sourcePolicyCategory": primary_category,
        "economicDirectionUnmodified": sign,
        "qualityStatus": (
            _STATUS[0] if any(f["priority"] == "CRITICAL" for f in findings)
            else _STATUS[1] if findings else _STATUS[2]
        ),
        "findings": findings,
        "semanticClaimsIndependentlyCertified": False,
        "manualReviewStillRequired": True,
    }


def audit_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schemaVersion") != "frozen_thirty_source_grounded_semantic_v1":
        raise ValueError("UNSUPPORTED_SEMANTIC_REPLAY_SCHEMA")
    original = report.get("articles")
    if not isinstance(original, list) or len(original) != report.get("articleCount"):
        raise ValueError("SEMANTIC_COHORT_COUNT_MISMATCH")
    audits = [audit_article(item) for item in original]
    status_counts: Counter[str] = Counter(
        item["qualityStatus"] for item in audits
    )
    codes: Counter[str] = Counter(
        row["code"] for item in audits for row in item["findings"]
    )
    return {
        "schemaVersion": "independent_semantic_quality_review_v1",
        "originalHoldoutSha256": report["originalHoldoutSha256"],
        "reviewedArticles": len(audits),
        "priceDataAccessed": False,
        "originalPredictionsModified": False,
        "modelWasRerun": False,
        "independentlyGoldLabeled": False,
        "qualityStatusCounts": dict(sorted(status_counts.items())),
        "findingCodeCounts": dict(sorted(codes.items())),
        "articles": audits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output.exists() or args.output.is_symlink()
            or args.semantic_report.is_symlink()):
        parser.error("Unsafe input or existing output report")
    if not args.semantic_report.is_file():
        parser.error("Semantic report does not exist")
    original = json.loads(args.semantic_report.read_text(encoding="utf-8"))
    result = audit_report(original)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as writer:
        json.dump(result, writer, indent=2, ensure_ascii=False)
    print(json.dumps({
        key: value for key, value in result.items() if key != "articles"
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
