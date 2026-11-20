"""Read-only crosscheck of frozen model statements against extracted BOK evidence.

This is a diagnostic, not a factual gold label or an update to predictions.
Presence of a numeral in a document never proves an economic interpretation.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from tris_market_event_intelligence.bok_source_numeric_inventory_cli import COMPARISONS, QUANTITY

UNITS = {"%": "PERCENT", "％": "PERCENT", "퍼센트": "PERCENT",
         "%p": "PERCENT_POINT", "퍼센트포인트": "PERCENT_POINT",
         "bp": "BASIS_POINT", "bps": "BASIS_POINT",
         "조원": "KRW_TRILLION", "억원": "KRW_HUNDRED_MILLION",
         "만원": "KRW_TEN_THOUSAND", "원": "KRW_WON",
         "달러": "USD"}
MAX_FINDINGS = 160


def numeric_mentions(text: str) -> list[dict[str, str]]:
    mentions: list[dict[str, str]] = []
    for match in QUANTITY.finditer(text):
        unit = re.sub(r"\s+", "", match.group("unit") or "")
        if not unit:
            continue  # Dates and unqualified digits are not safe cross-document comparisons.
        value = match.group("number").replace(",", "")
        # Preserve precision and currency *scale*. Do not equate 1억 with 1만원.
        mentions.append({"value": value, "unit": UNITS.get(unit, unit),
                         "verbatim": match.group(0)})
    return mentions


def periods(text: str) -> set[str]:
    return {key for key, regex in COMPARISONS.items() if regex.search(text)}


def compare_claim(text: str, quote: str, source_numbers: set[tuple[str, str]]) -> dict:
    claim = numeric_mentions(text)
    quote_numeric_mentions = numeric_mentions(quote)
    quote_mentions = {(m["value"], m["unit"]) for m in quote_numeric_mentions}
    claimed_mentions = {(m["value"], m["unit"]) for m in claim}
    omitted_quote_numbers = [
        mention for mention in quote_numeric_mentions
        if (mention["value"], mention["unit"]) not in claimed_mentions
    ]
    source_periods = periods(quote)
    claimed_periods = periods(text)
    # If explicit text states contradictory comparison bases, flag review.
    period_contradiction = bool(source_periods and claimed_periods
                                and source_periods.isdisjoint(claimed_periods))
    entries = []
    for mention in claim[:MAX_FINDINGS]:
        key = (mention["value"], mention["unit"])
        if key in quote_mentions:
            state = "MATCHED_IN_MODEL_QUOTE"
        elif key in source_numbers:
            state = "MATCHED_IN_ATTACHMENT"
        else:
            state = "NO_EXACT_UNIT_VALUE_MATCH"
        entries.append({**mention, "evidenceStatus": state})
    return {
        "numericClaims": entries,
        "sourceQuotedNumericCount": len(quote_numeric_mentions),
        "sourceQuotedNumbersNotRepeated": omitted_quote_numbers[:MAX_FINDINGS],
        "sourceQuotedNumberOmissionReview": bool(omitted_quote_numbers),
        "comparisonPeriodInQuotedEvidence": sorted(source_periods),
        "comparisonPeriodInInterpretation": sorted(claimed_periods),
        "comparisonPeriodConflict": period_contradiction,
        "reviewStatus": (
            "PERIOD_CONFLICT_REVIEW" if period_contradiction
            else "NUMERIC_SOURCE_REVIEW" if any(
                n["evidenceStatus"] == "NO_EXACT_UNIT_VALUE_MATCH" for n in entries
            )
            else "SOURCE_NUMBER_PRESENCE_ONLY" if entries
            else "NO_EXPLICIT_NUMERIC_CLAIM"
        ),
        "factualCorrectnessIndependentlyVerified": False,
    }


def audit(semantic: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    if semantic.get("schemaVersion") != "frozen_thirty_source_grounded_semantic_v1":
        raise ValueError("UNSUPPORTED_SEMANTIC_SCHEMA")
    if inventory.get("schemaVersion") != "bok_source_numeric_inventory_v1":
        raise ValueError("UNSUPPORTED_NUMERIC_INVENTORY_SCHEMA")
    cohort = semantic.get("originalHoldoutSha256")
    if not cohort or cohort != inventory.get("originalHoldoutSha256"):
        raise ValueError("FROZEN_COHORT_HASH_MISMATCH")
    semantic_rows = semantic.get("articles")
    numeric_rows = inventory.get("articles")
    if (not isinstance(semantic_rows, list) or not isinstance(numeric_rows, list)
            or len(semantic_rows) != semantic.get("articleCount")
            or len(numeric_rows) != inventory.get("articleCount")):
        raise ValueError("COHORT_SIZE_MISMATCH")
    semantic_by_id = {r["articleId"]: r for r in semantic_rows}
    if len(semantic_by_id) != len(semantic_rows):
        raise ValueError("DUPLICATE_SEMANTIC_ARTICLE")
    if len({r["articleId"] for r in numeric_rows}) != len(numeric_rows):
        raise ValueError("DUPLICATE_INVENTORY_ARTICLE")
    if not {r["articleId"] for r in numeric_rows}.issubset(semantic_by_id):
        raise ValueError("ARTICLE_ID_NOT_IN_FROZEN_COHORT")
    results: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for nr in numeric_rows:
        article_id = nr["articleId"]
        original = semantic_by_id[article_id]
        model = (original.get("semanticAnalysis") or {}).get("modelAnalysis")
        source_numbers: set[tuple[str, str]] = set()
        for attachment in nr.get("attachmentEvidence", []):
            for number in attachment.get("numericCandidates", []):
                unit = str(number.get("unit", ""))
                unit = UNITS.get(unit, unit)
                if unit != "NOT_EXPLICIT":
                    source_numbers.add((str(number.get("numberText", "")).replace(",", ""), unit))
        claims = []
        if isinstance(model, dict):
            for i, fact in enumerate(model.get("sourceStatements", [])):
                verdict = compare_claim(
                    str(fact.get("interpretation", "")),
                    str(fact.get("quote", "")), source_numbers,
                )
                claims.append({"location": f"sourceStatements[{i}].interpretation",
                               "sourceQuote": fact.get("quote"), **verdict})
            for i, mechanism in enumerate(model.get("economicMechanisms", [])):
                verdict = compare_claim(
                    str(mechanism.get("reasoning", "")),
                    str(mechanism.get("groundingQuote", "")), source_numbers,
                )
                claims.append({"location": f"economicMechanisms[{i}].reasoning",
                               "sourceQuote": mechanism.get("groundingQuote"), **verdict})
        for claim in claims:
            counts[claim["reviewStatus"]] += 1
        results.append({
            "articleId": article_id,
            "originalModelStatus": original.get("status"),
            "originalEconomicDirection": model.get("overallEconomicDirection")
            if isinstance(model, dict) else None,
            "attachmentSourceCount": len(nr.get("attachmentEvidence", [])),
            "numericClaimReviews": claims,
            "modelWasRerun": False,
            "economicDirectionUnmodified": True,
        })
    return {
        "schemaVersion": "bok_model_numeric_crosscheck_v1",
        "originalHoldoutSha256": cohort, "articleCount": len(results),
        "reviewStatusCounts": dict(sorted(counts.items())),
        "sourceQuoteNumberOmissionReviewCount": sum(
            claim["sourceQuotedNumberOmissionReview"]
            for article in results for claim in article["numericClaimReviews"]
        ),
        "modelWasRerun": False, "priceDataAccessed": False,
        "originalPredictionsModified": False, "pointInTimeCertified": False,
        "independentlyGoldLabeled": False,
        "articles": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-report", type=Path, required=True)
    parser.add_argument("--numeric-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--examples", type=int, default=8,
        help="Show up to N source-quote number omissions or factual-review cases",
    )
    args = parser.parse_args()
    if not 0 <= args.examples <= 50:
        parser.error("--examples must be between 0 and 50")
    for path in (args.semantic_report, args.numeric_inventory):
        if not path.is_file() or path.is_symlink():
            parser.error("Missing or unsafe input")
    if args.output.exists() or args.output.is_symlink():
        parser.error("Refusing to overwrite existing report")
    result = audit(json.loads(args.semantic_report.read_text(encoding="utf-8")),
                   json.loads(args.numeric_inventory.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items() if k != "articles"},
                     ensure_ascii=False))
    shown = 0
    for article in result["articles"]:
        for claim in article["numericClaimReviews"]:
            if shown >= args.examples:
                return
            if (not claim["sourceQuotedNumberOmissionReview"]
                    and claim["reviewStatus"] not in {
                        "PERIOD_CONFLICT_REVIEW", "NUMERIC_SOURCE_REVIEW"
                    }):
                continue
            print(json.dumps({
                "articleId": article["articleId"],
                "location": claim["location"],
                "reviewStatus": claim["reviewStatus"],
                "sourceQuotedNumbersNotRepeated": claim["sourceQuotedNumbersNotRepeated"][:8],
                "sourceQuote": claim["sourceQuote"],
            }, ensure_ascii=False))
            shown += 1


if __name__ == "__main__":
    main()
