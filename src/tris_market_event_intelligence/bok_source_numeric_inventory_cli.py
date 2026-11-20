"""Offline source-number inventory for retrospective BOK attachments.

Only extracted source spans are recorded. A matched number is NOT a confirmed
financial fact, source publication proof, interpretation or market forecast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

# Do not silently treat a date, quantity, interest rate and currency as interchangeable.
QUANTITY = re.compile(
    r"(?<![\w.])(?P<sign>[-+−]?)\s*(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(?P<unit>조\s*\d+(?:\.\d+)?\s*억\s*원|조\s*원|억\s*\d+(?:\.\d+)?\s*만\s*원|"
    r"억원|억\s*원|만원|만\s*원|원|달러|%|％|퍼센트포인트|%p|bp|bps|퍼센트)?",
    re.I,
)
COMPARISONS = {
    "QOQ": re.compile(r"전기\s*대비|전분기\s*대비|quarter.on.quarter|quarter.over.quarter|\bqoq\b", re.I),
    "YOY": re.compile(r"전년\s*동기\s*대비|전년\s*대비|year.on.year|year.over.year|\byoy\b", re.I),
    "MOM": re.compile(r"전월\s*대비|month.on.month|month.over.month|\bmom\b", re.I),
}
SIGNS = {
    "INCREASE": re.compile(r"증가|상승|확대|올랐|늘었", re.I),
    "DECREASE": re.compile(r"감소|하락|축소|내렸|줄었", re.I),
}
SCALES = {"조원": Decimal(10)**12, "억원": Decimal(10)**8,
          "만원": Decimal(10)**4, "원": Decimal(1)}
MAX_TEXT = 120_000
MAX_NUMBERS = 500


def scale_value(number: str, unit: str) -> str | None:
    """Normalize only simple unambiguous Korean won units; never infer FX."""
    clean = re.sub(r"\s+", "", unit)
    scale = SCALES.get(clean)
    if scale is None:
        return None
    value = Decimal(number.replace(",", "")) * scale
    return format(value, "f")


def extract_quantities(text: str) -> list[dict[str, Any]]:
    """Enumerate verbatim numeric candidates and local context, not model conclusions."""
    rows: list[dict[str, Any]] = []
    for match in QUANTITY.finditer(text[:MAX_TEXT]):
        if len(rows) >= MAX_NUMBERS:
            break
        original = match.group(0)
        number = match.group("number")
        unit = re.sub(r"\s+", "", match.group("unit") or "")
        before = text[max(0, match.start() - 75):match.start()]
        after = text[match.end():min(len(text), match.end() + 75)]
        context = before + original + after
        periods = [key for key, regex in COMPARISONS.items() if regex.search(context)]
        trends = [key for key, regex in SIGNS.items() if regex.search(context)]
        rows.append({
            "sourceStart": match.start(), "sourceEnd": match.end(),
            "verbatim": original, "numberText": number,
            "unit": unit or "NOT_EXPLICIT",
            "normalizedKrw": scale_value(number, unit) if unit else None,
            "comparisonCandidates": periods,
            "directionWordCandidates": trends,
            "sourceContext": context,
            "sourceSpanVerified": text[match.start():match.end()] == original,
            "economicMeaningVerified": False,
        })
    return rows


def audit_attachment_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schemaVersion") != "bok_attachment_evidence_v1":
        raise ValueError("UNSUPPORTED_ATTACHMENT_SCHEMA")
    articles = report.get("articles")
    if not isinstance(articles, list) or len(articles) != report.get("targetedCount"):
        raise ValueError("ATTACHMENT_COHORT_MISMATCH")
    if len({a.get("articleId") for a in articles}) != len(articles):
        raise ValueError("DUPLICATE_ARTICLE_ID")
    records: list[dict[str, Any]] = []
    status: Counter[str] = Counter()
    for article in articles:
        attachment_rows = []
        for entry in article.get("attachments", []):
            if entry.get("extractionStatus") != "TEXT_EXTRACTED":
                continue
            extracted = entry.get("extractedText")
            if not isinstance(extracted, str):
                raise ValueError("MISSING_EXTRACTED_TEXT")
            digest = hashlib.sha256(extracted.encode()).hexdigest()
            if digest != entry.get("extractedTextSha256"):
                raise ValueError("EXTRACTED_TEXT_HASH_MISMATCH")
            items = extract_quantities(extracted)
            attachment_rows.append({
                "sourceUrl": entry.get("sourceUrl"),
                "extractedTextSha256": digest,
                "sourceCharacters": len(extracted),
                "numericCandidateCount": len(items),
                "numericCandidates": items,
            })
        state = "NUMERIC_EVIDENCE_INVENTORIED" if attachment_rows else "NO_READABLE_ATTACHMENT"
        status[state] += 1
        records.append({
            "articleId": article.get("articleId"),
            "status": state,
            "attachmentEvidence": attachment_rows,
            "originalPublicationPITCertified": False,
            "numericInterpretationCertified": False,
            "economicDirectionPredicted": False,
        })
    return {
        "schemaVersion": "bok_source_numeric_inventory_v1",
        "originalHoldoutSha256": report.get("originalHoldoutSha256"),
        "articleCount": len(records),
        "statusCounts": dict(sorted(status.items())),
        "numericCandidates": sum(
            item["numericCandidateCount"]
            for a in records for item in a["attachmentEvidence"]
        ),
        "sourceIsRetrospective": True,
        "pointInTimeCertified": False,
        "priceDataAccessed": False,
        "originalPredictionsModified": False,
        "articles": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.attachments.is_file() or args.attachments.is_symlink():
        parser.error("Unsafe or missing attachment report")
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output already exists")
    report = json.loads(args.attachments.read_text(encoding="utf-8"))
    inventory = audit_attachment_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(inventory, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in inventory.items() if k != "articles"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
