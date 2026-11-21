"""Rebuild a quarantined BOK cohort from fresh official PDF downloads.

No fallback to prior attachment text, cached summaries, or silently clipped sources.
This CLI operates on the user's local PDF files. Official release URLs are recorded
for provenance and the user must independently download each PDF from that page.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader

RELEASES = {
    194: {
        "title": "2026년 5월 생산자물가지수(잠정)",
        "officialUrl": "https://www.bok.or.kr/portal/bbs/B0000501/view.do?menuNo=201264&nttId=10098549",
        "releasedDate": "2026-06-19",
        "searchTerms": ("생산자물가", "2026"),
    },
    204: {
        "title": "2026년 5월 기업경기조사 결과 및 경제심리지수(ESI)",
        "officialUrl": "https://www.bok.or.kr/portal/bbs/B0000501/view.do?menuNo=201264&nttId=10098124",
        "releasedDate": "2026-05-27",
        "searchTerms": ("기업", "심리"),
    },
    198: {
        "title": "2026년 1/4분기 국민소득(잠정)",
        "officialUrl": "https://www.bok.or.kr/portal/bbs/B0000501/view.do?menuNo=201264&nttId=10098384",
        "releasedDate": "2026-06-09",
        "searchTerms": ("국내총생산", "국민"),
    },
    117: {
        "title": "경제전망(2026년 8월)",
        "officialUrl": "https://www.bok.or.kr/portal/bbs/B0000502/view.do?menuNo=201263&nttId=11064211",
        "releasedDate": "2026-08-27",
        "searchTerms": ("경제전망", "2026"),
    },
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inspect_pdf(path: Path, record: dict) -> tuple[dict, dict]:
    """Preserve page layout, never truncate text, and reject uncertain extraction."""
    if path.is_symlink() or not path.is_file():
        return {}, {"status": "PDF_MISSING_OR_UNSAFE", "path": str(path)}
    raw = path.read_bytes()
    if not raw.startswith(b"%PDF-"):
        return {}, {"status": "NOT_PDF", "path": str(path)}
    try:
        reader = PdfReader(path, strict=True)
        if reader.is_encrypted:
            return {}, {"status": "ENCRYPTED_PDF", "path": str(path)}
        pages = []
        page_audit = []
        for index, page in enumerate(reader.pages, 1):
            layout = page.extract_text(extraction_mode="layout") or ""
            # Unlike plain_text(), retain page and row boundaries for tables.
            pages.append(f"[PDF_PAGE_{index}]\n{layout.rstrip()}")
            page_audit.append({
                "page": index,
                "textChars": len(layout.strip()),
                "lineCount": len(layout.splitlines()),
            })
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {}, {"status": "PDF_EXTRACTION_ERROR", "reason": type(exc).__name__}
    text = "\n\n".join(pages)
    normalized = re.sub(r"\s+", "", text)
    missing = [term for term in record["searchTerms"] if term not in normalized]
    empty_pages = [row["page"] for row in page_audit if row["textChars"] < 50]
    audit = {
        "status": "READY_FOR_MANUAL_TABLE_REVIEW",
        "sha256": digest(raw),
        "bytes": len(raw),
        "pageCount": len(page_audit),
        "textChars": len(text),
        "pages": page_audit,
        "emptyOrSparsePages": empty_pages,
        "missingExpectedTerms": missing,
        "officialUrl": record["officialUrl"],
        "releaseDate": record["releasedDate"],
        "pdfFile": str(path),
    }
    if not page_audit or len(normalized) < 300:
        audit["status"] = "INSUFFICIENT_PDF_TEXT"
    elif empty_pages or missing:
        audit["status"] = "NEEDS_MANUAL_PDF_INSPECTION"
    article = {
        "title": record["title"],
        "sourceUrl": record["officialUrl"],
        "publishedAtReported": record["releasedDate"],
        "source": "bok_official_fresh_pdf",
        "bodyFormat": "PDF_LAYOUT",
        "body": text,
        "officialPdfSha256": audit["sha256"],
        "originalPdfPageCount": len(page_audit),
    }
    return article, audit


def build_cohort(pdf_dir: Path) -> tuple[dict, dict]:
    articles = []
    reviews = []
    for article_id, record in RELEASES.items():
        path = pdf_dir / f"{article_id}.pdf"
        article, audit = inspect_pdf(path, record)
        audit["articleId"] = article_id
        reviews.append(audit)
        if article and audit["status"] == "READY_FOR_MANUAL_TABLE_REVIEW":
            articles.append({"articleId": article_id, **article})
    valid = len(articles) == len(RELEASES)
    report = {
        "schemaVersion": "bok_fresh_pdf_readiness_v1",
        "status": "AWAITING_MANUAL_TABLE_REVIEW" if valid else "BLOCKED_INVALID_SOURCE",
        "candidateCount": len(RELEASES),
        "structurallyExtractedCount": len(articles),
        "note": (
            "Page text has been extracted without clipping. Human must inspect PDF "
            "tables, numerical columns, units and release dates before approving."
        ),
        "items": reviews,
    }
    return {"articles": articles}, report


def quarantine_old(old_path: Path, quarantine_dir: Path) -> dict:
    """Explicitly isolate only named legacy attachment report; never erase it."""
    if old_path.is_symlink() or not old_path.is_file():
        return {"status": "OLD_REPORT_MISSING_OR_UNSAFE"}
    target = quarantine_dir / old_path.name
    if target.exists() or target.is_symlink():
        return {"status": "QUARANTINE_DESTINATION_EXISTS"}
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    checksum = digest(old_path.read_bytes())
    old_path.rename(target)
    return {
        "status": "QUARANTINED_NOT_FOR_REUSE",
        "file": str(target),
        "sha256": checksum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--quarantine-old", type=Path)
    parser.add_argument("--quarantine-dir", type=Path)
    args = parser.parse_args()
    for output in (args.output, args.audit):
        if output.exists() or output.is_symlink():
            parser.error(f"Refusing to overwrite {output}")
    if args.output == args.audit:
        parser.error("Input and audit outputs must differ")
    if args.quarantine_old and not args.quarantine_dir:
        parser.error("--quarantine-dir required with --quarantine-old")
    data, report = build_cohort(args.pdf_dir)
    if args.quarantine_old:
        report["legacyDataDisposal"] = quarantine_old(args.quarantine_old, args.quarantine_dir)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    if report["status"] != "AWAITING_MANUAL_TABLE_REVIEW":
        print(json.dumps(report, ensure_ascii=False))
        raise SystemExit("Source audit failed: no model input written")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    print(json.dumps({
        "status": report["status"],
        "articleCount": len(data["articles"]),
        "auditPath": str(args.audit),
        "outputPath": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
