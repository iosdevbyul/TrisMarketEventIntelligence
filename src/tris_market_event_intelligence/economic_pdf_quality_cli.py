"""Read-only integrity and OCR confidence inventory; not semantic verification."""
from __future__ import annotations

import argparse
import csv
import io
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from tris_market_event_intelligence.economic_pdf_recovery_cli import valid_evidence


def audit(path: Path) -> dict:
    counts = {"rows": 0, "valid": 0, "invalid": 0, "textPages": 0, "noTextPages": 0,
              "lowConfidencePages": 0, "noWordConfidencePages": 0}
    pages = []
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        for digest, page, engine, dpi, languages, image_hash, png, text, tsv, status in db.execute("""
            SELECT raw_sha256,page,engine,dpi,languages,image_sha256,image_png,
                   ocr_text,ocr_tsv,status FROM economic_pdf_recovery
            ORDER BY raw_sha256,page,engine,dpi,languages
        """):
            counts["rows"] += 1
            valid = valid_evidence(image_hash, png, text, tsv, status)
            confidences = []
            if valid:
                try:
                    for word in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
                        if word["level"] == "5" and word["text"].strip():
                            value = float(word["conf"])
                            if not 0 <= value <= 100:
                                raise ValueError("Invalid confidence")
                            confidences.append(value)
                except (ValueError, KeyError, AttributeError, TypeError):
                    valid = False
            counts["valid" if valid else "invalid"] += 1
            if valid:
                counts["textPages" if text.strip() else "noTextPages"] += 1
                if not confidences:
                    counts["noWordConfidencePages"] += 1
                elif any(value < 60 for value in confidences):
                    counts["lowConfidencePages"] += 1
            pages.append({
                "sha256": digest, "page": page, "engine": engine, "dpi": dpi,
                "languages": languages, "valid": valid, "status": status,
                "textChars": len(text), "wordCount": len(confidences),
                "meanConfidence": round(sum(confidences) / len(confidences), 2)
                if confidences else None,
                "visualReviewRequired": True,
            })
    return {**counts, "pages": pages, "numericAccuracyVerified": False,
            "databaseModified": False, "analysisPerformed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.db)
    print(json.dumps(report, ensure_ascii=False))
    if report["invalid"] or not report["rows"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
