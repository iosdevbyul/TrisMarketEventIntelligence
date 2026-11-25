"""Recover sparse official PDF pages into an independent evidence database."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import tempfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

SOURCES = ("bok_all", "bok_statistics", "bok_corrections", "fed_all", "fed_monetary")
SCHEMA = """
CREATE TABLE IF NOT EXISTS economic_pdf_recovery (
    raw_sha256 TEXT NOT NULL, page INTEGER NOT NULL, engine TEXT NOT NULL,
    dpi INTEGER NOT NULL, languages TEXT NOT NULL, image_sha256 TEXT NOT NULL,
    image_png BLOB NOT NULL, ocr_text TEXT NOT NULL, ocr_tsv TEXT NOT NULL,
    status TEXT NOT NULL, captured_at TEXT NOT NULL,
    PRIMARY KEY(raw_sha256,page,engine,dpi,languages)
);
"""


def recover(news: Path, output: Path, *, limit: int, dpi: int = 250) -> dict:
    if not news.is_file() or news.resolve() == output.resolve():
        raise ValueError("Separate existing source and recovery database required")
    if output.exists() and news.samefile(output):
        raise ValueError("Recovery database must not alias source")
    if not 1 <= limit <= 1000 or not 150 <= dpi <= 300:
        raise ValueError("Invalid page limit or dpi")
    import pymupdf

    tesseract = shutil.which("tesseract")
    if not tesseract:
        raise ValueError("Install Tesseract with Korean and English language data")
    langs = subprocess.run(
        [tesseract, "--list-langs"], capture_output=True, text=True, check=True, timeout=30,
    ).stdout.splitlines()
    if not {"kor", "eng"}.issubset(set(langs)):
        raise ValueError("Missing kor or eng language data")
    version = subprocess.run(
        [tesseract, "--version"], capture_output=True, text=True, check=True, timeout=30,
    ).stdout.splitlines()[0]
    engine = version + "; pymupdf " + pymupdf.VersionBind
    languages = "kor+eng"
    with closing(sqlite3.connect(news.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        rows = db.execute("""
            SELECT DISTINCT f.raw_sha256,f.sparse_pages
            FROM articles a JOIN article_full_text f ON f.article_id=a.id
            WHERE a.source IN (?,?,?,?,?) AND f.format='PDF'
              AND f.status='REVIEW_REQUIRED'
            ORDER BY f.raw_sha256
        """, SOURCES).fetchall()
        pending = {}
        for digest, sparse in rows:
            pages = json.loads(sparse)
            if not isinstance(pages, list) or any(type(p) is not int or p < 1 for p in pages):
                raise ValueError("Invalid sparse page metadata")
            pending.setdefault(digest, set()).update(pages)
        output.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(output)) as dest:
            dest.executescript(SCHEMA)
            counts = {"recoveredText": 0, "noText": 0, "failed": 0, "reused": 0}
            failures = []
            attempted = 0
            for digest, pages in pending.items():
                payload = db.execute(
                    "SELECT raw_payload FROM article_full_text WHERE raw_sha256=? LIMIT 1",
                    (digest,),
                ).fetchone()[0]
                if payload is None or hashlib.sha256(payload).hexdigest() != digest:
                    raise ValueError("Stored PDF digest mismatch")
                with pymupdf.open(stream=payload, filetype="pdf") as document:
                    for number in sorted(pages):
                        if dest.execute("""
                            SELECT 1 FROM economic_pdf_recovery
                            WHERE raw_sha256=? AND page=? AND engine=? AND dpi=? AND languages=?
                        """, (digest, number, engine, dpi, languages)).fetchone():
                            counts["reused"] += 1
                            continue
                        if attempted >= limit:
                            continue
                        attempted += 1
                        try:
                            if not 1 <= number <= len(document):
                                raise ValueError("Invalid page number")
                            page = document[number - 1]
                            if page.rect.width * dpi / 72 * page.rect.height * dpi / 72 > 20_000_000:
                                raise ValueError("Page exceeds pixel limit")
                            png = page.get_pixmap(dpi=dpi, alpha=False).tobytes("png")
                            with tempfile.TemporaryDirectory() as directory:
                                image = Path(directory) / "page.png"
                                image.write_bytes(png)
                                base = Path(directory) / "ocr"
                                subprocess.run([
                                    tesseract, str(image), str(base), "-l", languages,
                                    "--psm", "3", "txt", "tsv",
                                ], capture_output=True, check=True, timeout=120)
                                text = base.with_suffix(".txt").read_text(encoding="utf-8")
                                tsv = base.with_suffix(".tsv").read_text(encoding="utf-8")
                            status = "OCR_TEXT" if text.strip() else "OCR_NO_TEXT"
                            with dest:
                                dest.execute("""
                                    INSERT OR IGNORE INTO economic_pdf_recovery
                                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                                """, (
                                    digest, number, engine, dpi, languages,
                                    hashlib.sha256(png).hexdigest(), png, text, tsv,
                                    status, datetime.now(UTC).isoformat(),
                                ))
                            counts["recoveredText" if text.strip() else "noText"] += 1
                        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
                            counts["failed"] += 1
                            failures.append({
                                "sha256": digest, "page": number, "error": type(exc).__name__,
                            })
    total = sum(map(len, pending.values()))
    return {
        "uniqueDocuments": len(pending), "targetPages": total, "attempted": attempted,
        **counts, "remainingPages": total - counts["reused"] - counts["recoveredText"]
        - counts["noText"], "failures": failures, "sourceDatabaseModified": False,
        "captureStatusChanged": False, "visualReviewRequired": True,
        "analysisPerformed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output-db", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dpi", type=int, default=250)
    args = parser.parse_args()
    report = recover(args.db, args.output_db, limit=args.limit, dpi=args.dpi)
    print(json.dumps(report, ensure_ascii=False))
    if report["failed"] or report["remainingPages"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
