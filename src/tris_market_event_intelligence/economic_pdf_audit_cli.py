"""Inspect stored official PDF evidence without modifying the news database."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sqlite3
from pathlib import Path

from pypdf import PdfReader


def audit(path: Path) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = {}
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute("""
            SELECT a.id,a.title,f.raw_sha256,f.raw_payload,f.sparse_pages
            FROM articles a JOIN article_full_text f ON f.article_id=a.id
            WHERE a.source IN ('bok_all','bok_statistics','bok_corrections')
              AND f.status='REVIEW_REQUIRED' AND f.format='PDF'
            ORDER BY a.id
        """).fetchall()
    for article_id, title, digest, payload, sparse in rows:
        if payload is None or hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"Invalid stored PDF evidence for article {article_id}")
        groups.setdefault(digest, []).append({
            "articleId": article_id, "title": title,
            "payload": payload, "sparse": json.loads(sparse),
        })
    documents = []
    for digest, records in groups.items():
        reader = PdfReader(io.BytesIO(records[0]["payload"]), strict=True)
        pages = []
        for number in sorted({n for record in records for n in record["sparse"]}):
            page = reader.pages[number - 1]
            plain = (page.extract_text() or "").strip()
            contents = page.get_contents()
            operators = (
                [operator.decode("ascii", errors="replace") for _, operator in contents.operations]
                if contents is not None else []
            )
            pages.append({
                "page": number, "plainTextChars": len(plain),
                "imageCount": len(page.images),
                "contentOperators": len(operators),
                "hasPaintOperators": any(op in {
                    "Do", "BI", "INLINE IMAGE", "Tj", "TJ", "F", "f", "f*",
                    "S", "s", "B", "B*", "b", "b*", "sh", "'", '"',
                } for op in operators),
                "requiresVisualReview": True,
            })
        documents.append({
            "sha256": digest, "articleIds": [r["articleId"] for r in records],
            "title": records[0]["title"], "pageCount": len(reader.pages),
            "sparsePages": pages,
        })
    return {
        "reviewRows": len(rows), "uniqueDocuments": len(documents),
        "documents": documents, "databaseModified": False,
        "analysisPerformed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.db), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
