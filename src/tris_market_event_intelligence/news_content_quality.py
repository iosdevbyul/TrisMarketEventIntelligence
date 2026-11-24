"""Conservative content screening. Admission is not semantic verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from urllib.parse import urlsplit

POLICY = "content_quality_v1"
_BLOCK = re.compile(
    r"^(?:close|loading|advertisement|subscribe|sign in|log in|"
    r"not enough text|summary only|read more|watch the video|"
    r"click here|access denied|page not found|enable javascript|"
    r"로그인|광고|자세히 보기|영상에서 확인하세요)[.!\s]*$", re.I,
)
_ACCESS = re.compile(
    r"^(?:please (?:sign in|log in|subscribe|enable javascript)|"
    r"subscribe to (?:read|continue)|sign in to (?:read|continue)|"
    r"access to this (?:page|article) is denied|로그인 후|구독 후)", re.I,
)


def media_page(url: str) -> bool:
    path = urlsplit(url).path.lower()
    return path.startswith(("/iplayer/", "/news/videos/", "/news/av/"))


def classify(body: object, *, url: str = "", title: str = "",
             digest: str | None = None, sparse: object = None) -> str:
    """No minimum character count. Unknown completeness is not verified."""
    if not isinstance(body, str) or not body.strip():
        return "EMPTY_CONTENT"
    if digest is not None and hashlib.sha256(body.encode()).hexdigest() != digest:
        return "DIGEST_MISMATCH"
    if sparse not in (None, "[]"):
        return "PARTIAL_CONTENT"
    normalized = " ".join(body.split())
    if not any(char.isalnum() for char in normalized):
        return "EMPTY_CONTENT"
    if _BLOCK.fullmatch(normalized) or _ACCESS.match(normalized):
        return "NO_INFORMATION"
    # Video descriptions are not transcripts. Do not infer completeness from
    # text volume; keep them for source review, including potential short alerts.
    if media_page(url):
        return "MEDIA_REVIEW_REQUIRED"
    if normalized.endswith("?") or normalized.endswith("？"):
        if normalized == " ".join(title.split()) or "\n" not in body:
            return "QUESTION_ONLY_REVIEW"
    return "TEXT_CANDIDATE"


def admissible(body: object, **kwargs) -> bool:
    return classify(body, **kwargs) == "TEXT_CANDIDATE"


def audit(path: Path) -> dict:
    if not path.is_file():
        raise ValueError("Existing news DB required")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        completed = set()
        if "automatic_news_analyses" in tables:
            completed = set(db.execute(
                "SELECT DISTINCT article_id,body_sha256 FROM automatic_news_analyses "
                "WHERE status='COMPLETED'"
            ))
        counts: Counter[str] = Counter()
        impacted: Counter[str] = Counter()
        by_source: dict[str, Counter[str]] = {}
        excluded = []
        for article_id, source, title, url, status, body, digest, sparse in db.execute(
            """SELECT a.id,a.source,a.title,a.url,f.status,f.body,f.body_sha256,f.sparse_pages
               FROM articles a LEFT JOIN article_full_text f ON a.id=f.article_id
               ORDER BY a.id"""
        ):
            quality = classify(body, url=url, title=title, digest=digest, sparse=sparse)
            if status != "COMPLETE" and quality == "TEXT_CANDIDATE":
                quality = "CAPTURE_NOT_COMPLETE"
            if status == "COMPLETE" and not digest:
                quality = "DIGEST_MISSING"
            counts[quality] += 1
            by_source.setdefault(source, Counter())[quality] += 1
            if quality != "TEXT_CANDIDATE":
                analyzed = (article_id, digest) in completed
                if analyzed:
                    impacted[quality] += 1
                excluded.append({
                    "articleId": article_id, "source": source, "quality": quality,
                    "bodySha256": digest, "existingCompletedAnalysis": analyzed,
                })
    return {
        "policy": POLICY, "discovered": sum(counts.values()),
        "counts": dict(counts), "bySource": by_source,
        "existingCompletedAnalysesExcluded": dict(impacted),
        "excluded": excluded, "databaseModified": False,
        "semanticCompletenessVerified": False,
        "marketReturnsAnalyzed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.db)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "excluded"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
