"""Optional, read-only evidence acquisition from official article attachments.

Does not mutate the original RSS archive. An enriched historical document is
retrospectively acquired and must not be treated as certified PIT evidence.
Only directly linked public PDF documents are supported; HWP, JS-driven and
otherwise inaccessible attachments remain explicitly unresolved.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
from pypdf.errors import PdfReadError

from tris_market_event_intelligence.event_evidence_policy import (
    content_quality,
    plain_text,
)
from tris_market_event_intelligence.historical_news_cohort import historical_news_cohort

ALLOWED_HOSTS = {"www.bok.or.kr", "bok.or.kr"}
MAX_BYTES = 5_000_000


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.href: str | None = None
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.text = []

    def handle_data(self, data: str) -> None:
        if self.href is not None:
            self.text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.href is not None:
            self.links.append((self.href, " ".join(self.text)))
            self.href = None


def authorized(url: str) -> bool:
    parts = urlsplit(url)
    return (parts.scheme == "https" and parts.hostname in ALLOWED_HOSTS
            and parts.username is None and parts.password is None)


def fetch(client: httpx.Client, url: str) -> httpx.Response:
    if not authorized(url):
        raise ValueError("SOURCE_HOST_NOT_ALLOWED")
    response = client.get(url, headers={"User-Agent": "TrisMarketEventIntelligence/1.0"})
    if response.is_redirect:
        raise ValueError("SOURCE_REDIRECT_REQUIRES_REVIEW")
    response.raise_for_status()
    if len(response.content) > MAX_BYTES:
        raise ValueError("SOURCE_TOO_LARGE")
    return response


def extract_pdf(blob: bytes) -> str:
    # The extraction library never executes embedded document code.
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(blob), strict=True)
    if len(reader.pages) > 60:
        raise ValueError("SOURCE_TOO_MANY_PAGES")
    return plain_text(" ".join(page.extract_text() or "" for page in reader.pages))


def source_document(client: httpx.Client, row: dict) -> dict:
    url = str(row.get("sourceUrl") or "")
    if not authorized(url):
        return {"status": "UNAVAILABLE", "reasonCode": "NON_OFFICIAL_SOURCE"}
    try:
        page = fetch(client, url)
        if "html" not in page.headers.get("content-type", "").lower():
            raise ValueError("SOURCE_NOT_HTML")
        parser = Links()
        parser.feed(page.text)
        candidates: list[str] = []
        for href, name in parser.links:
            link = urljoin(url, href.strip())
            if not authorized(link):
                continue
            if (re.search(r"\.pdf(?:$|[?#])", link, re.IGNORECASE)
                    or re.search(r"\bpdf\b", name, re.IGNORECASE)):
                if link not in candidates:
                    candidates.append(link)
        if not candidates:
            return {"status": "UNAVAILABLE",
                    "reasonCode": "PDF_ATTACHMENT_NOT_LINKED_OR_UNSUPPORTED"}
        for link in candidates[:3]:
            response = fetch(client, link)
            pdf = response.content
            if not pdf.startswith(b"%PDF-"):
                continue
            body = extract_pdf(pdf)
            enriched = {**row, "summary": body}
            if content_quality(enriched)["status"] != "ELIGIBLE":
                continue
            return {
                "status": "AVAILABLE",
                "sourceUrl": url, "attachmentUrl": link,
                "contentText": body[:25000],
                "contentSha256": hashlib.sha256(pdf).hexdigest(),
                "contentTextSha256": hashlib.sha256(body[:25000].encode()).hexdigest(),
                "captureSha256": row.get("captureSha256"),
                "publishedAtReported": row.get("publishedAtReported"),
                "acquiredAt": datetime.now(UTC).isoformat(),
                "pointInTimeCertified": False,
            }
        return {"status": "UNAVAILABLE", "reasonCode": "PDF_TEXT_NOT_EXTRACTABLE"}
    except (ValueError, PdfReadError, httpx.HTTPError, OSError,
            KeyError, IndexError) as exc:
        return {"status": "UNAVAILABLE", "reasonCode": type(exc).__name__ +
                ":" + str(exc)[:120]}


def enrich(source_db: Path, output: Path, as_of: datetime, limit: int,
           *, client: httpx.Client) -> dict:
    if output.is_symlink() or source_db.is_symlink() or output.resolve() == source_db.resolve():
        raise ValueError("Immutable source archive must not be overwritten")
    if not 1 <= limit <= 1000:
        raise ValueError("Limit must be 1..1000")
    cohort = historical_news_cohort(source_db=source_db, as_of=as_of, limit=5000)
    items = sorted(cohort["items"],
                   key=lambda r: (r.get("publishedAtReported") or "", r["articleId"]))
    cache: dict[str, dict] = {}
    eligible = 0
    for row in items:
        if row.get("publishedAtReported") is None:
            continue
        if content_quality(row)["status"] == "ELIGIBLE":
            eligible += 1
            continue
        if len(cache) >= limit:
            break
        cache[str(row["articleId"])] = source_document(client, row)
    result = {
        "schemaVersion": "retrospective_official_attachment_cache_v1",
        "sourceDb": str(source_db.resolve()),
        "originalArchiveModified": False,
        "pointInTimeCertified": False,
        "alreadySubstantiveArticlesSeen": eligible,
        "articles": cache,
    }
    # atomic write, no mutation of the source DB or any earlier cache file
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return {
        "fetched": len(cache),
        "available": sum(x["status"] == "AVAILABLE" for x in cache.values()),
        "unavailable": sum(x["status"] != "AVAILABLE" for x in cache.values()),
        "output": str(output),
        "pointInTimeCertified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    cutoff = datetime.fromisoformat(args.as_of)
    if cutoff.tzinfo is None:
        parser.error("--as-of requires timezone")
    with httpx.Client(timeout=20, follow_redirects=False) as client:
        print(json.dumps(enrich(args.source_db, args.output, cutoff,
                                args.limit, client=client), ensure_ascii=False))


if __name__ == "__main__":
    main()
