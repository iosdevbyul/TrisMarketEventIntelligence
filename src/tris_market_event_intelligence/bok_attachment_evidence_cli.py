"""Read-only retrospective Bank of Korea attachment evidence acquisition.

Downloaded bytes are NOT point-in-time proof or independently verified truth.
No source DB, training journal, historical forecast or price data is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

HOSTS = {"bok.or.kr", "www.bok.or.kr"}
FILE_HOSTS = HOSTS | {"file-cdn.bok.or.kr"}
MAX_REDIRECTS = 3
KEYWORDS = ("붙임", "첨부", "별첨", "download", ".pdf", ".hwp", ".hwpx", ".txt")
ROUTES = ("download", "filedown", "atch", ".pdf", ".hwp", ".hwpx", ".txt")
MAX_PAGE = 2_000_000
MAX_FILE = 12_000_000


def allowed(url: str, *, file_url: bool = False) -> bool:
    p = urlparse(url)
    return (p.scheme == "https"
            and p.hostname in (FILE_HOSTS if file_url else HOSTS)
            and p.port in (None, 443) and p.username is None
            and p.password is None and not p.fragment)


def safe_url_label(url: str) -> str:
    """Do not expose temporary CDN query tokens in audit reports."""
    parsed = urlparse(url)
    return parsed._replace(query="", fragment="").geturl()


def failure_reason(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP_{exc.response.status_code}"
    if isinstance(exc, httpx.TimeoutException):
        return "DOWNLOAD_TIMEOUT"
    if isinstance(exc, httpx.ConnectError):
        return "DOWNLOAD_CONNECTION_ERROR"
    if isinstance(exc, httpx.TransportError):
        return "DOWNLOAD_TRANSPORT_ERROR"
    return str(exc)[:120] if isinstance(exc, ValueError) else type(exc).__name__


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[dict[str, str]] = []
        self.active: dict[str, str] | None = None
        self.words: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            a = dict(attrs)
            self.active = {k: a.get(k) or "" for k in ("href", "onclick", "title", "download")}
            self.words = []

    def handle_data(self, data: str) -> None:
        if self.active is not None:
            self.words.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.active is not None:
            self.active["text"] = " ".join(" ".join(self.words).split())
            self.links.append(self.active)
            self.active = None


def attachment_links(html: str, page: str) -> tuple[list[dict], list[dict]]:
    parser = Links()
    parser.feed(html)
    found: list[dict] = []
    unresolved: list[dict] = []
    seen: set[str] = set()
    for item in parser.links:
        href = item["href"].strip()
        label = " ".join((item["text"], item["title"], item["download"]))
        if not any(k in (label + href + item["onclick"]).lower() for k in KEYWORDS):
            continue
        if not href or href == "#" or href.lower().startswith(("javascript:", "data:")):
            unresolved.append({"label": label[:140], "reason": "SCRIPT_ONLY_LINK"})
            continue
        target = urljoin(page, href)
        if not allowed(target):
            unresolved.append({"label": label[:140], "reason": "UNSAFE_LINK"})
            continue
        if not any(k in urlparse(target).path.lower() for k in ROUTES) and not item["download"]:
            unresolved.append({"label": label[:140], "reason": "UNCONFIRMED_DOWNLOAD_LINK"})
            continue
        if target not in seen:
            found.append({"url": target, "label": label[:140]})
            seen.add(target)
    return found, unresolved


def fetch(
    client: httpx.Client, url: str, limit: int, *, file_url: bool = False,
) -> tuple[bytes, str]:
    """Allow only a small number of redirects to approved BOK domains."""
    current = url
    for hop in range(MAX_REDIRECTS + 1):
        if not allowed(current, file_url=file_url):
            raise ValueError("UNSAFE_BOK_DOWNLOAD_DESTINATION")
        with client.stream("GET", current, follow_redirects=False) as response:
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise ValueError("REDIRECT_LOCATION_MISSING")
                target = urljoin(current, location)
                if not allowed(target, file_url=file_url):
                    raise ValueError("UNSAFE_BOK_REDIRECT_DESTINATION")
                if hop == MAX_REDIRECTS:
                    raise ValueError("REDIRECT_LIMIT_EXCEEDED")
                current = target
                continue
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_bytes():
                data.extend(chunk)
                if len(data) > limit:
                    raise ValueError("MAX_DOWNLOAD_EXCEEDED")
            return bytes(data), response.headers.get("content-type", "").split(";")[0].lower()
    raise ValueError("REDIRECT_LIMIT_EXCEEDED")

def extract(data: bytes, mime: str, url: str) -> tuple[str, str]:
    if data.startswith(b"%PDF-"):
        try:
            import fitz  # type: ignore[import-not-found]
        except ImportError:
            return "", "PDF_EXTRACTOR_UNAVAILABLE"
        try:
            with fitz.open(stream=data, filetype="pdf") as doc:
                if len(doc) > 250:
                    return "", "PDF_PAGE_LIMIT_EXCEEDED"
                text = " ".join(page.get_text() for page in doc)
        except (RuntimeError, ValueError):
            return "", "PDF_TEXT_EXTRACTION_FAILED"
    elif mime == "text/plain" or urlparse(url).path.lower().endswith(".txt"):
        text = data.decode("utf-8", errors="replace")
    elif data.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        return "", "HTML_INSTEAD_OF_ATTACHMENT"
    else:
        return "", "UNSUPPORTED_ATTACHMENT_FORMAT"
    text = " ".join(text.split())[:120_000]
    return text, "TEXT_EXTRACTED" if len(text) >= 50 else "NO_SUBSTANTIVE_TEXT"


def capture(article: dict, client: httpx.Client) -> dict:
    url = str(article.get("sourceUrl", ""))
    result: dict[str, Any] = {
        "articleId": article.get("articleId"), "sourceUrl": url,
        "observedAt": datetime.now(timezone.utc).isoformat(),
        "reportedPublishedAt": article.get("publishedAtReported"),
        "firstPublicationPITCertified": False,
        "attachments": [], "unresolvedLinks": [],
    }
    if not str(article.get("source", "")).startswith("bok_") or not allowed(url):
        result["status"] = "UNSUPPORTED_SOURCE"
        return result
    try:
        raw, mime = fetch(client, url, MAX_PAGE)
        result["pageSha256"] = hashlib.sha256(raw).hexdigest()
        if mime not in ("text/html", "application/xhtml+xml"):
            result["status"] = "PAGE_NOT_HTML"
            return result
        links, unresolved = attachment_links(raw.decode("utf-8", "replace"), url)
        result["unresolvedLinks"] = unresolved
        result["candidateCount"] = len(links)
        for link in links[:12]:
            item: dict[str, Any] = {"sourceUrl": safe_url_label(link["url"]), "label": link["label"]}
            try:
                content, kind = fetch(client, link["url"], MAX_FILE, file_url=True)
                item["rawSha256"] = hashlib.sha256(content).hexdigest()
                item["rawBytes"] = len(content)
                item["contentType"] = kind
                text, status = extract(content, kind, link["url"])
                item["extractionStatus"] = status
                if text:
                    item["extractedTextSha256"] = hashlib.sha256(text.encode()).hexdigest()
                    item["extractedText"] = text
            except (httpx.HTTPError, ValueError) as exc:
                item["extractionStatus"] = "RETRIEVAL_FAILED"
                item["errorCode"] = failure_reason(exc)
            result["attachments"].append(item)
        if len(links) > 12:
            result["unresolvedLinks"].append({
                "reason": "CANDIDATE_LIMIT", "remaining": len(links) - 12,
            })
        result["status"] = (
            "TEXT_EVIDENCE_ACQUIRED"
            if any(x.get("extractionStatus") == "TEXT_EXTRACTED"
                   for x in result["attachments"]) else "ATTACHMENTS_UNVERIFIED"
        )
    except (httpx.HTTPError, ValueError) as exc:
        result["status"] = "PAGE_RETRIEVAL_FAILED"
        result["errorCode"] = failure_reason(exc)
    return result


def acquire(articles: list[dict], readiness: dict, client: httpx.Client) -> dict:
    rows = readiness.get("articles")
    if not isinstance(rows, list) or len(articles) != readiness.get("articleCount"):
        raise ValueError("COHORT_COUNT_MISMATCH")
    source_ids = {x["articleId"] for x in articles}
    if (len(source_ids) != len(articles) or len(rows) != len(articles)
            or {x["articleId"] for x in rows} != source_ids):
        raise ValueError("COHORT_IDENTITY_MISMATCH")
    index = {x["articleId"]: x for x in articles}
    targets = [index[x["articleId"]] for x in rows
               if x.get("sourceReadiness") == "SOURCE_REVIEW_REQUIRED"]
    evidence = [capture(x, client) for x in targets]
    return {
        "schemaVersion": "bok_attachment_evidence_v1",
        "originalHoldoutSha256": readiness.get("originalHoldoutSha256"),
        "articleCount": len(articles), "targetedCount": len(targets),
        "statusCounts": {s: sum(x["status"] == s for x in evidence)
                         for s in sorted({x["status"] for x in evidence})},
        "pointInTimeCertified": False, "priceDataAccessed": False,
        "originalPredictionsModified": False, "sourceDbModified": False,
        "articles": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-result", type=Path, required=True)
    parser.add_argument("--readiness-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output exists; refusing to overwrite")
    for p in (args.holdout_result, args.readiness_report):
        if not p.is_file() or p.is_symlink():
            parser.error("Unsafe or missing input")
    from tris_market_event_intelligence.source_grounded_semantic_replay_cli import source_items
    articles, original_hash, _ = source_items(args.holdout_result)
    readiness = json.loads(args.readiness_report.read_text(encoding="utf-8"))
    if (readiness.get("schemaVersion") != "historical_source_readiness_v1"
            or readiness.get("originalHoldoutSha256") != original_hash):
        parser.error("Frozen input provenance mismatch")
    with httpx.Client(timeout=20, follow_redirects=False) as client:
        report = acquire(articles, readiness, client)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as writer:
        json.dump(report, writer, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "articles"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
