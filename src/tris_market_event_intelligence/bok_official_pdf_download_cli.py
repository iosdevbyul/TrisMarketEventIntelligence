"""Download fresh BOK release PDFs from the official announcement pages.

Strictly fail on missing or ambiguous PDF links. Never use historical caches.
"""
from __future__ import annotations

import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from tris_market_event_intelligence.bok_official_recollection_cli import RELEASES, digest

ALLOWED_HOSTS = {"www.bok.or.kr", "bok.or.kr", "file-cdn.bok.or.kr"}
MAX_PDF_BYTES = 40_000_000


class PDFLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[dict[str, str], str]] = []
        self.current: dict[str, str] | None = None
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self.current = {key: (value or "") for key, value in attrs}
            self.text = []

    def handle_data(self, data: str) -> None:
        if self.current is not None:
            self.text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.current is not None:
            self.links.append((self.current, "".join(self.text).strip()))
            self.current = None
            self.text = []


def trusted(url: str) -> bool:
    parts = urlparse(url)
    return (parts.scheme == "https" and parts.hostname in ALLOWED_HOSTS
            and not parts.username and not parts.password)


def find_pdf_url(page_html: str, base_url: str) -> str:
    parser = PDFLinks()
    parser.feed(page_html)
    possible: list[str] = []
    for attrs, label in parser.links:
        # Only PDF attachment labels, not generic "PDF viewer" links.
        if ".pdf" not in label.casefold():
            continue
        for attr in ("href", "data-url", "data-href", "data-download-url"):
            value = attrs.get(attr, "").strip()
            if not value or value.startswith(("#", "javascript:", "data:")):
                continue
            candidate = urljoin(base_url, value)
            if trusted(candidate):
                possible.append(candidate)
                break
        else:
            # Some BOK pages put the actual URL in the onclick argument.
            onclick = attrs.get("onclick", "")
            matches = re.findall(r"['\"](https?://[^'\"]+|/[^'\"]+)['\"]", onclick)
            for match in matches:
                candidate = urljoin(base_url, match)
                if trusted(candidate):
                    possible.append(candidate)
                    break
    unique = list(dict.fromkeys(possible))
    if len(unique) != 1:
        raise ValueError("OFFICIAL_PDF_LINK_MISSING_OR_AMBIGUOUS")
    return unique[0]


def download_pdf(client: httpx.Client, official_url: str, target: Path) -> dict:
    """Download only from trusted BOK hosts with PDF content verification."""
    if target.exists() or target.is_symlink():
        return {"status": "EXISTING_FILE_NOT_OVERWRITTEN", "path": str(target)}
    if not trusted(official_url):
        raise ValueError("UNTRUSTED_OFFICIAL_RELEASE_URL")
    release = client.get(official_url, timeout=20)
    release.raise_for_status()
    attachment_url = find_pdf_url(release.text, str(release.url))
    if not trusted(attachment_url):
        raise ValueError("UNTRUSTED_PDF_LINK")
    response = client.get(attachment_url, timeout=60, follow_redirects=True)
    response.raise_for_status()
    if not trusted(str(response.url)):
        raise ValueError("UNTRUSTED_PDF_REDIRECT")
    payload = response.content
    if not payload.startswith(b"%PDF-") or len(payload) > MAX_PDF_BYTES:
        raise ValueError("INVALID_OR_OVERSIZED_PDF_RESPONSE")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(payload)
    return {
        "status": "DOWNLOADED",
        "path": str(target),
        "officialUrl": official_url,
        "downloadUrl": attachment_url,
        "sha256": digest(payload),
        "bytes": len(payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.audit.exists() or args.audit.is_symlink():
        parser.error("Audit output already exists; choose a new filename")
    results = []
    with httpx.Client(follow_redirects=True, headers={
        "User-Agent": "TrisMarketEventIntelligence/official-public-document-research",
    }) as client:
        for article_id, release in RELEASES.items():
            target = args.pdf_dir / f"{article_id}.pdf"
            try:
                result = download_pdf(client, release["officialUrl"], target)
            except (ValueError, httpx.HTTPError, OSError) as exc:
                result = {"status": "DOWNLOAD_FAILED", "reason": str(exc)[:250],
                          "path": str(target), "officialUrl": release["officialUrl"]}
            results.append({"articleId": article_id, **result})
    report = {
        "schemaVersion": "bok_official_pdf_download_v1",
        "downloadedCount": sum(row["status"] == "DOWNLOADED" for row in results),
        "items": results,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))
    if any(item["status"] == "DOWNLOAD_FAILED" for item in results):
        raise SystemExit("Official PDF retrieval incomplete; inspect the download audit")


if __name__ == "__main__":
    main()
