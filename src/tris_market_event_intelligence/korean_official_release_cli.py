"""Keyless official Korean board collection; retain originals and date uncertainty."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from contextlib import closing
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit

import httpx

from tris_market_event_intelligence.collector import Article, NewsStore, utc_now
from tris_market_event_intelligence.source_article_content import extract_pdf

BOARDS = {
    "mods_cpi": "https://mods.go.kr/board.es?mid=b70203010000&bid=b702030100&ref_bid=213",
    "mods_industry": "https://mods.go.kr/board.es?bid=216&mid=a10301050100",
    "mods_all": "https://mods.go.kr/board.es?bid=216&mid=a10301010000",
}
HOSTS = {"mods.go.kr", "www.mods.go.kr", "kostat.go.kr", "www.kostat.go.kr"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS official_release_originals (
    url TEXT NOT NULL, sha256 TEXT NOT NULL, payload BLOB NOT NULL,
    observed_at TEXT NOT NULL, media_type TEXT NOT NULL,
    PRIMARY KEY(url,sha256)
);
CREATE TABLE IF NOT EXISTS official_release_documents (
    article_id INTEGER NOT NULL, url TEXT NOT NULL, sha256 TEXT NOT NULL,
    body TEXT, sparse_pages TEXT NOT NULL, status TEXT NOT NULL,
    PRIMARY KEY(article_id,url,sha256)
);
"""


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.href: str | None = None
        self.words: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.words = []

    def handle_data(self, data):
        if self.href:
            self.words.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.href:
            self.links.append((self.href, " ".join(" ".join(self.words).split())))
            self.href = None


def allowed(url: str) -> bool:
    p = urlsplit(url)
    return (p.scheme == "https" and p.hostname in HOSTS and not p.username
            and not p.password and p.port in (None, 443))


def discover(payload: bytes, base: str) -> list[tuple[str, str]]:
    parser = Links()
    parser.feed(payload.decode("utf-8-sig", errors="replace"))
    results = {}
    for href, label in parser.links:
        url = urljoin(base, href)
        p = urlsplit(url)
        q = parse_qs(p.query)
        if (allowed(url) and p.path == "/board.es" and q.get("act") == ["view"]
                and q.get("list_no", [""])[0].isdigit() and label):
            results.setdefault(url, label)
    # Some board templates expose downloads but not clickable detail anchors.
    # Derive the detail URL only from official attachment IDs.
    for href, label in parser.links:
        url = urljoin(base, href)
        p = urlsplit(url)
        q = parse_qs(p.query)
        identifier = q.get("list_no", [""])[0]
        bid = q.get("bid", [""])[0]
        if not (allowed(url) and p.path == "/boardDownload.es"
                and identifier.isdigit() and bid.isdigit()):
            continue
        title = label.rsplit("의 ", 1)[0].strip()
        if not title or title == label:
            continue
        if any(parse_qs(urlsplit(key).query).get("list_no") == [identifier]
               for key in results):
            continue
        query = parse_qs(urlsplit(base).query)
        detail = urljoin(base, "/board.es") + "?" + urlencode({
            "mid": query.get("mid", [""])[0], "bid": bid,
            "act": "view", "list_no": identifier,
        })
        results[detail] = title
    return list(results.items())


def attachments(payload: bytes, base: str) -> list[str]:
    parser = Links()
    parser.feed(payload.decode("utf-8-sig", errors="replace"))
    return list(dict.fromkeys(
        url for href, label in parser.links
        if allowed(url := urljoin(base, href))
        and urlsplit(url).path == "/boardDownload.es"
    ))


def fetch(client: httpx.Client, url: str, limit: int) -> tuple[bytes, str, str]:
    for _ in range(4):
        if not allowed(url):
            raise ValueError("Untrusted official URL")
        with client.stream("GET", url, follow_redirects=False) as response:
            if response.is_redirect:
                url = urljoin(url, response.headers["location"])
                continue
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_bytes():
                data.extend(chunk)
                if len(data) > limit:
                    raise ValueError("Response too large")
            return bytes(data), url, response.headers.get("content-type", "")
    raise ValueError("Redirect limit")


def save_original(db: sqlite3.Connection, url: str, payload: bytes, kind: str) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    db.execute("INSERT OR IGNORE INTO official_release_originals VALUES (?,?,?,?,?)",
               (url, digest, payload, utc_now(), kind))
    return digest


def collect(path: Path, sources: tuple[str, ...], limit: int) -> dict:
    store = NewsStore(path)
    store.initialize()
    reports = []
    with closing(store.connect()) as db, httpx.Client(
        timeout=30, headers={"User-Agent": "TrisMarketEventIntelligence/0.1 (+local research)"},
    ) as client:
        db.executescript(SCHEMA)
        for source in sources:
            report = {"source": source, "discovered": 0, "inserted": 0,
                      "documents": 0, "pdfText": 0, "errors": [], "status": "FAILED"}
            reports.append(report)
            try:
                raw, base, kind = fetch(client, BOARDS[source], 2_000_000)
                with db:
                    save_original(db, base, raw, kind)
                report["responseUrl"] = base
                report["responseBytes"] = len(raw)
                items = discover(raw, base)
                if not items:
                    raise ValueError("No release links discovered")
                report["discovered"] = len(items)
                now = utc_now()
                records = [
                    Article(source, parse_qs(urlsplit(url).query)["list_no"][0],
                            url, title, None, None, now)
                    for url, title in items[:limit]
                ]
                report["inserted"] = store.save(source, base, raw, records, now, None, None)
                with db:
                    save_original(db, base, raw, kind)
                for record in records:
                    try:
                        time.sleep(0.3)
                        page, url, kind = fetch(client, record.url, 2_000_000)
                        article_id = db.execute(
                            "SELECT id FROM articles WHERE source=? AND external_id=?",
                            (source, record.external_id),
                        ).fetchone()[0]
                        with db:
                            save_original(db, url, page, kind)
                        identifier = record.external_id
                        links = list(dict.fromkeys([
                            *attachments(page, url),
                            *(link for link in attachments(raw, base)
                              if parse_qs(urlsplit(link).query).get("list_no") == [identifier]),
                        ]))
                        if not links:
                            report["errors"].append({"id": article_id, "error": "NO_ATTACHMENT_LINK"})
                        for link in links:
                            try:
                                time.sleep(0.3)
                                data, final, media = fetch(client, link, 30_000_000)
                                with db:
                                    digest = save_original(db, final, data, media)
                                body, sparse, status = None, (), "RAW_ONLY"
                                if data.startswith(b"%PDF-"):
                                    body, sparse = extract_pdf(data)
                                    status = "REVIEW_REQUIRED" if sparse else "TEXT_EXTRACTED"
                                    report["pdfText"] += 1
                                with db:
                                    db.execute(
                                        "INSERT OR IGNORE INTO official_release_documents "
                                        "VALUES (?,?,?,?,?,?)",
                                        (article_id, final, digest, body, json.dumps(sparse), status),
                                    )
                                report["documents"] += 1
                            except (ValueError, httpx.HTTPError, OSError) as exc:
                                report["errors"].append({"id": article_id, "error": type(exc).__name__, "detail": str(exc)[:300]})
                    except (ValueError, httpx.HTTPError, OSError) as exc:
                        report["errors"].append({"url": record.url, "error": type(exc).__name__, "detail": str(exc)[:300]})
                report["status"] = "PARTIAL" if report["errors"] else "COLLECTED"
            except (ValueError, httpx.HTTPError, OSError) as exc:
                report["errors"].append({"error": type(exc).__name__, "detail": str(exc)[:300]})
    return {"sources": reports, "analysisPerformed": False,
            "firstPublicAvailabilityVerified": False, "allAttachmentsCaptured": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--source", choices=(*BOARDS, "all"), default="all")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.limit <= 50:
        parser.error("limit must be 1..50")
    sources = tuple(BOARDS) if args.source == "all" else (args.source,)
    report = collect(args.db, sources, args.limit)
    print(json.dumps(report, ensure_ascii=False))
    if any(r["status"] != "COLLECTED" for r in report["sources"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
