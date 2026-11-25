"""Unified keyless macro collection with independent source failure reporting."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from datetime import date
from decimal import Decimal
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx

from tris_market_event_intelligence.collector import NewsStore, parse_rss, utc_now
from tris_market_event_intelligence.economic_collection_cli import (
    KOREA_WB_SERIES, SCHEMA, save,
)
from tris_market_event_intelligence.korean_official_release_cli import (
    Links, SCHEMA as RELEASE_SCHEMA, save_original,
)
from tris_market_event_intelligence.source_article_content import extract_pdf

COUNTRIES = ("USA", "EMU", "JPN", "CHN", "WLD", "OED")
FEEDS = {
    "ecb_press": ("https://www.ecb.europa.eu/rss/press.html", "ecb.europa.eu"),
    "eurostat_updates": (
        "https://ec.europa.eu/eurostat/api/dissemination/catalogue/rss/en/statistics-update.rss",
        "ec.europa.eu",
    ),
    "boj_updates": ("https://www.boj.or.jp/en/rss/whatsnew.xml", "boj.or.jp"),
    "bls_cpi": ("https://www.bls.gov/feed/cpi.rss", "bls.gov"),
    "bls_ppi": ("https://www.bls.gov/feed/ppi.rss", "bls.gov"),
    "bls_employment": ("https://www.bls.gov/feed/empsit.rss", "bls.gov"),
    "bea_news": ("https://www.bea.gov/rss/rss.xml", "bea.gov"),
    "eia_news": ("https://www.eia.gov/rss/todayinenergy.xml", "eia.gov"),
}
BOARDS = {
    "customs": (
        "https://www.customs.go.kr/kcs/na/ntt/selectNttList.do?bbsId=1362&mi=2891",
        "customs.go.kr", "/kcs/na/ntt/selectNttInfo.do",
    ),
    "motir": (
        "https://www.motir.go.kr/kor/article/ATCL3f49a5a8c/list",
        "motir.go.kr", "/kor/article/ATCL3f49a5a8c/",
    ),
    "china_nbs": (
        "https://www.stats.gov.cn/english/PressRelease/", "stats.gov.cn",
        "/english/PressRelease/",
    ),
    "imf_news": ("https://www.imf.org/en/news", "imf.org", "/en/news/articles/"),
    "oecd_news": (
        "https://www.oecd.org/en/about/newsroom.html", "oecd.org", "/en/about/news/press-releases/",
    ),
}
NOT_CONNECTED = (
    "KR_fiscal", "KR_fsc", "KR_fss", "US_census", "US_treasury",
    "JP_statistics", "CN_pboc", "CN_customs", "IMF_numeric", "OECD_numeric",
)


def trusted(url: str, host: str) -> bool:
    p = urlsplit(url)
    return (
        p.scheme == "https" and p.hostname in (host, "www." + host)
        and not p.username and not p.password and p.port in (None, 443)
    )


def fetch(client, url, host, limit=30_000_000):
    for _ in range(5):
        if not trusted(url, host):
            raise ValueError("Untrusted redirect or link")
        with client.stream("GET", url, follow_redirects=False) as response:
            if response.is_redirect:
                url = urljoin(url, response.headers["location"])
                continue
            response.raise_for_status()
            payload = bytearray()
            for chunk in response.iter_bytes():
                payload.extend(chunk)
                if len(payload) > limit:
                    raise ValueError("Response size limit")
            return bytes(payload), url, response.headers.get("content-type", "")
    raise ValueError("Redirect limit")


def world_bank(client, db, country, series, start, end):
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{series}"
    page = 1
    added = seen = observed = 0
    while page <= 100:
        response = client.get(url, params={
            "format": "json", "source": 2, "date": f"{start.year}:{end.year}",
            "per_page": 100, "page": page,
        })
        response.raise_for_status()
        metadata, records = response.json()
        if int(metadata["page"]) != page or not isinstance(records, list):
            raise ValueError("Invalid World Bank page")
        rows = []
        for record in records:
            if record["countryiso3code"] != country or record["indicator"]["id"] != series:
                raise ValueError("World Bank identity mismatch")
            year = int(record["date"])
            value = record["value"]
            if start.year <= year <= end.year and value is not None:
                if not Decimal(str(value)).is_finite():
                    raise ValueError("Nonfinite value")
                rows.append((f"{year:04}-01-01", "", "", str(value)))
        observed += len(rows)
        added += save(db, "WORLD_BANK_" + country, series, {
            "response": response.json(), "url": str(response.request.url),
            "frequency": "annual", "vintageKnown": False,
        }, rows)
        seen += len(records)
        if page >= int(metadata["pages"]):
            if seen != int(metadata["total"]):
                raise ValueError("Incomplete pagination")
            return {"observed": observed, "inserted": added,
                    "status": "COLLECTED" if observed else "EMPTY"}
        if not records:
            raise ValueError("Empty intermediate page")
        page += 1
    raise ValueError("Pagination limit")


def board_links(raw, base, host, prefix):
    parser = Links()
    parser.feed(raw.decode("utf-8-sig", errors="replace"))
    result = {}
    for href, label in parser.links:
        url = urljoin(base, href)
        path = urlsplit(url).path
        if trusted(url, host) and path.startswith(prefix) and label and url != base:
            if path.endswith((".html", ".htm", "/view", "Info.do", ".do")) or (
                "/news/articles/" in path or "/press-releases/" in path
            ):
                result.setdefault(url, label)
    return result


def releases(client, store, db, source, spec, limit, rss):
    from tris_market_event_intelligence.collector import Article

    entry, host = spec[:2]
    raw, base, media = fetch(client, entry, host)
    with db:
        save_original(db, base, raw, media)
    now = utc_now()
    if rss:
        records = [
            a for a in parse_rss(source, raw, now) if trusted(a.url, host)
        ][:limit]
    else:
        records = [
            Article(source, hashlib.sha256(url.encode()).hexdigest(), url, title, None, None, now)
            for url, title in list(board_links(raw, base, host, spec[2]).items())[:limit]
        ]
    if not records:
        raise ValueError("No supported release links")
    inserted = store.save(source, base, raw, records, now, None, None)
    errors = []
    documents = 0
    for article in records:
        try:
            time.sleep(0.25)
            body, url, media = fetch(client, article.url, host)
            aid = db.execute(
                "SELECT id FROM articles WHERE source=? AND url=?", (source, article.url),
            ).fetchone()[0]
            candidates = [(body, url, media)]
            if "html" in media.lower():
                parser = Links()
                parser.feed(body.decode("utf-8-sig", errors="replace"))
                links = list(dict.fromkeys(
                    target for href, _ in parser.links
                    if trusted(target := urljoin(url, href), host)
                    and urlsplit(target).path.lower().endswith(
                        (".pdf", ".hwp", ".hwpx", ".xlsx", ".xls", ".csv", ".zip")
                    )
                ))
                for link in links[:30]:
                    try:
                        time.sleep(0.25)
                        candidates.append(fetch(client, link, host))
                    except (ValueError, httpx.HTTPError, OSError) as exc:
                        errors.append({"url": link, "error": str(exc)[:250]})
            for payload, final, kind in candidates:
                with db:
                    digest = save_original(db, final, payload, kind)
                text, sparse, status = None, (), "RAW_ONLY"
                if payload.startswith(b"%PDF-"):
                    try:
                        text, sparse = extract_pdf(payload)
                        status = "REVIEW_REQUIRED" if sparse else "TEXT_EXTRACTED"
                    except Exception as exc:
                        errors.append({"url": final, "error": type(exc).__name__})
                with db:
                    db.execute("INSERT OR IGNORE INTO official_release_documents VALUES (?,?,?,?,?,?)",
                               (aid, final, digest, text, json.dumps(sparse), status))
                documents += 1
        except (ValueError, httpx.HTTPError, OSError, sqlite3.Error) as exc:
            errors.append({"url": article.url, "error": str(exc)[:250]})
    return {"status": "PARTIAL" if errors else "CAPTURED", "discovered": len(records),
            "inserted": inserted, "documents": documents, "errors": errors,
            "firstPublicAvailabilityVerified": False, "allAttachmentsCaptured": False}


def run_module(module, args):
    result = subprocess.run(
        [sys.executable, "-m", "tris_market_event_intelligence." + module, *args],
        capture_output=True, text=True, timeout=7200,
    )
    try:
        report = json.loads(result.stdout)
    except ValueError:
        report = {"error": result.stderr[-1000:]}
    return {"module": module, "exitCode": result.returncode, "report": report}


def cycle(args):
    results = []
    for module, params in [
        ("economic_collection_cli", ["--db", str(args.macro_db), "--start", str(args.start),
                                     "--end", str(args.end)]),
        ("economic_release_collection_cli", ["--db", str(args.news_db), "--region", "all",
                                             "--max-batches", "100"] +
         (["--ocr-db", str(args.ocr_db)] if args.ocr_db else [])),
        ("korean_official_release_cli", ["--db", str(args.news_db), "--limit", str(args.limit)]),
    ]:
        try:
            results.append(run_module(module, params))
        except (OSError, subprocess.SubprocessError) as exc:
            results.append({"module": module, "status": "FAILED", "error": type(exc).__name__})
    store = NewsStore(args.news_db)
    store.initialize()
    args.macro_db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(args.macro_db)) as macro, closing(store.connect()) as news, \
            httpx.Client(timeout=30, headers={"User-Agent": "TrisMarketEventIntelligence/0.1"}) as client:
        macro.executescript(SCHEMA)
        news.executescript(RELEASE_SCHEMA)
        for country in COUNTRIES:
            for series in KOREA_WB_SERIES:
                job = {"source": "WORLD_BANK_" + country, "series": series}
                try:
                    job.update(world_bank(client, macro, country, series, args.start, args.end))
                except (ValueError, KeyError, TypeError, httpx.HTTPError, ArithmeticError) as exc:
                    job.update(status="FAILED", error=str(exc)[:250])
                results.append(job)
        for registry, rss in ((FEEDS, True), (BOARDS, False)):
            for source, spec in registry.items():
                job = {"source": source}
                try:
                    job.update(releases(client, store, news, source, spec, args.limit, rss))
                except Exception as exc:
                    job.update(status="FAILED", error=str(exc)[:250])
                results.append(job)
    return {"observedAt": utc_now(), "results": results,
            "notConnected": list(NOT_CONNECTED), "analysisPerformed": False,
            "firstPublicAvailabilityVerified": False, "allAttachmentsCaptured": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news-db", required=True, type=Path)
    parser.add_argument("--macro-db", required=True, type=Path)
    parser.add_argument("--ocr-db", type=Path)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2000, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=3600)
    args = parser.parse_args()
    if args.start > args.end or not 1 <= args.limit <= 50 or args.interval < 300:
        parser.error("Invalid dates, limit or interval")
    while True:
        report = cycle(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_suffix(".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(args.output)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
