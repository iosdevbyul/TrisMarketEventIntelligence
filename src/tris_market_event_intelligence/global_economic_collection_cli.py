"""Unified keyless macro collection with independent source failure reporting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sqlite3
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from contextlib import closing
from datetime import date
from decimal import Decimal
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from pypdf.errors import PdfReadError

from tris_market_event_intelligence.collector import Article, NewsStore, parse_rss, utc_now
from tris_market_event_intelligence.economic_collection_cli import (
    KOREA_WB_SERIES,
    SCHEMA,
    save,
)
from tris_market_event_intelligence.korean_official_release_cli import (
    SCHEMA as RELEASE_SCHEMA,
)
from tris_market_event_intelligence.korean_official_release_cli import (
    Links,
    save_original,
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
    "US_census": ("https://www.census.gov/economic-indicators/indicator.xml", "census.gov"),
    "KR_fiscal": ("https://english.mofe.go.kr/pc/engmosfrss.do?boardCd=N0001", "mofe.go.kr"),
    "eia_news": ("https://www.eia.gov/rss/todayinenergy.xml", "eia.gov"),
}
BOARDS = {
    "customs": (
        "https://www.customs.go.kr/kcs/na/ntt/selectNttList.do?bbsId=1362&mi=2891",
        "customs.go.kr", "/kcs/na/ntt/selectNttInfo.do",
    ),
    "bea_news": ("https://www.bea.gov/news/archive", "bea.gov", "/news/"),
    "motir": (
        "https://www.motir.go.kr/kor/article/ATCL3f49a5a8c",
        "motir.go.kr", "/kor/article/ATCL3f49a5a8c/",
    ),
    "KR_fsc": ("https://www.fsc.go.kr/no010101", "fsc.go.kr", "/no010101/"),
    "JP_statistics": ("https://www.stat.go.jp/english/data/index.html", "stat.go.jp", "/english/data/"),
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
    "KR_fss",
    "CN_pboc", "CN_customs",
)


def trusted(url: str, host: str) -> bool:
    p = urlsplit(url)
    return (
        p.scheme == "https" and p.hostname in (host, "www." + host, "english." + host)
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


def retry_get(client, url, **kwargs):
    for attempt in range(3):
        try:
            response = client.get(url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except httpx.TransportError:
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)
    raise AssertionError("Unreachable retry state")


def feed_records(source, raw, base, host, now):
    records = parse_rss(source, raw, now)
    if not records:
        # RSS 1.0 / RDF uses namespaced items and Dublin Core dates.
        root = ET.fromstring(raw)
        for item in root.findall(".//{http://purl.org/rss/1.0/}item"):
            title = item.findtext("{http://purl.org/rss/1.0/}title") or ""
            link = item.findtext("{http://purl.org/rss/1.0/}link") or ""
            if title and link:
                records.append(Article(source, link, link, title, None, None, now))
    result = []
    for record in records:
        url = urljoin(base, record.url)
        parsed = urlsplit(url)
        if parsed.scheme == "http":
            url = urlunsplit(parsed._replace(scheme="https"))
        if trusted(url, host):
            result.append(Article(source, record.external_id, url, record.title,
                                  record.summary, record.published_at, record.detected_at))
    return result


def world_bank(client, db, country, series, start, end):
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{series}"
    page = 1
    added = seen = observed = 0
    while page <= 100:
        response = retry_get(client, url, params={
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
        if (
            trusted(url, host) and path.startswith(prefix) and label and url != base
            and (path.endswith((".html", ".htm", "/view", "Info.do", ".do"))
                 or "/news/articles/" in path or "/press-releases/" in path
                 or (host == "bea.gov" and len(path.split("/")) > 3
                     and path.split("/")[2].isdigit())
                 or (host == "fsc.go.kr" and path.rsplit("/", 1)[-1].isdigit()))
        ):
            result.setdefault(url, label)
    return result


def releases(client, store, db, source, spec, limit, rss, *, reuse=False):
    entry, host = spec[:2]
    raw, base, media = fetch(client, entry, host)
    with db:
        save_original(db, base, raw, media)
    now = utc_now()
    if rss:
        records = feed_records(source, raw, base, host, now)[:limit]
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
    reused = 0
    for article in records:
        try:
            if reuse and db.execute("""
                SELECT 1 FROM articles a JOIN official_release_documents d ON d.article_id=a.id
                WHERE a.source=? AND a.url=? AND d.url=a.url
                LIMIT 1
            """, (source, article.url)).fetchone():
                reused += 1
                continue
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
                    and (urlsplit(target).path.lower().endswith(
                        (".pdf", ".hwp", ".hwpx", ".xlsx", ".xls", ".csv", ".zip")
                    ) or (host == "motir.go.kr" and urlsplit(target).path.startswith("/attach/down/")))
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
                    except (PdfReadError, ValueError, OSError, RuntimeError) as exc:
                        errors.append({"url": final, "error": type(exc).__name__})
                with db:
                    db.execute("INSERT OR IGNORE INTO official_release_documents VALUES (?,?,?,?,?,?)",
                               (aid, final, digest, text, json.dumps(sparse), status))
                documents += 1
        except (ValueError, httpx.HTTPError, OSError, sqlite3.Error) as exc:
            errors.append({"url": article.url, "error": str(exc)[:250]})
    return {"status": "PARTIAL" if errors else "CAPTURED", "discovered": len(records),
            "inserted": inserted, "documents": documents, "reusedArticles": reused, "errors": errors,
            "firstPublicAvailabilityVerified": False, "allAttachmentsCaptured": False}


def run_module(module, args):
    result = subprocess.run(
        [sys.executable, "-m", "tris_market_event_intelligence." + module, *args],
        check=False, capture_output=True, text=True, timeout=7200,
    )
    try:
        report = json.loads(result.stdout)
    except ValueError:
        report = {"error": result.stderr[-1000:]}
    return {"module": module, "exitCode": result.returncode, "report": report}


def imf_numeric(client, db, start, end):
    series = ("NGDP_RPCH", "PCPIPCH", "LUR")
    for indicator in series:
        job = {"source": "IMF_DATAMAPPER", "series": indicator}
        try:
            url = "https://www.imf.org/external/datamapper/api/v1/" + indicator
            response = retry_get(client, url)
            payload = response.json()
            values = payload["values"][indicator]
            for country in ("KOR", "USA", "JPN", "CHN"):
                rows = [
                    (year + "-01-01", "", "", str(value))
                    for year, value in values.get(country, {}).items()
                    if year.isdigit() and start.year <= int(year) < end.year
                    and value is not None and Decimal(str(value)).is_finite()
                ]
                added = save(db, "IMF_DATAMAPPER", indicator + ":" + country, {
                    "response": payload, "url": url, "vintageKnown": False,
                    "historicalEstimatesMayIncludeRevisions": True,
                }, rows)
                yield {**job, "country": country, "observed": len(rows),
                       "inserted": added, "status": "COLLECTED" if rows else "EMPTY"}
        except (httpx.HTTPError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
            yield {**job, "status": "FAILED", "error": str(exc)[:250]}


def treasury_numeric(client, db, start, end):
    url = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
           "v2/accounting/od/debt_to_penny")
    page = 1
    added = observed = 0
    while page <= 100:
        response = retry_get(client, url, params={
            "filter": f"record_date:gte:{start},record_date:lte:{end}",
            "fields": "record_date,tot_pub_debt_out_amt",
            "sort": "record_date", "page[size]": 1000, "page[number]": page,
        })
        payload = response.json()
        rows = [
            (r["record_date"], "", "", r["tot_pub_debt_out_amt"])
            for r in payload["data"] if r["tot_pub_debt_out_amt"] is not None
        ]
        observed += len(rows)
        added += save(db, "US_TREASURY", "TOTAL_PUBLIC_DEBT", {
            "response": payload, "url": str(response.request.url), "vintageKnown": False,
        }, rows)
        if not payload["links"]["next"]:
            return {"source": "US_TREASURY", "series": "TOTAL_PUBLIC_DEBT",
                    "observed": observed, "inserted": added,
                    "status": "COLLECTED" if observed else "EMPTY"}
        page += 1
    raise ValueError("Treasury pagination limit")


def oecd_numeric(client, db, start, end):
    flow = "OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,1.0"
    key = "KOR+USA+JPN+OECD..._Z.Y._T.Y_GE15..M"
    url = f"https://sdmx.oecd.org/public/rest/data/{flow}/{key}"
    response = retry_get(client, url, params={
        "startPeriod": start.strftime("%Y-%m"), "endPeriod": end.strftime("%Y-%m"),
        "format": "csvfile",
    }, headers={"Accept": "text/csv"})
    reader = csv.DictReader(io.StringIO(response.text.lstrip("\ufeff")))
    if not {"REF_AREA", "TIME_PERIOD", "OBS_VALUE"}.issubset(reader.fieldnames or []):
        raise ValueError("Unexpected OECD CSV schema")
    groups = {}
    for row in reader:
        country = row["REF_AREA"]
        if country not in ("KOR", "USA", "JPN", "OECD"):
            raise ValueError("Unexpected OECD area")
        period = date.fromisoformat(row["TIME_PERIOD"] + "-01")
        value = row["OBS_VALUE"]
        if not value or not start.replace(day=1) <= period <= end:
            continue
        if not Decimal(value).is_finite():
            raise ValueError("Invalid OECD value")
        # Preserve every dimension; never collapse different statistical series.
        dimensions = tuple((k, v) for k, v in sorted(row.items())
                           if k not in ("TIME_PERIOD", "OBS_VALUE") and k in (
                               "REF_AREA", "MEASURE", "UNIT_MEASURE", "ACTIVITY",
                               "ADJUSTMENT", "SEX", "AGE", "FREQ", "TRANSFORMATION",
                               "LABOUR_FORCE_STATUS", "COUNTERPART_AREA",
                           ))
        series = json.dumps(dimensions, separators=(",", ":"))
        groups.setdefault(series, []).append((period.isoformat(), "", "", value))
    added = 0
    for series, rows in groups.items():
        added += save(db, "OECD_SDMX", series, {
            "url": str(response.request.url), "csv": response.text, "vintageKnown": False,
        }, rows)
    return {"source": "OECD_SDMX", "status": "COLLECTED" if groups else "EMPTY",
            "seriesCount": len(groups), "observed": sum(map(len, groups.values())), "inserted": added}


def cycle(args):
    results = []
    for module, params in [
        ("economic_collection_cli", ["--db", str(args.macro_db), "--start", str(args.start),
                                     "--end", str(args.end)]),
        ("economic_release_collection_cli", ["--db", str(args.news_db), "--region", "all",
                                             "--max-batches", "100"]),
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
        results.extend(imf_numeric(client, macro, args.start, args.end))
        try:
            results.append(oecd_numeric(client, macro, args.start, args.end))
        except (httpx.HTTPError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
            results.append({"source": "OECD_SDMX", "status": "FAILED", "error": str(exc)[:250]})
        try:
            results.append(treasury_numeric(client, macro, args.start, args.end))
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            results.append({"source": "US_TREASURY", "status": "FAILED", "error": str(exc)[:250]})
        for registry, rss in ((FEEDS, True), (BOARDS, False)):
            for source, spec in registry.items():
                job = {"source": source}
                try:
                    job.update(releases(client, store, news, source, spec, args.limit, rss, reuse=args.reuse_captured))
                except (ValueError, KeyError, TypeError, OSError, httpx.HTTPError,
                        sqlite3.Error, RuntimeError, ET.ParseError) as exc:
                    job.update(status="FAILED", error=str(exc)[:250])
                results.append(job)
    if args.ocr_db:
        try:
            from tris_market_event_intelligence.economic_pdf_recovery_cli import recover

            results.append({"module": "all_official_ocr", **recover(args.news_db, args.ocr_db, limit=100)})
        except (ValueError, OSError, RuntimeError, ImportError, sqlite3.Error, subprocess.SubprocessError) as exc:
            results.append({"module": "all_official_ocr", "status": "FAILED", "error": str(exc)[:250]})
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
    parser.add_argument("--reuse-captured", action="store_true",
                        help="Skip previously captured article URLs; may miss revisions")
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
