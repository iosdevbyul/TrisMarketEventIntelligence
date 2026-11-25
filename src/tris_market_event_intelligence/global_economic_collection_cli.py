"""Unified keyless macro collection with independent source failure reporting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sqlite3
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from contextlib import closing
from datetime import date, timedelta
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit, urlunsplit

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
from tris_market_event_intelligence.source_article_content import ContentError, extract_pdf

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
    "bea_news": ("https://www.bea.gov/news/archive?created_1=All&field_related_product_target_id=All&page=0&title=",
        "bea.gov", "/news/"),
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
    "imf_data_news": ("https://data.imf.org/news", "data.imf.org", "/news/"),
    "oecd_statistics": ("https://www.oecd.org/en/data/insights.html", "oecd.org",
                        "/en/data/insights/statistical-releases/"),
    "KR_fss": ("https://dart.fss.or.kr/info/searchBodo.do", "dart.fss.or.kr", "/info/selectBodo.do"),
    "oecd_news": (
        "https://www.oecd.org/en/about/newsroom.html", "oecd.org", "/en/about/news/press-releases/",
    ),
}
NOT_CONNECTED = (
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
        if host == "customs.go.kr" and href.lower().startswith("javascript:"):
            # Only accept an explicit complete detail URL embedded in an anchor.
            match = re.search(r"""(/kcs/na/ntt/selectNttInfo\.do\?[^'"]+)""", href)
            if match:
                href = match.group(1)
        url = urljoin(base, href)
        path = urlsplit(url).path.removeprefix("/index.php")
        if (
            trusted(url, host) and path.startswith(prefix) and label and url != base
            and (host != "customs.go.kr" or (
                parse_qs(urlsplit(url).query).get("nttSn", [""])[0].isdigit()
                and bool(parse_qs(urlsplit(url).query).get("nttSnUrl", [""])[0])
            ))
            and (path.endswith((".html", ".htm", "/view", "Info.do", ".do"))
                 or "/news/articles/" in path or "/press-releases/" in path
                 or (host == "bea.gov" and len(path.split("/")) > 3
                     and path.split("/")[2].isdigit())
                 or (host == "fsc.go.kr" and path.rsplit("/", 1)[-1].isdigit()))
        ):
            result.setdefault(url, label)
    return result


class PublicationMetadata(HTMLParser):
    """Keep publisher-declared dates as evidence, not verified first availability."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.values = []

    def handle_starttag(self, tag, attrs):
        fields = dict(attrs)
        if tag == "meta":
            key = fields.get("property") or fields.get("name") or fields.get("itemprop")
            value = fields.get("content")
        elif tag == "time":
            key = fields.get("itemprop") or "time"
            value = fields.get("datetime")
        else:
            return
        if key and value and key.lower() in {
            "article:published_time", "article:modified_time", "datepublished",
            "datemodified", "pubdate", "publication_date", "time",
        }:
            self.values.append((key, value))


CAPTURE_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS official_release_date_evidence (
    article_id INTEGER NOT NULL, sha256 TEXT NOT NULL, field TEXT NOT NULL,
    value TEXT NOT NULL, observed_at TEXT NOT NULL,
    PRIMARY KEY(article_id,sha256,field,value)
);
CREATE TABLE IF NOT EXISTS official_release_capture_state (
    article_id INTEGER PRIMARY KEY, checked_at TEXT NOT NULL,
    status TEXT NOT NULL, errors TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS official_release_capture_attempt (
    id INTEGER PRIMARY KEY, article_id INTEGER NOT NULL, checked_at TEXT NOT NULL,
    status TEXT NOT NULL, errors TEXT NOT NULL
);
"""


def capture_state(db, aid, errors):
    status = "PARTIAL" if errors else "CAPTURED"
    now = utc_now()
    with db:
        db.execute("INSERT OR REPLACE INTO official_release_capture_state VALUES (?,?,?,?)",
                   (aid, now, status, json.dumps(errors, ensure_ascii=False)))
        db.execute("""INSERT INTO official_release_capture_attempt
                   (article_id,checked_at,status,errors) VALUES (?,?,?,?)""",
                   (aid, now, status, json.dumps(errors, ensure_ascii=False)))


def archive_url(entry, page):
    parsed = urlsplit(entry)
    query = parse_qs(parsed.query, keep_blank_values=True)
    query["page"] = [str(page)]
    return urlunsplit(parsed._replace(query=urlencode(query, doseq=True)))


def releases(client, store, db, source, spec, limit, rss, *, reuse=False, archive_pages=1):
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
    if not records and source == "customs":
        home, home_url, home_media = fetch(client, "https://customs.go.kr/", host)
        with db:
            save_original(db, home_url, home, home_media)
        records = [
            Article(source, hashlib.sha256(url.encode()).hexdigest(), url, title, None, None, now)
            for url, title in list(board_links(home, home_url, host, spec[2]).items())[:limit]
        ]
        if records:
            raw, base = home, home_url
    if not records and source == "customs":
        # Official detail URLs independently located on the publisher's domain.
        # A bounded historical seed, not a substitute for live discovery.
        seeds = (
            ("10183383", "543e432d867f6dd472a69c7a3cfff59c", "2026년 10월 월간 수출입 현황 [확정치]"),
            ("10182926", "f245ffc86cc10295d215ed0379c4a630", "2026년 11월 1일 ~ 11월 10일 수출입 현황 [잠정치]"),
            ("10181824", "4c5427b9707dd3a4c5c0a47f53dbf1c7", "2026년 10월 수출입 현황 [잠정치]"),
            ("10176444", "8599e04facbad02816d3af783a538778", "2026년 9월 1일 ~ 9월 10일 수출입 현황 [잠정치]"),
        )
        for number, token, title in seeds[:limit]:
            url = urljoin(entry, spec[2]) + "?" + urlencode({
                "bbsId": "1362", "mi": "2891", "nttSn": number, "nttSnUrl": token,
            })
            records.append(Article(source, hashlib.sha256(url.encode()).hexdigest(),
                                   url, title, None, None, now))
        seeded = True
    else:
        seeded = False
    if not records:
        raise ValueError("No supported release links")
    inserted = store.save(source, base, raw, records, now, None, None)
    db.executescript(CAPTURE_STATE_SCHEMA)
    archive_errors = []
    if source == "bea_news" and not rss:
        seen = {record.url for record in records}
        for index in range(1, archive_pages):
            try:
                time.sleep(0.5)
                page_raw, page_url, page_media = fetch(client, archive_url(entry, index), host)
                with db:
                    save_original(db, page_url, page_raw, page_media)
                found = [Article(source, hashlib.sha256(url.encode()).hexdigest(), url, title,
                                 None, None, utc_now())
                         for url, title in board_links(page_raw, page_url, host, spec[2]).items()
                         if url not in seen][:limit]
                if not found:
                    break
                inserted += store.save(source, page_url, page_raw, found, utc_now(), None, None)
                records.extend(found)
                seen.update(record.url for record in found)
            except (ValueError, httpx.HTTPError, OSError) as exc:
                archive_errors.append({"page": index, "error": str(exc)[:250]})
                break
    current_urls = {record.url for record in records}
    backlog = db.execute("""SELECT a.external_id,a.url,a.title,a.summary,a.published_at,a.detected_at
        FROM articles a JOIN official_release_capture_state s ON s.article_id=a.id
        WHERE a.source=? AND s.status='PARTIAL' ORDER BY s.checked_at LIMIT ?""",
        (source, limit)).fetchall()
    records.extend(Article(source, *row) for row in backlog if row[1] not in current_urls)
    errors = list(archive_errors)
    documents = 0
    reused = 0
    for article in records:
        aid = db.execute("SELECT id FROM articles WHERE source=? AND url=?",
                         (source, article.url)).fetchone()[0]
        previous = len(errors)
        try:
            if reuse and db.execute("""
                SELECT 1 FROM official_release_capture_state
                WHERE article_id=? AND status='CAPTURED'
                  AND checked_at >= ?
            """, (aid, (date.today() - timedelta(days=7)).isoformat())).fetchone():
                reused += 1
                continue
            time.sleep(0.25)
            body, url, media = fetch(client, article.url, host)
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
                if len(links) > 30:
                    errors.append({"url": url, "error": "ATTACHMENT_LIMIT"})
                for link in links[:30]:
                    try:
                        time.sleep(0.25)
                        candidates.append(fetch(client, link, host))
                    except (ValueError, httpx.HTTPError, OSError) as exc:
                        errors.append({"url": link, "error": str(exc)[:250]})
            for payload, final, kind in candidates:
                with db:
                    digest = save_original(db, final, payload, kind)
                    if final == url and "html" in kind.lower():
                        metadata = PublicationMetadata()
                        metadata.feed(payload.decode("utf-8-sig", errors="replace"))
                        db.executemany("""INSERT OR IGNORE INTO official_release_date_evidence
                            VALUES (?,?,?,?,?)""", [(aid, digest, field, value, utc_now())
                            for field, value in metadata.values])
                text, sparse, status = None, (), "RAW_ONLY"
                if payload.startswith(b"%PDF-"):
                    try:
                        text, sparse = extract_pdf(payload)
                        status = "REVIEW_REQUIRED" if sparse else "TEXT_EXTRACTED"
                    except (ContentError, PdfReadError, ValueError, OSError, RuntimeError) as exc:
                        status = "EXTRACTION_FAILED"
                        errors.append({"url": final, "error": type(exc).__name__})
                with db:
                    db.execute("""INSERT INTO official_release_documents VALUES (?,?,?,?,?,?)
                               ON CONFLICT(article_id,url,sha256) DO UPDATE SET
                               body=excluded.body,sparse_pages=excluded.sparse_pages,status=excluded.status
                               WHERE official_release_documents.status IN ('RAW_ONLY','EXTRACTION_FAILED')""",
                               (aid, final, digest, text, json.dumps(sparse), status))
                documents += 1
        except (ValueError, httpx.HTTPError, OSError, sqlite3.Error) as exc:
            errors.append({"url": article.url, "error": str(exc)[:250]})
        capture_state(db, aid, errors[previous:])
    return {"status": "PARTIAL" if errors or seeded else "CAPTURED", "discovered": len(records),
            "historicalSeedOnly": seeded,
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


DEFERRED_RELEASES = frozenset({
    "customs", "KR_fss", "imf_news", "imf_data_news", "oecd_news", "oecd_statistics",
})


def cycle(args):
    results = []
    selected = getattr(args, "release_source", None)
    operational = getattr(args, "operational", False)
    effective_start = max(args.start, args.end - timedelta(days=730)) if args.incremental else args.start
    modules = [
        ("economic_collection_cli", ["--db", str(args.macro_db), "--start", str(args.start),
                                     "--end", str(args.end)] + (["--incremental"] if args.incremental else [])),
        ("economic_release_collection_cli", ["--db", str(args.news_db), "--region", "all",
                                             "--max-batches", "100"]),
        ("korean_official_release_cli", ["--db", str(args.news_db), "--limit", str(args.limit)]),
    ]
    for module, params in ([] if selected else modules):
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
        if not selected:
            for country in COUNTRIES:
                for series in KOREA_WB_SERIES:
                    job = {"source": "WORLD_BANK_" + country, "series": series}
                    try:
                        job.update(world_bank(client, macro, country, series, effective_start, args.end))
                    except (ValueError, KeyError, TypeError, httpx.HTTPError, ArithmeticError) as exc:
                        job.update(status="FAILED", error=str(exc)[:250])
                    results.append(job)
            results.extend(imf_numeric(client, macro, effective_start, args.end))
            try:
                results.append(oecd_numeric(client, macro, effective_start, args.end))
            except (httpx.HTTPError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
                results.append({"source": "OECD_SDMX", "status": "FAILED", "error": str(exc)[:250]})
            try:
                results.append(treasury_numeric(client, macro, effective_start, args.end))
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                results.append({"source": "US_TREASURY", "status": "FAILED", "error": str(exc)[:250]})
        for registry, rss in ((FEEDS, True), (BOARDS, False)):
            for source, spec in registry.items():
                if selected and source not in selected:
                    continue
                if operational and not selected and source in DEFERRED_RELEASES:
                    continue
                job = {"source": source}
                try:
                    job.update(releases(client, store, news, source, spec, args.limit, rss, reuse=args.reuse_captured,
                                        archive_pages=args.archive_pages))
                except (ValueError, KeyError, TypeError, OSError, httpx.HTTPError,
                        sqlite3.Error, RuntimeError, ET.ParseError) as exc:
                    job.update(status="FAILED", error=str(exc)[:250])
                results.append(job)
    if args.ocr_db and not selected and not operational:
        try:
            from tris_market_event_intelligence.economic_pdf_recovery_cli import recover

            results.append({"module": "all_official_ocr", **recover(args.news_db, args.ocr_db, limit=args.ocr_limit)})
        except (ValueError, OSError, RuntimeError, ImportError, sqlite3.Error, subprocess.SubprocessError) as exc:
            results.append({"module": "all_official_ocr", "status": "FAILED", "error": str(exc)[:250]})
    return {"observedAt": utc_now(), "results": results,
            "notConnected": list(NOT_CONNECTED),
            "deferredSources": sorted(DEFERRED_RELEASES) if operational else [],
            "mode": "selected_releases" if selected else ("incremental" if args.incremental else "backfill"),
            "effectiveStart": str(effective_start), "analysisPerformed": False,
            "firstPublicAvailabilityVerified": False, "allAttachmentsCaptured": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news-db", required=True, type=Path)
    parser.add_argument("--macro-db", required=True, type=Path)
    parser.add_argument("--ocr-db", type=Path)
    parser.add_argument("--release-source", action="append", choices=sorted(FEEDS.keys() | BOARDS.keys()),
                        help="Collect only selected global release sources; skip numeric and OCR jobs")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2000, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--operational", action="store_true",
                        help="Incremental core collection; defer known blocked releases and OCR")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ocr-limit", type=int, default=100)
    parser.add_argument("--archive-pages", type=int, default=1,
                        help="BEA archive pages, where supported")
    parser.add_argument("--incremental", action="store_true", help="Refresh recent two years only")
    parser.add_argument("--reuse-captured", action="store_true",
                        help="Reuse successful captures for seven days; failed captures are retried")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=3600)
    args = parser.parse_args()
    rolling_end = args.end is None
    args.end = args.end or date.today()
    if args.operational:
        args.incremental = True
        args.reuse_captured = True
        args.archive_pages = 1
    paths = [args.news_db, args.macro_db, args.output] + ([args.ocr_db] if args.ocr_db else [])
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error("Database and report paths must be distinct")
    if (args.start > args.end or not 1 <= args.limit <= 50 or args.interval < 300
            or not 1 <= args.archive_pages <= 50 or not 1 <= args.ocr_limit <= 1000):
        parser.error("Invalid dates, limit or interval")
    while True:
        if rolling_end:
            args.end = date.today()
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
