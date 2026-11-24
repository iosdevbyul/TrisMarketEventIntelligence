"""One bounded SEC submissions discovery, original capture and optional Gemma analysis."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

from tris_market_event_intelligence.automatic_news_pipeline import process
from tris_market_event_intelligence.collector import Article, NewsStore
from tris_market_event_intelligence.sec_operational_content import canonical_index
from tris_market_event_intelligence.source_article_content import collect_full_text

DEFAULT_CIKS = (1045810, 1326801, 1652044, 2488, 320193, 789019)
BASE = "https://data.sec.gov/submissions/"


def discover(store: NewsStore, client: httpx.Client, *, cik: int, start: date,
             end: date) -> dict:
    if not 1 <= cik <= 9999999999 or start > end:
        raise ValueError("SEC_DISCOVERY_SCOPE_INVALID")
    source = f"sec_filing_{cik}"
    name = f"CIK{cik:010d}.json"
    seen = set()
    forms: Counter[str] = Counter()
    inserted = 0
    pages = 0
    duplicates = 0
    store.initialize()
    with store.connect() as db:
        existing = db.execute(
            "SELECT url,source FROM articles WHERE source IN (?,?)",
            (source, f"sec_8k_{cik}"),
        ).fetchall()
    known = set()
    for url, old_source in existing:
        try:
            known.add(canonical_index(url, old_source))
        except ValueError:
            continue

    def fetch_page(filename: str) -> tuple[bytes, dict, str]:
        time.sleep(0.25)
        with client.stream("GET", BASE + filename, follow_redirects=False) as response:
            response.raise_for_status()
            raw = bytearray()
            for chunk in response.iter_bytes():
                raw.extend(chunk)
                if len(raw) > 8_000_000:
                    raise ValueError("SEC_SUBMISSIONS_TOO_LARGE")
        packet = json.loads(raw)
        if not isinstance(packet, dict):
            raise TypeError("SEC_SUBMISSIONS_SCHEMA")
        return bytes(raw), packet, datetime.now(UTC).isoformat()

    raw, main, stamp = fetch_page(name)
    if str(main.get("cik", "")).lstrip("0") != str(cik):
        raise ValueError("SEC_SUBMISSIONS_CIK_MISMATCH")

    def persist(filename: str, payload: bytes, columns: dict, observed: str) -> None:
        nonlocal inserted, pages, duplicates
        fields = ("accessionNumber", "filingDate", "acceptanceDateTime", "form")
        values = [columns.get(key) for key in fields]
        if not all(isinstance(v, list) for v in values) or len({len(v) for v in values}) != 1:
            raise ValueError("SEC_SUBMISSIONS_COLUMNS")
        articles = []
        for accession, filed, accepted, form in zip(*values, strict=True):
            if not isinstance(accession, str) or not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
                raise ValueError("SEC_ACCESSION_INVALID")
            if not start <= date.fromisoformat(filed) <= end:
                continue
            if not isinstance(form, str) or not form.strip():
                raise ValueError("SEC_FORM_INVALID")
            if accession in seen:
                continue
            seen.add(accession)
            forms[form] += 1
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{accession}-index.htm"
            if url in known:
                duplicates += 1
                continue
            # Acceptance is a publisher metadata clock, not independently proven release.
            published = None
            try:
                parsed = datetime.fromisoformat(accepted)
                if parsed.utcoffset() is not None and parsed <= datetime.fromisoformat(observed):
                    published = parsed.astimezone(UTC).isoformat()
            except (ValueError, TypeError):
                pass
            articles.append(Article(
                source=source, external_id=accession, url=url,
                title=f"{main.get('name', cik)} SEC {form}",
                summary=json.dumps({"form": form, "filingDate": filed,
                                    "secAcceptanceDateTime": accepted}),
                published_at=published, detected_at=observed,
            ))
        inserted += store.save(source, BASE + filename, payload, articles, observed, None, None)
        pages += 1

    filings = main["filings"]
    persist(name, raw, filings["recent"], stamp)
    for item in filings.get("files", []):
        if date.fromisoformat(item["filingFrom"]) > end or date.fromisoformat(item["filingTo"]) < start:
            continue
        filename = item["name"]
        if not re.fullmatch(rf"CIK{cik:010d}-submissions-\d+\.json", filename):
            raise ValueError("SEC_HISTORY_NAME_INVALID")
        if pages >= 32:
            raise ValueError("SEC_HISTORY_PAGE_LIMIT")
        raw, packet, stamp = fetch_page(filename)
        persist(filename, raw, packet, stamp)
    return {"cik": cik, "forms": dict(forms), "filings": len(seen),
            "inserted": inserted, "alreadyKnown": duplicates, "pages": pages}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--cik", type=int, action="append")
    parser.add_argument("--limit", type=int, choices=range(1, 13), default=12)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    agent = os.environ.get("MARKET_EVENT_SEC_USER_AGENT", "")
    if "@" not in agent or not args.db.is_file() or args.db.is_symlink():
        parser.error("Existing DB and identifying MARKET_EVENT_SEC_USER_AGENT required")
    if args.start > args.end or args.end > datetime.now(UTC).date():
        parser.error("Invalid discovery period")
    store = NewsStore(args.db)
    with httpx.Client(timeout=30, trust_env=False, headers={"User-Agent": agent}) as client:
        results = [discover(store, client, cik=cik, start=args.start, end=args.end)
                   for cik in (args.cik or DEFAULT_CIKS)]
        capture = collect_full_text(args.db, client=client, sec_only=True, limit=args.limit)
    analysis = process(args.db, collect=False, sec_only=True, model=args.model,
                       limit=args.limit) if args.analyze else None
    print(json.dumps({"discovery": results, "capture": capture, "analysis": analysis,
                      "allFormsRequested": True, "firstPublicTimeVerified": False,
                      "prospectiveForecasts": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
