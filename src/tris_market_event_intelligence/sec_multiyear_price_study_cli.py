"""Opt-in SEC raw-byte custody and retrospective multi-year EOD event outcomes.

Official EDGAR received time is not the first publisher release or a tradable
market feed. This is a historic as-of audit, not a backtested forecast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from bisect import bisect_right
from datetime import UTC, date, datetime
from datetime import time as clock_time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zoneinfo import ZoneInfo

import httpx

from tris_market_event_intelligence.historical_issuer_corroboration_cli import _read, _write
from tris_market_event_intelligence.historical_shock_research import read_daily_prices

NY = ZoneInfo("America/New_York")
ACC = re.compile(r"0001045810-\d{2}-\d{6}\Z")
MAX_SOURCE_BYTES = 1_500_000
MAX_EVENTS = 20
MAX_SOURCE_FETCHES = 40
HORIZONS = (1, 3, 5)


class _VisibleHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.words: list[str] = []
        self.links: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip += 1
        if tag == "a":
            self.links.extend(v for k, v in attrs if k == "href" and v is not None)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1

    def handle_data(self, value: str) -> None:
        if not self._skip:
            self.words.append(value)


def _html(raw: bytes) -> _VisibleHTML:
    if not raw or len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("SEC_RAW_SIZE_INVALID")
    doc = _VisibleHTML()
    doc.feed(raw.decode("utf-8-sig"))
    return doc


def _approved_url(value: object, accession: str, *, index: bool) -> bool:
    if not isinstance(value, str) or not ACC.fullmatch(accession):
        return False
    parsed = urlsplit(value)
    prefix = f"/Archives/edgar/data/1045810/{accession.replace('-', '')}/"
    # Two approved SEC archive index naming conventions exist.
    flat_index = f"/Archives/edgar/data/1045810/{accession}-index.htm"
    if (parsed.scheme != "https" or parsed.netloc != "www.sec.gov"
            or parsed.query or parsed.fragment or parsed.username or parsed.password
            or unquote(parsed.path) != parsed.path):
        return False
    if index and parsed.path == flat_index:
        return True
    if not parsed.path.startswith(prefix):
        return False
    suffix = parsed.path.removeprefix(prefix)
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", suffix)) and (
        suffix == f"{accession}-index.htm" if index else suffix.lower().endswith(".htm")
    )


def _validated_inputs(review: dict, inventory: dict) -> list[dict]:
    if (review.get("schemaVersion") != "sec_multiyear_eight_k_primary_source_review_v1"
            or review.get("ticker") != "NVDA"
            or review.get("companyCik") != "0001045810"
            or review.get("selectionRule") !=
            "EX_ANTE_FIRST_EIGHT_K_PER_YEAR_FROM_SEC_COVERAGE_REVIEW_QUEUE"
            or review.get("trainingReady") is not False
            or review.get("rawIndexAndExhibitSha256Pinned") is not False
            or inventory.get("schemaVersion") != "sec_historical_filing_inventory_v1"
            or inventory.get("companyCik") != "0001045810"
            or inventory.get("ticker") != "NVDA"
            or inventory.get("trainingReady") is not False):
        raise ValueError("SEC_RESEARCH_SOURCE_CONTRACT_INVALID")
    manifest = review.get("entries")
    filings = inventory.get("filings")
    if (not isinstance(manifest, list) or not 2 <= len(manifest) <= MAX_EVENTS
            or not isinstance(filings, list)
            or inventory.get("candidateFilings") != len(filings)):
        raise ValueError("SEC_RESEARCH_MANIFEST_INVALID")
    inventory_by_acc = {row["accession"]: row for row in filings}
    if len(inventory_by_acc) != len(filings):
        raise ValueError("SEC_INVENTORY_ACCESSION_DUPLICATE")
    seen: set[str] = set()
    years: set[int] = set()
    checked: list[dict] = []
    for event in manifest:
        if not isinstance(event, dict):
            raise TypeError("SEC_EVENT_RECORD_INVALID")
        acc = event.get("accession")
        if not isinstance(acc, str) or not ACC.fullmatch(acc) or acc in seen:
            raise ValueError("SEC_EVENT_ACCESSION_INVALID")
        seen.add(acc)
        row = inventory_by_acc.get(acc)
        day = event.get("filingDate")
        if (not isinstance(day, str) or row is None
                or row.get("filingDate") != day or row.get("form") != "8-K"):
            raise ValueError("SEC_EVENT_INVENTORY_MISMATCH")
        filing_day = date.fromisoformat(day)
        if filing_day.year in years:
            raise ValueError("SEC_EVENT_YEAR_DUPLICATE")
        years.add(filing_day.year)
        accepted = datetime.fromisoformat(event["secAcceptedEastern"])
        if (accepted.utcoffset() is None
                or accepted.astimezone(NY).date() != filing_day
                or accepted.astimezone(UTC).isoformat() != event.get("secAcceptedUtc")
                or accepted.astimezone(NY).time() <= clock_time(16, 0)):
            raise ValueError("SEC_EVENT_ACCEPTANCE_NOT_AFTER_CLOSE")
        earnings = event.get("category") == "EARNINGS"
        if (event.get("category") not in ("EARNINGS", "OTHER_EVENT", "GOVERNANCE")
                or not _approved_url(event.get("indexUrl"), acc, index=True)
                or (earnings and not _approved_url(
                    event.get("exhibit99_1Url"), acc, index=False))
                or (not earnings and event.get("exhibit99_1Url") is not None)
                or event.get("exhibitDocumentReviewed") is not earnings):
            raise ValueError("SEC_EVENT_DOCUMENT_CONTRACT_INVALID")
        checked.append(event)
    return checked


def _verify_index(raw: bytes, event: dict) -> None:
    document = _html(raw)
    text = re.sub(r"\s+", " ", " ".join(document.words))
    accepted = datetime.fromisoformat(event["secAcceptedEastern"]).astimezone(NY)
    acc = event["accession"]
    accepted_text = accepted.strftime("%Y-%m-%d %H:%M:%S")
    if (acc not in text or not re.search(r"\bForm 8-K\b", text, re.IGNORECASE)
            or not re.search(r"Accepted\s+" + re.escape(accepted_text), text, re.IGNORECASE)
            or not re.search(r"Filing Date\s+" + re.escape(event["filingDate"]),
                             text, re.IGNORECASE)):
        raise ValueError("SEC_INDEX_FACTS_OR_ACCEPTANCE_MISMATCH")
    type_pattern = {"EARNINGS": "Item 2.02", "OTHER_EVENT": "Item 8.01",
                    "GOVERNANCE": "Item 5.02"}[event["category"]]
    if type_pattern not in text:
        raise ValueError("SEC_INDEX_EVENT_TYPE_MISMATCH")
    exhibit = event.get("exhibit99_1Url")
    if exhibit is not None:
        target = urlsplit(exhibit).path.rsplit("/", 1)[-1]
        if ("EX-99.1" not in text or not any(
            urlsplit(link).path.rsplit("/", 1)[-1] == target
            for link in document.links
        )):
            raise ValueError("SEC_EXHIBIT_NOT_LISTED_IN_INDEX")


def _verify_exhibit(raw: bytes, event: dict) -> None:
    if len(raw) < 200:
        raise ValueError("SEC_EXHIBIT_TOO_SHORT")
    document = _html(raw)
    text = re.sub(r"\s+", " ", " ".join(document.words)).lower()
    if "nvidia" not in text or "revenue" not in text:
        raise ValueError("SEC_EXHIBIT_ISSUER_OR_RESULT_MISSING")
    if event["category"] != "EARNINGS":
        raise ValueError("SEC_EXHIBIT_NON_EARNINGS")


def _raw_name(event: dict, kind: str) -> str:
    return f"sec-{event['accession']}-{kind}.html"


def _get_raw(client: httpx.Client, url: str, user_agent: str) -> bytes:
    with client.stream("GET", url, follow_redirects=False, headers={
        "User-Agent": user_agent, "Accept": "text/html",
    }) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("SEC_SOURCE_NON_200")
        content = bytearray()
        for chunk in response.iter_bytes():
            content.extend(chunk)
            if len(content) > MAX_SOURCE_BYTES:
                raise ValueError("SEC_SOURCE_OVERSIZED")
    return bytes(content)


def _read_or_download(
    *, path: Path, url: str, client: httpx.Client | None,
    user_agent: str | None, fetch_count: list[int],
) -> bytes:
    if path.is_symlink():
        raise ValueError("SEC_RAW_SYMLINK_FORBIDDEN")
    if path.exists():
        if not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError("SEC_RAW_FILE_INVALID")
        return path.read_bytes()
    if client is None or user_agent is None:
        raise ValueError("SEC_RAW_INPUT_MISSING")
    if fetch_count[0] >= MAX_SOURCE_FETCHES:
        raise ValueError("SEC_FETCH_LIMIT")
    if fetch_count[0]:
        time.sleep(0.55)
    raw = _get_raw(client, url, user_agent)
    fetch_count[0] += 1
    return raw


def build_multiyear_sec_study(
    *, review: dict, review_sha: str, inventory: dict, inventory_sha: str,
    stock_csv: Path, benchmark_csv: Path, raw_dir: Path,
    client: httpx.Client | None = None, user_agent: str | None = None,
    persist_raw: bool = False,
) -> dict:
    events = _validated_inputs(review, inventory)
    if not raw_dir.is_dir() or raw_dir.is_symlink():
        raise ValueError("SEC_PRIVATE_RAW_DIRECTORY_INVALID")
    stock, stock_sha = read_daily_prices(stock_csv)
    market, market_sha = read_daily_prices(benchmark_csv)
    stocks = {bar.day: bar for bar in stock}
    markets = {bar.day: bar for bar in market}
    sessions = sorted(set(stocks) & set(markets))
    fetch_count = [0]
    rows = []
    seen_sessions: set[date] = set()
    for event in events:
        acc = event["accession"]
        index_path = raw_dir / _raw_name(event, "index")
        raw_index = _read_or_download(
            path=index_path, url=event["indexUrl"], client=client,
            user_agent=user_agent, fetch_count=fetch_count,
        )
        _verify_index(raw_index, event)
        if persist_raw and not index_path.exists():
            from tris_market_event_intelligence.historical_source_acquisition import (
                create_private_artifact,
            )
            create_private_artifact(index_path, raw_index)
        exhibit_sha = None
        if event["category"] == "EARNINGS":
            exhibit_path = raw_dir / _raw_name(event, "exhibit99-1")
            raw_exhibit = _read_or_download(
                path=exhibit_path, url=event["exhibit99_1Url"], client=client,
                user_agent=user_agent, fetch_count=fetch_count,
            )
            _verify_exhibit(raw_exhibit, event)
            if persist_raw and not exhibit_path.exists():
                from tris_market_event_intelligence.historical_source_acquisition import (
                    create_private_artifact,
                )
                create_private_artifact(exhibit_path, raw_exhibit)
            exhibit_sha = hashlib.sha256(raw_exhibit).hexdigest()
        accepted = datetime.fromisoformat(event["secAcceptedEastern"]).astimezone(UTC)
        filing_day = date.fromisoformat(event["filingDate"])
        start_index = bisect_right(sessions, filing_day)
        if start_index == 0 or start_index + max(HORIZONS) - 1 >= len(sessions):
            raise ValueError("SEC_EVENT_PRICE_WINDOW_MISSING")
        event_day = sessions[start_index]
        if event_day in seen_sessions:
            raise ValueError("SEC_EVENT_DUPLICATE_PRICE_SESSION")
        seen_sessions.add(event_day)
        cutoff = datetime.combine(event_day, clock_time(9, 30), tzinfo=NY).astimezone(UTC)
        if accepted >= cutoff:
            raise ValueError("SEC_EVENT_FUTURE_CLOCK")
        prior = sessions[start_index - 1]
        results = {}
        for horizon in HORIZONS:
            end = sessions[start_index + horizon - 1]
            sr = stocks[end].adjusted_close / stocks[prior].adjusted_close - 1
            mr = markets[end].adjusted_close / markets[prior].adjusted_close - 1
            results[str(horizon)] = {
                "targetSession": end.isoformat(),
                "stockReturn": sr,
                "benchmarkReturn": mr,
                "excessReturn": sr - mr,
                "directionVersusBenchmark": "UP" if sr > mr else (
                    "DOWN" if sr < mr else "FLAT"
                ),
            }
        rows.append({
            "accession": acc,
            "category": event["category"],
            "secAcceptedAt": accepted.isoformat(),
            "decisionAt": cutoff.isoformat(),
            "eventSession": event_day.isoformat(),
            "secIndexUrl": event["indexUrl"],
            "secIndexSha256": hashlib.sha256(raw_index).hexdigest(),
            "secExhibit99_1Url": event["exhibit99_1Url"],
            "secExhibit99_1Sha256": exhibit_sha,
            "issuerFilingIndexContentVerified": True,
            "issuerExhibitContentVerified": exhibit_sha is not None,
            "sourceAvailableByDecisionCutoff": True,
            "publisherFirstReleaseTimeVerified": False,
            "firstTradableAvailabilityVerified": False,
            "features": {
                "lastKnownPreviousStockAdjustedClose": stocks[prior].adjusted_close,
                "lastKnownPreviousBenchmarkAdjustedClose": markets[prior].adjusted_close,
                "previousSession": prior.isoformat(),
            },
            "outcome": {
                "returnWindow": "PREVIOUS_CLOSE_TO_TARGET_CLOSE",
                "includesMovementBeforeDecision": True,
                "horizonsTradingSessions": results,
            },
            "trainingEligible": False,
        })
    return {
        "schemaVersion": "historical_sec_multiyear_price_study_v1",
        "status": "RAW_SEC_SOURCES_PINNED_RETROSPECTIVE_PRICE_OUTCOMES",
        "ticker": "NVDA",
        "benchmark": "SPY",
        "selectionRule": review["selectionRule"],
        "reviewManifestSha256": review_sha,
        "inventorySha256": inventory_sha,
        "stockSha256": stock_sha,
        "benchmarkSha256": market_sha,
        "eventCount": len(rows),
        "earningsEventCount": sum(row["category"] == "EARNINGS" for row in rows),
        "nonEarningsEventCount": sum(row["category"] != "EARNINGS" for row in rows),
        "rawSourceCount": sum(1 + (row["secExhibit99_1Sha256"] is not None)
                              for row in rows),
        "rawSecSourcesStoredLocally": persist_raw,
        "newsAbsenceVerifiedForControls": False,
        "retrospectiveOutcomeAlreadyKnown": True,
        "priceMovementBeforeDecisionExcluded": False,
        "firstPublisherPublicationTimeVerified": False,
        "trainingReady": False,
        "predictionReady": False,
        "finalTestTouched": False,
        "modelPromotionAllowed": False,
        "causalPriceImpactEstablished": False,
        "productionDbModified": False,
        "events": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SEC raw index/exhibit custody and multi-year retrospective outcomes"
    )
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--stock-csv", type=Path, required=True)
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preview", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        review, review_sha = _read(args.review)
        # SEC inventory is larger than _read()'s 700K input bound.
        from tris_market_event_intelligence.sec_coverage_reconciliation_cli import (
            read_inventory,
        )
        inventory, inventory_sha = read_inventory(args.inventory)
        user_agent = os.getenv("TRIS_SEC_USER_AGENT", "")
        if args.run and (
            not user_agent or len(user_agent) < 12 or len(user_agent) > 200
            or "@" not in user_agent or "\n" in user_agent or "\r" in user_agent
        ):
            raise ValueError("SEC_USER_AGENT_INVALID")
        if args.run and (args.output is None or args.output.exists()
                         or args.output.is_symlink()):
            raise ValueError("SEC_OUTPUT_NOT_NEW")
        with httpx.Client(timeout=httpx.Timeout(30), trust_env=False) as client:
            result = build_multiyear_sec_study(
                review=review, review_sha=review_sha,
                inventory=inventory, inventory_sha=inventory_sha,
                stock_csv=args.stock_csv, benchmark_csv=args.benchmark_csv,
                raw_dir=args.raw_dir, client=client if args.run else None,
                user_agent=user_agent if args.run else None,
                persist_raw=args.run,
            )
        if args.run:
            _write(args.output, result)
        print(json.dumps({
            "status": result["status"],
            "events": result["eventCount"],
            "earnings": result["earningsEventCount"],
            "other": result["nonEarningsEventCount"],
            "rawSources": result["rawSourceCount"],
            "trainingReady": False,
            "outputCreated": args.run,
        }))
    except httpx.HTTPStatusError as exc:
        parser.exit(2, f"sec-multiyear-study: SEC_HTTP_{exc.response.status_code}\n")
    except httpx.HTTPError:
        parser.exit(2, "sec-multiyear-study: SEC_NETWORK_FAILED\n")
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, json.JSONDecodeError):
        parser.exit(2, "sec-multiyear-study: SOURCE_OR_PIT_VALIDATION_FAILED\n")


if __name__ == "__main__":
    main()
