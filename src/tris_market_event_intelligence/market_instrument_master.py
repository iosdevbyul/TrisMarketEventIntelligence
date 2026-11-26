"""Current official issuer snapshots, not historical listing or event proof."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import httpx

from tris_market_event_intelligence.listing_research_workflow import (
    US_DIRECTORY,
    _fetch_official_directory,
)
from tris_market_event_intelligence.listing_snapshot_import import (
    MAX_SOURCE_BYTES,
    _us_rows,
)

KIND = "https://kind.krx.co.kr/corpgeneral/corpList.do"
KRX_MARKETS = {"stockMkt": ".KS", "kosdaqMkt": ".KQ"}


class ListingTable(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None and self.row is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def parse_krx(raw: bytes, market: str) -> list[dict]:
    if market not in KRX_MARKETS:
        raise ValueError("Unsupported KRX market")
    try:
        html = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        html = raw.decode("cp949")
    parser = ListingTable()
    parser.feed(html)
    header = None
    output = []
    for row in parser.rows:
        if "회사명" in row and "종목코드" in row:
            header = row
            continue
        if header is None or len(row) != len(header):
            continue
        item = dict(zip(header, row, strict=True))
        code = item["종목코드"].strip()
        name = item["회사명"].strip()
        if not re.fullmatch(r"[0-9]{6}", code) or not name:
            continue
        output.append({
            "market": "XKRX", "symbol": code + KRX_MARKETS[market],
            "name": name, "benchmark": "^KS11" if market == "stockMkt" else "^KQ11",
            "source": KIND, "listingDate": item.get("상장일", ""),
        })
    if not output:
        raise ValueError("No valid KRX listed-company rows")
    return output


def fetch_krx(client: httpx.Client, market: str) -> bytes:
    with client.stream("GET", KIND, params={
        "method": "download", "marketType": market, "searchType": "13",
    }, follow_redirects=False) as response:
        response.raise_for_status()
        chunks = []
        size = 0
        for chunk in response.iter_bytes(65536):
            size += len(chunk)
            if size > MAX_SOURCE_BYTES:
                raise ValueError("KRX listing export exceeds size limit")
            chunks.append(chunk)
    return b"".join(chunks)


def acquire_master(directory: Path, client: httpx.Client) -> dict:
    directory.mkdir()
    sources = []
    instruments = []
    failures = []
    for kind, (url, filename) in US_DIRECTORY.items():
        try:
            raw = _fetch_official_directory(client, kind)
            rows, _ = _us_rows(raw, kind)
            records = [{
                "market": market, "symbol": symbol, "name": name,
                "benchmark": "SPY", "source": url, "listingDate": "",
            } for market, symbol, name in rows]
        except (ValueError, UnicodeError, httpx.HTTPError) as exc:
            failures.append({"source": kind, "error": str(exc)[:300]})
            continue
        (directory / filename).write_bytes(raw)
        sources.append({"source": kind, "sha256": hashlib.sha256(raw).hexdigest(),
                        "rows": len(records), "url": url})
        instruments.extend(records)
    for market in KRX_MARKETS:
        try:
            raw = fetch_krx(client, market)
            records = parse_krx(raw, market)
        except (ValueError, UnicodeError, httpx.HTTPError) as exc:
            failures.append({"source": market, "error": str(exc)[:300]})
            continue
        (directory / (market + ".html")).write_bytes(raw)
        sources.append({"source": market, "sha256": hashlib.sha256(raw).hexdigest(),
                        "rows": len(records), "url": KIND})
        instruments.extend(records)
    if not instruments:
        raise ValueError("No official listing sources available")
    master = {
        "observedAt": datetime.now(UTC).isoformat(), "sources": sources,
        "instruments": instruments, "failures": failures,
        "historicalListingVerified": False, "researchOnly": True,
    }
    (directory / "instrument-master.json").write_text(
        json.dumps(master, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return master
