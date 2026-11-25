"""Collect official macroeconomic observations without running analysis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sqlite3
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx

FRED_SERIES = (
    "FEDFUNDS", "CPIAUCSL", "CPILFESL", "PPIACO", "PCEPI", "PCEPILFE",
    "UNRATE", "PAYEMS", "ICSA", "GDPC1", "RSAFS", "PCEC96", "INDPRO",
    "DGS2", "DGS10", "DEXKOUS", "M2SL", "HOUST", "PERMIT", "UMCSENT",
    "BOPGSTB",
)
KOREA_WB_SERIES = (
    "NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG", "SL.UEM.TOTL.ZS",
    "NE.EXP.GNFS.KD.ZG", "NE.IMP.GNFS.KD.ZG", "BN.CAB.XOKA.GD.ZS",
    "NV.IND.TOTL.KD.ZG", "NE.CON.PRVT.KD.ZG",
)
BLS_SERIES = ("CUUR0000SA0", "CUSR0000SA0", "LNS14000000", "CES0000000001")
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
BLS_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
ECOS_URL = "https://ecos.bok.or.kr/api/StatisticSearch"

SCHEMA = """
CREATE TABLE IF NOT EXISTS macro_raw (
    source TEXT NOT NULL, series TEXT NOT NULL, digest TEXT NOT NULL,
    fetched_at TEXT NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY (source, series, digest)
);
CREATE TABLE IF NOT EXISTS macro_observation (
    source TEXT NOT NULL, series TEXT NOT NULL, period TEXT NOT NULL,
    vintage_start TEXT NOT NULL, vintage_end TEXT NOT NULL,
    value TEXT NOT NULL, fetched_at TEXT NOT NULL, raw_digest TEXT NOT NULL,
    PRIMARY KEY (source, series, period, vintage_start, vintage_end, value)
);
"""


def save(
    db: sqlite3.Connection, source: str, series: str, payload: dict,
    rows: list[tuple[str, str, str, str]],
) -> int:
    now = datetime.now(UTC).isoformat()
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(raw.encode()).hexdigest()
    with db:
        db.execute(
            "INSERT OR IGNORE INTO macro_raw VALUES (?, ?, ?, ?, ?)",
            (source, series, digest, now, raw),
        )
        before = db.total_changes
        db.executemany(
            "INSERT OR IGNORE INTO macro_observation VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(source, series, *row, now, digest) for row in rows],
        )
        return db.total_changes - before


def fred_collect(
    client: httpx.Client, db: sqlite3.Connection, series: str,
    start: date, end: date, key: str,
) -> int:
    offset = total = 0
    while True:
        response = client.get(FRED_URL, params={
            "api_key": key, "file_type": "json", "series_id": series,
            "observation_start": start.isoformat(), "observation_end": end.isoformat(),
            "realtime_start": "1776-07-04", "realtime_end": end.isoformat(),
            "limit": 100000, "offset": offset,
        })
        response.raise_for_status()
        payload = response.json()
        observations = payload["observations"]
        rows = [
            (r["date"], r["realtime_start"], r["realtime_end"], r["value"])
            for r in observations
        ]
        total += save(db, "FRED", series, payload, rows)
        offset += len(observations)
        if offset >= int(payload["count"]):
            return total
        if not observations:
            raise ValueError("incomplete FRED pagination")


def fred_csv_collect(
    client: httpx.Client, db: sqlite3.Connection, series: str,
    start: date, end: date,
) -> int:
    response = client.get(FRED_CSV_URL, params={
        "id": series, "cosd": start.isoformat(), "coed": end.isoformat(),
    })
    response.raise_for_status()
    reader = csv.DictReader(io.StringIO(response.text.lstrip("\ufeff")))
    if reader.fieldnames not in (["observation_date", series], ["DATE", series]):
        raise ValueError("unexpected FRED CSV columns")
    rows = []
    for record in reader:
        period = date.fromisoformat(record[reader.fieldnames[0]])
        value = record[series]
        if start <= period <= end and value not in ("", "."):
            try:
                if not Decimal(value).is_finite():
                    raise ValueError("nonfinite observation")
            except InvalidOperation as exc:
                raise ValueError("invalid observation") from exc
            rows.append((period.isoformat(), "", "", value))
    if not rows:
        raise ValueError("empty FRED CSV observations")
    return save(db, "FRED_CSV", series, {
        "url": str(response.request.url), "format": "csv",
        "vintageKnown": False, "csv": response.text,
    }, rows)


def korea_world_bank_collect(
    client: httpx.Client, db: sqlite3.Connection, series: str,
    start: date, end: date,
) -> int:
    url = f"https://api.worldbank.org/v2/country/KOR/indicator/{series}"
    page, total, seen = 1, 0, 0
    while True:
        response = client.get(url, params={
            "format": "json", "date": f"{start.year}:{end.year}",
            "per_page": 100, "page": page, "source": 2,
        })
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) != 2:
            raise ValueError("unexpected World Bank response")
        metadata, records = payload
        if not isinstance(records, list) or int(metadata["page"]) != page:
            raise ValueError("invalid World Bank page")
        rows = []
        for record in records:
            if record["countryiso3code"] != "KOR" or record["indicator"]["id"] != series:
                raise ValueError("World Bank series mismatch")
            year = int(record["date"])
            value = record["value"]
            if start.year <= year <= end.year and value is not None:
                rows.append((f"{year:04}-01-01", "", "", str(value)))
        total += save(db, "WORLD_BANK_KOR", series, {
            "url": str(response.request.url), "frequency": "annual",
            "vintageKnown": False, "response": payload,
        }, rows)
        seen += len(records)
        if page >= int(metadata["pages"]):
            if seen != int(metadata["total"]):
                raise ValueError("incomplete World Bank pagination")
            return total
        if not records:
            raise ValueError("empty World Bank page")
        page += 1


def bls_collect(
    client: httpx.Client, db: sqlite3.Connection, series: str,
    start: date, end: date,
) -> int:
    total = 0
    for year in range(start.year, end.year + 1, 10):
        response = client.post(BLS_URL, json={
            "seriesid": [series], "startyear": str(year),
            "endyear": str(min(year + 9, end.year)),
        })
        response.raise_for_status()
        payload = response.json()
        if payload["status"] != "REQUEST_SUCCEEDED":
            raise ValueError("BLS request unsuccessful")
        records = payload["Results"]["series"]
        if len(records) != 1 or records[0]["seriesID"] != series:
            raise ValueError("BLS series mismatch")
        rows = []
        for r in records[0]["data"]:
            month = r["period"]
            if month not in {f"M{i:02}" for i in range(1, 13)}:
                continue
            period = date(int(r["year"]), int(month[1:]), 1)
            if start.replace(day=1) <= period <= end:
                rows.append((period.isoformat(), "", "", r["value"]))
        total += save(db, "BLS", series, payload, rows)
    return total


def ecos_collect(
    client: httpx.Client, db: sqlite3.Connection, key: str,
    spec: str, start: date, end: date,
) -> int:
    table, cycle, item = spec.split(":")
    fmt = {"D": "%Y%m%d", "M": "%Y%m", "A": "%Y"}[cycle]
    offset, total = 1, 0
    while True:
        url = (
            f"{ECOS_URL}/{key}/json/kr/{offset}/{offset + 999}/"
            f"{table}/{cycle}/{start.strftime(fmt)}/{end.strftime(fmt)}/{item}"
        )
        response = client.get(url)
        response.raise_for_status()
        payload = response.json()
        result = payload["StatisticSearch"]
        records = result["row"]
        rows = [(r["TIME"], "", "", r["DATA_VALUE"]) for r in records]
        total += save(db, "ECOS", spec, payload, rows)
        offset += len(records)
        if offset > int(result["list_total_count"]):
            return total
        if not records:
            raise ValueError("incomplete ECOS pagination")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument(
        "--source", choices=("all", "fred", "fred-vintage", "bls", "ecos", "korea"),
        default="all",
    )
    parser.add_argument("--fred-series", action="append", choices=FRED_SERIES)
    parser.add_argument("--ecos-series", action="append", default=[])
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("start must not exceed end")
    args.db.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with sqlite3.connect(args.db) as db, httpx.Client(timeout=30) as client:
        db.executescript(SCHEMA)
        jobs = []
        if args.source in ("all", "fred"):
            jobs.extend(
                ("FRED_CSV", s, lambda s=s: fred_csv_collect(
                    client, db, s, args.start, args.end
                )) for s in (args.fred_series or FRED_SERIES)
            )
        if args.source == "fred-vintage":
            key = os.getenv("FRED_API_KEY")
            if key:
                jobs.extend(
                    ("FRED", s, lambda s=s: fred_collect(
                        client, db, s, args.start, args.end, key
                    )) for s in (args.fred_series or FRED_SERIES)
                )
            else:
                results.append({"source": "FRED", "status": "MISSING_API_KEY"})
        if args.source in ("all", "bls"):
            jobs.extend(
                ("BLS", s, lambda s=s: bls_collect(
                    client, db, s, args.start, args.end
                )) for s in BLS_SERIES
            )
        if args.source in ("all", "korea"):
            jobs.extend(
                ("WORLD_BANK_KOR", s, lambda s=s: korea_world_bank_collect(
                    client, db, s, args.start, args.end
                )) for s in KOREA_WB_SERIES
            )
        if args.source == "ecos":
            ecos_key = os.getenv("ECOS_API_KEY")
            if ecos_key:
                jobs.extend(
                    ("ECOS", s, lambda s=s: ecos_collect(
                        client, db, ecos_key, s, args.start, args.end
                    )) for s in (args.ecos_series or ["722Y001:D:0101000"])
                )
            else:
                results.append({"source": "ECOS", "status": "MISSING_API_KEY"})
        for source, series, collect in jobs:
            try:
                added = collect()
                count = db.execute(
                    "SELECT COUNT(*) FROM macro_observation WHERE source=? AND series=?",
                    (source, series),
                ).fetchone()[0]
                results.append({
                    "source": source, "series": series,
                    "status": "COLLECTED" if count else "EMPTY",
                    "newObservations": added, "storedObservations": count,
                })
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                # Do not print URLs or exception messages containing credentials.
                results.append({
                    "source": source, "series": series, "status": "FAILED",
                    "errorType": type(exc).__name__,
                })
        print(json.dumps({"results": results, "analysisPerformed": False}, ensure_ascii=False))
    if any(r["status"] != "COLLECTED" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
