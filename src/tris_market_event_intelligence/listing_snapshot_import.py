"""Offline audited symbol-directory snapshots into review-only issuer CSV.

No web requests, DB writes, issuer backdating, or automatic approved links.
Source/observed times are supplied assertions and are NOT authenticated.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path

from tris_market_event_intelligence.event_ticker_proposals import (
    FIELDS,
    MAX_ROWS,
    _names,
    _when,
    load_reviewed_issuers,
)

MAX_SOURCE_BYTES = 3_000_000
MAX_MANIFEST_BYTES = 65_536
SHA = re.compile(r"[a-f0-9]{64}\Z")
KOREAN = re.compile(r"[0-9]{6}\Z")
US = re.compile(r"[A-Z][A-Z0-9.\-]{0,19}\Z")
TYPES = {"NASDAQ_DIRECTORY", "NYSE_DIRECTORY", "KRX_REVIEWED_EXPORT"}
NASDAQ_FIELDS = {
    "Symbol", "Security Name", "Market Category", "Test Issue", "Financial Status",
    "Round Lot Size", "ETF", "NextShares",
}
OTHER_FIELDS = {
    "ACT Symbol", "Security Name", "Exchange", "CQS Symbol", "ETF",
    "Round Lot Size", "Test Issue", "NASDAQ Symbol",
}
KRX_FIELDS = {"market", "symbol", "company_name", "security_type"}


def _read_snapshot(root: Path, filename: str, digest: str) -> bytes:
    if not isinstance(filename, str) or filename != Path(filename).name or (
        filename in {"", ".", ".."}
    ):
        raise ValueError("Snapshot filename must be a plain local basename")
    if not isinstance(digest, str) or not SHA.fullmatch(digest):
        raise ValueError("Explicit lowercase SHA-256 required")
    path = root / filename
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("Snapshot must be a bounded non-symlink file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Snapshot SHA-256 mismatch")
    return raw


def _rows(raw: bytes, *, delimiter: str, expected: set[str]) -> list[dict[str, str]]:
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter)
    if reader.fieldnames is None or set(reader.fieldnames) != expected or (
        len(reader.fieldnames) != len(expected)
    ):
        raise ValueError("Snapshot columns are not the expected source schema")
    result: list[dict[str, str]] = []
    for index, row in enumerate(reader, start=1):
        if index > MAX_ROWS:
            raise ValueError("Snapshot exceeds issuer row limit")
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed snapshot row")
        result.append(row)
    return result


def _us_rows(raw: bytes, kind: str) -> tuple[list[tuple[str, str, str]], dict[str, int]]:
    expected = NASDAQ_FIELDS if kind == "NASDAQ_DIRECTORY" else OTHER_FIELDS
    rows = _rows(raw, delimiter="|", expected=expected)
    output: list[tuple[str, str, str]] = []
    counts = {"test": 0, "etf": 0, "non_nyse": 0, "unsupported": 0}
    footer_found = False
    for item in rows:
        symbol = item["Symbol"] if kind == "NASDAQ_DIRECTORY" else item["ACT Symbol"]
        if symbol.startswith("File Creation Time:"):
            footer_found = True
            continue
        if not US.fullmatch(symbol) or not item["Security Name"].strip():
            counts["unsupported"] += 1
            continue
        if kind == "NYSE_DIRECTORY" and item["Exchange"] != "N":
            counts["non_nyse"] += 1
            continue
        if item["Test Issue"] != "N":
            counts["test"] += 1
            continue
        if item["ETF"] != "N":
            counts["etf"] += 1
            continue
        name = item["Security Name"].strip()
        # Listing files include warrants, debt, rights, units, ADRs etc.
        # Fail closed: do not assume that 'ETF=N' means common stock.
        suffix = " - Common Stock"
        if not name.endswith(suffix):
            counts["unsupported"] += 1
            continue
        candidate = name[:-len(suffix)].strip()
        if not candidate:
            counts["unsupported"] += 1
            continue
        output.append(("XNAS" if kind == "NASDAQ_DIRECTORY" else "XNYS", symbol, candidate))
    if not footer_found:
        raise ValueError("Official US symbol directory creation footer is required")
    return output, counts


def _krx_rows(raw: bytes) -> tuple[list[tuple[str, str, str]], dict[str, int]]:
    # This is a deliberately explicit normalized *reviewed export* contract,
    # not a guess at the KRX website's Korean-language CSV headings.
    rows = _rows(raw, delimiter=",", expected=KRX_FIELDS)
    output: list[tuple[str, str, str]] = []
    counts = {"test": 0, "etf": 0, "non_nyse": 0, "unsupported": 0}
    for row in rows:
        if (
            row["market"] != "XKRX"
            or not KOREAN.fullmatch(row["symbol"])
            or row["security_type"] != "COMMON"
            or not row["company_name"].strip()
        ):
            counts["unsupported"] += 1
            continue
        output.append(("XKRX", row["symbol"], row["company_name"].strip()))
    return output, counts


def preview_listing_snapshot(manifest: Path) -> tuple[dict[str, object], list[dict[str, str]]]:
    """Build unverified snapshot-only issuer rows without changing any files."""
    if (
        manifest.is_symlink() or not manifest.is_file()
        or manifest.stat().st_size > MAX_MANIFEST_BYTES
    ):
        raise ValueError("Bounded existing non-symlink manifest required")
    raw_manifest = manifest.read_bytes()
    document = json.loads(raw_manifest)
    if not isinstance(document, dict) or set(document) != {"schema_version", "sources"} or (
        document["schema_version"] != "listing_snapshot_v1"
    ):
        raise ValueError("Invalid listing snapshot manifest schema")
    sources = document["sources"]
    if not isinstance(sources, list) or not 1 <= len(sources) <= 3:
        raise ValueError("Expected 1..3 source snapshots")
    output: list[dict[str, str]] = []
    reports: list[dict[str, object]] = []
    seen_files: set[str] = set()
    seen_symbols: set[tuple[str, str]] = set()
    for source in sources:
        if not isinstance(source, dict) or set(source) != {
            "kind", "filename", "sha256", "observed_at", "source_id",
        }:
            raise ValueError("Invalid source manifest fields")
        kind = source["kind"]
        if kind not in TYPES or not isinstance(source["filename"], str):
            raise ValueError("Unsupported snapshot source type")
        if kind in seen_files:
            raise ValueError("At most one snapshot per source kind")
        seen_files.add(kind)
        evidence_at = _when(source["observed_at"])
        source_id = source["source_id"]
        if not isinstance(source_id, str) or not source_id.strip() or len(source_id) > 200:
            raise ValueError("Source identifier must be bounded")
        data = _read_snapshot(manifest.parent, source["filename"], source["sha256"])
        rows, excluded = (
            _krx_rows(data) if kind == "KRX_REVIEWED_EXPORT"
            else _us_rows(data, kind)
        )
        accepted = 0
        for market, symbol, name in rows:
            try:
                cleaned = _names(name, "")
            except ValueError:
                excluded["unsupported"] += 1
                continue
            key = market, symbol
            if key in seen_symbols:
                raise ValueError("Duplicate market-symbol across sources")
            seen_symbols.add(key)
            output.append({
                "market": market, "symbol": symbol, "canonical_name": cleaned[0],
                "aliases": "", "valid_from": evidence_at.isoformat(),
                "valid_until": "", "evidence_seen_at": evidence_at.isoformat(),
                "source_id": source_id,
            })
            accepted += 1
        reports.append({
            "kind": kind, "sha256": source["sha256"],
            "observed_at": evidence_at.isoformat(), "source_id": source_id,
            "accepted": accepted, "excluded": excluded,
        })
    if manifest.read_bytes() != raw_manifest:
        raise ValueError("Manifest changed during read")
    if not output:
        raise ValueError("No supported common stock listings in snapshots")
    if len(output) > MAX_ROWS:
        raise ValueError("Combined listing rows exceed existing matcher limit")
    output.sort(key=lambda r: (r["market"], r["symbol"]))
    return ({
        "schema_version": "listing_snapshot_preview_v1",
        "status": "REVIEW_REQUIRED", "source_reports": reports,
        "issuer_rows": len(output), "source_hashes_checked": True,
        "source_origin_authenticated": False,
        "market_listing_history_verified": False,
        "snapshot_evidence_times_independently_verified": False,
        "historical_backfill_allowed": False,
        "approved_ticker_links_created": False,
        "database_modified": False,
        "note": "Snapshot observed_at is the earliest eligible time, never listing inception.",
    }, output)


def export_snapshot_registry(
    manifest: Path, destination: Path, *, write: bool,
) -> dict[str, object]:
    report, rows = preview_listing_snapshot(manifest)
    if not write:
        return report
    if destination.is_symlink():
        raise ValueError("Registry output symlinks prohibited")
    # Exclusive create: never overwrite the operator's existing artifacts.
    with destination.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    # Validate with exactly the same contract as the existing matcher.
    load_reviewed_issuers(destination)
    report["registry_created"] = str(destination)
    return report
