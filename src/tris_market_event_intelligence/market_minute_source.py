"""Offline, hash-pinned KR/US market-minute import readiness boundary.

A supplied SHA-256 and CSV metadata cannot independently prove exchange
authenticity, corporate-action adjustments, or license to use the data.
No news, market data or credentials are downloaded. Nothing is written.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime
from math import isfinite
from pathlib import Path

from tris_market_event_intelligence.exchange_sessions import MARKET_ZONES, utc

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MAX_SIZE = 32 * 1024 * 1024
_MAX_ROWS = 100_000
_ALLOWED = {"XKRX", "XNYS", "XNAS"}
_KINDS = {"STOCK", "BENCHMARK"}
_REQUIRED = {
    "market", "symbol", "close_at", "adjusted_close", "adjustment",
    "instrument_kind", "provider", "observed_at",
}


class MarketSourceContractError(ValueError):
    pass


def _checked_path(root: Path, filename: str) -> Path:
    if not filename or filename != Path(filename).name or filename in {".", ".."}:
        raise MarketSourceContractError("Source filename must be a plain local file name")
    path = root / filename
    if path.is_symlink() or not path.is_file():
        raise MarketSourceContractError("Source must be an existing non-symlink file")
    if path.stat().st_size > _MAX_SIZE:
        raise MarketSourceContractError("Source CSV exceeds bounded file size")
    return path


def _check_sha(path: Path, expected: str) -> None:
    if not isinstance(expected, str) or not _SHA.fullmatch(expected):
        raise MarketSourceContractError("Source requires explicit lowercase SHA-256")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise MarketSourceContractError("Source fingerprint mismatch")


def audit_market_source(manifest: Path) -> dict[str, object]:
    """Audit *provided* market-minute files; never imply verified provenance."""
    if manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size > 65536:
        raise MarketSourceContractError("Existing bounded non-symlink manifest required")
    raw = manifest.read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict) or document.get("schema_version") != "minute_source_v1":
        raise MarketSourceContractError("Unsupported minute-source manifest")
    entries = document.get("sources")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 12:
        raise MarketSourceContractError("Provide 1..12 bounded market-minute sources")
    root = manifest.resolve().parent
    checked_files: set[str] = set()
    output: list[dict[str, object]] = []
    seen_rows: set[tuple[str, str, datetime]] = set()
    seen_symbols: dict[tuple[str, str], str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "filename", "sha256", "market", "provider", "source_kind",
        }:
            raise MarketSourceContractError("Unexpected market-source metadata fields")
        filename, market, provider = (
            entry["filename"], entry["market"], entry["provider"]
        )
        kind = entry["source_kind"]
        if market not in _ALLOWED or kind not in _KINDS:
            raise MarketSourceContractError("Market and instrument kind must be explicit")
        if not isinstance(provider, str) or not provider.strip() or len(provider) > 80:
            raise MarketSourceContractError("Provider identity required")
        if filename in checked_files:
            raise MarketSourceContractError("Repeated source file")
        checked_files.add(filename)
        path = _checked_path(root, filename)
        _check_sha(path, entry["sha256"])
        count = 0
        min_at: datetime | None = None
        max_at: datetime | None = None
        symbol_set: set[str] = set()
        adjustments: set[str] = set()
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None or set(reader.fieldnames) != _REQUIRED:
                raise MarketSourceContractError("Minute CSV requires exact contract columns")
            for row in reader:
                count += 1
                if count > _MAX_ROWS:
                    raise MarketSourceContractError("Minute CSV exceeds row bound")
                try:
                    if row["market"] != market or row["provider"] != provider:
                        raise ValueError("Market/provider metadata mismatch")
                    if row["instrument_kind"] != kind:
                        raise ValueError("Instrument kind mismatch")
                    symbol = row["symbol"]
                    if not symbol or len(symbol) > 24 or not re.fullmatch(
                        r"[0-9A-Z._^-]+", symbol
                    ):
                        raise ValueError("Invalid stock or index symbol")
                    close_at = utc(datetime.fromisoformat(row["close_at"]))
                    observed_at = utc(datetime.fromisoformat(row["observed_at"]))
                    if close_at.second != 0 or close_at.microsecond != 0:
                        raise ValueError("Expected minute-end close timestamp")
                    if observed_at < close_at:
                        raise ValueError("Bar cannot be observed before close")
                    price = float(row["adjusted_close"])
                    if not isfinite(price) or price <= 0:
                        raise ValueError("Price must be positive and finite")
                    adjustment = row["adjustment"]
                    if adjustment not in {"adjusted", "unadjusted", "unknown"}:
                        raise ValueError("Invalid adjustment metadata")
                except (ValueError, TypeError, KeyError, OverflowError) as exc:
                    raise MarketSourceContractError(
                        f"Invalid minute source row {count}"
                    ) from exc
                key = market, symbol, close_at
                if key in seen_rows:
                    raise MarketSourceContractError("Duplicate minute across source files")
                seen_rows.add(key)
                symbol_key = market, symbol
                if symbol_key in seen_symbols and seen_symbols[symbol_key] != kind:
                    raise MarketSourceContractError("Instrument kind conflicts for one symbol")
                seen_symbols[symbol_key] = kind
                symbol_set.add(symbol)
                adjustments.add(adjustment)
                min_at = close_at if min_at is None else min(min_at, close_at)
                max_at = close_at if max_at is None else max(max_at, close_at)
        _check_sha(path, entry["sha256"])
        if not count:
            status = "NO_MINUTE_BARS"
        elif adjustments != {"adjusted"}:
            status = "ADJUSTED_PRICE_PROVENANCE_REQUIRED"
        else:
            status = "SCHEMA_COMPATIBLE_SOURCE_UNVERIFIED"
        output.append({
            "filename": filename,
            "sha256": entry["sha256"],
            "market": market, "source_kind": kind, "provider": provider,
            "rows": count, "symbols": sorted(symbol_set),
            "first_close_at": min_at.isoformat() if min_at else None,
            "last_close_at": max_at.isoformat() if max_at else None,
            "adjustments": sorted(adjustments), "status": status,
        })
    # Manifest itself is re-checked after reads to fail if the manifest changed.
    if manifest.read_bytes() != raw:
        raise MarketSourceContractError("Manifest modified during inspection")
    markets = sorted({r["market"] for r in output})
    return {
        "schema_version": "minute_source_audit_v1",
        "markets_inspected": markets,
        "sources": output,
        "source_files_hashed": True,
        "source_origin_authenticated": False,
        "adjusted_prices_independently_verified": False,
        "exchange_calendar_verified": False,
        "benchmark_comparability_verified": False,
        "news_first_seen_verified": False,
        "real_market_training_ready": False,
        "prediction_model_trained": False,
        "note": (
            "Observed CSV rows are hash-pinned and schema-audited only. "
            "The existing DonghakStockVision public KRX OHLCV feed is daily "
            "and unadjusted, NOT the required real-time KR/US minute feed."
        ),
    }
