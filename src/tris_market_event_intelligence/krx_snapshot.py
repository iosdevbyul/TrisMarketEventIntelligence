"""Read-only, hash-pinned DonghakStockVision KRX snapshot adapter.

Never open a live operational DB, change a record, synthesize corporate
action adjustments, or treat raw KRX stock bars as a KOSPI index.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

from tris_market_event_intelligence.event_study import DailyClose

OPERATIONAL_APPLICATION_ID = 0x4453564F
TICKER = re.compile(r"[0-9A-Z]{6}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")


class MarketSnapshotError(ValueError):
    pass


class PinnedKRXSnapshot:
    """Read one explicitly approved immutable SQLite snapshot, without writes."""

    def __init__(self, path: Path, *, expected_sha256: str) -> None:
        if not SHA.fullmatch(expected_sha256):
            raise MarketSnapshotError("explicit lowercase SHA-256 fingerprint required")
        if path.is_symlink() or not path.is_file():
            raise MarketSnapshotError("existing non-symlink snapshot file required")
        self.path = path.resolve()
        self.sha256 = expected_sha256
        self._check_file()
        with closing(self._connect()) as db:
            if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise MarketSnapshotError("SQLite integrity check failed")
            app_id = db.execute("PRAGMA application_id").fetchone()[0]
            tables = {
                row[0] for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if app_id == OPERATIONAL_APPLICATION_ID:
                if not path.name.endswith(".snapshot.sqlite3"):
                    raise MarketSnapshotError("operational data requires offline .snapshot.sqlite3")
                if db.execute("PRAGMA user_version").fetchone()[0] != 1:
                    raise MarketSnapshotError("unsupported operational database version")
                table, required = "market_observations", {
                    "market", "ticker", "trading_date", "payload", "validation_status",
                }
                self.kind = "operational_snapshot"
            else:
                table, required = "bars", {"ticker", "trading_date", "payload"}
                self.kind = "research_snapshot"
            if table not in tables:
                raise MarketSnapshotError("unsupported market snapshot schema")
            columns = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
            if not required.issubset(columns):
                raise MarketSnapshotError("market snapshot required columns missing")
            self.table = table
        self._check_file()

    def _check_file(self) -> None:
        for suffix in ("-wal", "-journal"):
            sidecar = Path(str(self.path) + suffix)
            if sidecar.exists() and sidecar.stat().st_size:
                raise MarketSnapshotError("active SQLite sidecar; use an offline snapshot")
        with self.path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != self.sha256:
            raise MarketSnapshotError("market snapshot fingerprint mismatch")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(f"{self.path.as_uri()}?mode=ro&immutable=1", uri=True)
        db.execute("PRAGMA query_only=ON")
        return db

    def series(self, ticker: str, *, limit: int = 2500) -> dict[str, object]:
        if not TICKER.fullmatch(ticker):
            raise MarketSnapshotError("ticker must be a six-character KRX short code")
        if not 1 <= limit <= 2500:
            raise ValueError("bounded limit must be 1..2500")
        self._check_file()
        with closing(self._connect()) as db:
            if self.kind == "operational_snapshot":
                rows = db.execute(
                    "SELECT trading_date,payload,validation_status FROM market_observations "
                    "WHERE market='KOSPI' AND ticker=? ORDER BY trading_date DESC LIMIT ?",
                    (ticker, limit),
                ).fetchall()
            else:
                rows = [
                    (day, payload, None)
                    for day, payload in db.execute(
                        "SELECT trading_date,payload FROM bars WHERE ticker=? "
                        "ORDER BY trading_date DESC LIMIT ?",
                        (ticker, limit),
                    )
                ]
        self._check_file()
        adjustments: set[str] = set()
        days: set[date] = set()
        eligible: list[DailyClose] = []
        validation_statuses: set[str] = set()
        no_trade = 0
        for day_text, raw, validation in rows:
            try:
                row = json.loads(raw)
                day = date.fromisoformat(day_text)
                if (
                    type(row) is not dict or row.get("ticker") != ticker
                    or row.get("trading_date") != day_text
                    or row.get("market") != "KOSPI"
                    or row.get("provider") != "krx"
                    or row.get("adjustment") not in {"adjusted", "unadjusted", "unknown"}
                    or type(row.get("close")) is not int or row["close"] <= 0
                    or type(row.get("volume")) is not int or row["volume"] < 0
                ):
                    raise ValueError("invalid bar provenance or values")
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                raise MarketSnapshotError("invalid persisted KRX observation") from exc
            if day in days:
                raise MarketSnapshotError("duplicate trading day")
            days.add(day)
            adjustments.add(row["adjustment"])
            if validation is not None:
                validation_statuses.add(validation)
            if row["volume"] == 0:
                no_trade += 1
            eligible.append(DailyClose(ticker, day, float(row["close"])))
        if len(adjustments) > 1:
            raise MarketSnapshotError("mixed adjustment modes in one ticker series")
        adjustment = next(iter(adjustments)) if adjustments else None
        eligible_for_labels = adjustment == "adjusted" and no_trade == 0
        # Operational observations may have validation statuses other than
        # accepted; never promote them to research-ready just by reading.
        if self.kind == "operational_snapshot":
            eligible_for_labels = False
        if not rows:
            status = "NO_OBSERVATIONS"
        elif adjustment != "adjusted":
            status = "CORPORATE_ACTION_ADJUSTMENT_REQUIRED"
        elif no_trade:
            status = "NO_TRADE_BARS_REQUIRE_REVIEW"
        elif self.kind == "operational_snapshot":
            status = "OPERATIONAL_SOURCE_NOT_RESEARCH_APPROVED"
        else:
            status = "ADJUSTED_SERIES_AVAILABLE"
        return {
            "ticker": ticker,
            "source_kind": self.kind,
            "source_sha256": self.sha256,
            "observation_count": len(rows),
            "first_trading_date": min(days).isoformat() if days else None,
            "last_trading_date": max(days).isoformat() if days else None,
            "adjustment": adjustment,
            "no_trade_bars": no_trade,
            "validation_statuses": sorted(validation_statuses),
            "event_study_price_eligible": eligible_for_labels,
            "status": status,
        }

    def adjusted_closes(self, ticker: str, *, limit: int = 2500) -> list[DailyClose]:
        """Fail closed if any observation cannot safely become a return label."""
        info = self.series(ticker, limit=limit)
        if not info["event_study_price_eligible"]:
            raise MarketSnapshotError(f"cannot label unqualified price series: {info['status']}")
        self._check_file()
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT trading_date,payload FROM bars WHERE ticker=? "
                "ORDER BY trading_date DESC LIMIT ?", (ticker, limit),
            ).fetchall()
        self._check_file()
        return [
            DailyClose(ticker, date.fromisoformat(day), float(json.loads(raw)["close"]))
            for day, raw in reversed(rows)
        ]
