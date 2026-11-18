"""Append-only read/write boundary for independently generated market-pressure research.

Opt-in table beside existing operational data; no writes on API GET and
no coupling to legacy market_impacts rows or ticker price databases.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path


class MarketPressureResultStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("Existing non-symlink news database required")
        db = sqlite3.connect(self.path.resolve().as_uri() + (
            "?mode=ro" if readonly else "?mode=rw"
        ), uri=True, timeout=10)
        if readonly:
            db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def initialize(self) -> None:
        """One explicit additive migration; never called by a GET request."""
        with closing(self._connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS market_pressure_research (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_digest TEXT NOT NULL,
                model_id TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                cutoff_date TEXT NOT NULL,
                report_sha256 TEXT NOT NULL,
                report_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                UNIQUE(source_digest, model_id, prompt_version, cutoff_date)
            )""")

    @staticmethod
    def _validate(report: dict) -> None:
        if (
            report.get("schemaVersion") != "market_pressure_auto_cycle_v1"
            or report.get("status") != "CANDIDATE_EVENTS_REQUIRE_REVIEW"
            or report.get("marketReturnsAnalyzed") is not False
            or report.get("historicalPITVerified") is not False
            or not isinstance(report.get("events"), list)
            or not report["events"]
            or not isinstance(report.get("asOfDateExclusive"), str)
            or not isinstance(report.get("modelCalls"), int)
        ):
            raise ValueError("MARKET_PRESSURE_REPORT_UNVERIFIED_OR_INVALID")
        for group in report["events"]:
            analysis = group.get("analysis", {})
            if (
                analysis.get("schemaVersion") != "multi_source_market_pressure_v2"
                or analysis.get("marketReturnsAnalyzed") is not False
                or analysis.get("semanticAccuracyVerified") is not False
                or analysis.get("historicalPITVerified") is not False
                or analysis.get("status") != "CONDITIONAL_MARKET_PRESSURE_REQUIRES_REVIEW"
                or analysis.get("asOfDateExclusive") != report["asOfDateExclusive"]
                or not isinstance(analysis.get("sourceDigestSha256"), str)
                or len(analysis["sourceDigestSha256"]) != 64
                or not isinstance(analysis.get("impacts"), list)
            ):
                raise ValueError("MARKET_PRESSURE_ANALYSIS_INVALID")
            for item in analysis["impacts"]:
                if (
                    item.get("observedMarketReaction") is not False
                    or item.get("pressureNotPriceForecast") is not True
                    or not item.get("supportingSourceIds")
                ):
                    raise ValueError("MARKET_PRESSURE_PROVENANCE_INVALID")

    def save(self, report: dict) -> dict:
        self._validate(report)
        first = report["events"][0]["analysis"]
        # Digest includes all groups to avoid checkpoint aliasing between
        # runs with a common first article but different related evidence.
        manifest = [
            {"digest": group["analysis"]["sourceDigestSha256"],
             "sources": group.get("groupSourceIds", [])}
            for group in report["events"]
        ]
        source_digest = hashlib.sha256(json.dumps(
            manifest, sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        raw = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        checksum = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        key = (source_digest, first["modelId"], first["promptVersion"],
               report["asOfDateExclusive"])
        with closing(self._connect()) as db, db:
            row = db.execute(
                """SELECT id, report_sha256 FROM market_pressure_research
                   WHERE source_digest=? AND model_id=? AND prompt_version=?
                     AND cutoff_date=?""", key,
            ).fetchone()
            if row is not None:
                if row[1] != checksum:
                    raise ValueError("IMMUTABLE_MARKET_PRESSURE_RESEARCH_CONFLICT")
                return {"id": row[0], "inserted": False}
            cur = db.execute(
                """INSERT INTO market_pressure_research
                   (source_digest, model_id, prompt_version, cutoff_date,
                    report_sha256, report_json) VALUES (?,?,?,?,?,?)""",
                (*key, checksum, raw),
            )
            return {"id": cur.lastrowid, "inserted": True}

    def list_results(self, *, after_id: int = 0, limit: int = 100) -> dict:
        if (
            type(after_id) is not int or after_id < 0
            or type(limit) is not int or not 1 <= limit <= 500
        ):
            raise ValueError("INVALID_MARKET_PRESSURE_CURSOR")
        with closing(self._connect(readonly=True)) as db:
            rows = db.execute(
                """SELECT id, report_json, report_sha256
                   FROM market_pressure_research WHERE id > ?
                   ORDER BY id LIMIT ?""", (after_id, limit + 1),
            ).fetchall()
        items = []
        for row_id, raw, checksum in rows[:limit]:
            if hashlib.sha256(raw.encode("utf-8")).hexdigest() != checksum:
                raise ValueError("MARKET_PRESSURE_CHECKSUM_MISMATCH")
            report = json.loads(raw)
            self._validate(report)
            items.append({"id": row_id, **report})
        return {
            "schemaVersion": "market_pressure_research_feed_v1",
            "items": items,
            "nextCursor": items[-1]["id"] if items else after_id,
            "hasMore": len(rows) > limit,
        }
