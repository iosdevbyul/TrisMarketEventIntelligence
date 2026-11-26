"""Research-only memory of locally committed market outcomes."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path


def digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


class ExperienceMemory:
    """Keep outcome observations separate from unverified causal hypotheses."""

    def __init__(self, path: Path):
        if path.is_symlink():
            raise ValueError("Symlink memory path is not supported")
        self.path = path
        with closing(sqlite3.connect(path)) as db, db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS learning_experiences (
                id TEXT PRIMARY KEY, commitment TEXT NOT NULL,
                payload TEXT NOT NULL)"""
            )

    def record(self, row: dict, report: dict, reflection: dict) -> str:
        if (
            report.get("schemaVersion") != "committed_forecast_daily_price_evaluation_v1"
            or report.get("commitmentChainVerifiedLocally") is not True
            or row.get("status") != "OBSERVED_UNAUTHENTICATED_PRECOMMITMENT"
            or row.get("sourceIdentity") != "MATCHED_ORIGINAL_MODEL_AND_BODY"
            or row.get("verdict") not in {"CORRECT", "INCORRECT"}
            or row not in report.get("rows", [])
        ):
            raise ValueError("Expected a matured locally committed observation")
        payload = {
            "outcome": row,
            "reportSha256": digest(report),
            "reflection": reflection,
            "causalStatus": "UNVERIFIED",
            "trainingEligible": False,
            "promotionAllowed": False,
        }
        key = digest(payload)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "INSERT OR IGNORE INTO learning_experiences VALUES (?, ?, ?)",
                (key, row["commitmentSha256"], json.dumps(payload, ensure_ascii=False)),
            )
        return key

    def records(self) -> list[dict]:
        with closing(sqlite3.connect(self.path)) as db:
            return [
                {"id": key, **json.loads(payload)}
                for key, payload in db.execute(
                    "SELECT id, payload FROM learning_experiences ORDER BY id"
                )
            ]
