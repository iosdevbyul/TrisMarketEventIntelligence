"""Persistent local gate for reserved news-only TEST exposure.

The registry prevents a *cooperating* research runner from repeatedly
reserving one cohort's final TEST. It cannot detect use outside the registry.
No gold labels, news text, predictions, or market prices are stored here.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.news_importance_review import _read

SCHEMA = "news_test_usage_ledger_v1"
_PURPOSES = {"TRAIN", "VALIDATION", "TEST_FINAL"}
_SQL = """CREATE TABLE IF NOT EXISTS news_test_usage (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL UNIQUE,
    cohort_sha256 TEXT NOT NULL,
    plan_sha256 TEXT NOT NULL,
    candidate_sha256 TEXT NOT NULL,
    purpose TEXT NOT NULL CHECK(purpose IN ('TRAIN','VALIDATION','TEST_FINAL')),
    reserved_at TEXT NOT NULL,
    justification TEXT NOT NULL
)"""
_UNIQUE = """CREATE UNIQUE INDEX IF NOT EXISTS one_final_test_per_cohort
ON news_test_usage(cohort_sha256) WHERE purpose='TEST_FINAL'"""


def _sha(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError("Expected lowercase 64-character SHA-256")
    return value


def _plan(path: Path) -> tuple[str, str, dict[str, int]]:
    raw = _read(path)
    document = json.loads(raw)
    if not isinstance(document, dict) or document.get("schemaVersion") != (
        "news_holdout_plan_v1"
    ) or document.get("status") != "PLANNED_NOT_AUDITED" or (
        document.get("trainingDatasetEligible") is not False
        or document.get("stockPricesUsed") is not False
        or document.get("modelTrained") is not False
    ):
        raise ValueError("Expected the original, research-only news holdout plan")
    cohort_sha = _sha(document.get("cohortSha256"))
    items = document.get("items")
    if not isinstance(items, list) or not items or len(items) > 5000:
        raise ValueError("Nonempty bounded news partition plan required")
    counts: dict[str, int] = {}
    seen: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Invalid partition item")
        article_id = item.get("articleId")
        if type(article_id) is not int or article_id < 1 or article_id in seen:
            raise ValueError("Duplicate or invalid partition article")
        seen.add(article_id)
        partition = item.get("split")
        if partition not in {"TRAIN", "VALIDATION", "TEST", "EMBARGO", "QUARANTINED"}:
            raise ValueError("Unexpected news partition")
        counts[partition] = counts.get(partition, 0) + 1
    if counts != document.get("counts"):
        raise ValueError("Holdout partition counts do not match items")
    return hashlib.sha256(raw).hexdigest(), cohort_sha, counts


def _ledger(path: Path, *, create: bool) -> sqlite3.Connection:
    if path.is_symlink() or path.is_dir():
        raise ValueError("Ledger path cannot be a symlink or directory")
    if create:
        if path.exists() or not path.parent.is_dir():
            raise ValueError("Initialize only a new file in an existing directory")
        uri = path.resolve().as_uri() + "?mode=rwc"
    else:
        if not path.is_file():
            raise ValueError("Initialized, existing research ledger required")
        uri = path.resolve().as_uri() + "?mode=rw"
    db = sqlite3.connect(uri, uri=True, timeout=20, isolation_level=None)
    db.execute("PRAGMA busy_timeout=20000")
    return db


def initialize_usage_ledger(path: Path) -> dict[str, object]:
    """Initialize new research state; never touch production SQLite."""
    with closing(_ledger(path, create=True)) as db:
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute(_SQL)
            db.execute(_UNIQUE)
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
    return {"schemaVersion": SCHEMA, "status": "INITIALIZED",
            "privateResearchOnly": True, "stockPricesUsed": False}


def reserve_news_test_use(
    *, ledger_path: Path, plan_path: Path, experiment_id: str,
    candidate_sha256: str, purpose: str, justification: str,
) -> dict[str, object]:
    """Atomically reserve an experiment; final TEST once per cohort forever.

    Reservation does NOT grant permission to reveal human gold; the caller's
    training/evaluation logic must separately enforce its own data access.
    """
    plan_sha, cohort_sha, counts = _plan(plan_path)
    if purpose not in _PURPOSES or not counts.get(
        "TEST" if purpose == "TEST_FINAL" else purpose
    ):
        raise ValueError("Purpose invalid or requested news partition empty")
    if not isinstance(experiment_id, str) or not 1 <= len(
        experiment_id.strip()
    ) <= 100:
        raise ValueError("Unique experiment identity required")
    if not isinstance(justification, str) or not 20 <= len(
        justification.strip()
    ) <= 1000:
        raise ValueError("Substantive purpose justification required")
    candidate_sha = _sha(candidate_sha256)
    with closing(_ledger(ledger_path, create=False)) as db:
        db.execute("BEGIN IMMEDIATE")
        try:
            tables = {row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            if "news_test_usage" not in tables:
                raise ValueError("Research usage ledger not initialized")
            # Fail closed even if someone removed the defensive unique index.
            if purpose == "TEST_FINAL" and db.execute(
                "SELECT 1 FROM news_test_usage "
                "WHERE cohort_sha256=? AND purpose='TEST_FINAL'",
                (cohort_sha,),
            ).fetchone():
                raise ValueError("Final TEST already reserved for this news cohort")
            try:
                db.execute(
                    "INSERT INTO news_test_usage "
                    "(experiment_id,cohort_sha256,plan_sha256,candidate_sha256,"
                    "purpose,reserved_at,justification) VALUES (?,?,?,?,?,?,?)",
                    (experiment_id, cohort_sha, plan_sha, candidate_sha,
                     purpose, datetime.now(UTC).isoformat(), justification.strip()),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    "Duplicate experiment or previously reserved final TEST"
                ) from exc
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
    return {
        "schemaVersion": SCHEMA,
        "status": "RESERVATION_RECORDED_NOT_EVALUATED",
        "experimentId": experiment_id,
        "purpose": purpose,
        "cohortSha256": cohort_sha,
        "planSha256": plan_sha,
        "candidateSha256": candidate_sha,
        "finalTestReuseBlockedByLedger": purpose == "TEST_FINAL",
        "crossProcessAccessOutsideLedgerAudited": False,
        "goldValuesExposed": False,
        "modelQualityVerified": False,
        "stockPricesUsed": False,
        "productionDbModified": False,
    }


def audit_news_test_usage(path: Path) -> dict[str, object]:
    """Read recorded reservations without opening news or gold data."""
    with closing(_ledger(path, create=False)) as db:
        if not db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='news_test_usage'"
        ).fetchone():
            raise ValueError("Research usage ledger not initialized")
        records = db.execute(
            "SELECT experiment_id,cohort_sha256,plan_sha256,candidate_sha256,"
            "purpose,reserved_at FROM news_test_usage ORDER BY sequence"
        ).fetchall()
    return {
        "schemaVersion": SCHEMA,
        "reservations": [{
            "experimentId": x[0], "cohortSha256": x[1], "planSha256": x[2],
            "candidateSha256": x[3], "purpose": x[4], "reservedAt": x[5],
        } for x in records],
        "totalReservations": len(records),
        "testSetAccessOutsideLedgerVerified": False,
        "modelQualityVerified": False,
        "stockPricesUsed": False,
    }
