"""Learning memory report is read-only and exposes rule evidence."""
from __future__ import annotations

import sqlite3

import pytest

from tris_market_event_intelligence.learning_memory_report_cli import inspect


def test_report_requires_initialized_journal(tmp_path):
    path = tmp_path / "study.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE unrelated (id INTEGER)")
    with pytest.raises(ValueError, match="not yet initialized"):
        inspect(path)


def test_report_summarizes_rules_without_mutating_journal(tmp_path):
    path = tmp_path / "study.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE steps (evaluation_json TEXT)"
        )
        db.execute(
            "INSERT INTO steps VALUES (?)",
            (('{"impacts":[{"status":"CORRECT"},{"status":"WRONG"},'
              '{"status":"UNVERIFIED"}]}'),),
        )
        from tris_market_event_intelligence import learning_memory as memory

        memory.initialize(db)
        db.execute(
            "INSERT INTO memory_rules "
            "(family,sector,lesson,diagnosis,status,created_seq,eligible_after) "
            "VALUES ('RATES','FINANCIALS','Compare rate expectations',"
            "'Ignored duration risk','CANDIDATE',1,'2025-01-02T00:00:00+00:00')"
        )
        db.commit()
    report = inspect(path)
    assert report["learningMemory"]["correct"] == 1
    assert report["learningMemory"]["wrong"] == 1
    assert report["learningMemory"]["unverified"] == 1
    assert report["rules"][0]["status"] == "CANDIDATE"
    assert report["rules"][0]["version"] == 1
