"""Production model inference singleflight and live-event expiry regressions."""
from __future__ import annotations

import fcntl
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from test_local_impact import StubLocalModel, insert_fixture
from test_source_health_gates import NoImpactProvider, observed, seed_event

from tris_market_event_intelligence.collector import Article
from tris_market_event_intelligence.impact_store import ImpactProcessor


def test_competing_process_is_denied_before_inference_or_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    insert_fixture(path)
    model = StubLocalModel()
    lock_path = path.with_name(path.name + ".analysis.lock")
    lock_path.touch(mode=0o600)
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        stats = ImpactProcessor(path, model).process_pending()
        assert stats.already_running is True
        assert stats.processed == stats.failed == stats.impacts == 0
        assert model.calls == 0
    with sqlite3.connect(path) as db:
        assert db.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='impact_runs'"
        ).fetchone()[0] == 0
    stats = ImpactProcessor(path, model).process_pending()
    assert stats.already_running is False
    assert stats.processed == 2
    assert model.calls == 2
    assert ImpactProcessor(path, model).process_pending().processed == 0


def test_explicit_selected_event_lock_never_misreports_success(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    insert_fixture(path)
    model = StubLocalModel()
    lock_path = path.with_name(path.name + ".analysis.lock")
    lock_path.touch(mode=0o600)
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        stats = ImpactProcessor(path, model).process_pending(
            event_ids=(1,), batch_size=1,
        )
        assert stats.already_running is True
        assert model.calls == 0


def test_each_new_live_model_call_rechecks_current_freshness(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    base = datetime.now(UTC)
    first_seen = (base - timedelta(seconds=10)).isoformat()
    published = (base - timedelta(seconds=20)).isoformat()
    store = seed_event(path, first_seen=first_seen, published=published)
    store.save(
        "bbc_business", "https://example.org/feed", b"<new/>",
        [Article(
            "bbc_business", "id-2", "https://example.org/2",
            "NVIDIA executive leadership change", None, published, first_seen,
        )], first_seen, None, None,
    )
    with store.connect() as db:
        db.execute(
            "INSERT INTO economic_events("
            "article_id,extractor_version,classification,status,evidence,"
            "source,source_url,published_at,detected_at,processed_at,capture_sha256"
            ") VALUES (2,'test','CORPORATE_EVENT','CANDIDATE',"
            "'NVIDIA executive leadership change','bbc_business',"
            "'https://example.org/2',?,?,?,'sha')",
            (published, first_seen, first_seen),
        )
        db.execute(
            "INSERT INTO event_metadata(event_id,claim_kind,tags_json,verification_note)"
            " VALUES (2,'REPORTED','[]','test')",
        )
    class AdvancingClock:
        calls = 0

        @classmethod
        def now(cls, timezone: object) -> datetime:
            cls.calls += 1
            return base if cls.calls <= 2 else base + timedelta(minutes=20)

    model = NoImpactProvider()
    with patch("tris_market_event_intelligence.impact_store.datetime", AdvancingClock):
        stats = ImpactProcessor(path, model).process_pending(
            live_freshness_seconds=300,
        )
    assert stats.processed == 1
    assert stats.deferred == 1
    assert model.calls == 1
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM impact_runs").fetchone()[0] == 1


def test_invalid_parameters_rejected_before_lock_file_is_created(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    try:
        ImpactProcessor(path, StubLocalModel()).process_pending(batch_size=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected validation error")
    assert not path.parent.joinpath(path.name + ".analysis.lock").exists()
