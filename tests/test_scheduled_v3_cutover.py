"""Stage 2-M: no automatic scheduler installation or implicit v3 rollout."""
from __future__ import annotations

import sqlite3
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from tris_market_event_intelligence.collector import NewsStore, parse_rss
from tris_market_event_intelligence.event_store import EventProcessor
from tris_market_event_intelligence.integrated_launchd_config import (
    _confirm_unloaded,
    integrated_launchd_configuration,
    verified_v3_activation,
)


def test_v3_schedule_uses_existing_integrated_job_with_explicit_review_inputs(
    tmp_path: Path,
) -> None:
    db, status = tmp_path / "news.sqlite3", tmp_path / "status.json"
    bundle, output = tmp_path / "listing", tmp_path / "reports"
    config = integrated_launchd_configuration(
        Path("/usr/bin/python3"), db_path=db, status_path=status,
        interval_seconds=900, enable_v3_cutover_catchup=True,
        listing_bundle=bundle, review_output_root=output,
    )
    args = config["ProgramArguments"]
    assert args[2] == "tris_market_event_intelligence.integrated_pipeline_cli"
    assert args.count("--enable-v3-cutover-catchup") == 1
    assert "--enable-ticker-review" in args
    assert args[args.index("--listing-bundle") + 1] == str(bundle)
    assert args[args.index("--review-output-root") + 1] == str(output)
    assert "--enable-local-analysis" not in args
    assert config["StartInterval"] == 900
    assert config["RunAtLoad"] is False
    assert "EnvironmentVariables" not in config
    assert not db.exists()


def test_normal_schedule_stays_v2_without_other_flags(tmp_path: Path) -> None:
    config = integrated_launchd_configuration(
        Path("/usr/bin/python3"),
        db_path=tmp_path / "db.sqlite3",
        status_path=tmp_path / "status.json",
    )
    assert "--enable-v3-cutover-catchup" not in config["ProgramArguments"]
    assert "--enable-ticker-review" not in config["ProgramArguments"]
    assert "--enable-local-analysis" not in config["ProgramArguments"]


@pytest.mark.parametrize("bad", [
    {"enable_v3_cutover_catchup": True},
    {"listing_bundle": Path("/tmp/listing")},
    {"enable_v3_cutover_catchup": True, "enable_local_analysis": True,
     "confirmed_background_inference": True, "model_name": "gemma2:latest",
     "listing_bundle": Path("/tmp/listing"), "review_output_root": Path("/tmp/reviews")},
])
def test_schedule_rejects_incomplete_or_model_incompatible_v3(
    tmp_path: Path, bad: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        integrated_launchd_configuration(
            Path("/usr/bin/python3"),
            db_path=tmp_path / "db.sqlite3",
            status_path=tmp_path / "status.json", **bad,
        )


def test_scheduled_v3_requires_existing_persisted_utc_activation(
    tmp_path: Path,
) -> None:
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    with pytest.raises(ValueError, match="activated manually first"):
        verified_v3_activation(db)
    cutoff, floor, created = EventProcessor(db).activate_v3_rollout()
    assert created and floor == 0
    with sqlite3.connect(db) as conn:
        payload = tuple(conn.iterdump())
    observed = verified_v3_activation(db)
    assert observed == {
        "activated_at": cutoff.isoformat(),
        "article_id_floor": floor,
        "read_only": True,
    }
    with sqlite3.connect(db) as conn:
        assert tuple(conn.iterdump()) == payload
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE event_extractor_rollout SET activated_at=?",
            ((datetime.now(UTC) + timedelta(days=1)).isoformat(),),
        )
    with pytest.raises(ValueError, match="Invalid v3 activation fence"):
        verified_v3_activation(db)


def test_v3_plist_rejects_symlinked_review_paths(tmp_path: Path) -> None:
    actual = tmp_path / "real-bundle"
    actual.mkdir()
    linked = tmp_path / "alias"
    linked.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink"):
        integrated_launchd_configuration(
            Path("/usr/bin/python3"),
            db_path=tmp_path / "db.sqlite3",
            status_path=tmp_path / "status.json",
            enable_v3_cutover_catchup=True,
            listing_bundle=linked, review_output_root=actual,
        )


def test_scheduled_v3_refuses_loaded_or_unknown_launchagent() -> None:
    label = "com.tris.market-event-intelligence-integrated"
    with patch(
        "tris_market_event_intelligence.integrated_launchd_config.subprocess.run",
        return_value=subprocess.CompletedProcess(
            [], 0, stdout="loaded", stderr="",
        ),
    ), pytest.raises(ValueError, match="still loaded"):
        _confirm_unloaded(label)
    with patch(
        "tris_market_event_intelligence.integrated_launchd_config.subprocess.run",
        return_value=subprocess.CompletedProcess(
            [], 3, stdout="", stderr="Permission denied",
        ),
    ), pytest.raises(ValueError, match="unknown"):
        _confirm_unloaded(label)
    with patch(
        "tris_market_event_intelligence.integrated_launchd_config.subprocess.run",
        return_value=subprocess.CompletedProcess(
            [], 113, stdout="", stderr=f'Could not find service "{label}" in domain',
        ),
    ):
        _confirm_unloaded(label)


def test_rollback_v2_does_not_reprocess_v3_checkpointed_articles(
    tmp_path: Path,
) -> None:
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    activated_at, floor, _ = EventProcessor(db).activate_v3_rollout()
    receipt = datetime.now(UTC).isoformat()
    payload = (
        "<rss><channel><item><guid>meta</guid>"
        "<title>Italy fines Meta €805m over accessing users' photos</title>"
        "<link>https://example.org/meta</link></item></channel></rss>"
    ).encode()
    store.save(
        "bbc_world", "https://example.org/feed", payload,
        parse_rss("bbc_world", payload, receipt), receipt, None, None,
    )
    v3 = EventProcessor(db).process_pending(
        extractor_version="multi-v3",
        received_since=activated_at, after_article_id=floor,
    )
    assert v3.processed == 1
    assert v3.events == 1
    v2 = EventProcessor(db).process_pending()
    assert v2.processed == 0
    assert v2.events == 0
    with sqlite3.connect(db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM economic_events"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM extraction_runs"
        ).fetchone()[0] == 1
