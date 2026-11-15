"""Real read-only NEWS SQLite refresh uses existing source validation."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_historical_news_cohort import seeded

from tris_market_event_intelligence.historical_news_cohort import historical_news_cohort
from tris_market_event_intelligence.news_coverage_refresh import refresh_news_coverage


def _baseline(tmp_path: Path) -> tuple[Path, Path]:
    source_db = seeded(tmp_path)
    report = historical_news_cohort(
        source_db=source_db,
        as_of=datetime(2026, 11, 1, 13, tzinfo=UTC),
    )
    baseline = tmp_path / "original.json"
    baseline.write_text(json.dumps(report), encoding="utf-8")
    return source_db, baseline


def _refresh(db: Path, baseline: Path, output: Path, day: int):
    return refresh_news_coverage(
        baseline_path=baseline,
        source_db=db,
        output_path=output,
        as_of=datetime(2026, 11, day, 13, tzinfo=UTC),
    )


def test_no_fresh_news_creates_no_file_and_preserves_originals(
    tmp_path: Path,
) -> None:
    db, baseline = _baseline(tmp_path)
    before = baseline.read_bytes()
    output = tmp_path / "refreshed.json"
    result = _refresh(db, baseline, output, 2)
    assert result["status"] == "NO_NEW_NEWS_SNAPSHOT_NOT_CREATED"
    assert result["baselineArticleCount"] == 3
    assert result["latestObservedArticleCount"] == 3
    assert result["growth"]["newArticles"] == 0
    assert result["modelCalls"] == 0
    assert result["stockPricesUsed"] is False
    assert not output.exists()
    assert baseline.read_bytes() == before
    assert not list(tmp_path.glob(".news-coverage-refresh-*"))


def test_new_news_persists_only_after_immutable_history_audit(
    tmp_path: Path,
) -> None:
    db, baseline = _baseline(tmp_path)
    output = tmp_path / "refreshed.json"
    before = baseline.read_bytes()
    result = _refresh(db, baseline, output, 10)
    assert result["status"] == "NEW_NEWS_SNAPSHOT_VERIFIED_AND_CREATED"
    assert result["growth"]["newArticles"] == 1
    assert result["growth"]["unchangedPinnedOriginalArticles"] == 3
    assert result["outputCreated"] is True
    assert result["modelCalls"] == 0
    assert result["productionDbModified"] is False
    assert output.exists()
    assert os.stat(output).st_mode & 0o777 == 0o600
    assert len(json.loads(output.read_text())["items"]) == 4
    assert baseline.read_bytes() == before
    assert not list(tmp_path.glob(".news-coverage-refresh-*"))


def test_modified_older_article_refuses_output(tmp_path: Path) -> None:
    db, baseline = _baseline(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE articles SET title='Altered historical headline' WHERE id=1")
    output = tmp_path / "blocked.json"
    with pytest.raises(ValueError, match="content, identity or receipt"):
        _refresh(db, baseline, output, 10)
    assert not output.exists()
    assert not list(tmp_path.glob(".news-coverage-refresh-*"))


def test_output_is_create_only_and_not_overwritten(tmp_path: Path) -> None:
    db, baseline = _baseline(tmp_path)
    output = tmp_path / "existing.json"
    output.write_text("DO NOT TOUCH")
    with pytest.raises(ValueError, match="NEW private JSON"):
        _refresh(db, baseline, output, 10)
    assert output.read_text() == "DO NOT TOUCH"


def test_old_cutoff_rejected_before_persisting(tmp_path: Path) -> None:
    db, baseline = _baseline(tmp_path)
    output = tmp_path / "bad-cutoff.json"
    with pytest.raises(ValueError, match="strictly later"):
        _refresh(db, baseline, output, 1)
    assert not output.exists()


def test_source_db_capture_corruption_refused(tmp_path: Path) -> None:
    db, baseline = _baseline(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE captures SET payload=? WHERE id=1", (b"changed",))
    output = tmp_path / "corrupt.json"
    with pytest.raises(ValueError, match="checksum mismatch"):
        _refresh(db, baseline, output, 10)
    assert not output.exists()


def test_symlink_output_refused(tmp_path: Path) -> None:
    db, baseline = _baseline(tmp_path)
    target = tmp_path / "keep.json"
    target.write_text("KEEP")
    link = tmp_path / "alias.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="NEW private JSON"):
        _refresh(db, baseline, link, 10)
    assert target.read_text() == "KEEP"
