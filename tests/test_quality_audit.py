"""Quality scores require independent labels, never self-referential predictions."""
import csv
from pathlib import Path

import pytest

from tris_market_event_intelligence.collector import Article, NewsStore
from tris_market_event_intelligence.quality_audit import (
    create_review_sheet,
    score_review_sheet,
)


def _db(tmp_path: Path, num: int = 120) -> Path:
    db_path = tmp_path / "articles.sqlite3"
    store = NewsStore(db_path)
    store.initialize()
    for i in range(num):
        source = "bbc_business" if i % 2 else "fed_all"
        title = (
            f"Company {i} quarterly earnings released"
            if i % 4 == 0 else f"General news update number {i}"
        )
        a = Article(
            source, f"item-{i}", f"https://example.org/{i}", title,
            "A summary with context.", "2026-11-13T00:00:00+00:00", "now",
        )
        store.save(
            source, f"https://example.org/feed/{source}",
            f"capture {i}".encode(), [a], "now", None, None,
        )
    return db_path


def _set_gold(sheet: Path, *, omit_last: bool = False) -> None:
    with sheet.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows):
        row["gold_categories"] = (
            "" if omit_last and i == len(rows) - 1 else row["predicted_categories"]
        )
    with sheet.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_audit_requires_human_gold_and_sample_threshold(tmp_path: Path) -> None:
    db_path = _db(tmp_path)
    sheet = tmp_path / "review.csv"
    result = create_review_sheet(db_path, sheet, count=100, seed=42)
    assert result["exported_articles"] == 100
    assert result["reviewed_articles"] == 0
    with pytest.raises(ValueError, match="Unlabelled"):
        score_review_sheet(db_path, sheet)
    _set_gold(sheet)
    metrics = score_review_sheet(db_path, sheet)
    assert metrics["reviewed_articles"] == 100
    assert metrics["precision"] == 1
    assert metrics["recall"] == 1
    assert metrics["quality_verified"]
    assert not metrics["gate_6_ready"]


def test_audit_catches_independent_false_positives_and_missing_events(
    tmp_path: Path,
) -> None:
    db_path = _db(tmp_path, num=10)
    sheet = tmp_path / "small.csv"
    create_review_sheet(db_path, sheet, count=10)
    _set_gold(sheet)
    with sheet.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    positive = next(row for row in rows if row["predicted_categories"] != "NONE")
    negative = next(row for row in rows if row["predicted_categories"] == "NONE")
    positive["gold_categories"] = "NONE"
    negative["gold_categories"] = "EARNINGS"
    with sheet.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    scores = score_review_sheet(db_path, sheet)
    assert scores["false_positive"] == 1
    assert scores["false_negative"] == 1
    assert not scores["quality_verified"]
    assert len(scores["discrepancies"]) == 2


def test_review_sheet_rejects_missing_and_altered_rows(tmp_path: Path) -> None:
    db_path = _db(tmp_path, num=5)
    sheet = tmp_path / "audit.csv"
    create_review_sheet(db_path, sheet, count=5)
    _set_gold(sheet, omit_last=True)
    with pytest.raises(ValueError, match="Unlabelled"):
        score_review_sheet(db_path, sheet)
    _set_gold(sheet)
    with sheet.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["title"] = "altered title"
    with sheet.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="changed"):
        score_review_sheet(db_path, sheet)
