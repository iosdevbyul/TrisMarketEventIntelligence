"""Stage 3-B offline news/market-minute join does not fabricate labels."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tris_market_event_intelligence.news_price_coverage import inspect_news_price_coverage

CUTOFF = datetime(2026, 11, 16, 17, 0, tzinfo=UTC)
RECEIVED = "2026-11-16T15:15:45+00:00"
COLUMNS = [
    "market", "symbol", "close_at", "adjusted_close", "adjustment",
    "instrument_kind", "provider", "observed_at",
]


def make_review(tmp_path: Path, *, symbol: str = "MSFT", truncated: bool = False,
                approved: bool = False) -> Path:
    payload = {
        "schema_version": "article_issuer_review_bundle_v1",
        "status": "REVIEW_ONLY_NOT_APPROVED",
        "listing_observed_at": "2026-11-15T00:00:00+00:00",
        "evaluation_cutoff": "2026-11-16T16:00:00+00:00",
        "items_total": 1,
        "items_truncated": truncated,
        "human_review_required": True,
        "approved_ticker_links_created": 0,
        "point_in_time_training_eligible": False,
        "items": [{
            "article_id": 1,
            "market": "XNAS",
            "symbol": symbol,
            "article_detected_at": RECEIVED,
            "registry_evidence_seen_at": "2026-11-15T00:00:00+00:00",
            "candidate_type": "EXACT_REVIEWED_NAME",
            "approved": approved,
            "human_review_required": True,
            "point_in_time_training_eligible": False,
        }],
    }
    path = tmp_path / "article-review.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def make_minute_source(tmp_path: Path, *, adjusted: bool = True,
                       stock_only: bool = False, prior_only: bool = False) -> Path:
    entries = []
    for symbol, kind in (
        [("MSFT", "STOCK")] if stock_only else
        [("MSFT", "STOCK"), ("SPY", "BENCHMARK")]
    ):
        filename = f"{symbol}.csv"
        path = tmp_path / filename
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerow({
                "market": "XNAS", "symbol": symbol,
                "close_at": (
                    "2026-11-16T15:15:00+00:00" if prior_only
                    else "2026-11-16T15:16:00+00:00"
                ),
                "adjusted_close": "100",
                "adjustment": "adjusted" if adjusted else "unadjusted",
                "instrument_kind": kind, "provider": "offline-provider",
                "observed_at": "2026-11-16T15:17:00+00:00",
            })
        entries.append({
            "filename": filename,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "market": "XNAS",
            "provider": "offline-provider",
            "source_kind": kind,
        })
    path = tmp_path / "minute-manifest.json"
    path.write_text(json.dumps({
        "schema_version": "minute_source_v1", "sources": entries,
    }), encoding="utf-8")
    return path


def test_stock_and_benchmark_available_without_training_or_price_label(
    tmp_path: Path,
) -> None:
    news = make_review(tmp_path)
    minute = make_minute_source(tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    result = inspect_news_price_coverage(
        news, minute, benchmark_symbols={"XNAS": "SPY"}, as_of=CUTOFF,
    )
    assert result["status"] == "REVIEW_ONLY_PRICE_COVERAGE"
    assert result["candidates_inspected"] == 1
    assert result["status_counts"] == {
        "POST_NEWS_STOCK_AND_BENCHMARK_OBSERVED_UNVERIFIED": 1,
    }
    assert result["review_bundle_sha256"] == hashlib.sha256(
        before[news.name]
    ).hexdigest()
    assert result["minute_manifest_sha256"] == hashlib.sha256(
        before[minute.name]
    ).hexdigest()
    assert result["return_labels_generated"] == 0
    assert result["training_dataset_eligible"] is False
    assert result["issuer_identity_independently_verified"] is False
    assert result["database_modified"] is False
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize(("stock_only", "adjusted", "prior_only", "benchmark", "expected"), [
    (True, True, False, "SPY", "BENCHMARK_MINUTES_NOT_OBSERVED"),
    (False, False, False, "SPY", "ADJUSTMENT_PROVENANCE_REQUIRED"),
    (False, True, True, "SPY", "NO_POST_NEWS_MINUTE_OBSERVATIONS"),
    (False, True, False, None, "BENCHMARK_NOT_CONFIGURED"),
    (False, True, False, "MSFT", "BENCHMARK_EQUALS_STOCK"),
])
def test_missing_sources_and_adjustment_report_explicitly(
    tmp_path: Path, stock_only: bool, adjusted: bool, prior_only: bool,
    benchmark: str | None, expected: str,
) -> None:
    news = make_review(tmp_path)
    minute = make_minute_source(
        tmp_path, stock_only=stock_only, adjusted=adjusted,
        prior_only=prior_only,
    )
    result = inspect_news_price_coverage(
        news, minute, benchmark_symbols=(
            {"XNAS": benchmark} if benchmark else {}
        ), as_of=CUTOFF,
    )
    assert result["status_counts"] == {expected: 1}
    assert result["return_labels_generated"] == 0


def test_incomplete_or_approved_candidate_cannot_become_training_data(
    tmp_path: Path,
) -> None:
    minute = make_minute_source(tmp_path)
    news = make_review(tmp_path, truncated=True)
    result = inspect_news_price_coverage(
        news, minute, benchmark_symbols={"XNAS": "SPY"}, as_of=CUTOFF,
    )
    assert result["status"] == "INCOMPLETE_NEWS_REVIEW_BUNDLE"
    assert result["training_dataset_eligible"] is False
    make_review(tmp_path, approved=True)
    with pytest.raises(ValueError, match="unapproved"):
        inspect_news_price_coverage(
            news, minute, benchmark_symbols={"XNAS": "SPY"}, as_of=CUTOFF,
        )


def test_hash_tampering_and_future_registry_evidence_fail_closed(tmp_path: Path) -> None:
    news = make_review(tmp_path)
    minute = make_minute_source(tmp_path)
    (tmp_path / "MSFT.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        inspect_news_price_coverage(
            news, minute, benchmark_symbols={"XNAS": "SPY"}, as_of=CUTOFF,
        )
    minute = make_minute_source(tmp_path)
    data = json.loads(news.read_text())
    data["items"][0]["registry_evidence_seen_at"] = "2026-11-16T18:00:00+00:00"
    news.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Future or historical"):
        inspect_news_price_coverage(
            news, minute, benchmark_symbols={"XNAS": "SPY"}, as_of=CUTOFF,
        )


def test_missing_news_and_naive_clock_fail_closed(tmp_path: Path) -> None:
    minute = make_minute_source(tmp_path)
    news = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="non-symlink"):
        inspect_news_price_coverage(
            news, minute, benchmark_symbols={"XNAS": "SPY"}, as_of=CUTOFF,
        )
    assert not news.exists()
    make_review(tmp_path)
    with pytest.raises(ValueError, match="Timezone-aware"):
        inspect_news_price_coverage(
            tmp_path / "article-review.json", minute,
            benchmark_symbols={"XNAS": "SPY"},
            as_of=datetime(2026, 11, 16, 17),
        )
