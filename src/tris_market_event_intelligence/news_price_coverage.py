"""Stage 3-B read-only join of review-only news with supplied minute source evidence.

A candidate name is NOT a point-in-time issuer link, and hash-pinned CSV
is NOT authenticated price history. This produces readiness diagnostics,
never labels, price predictions, or silently approved associations.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.event_ticker_proposals import MARKETS, SYMBOL, _when
from tris_market_event_intelligence.market_minute_source import audit_market_source

MAX_BUNDLE_BYTES = 3_000_000
MAX_NEWS_ITEMS = 1000


def _read(path: Path, *, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Existing bounded non-symlink input required")
    data = path.read_bytes()
    if len(data) > limit:
        raise ValueError("Input grew beyond size limit")
    return data


def inspect_news_price_coverage(
    review_bundle: Path, minute_manifest: Path, *,
    benchmark_symbols: dict[str, str],
    as_of: datetime,
) -> dict[str, object]:
    """Join supplied article candidates to locally supplied minute bar coverage.

    No return calculation is allowed, even when stock and benchmark exist:
    both issuer identity and independent price provenance remain unverified.
    """
    if as_of.utcoffset() is None:
        raise ValueError("Timezone-aware report cutoff required")
    cutoff = as_of.astimezone(UTC)
    if not isinstance(benchmark_symbols, dict) or any(
        market not in MARKETS or not isinstance(symbol, str)
        or SYMBOL.fullmatch(symbol) is None
        for market, symbol in benchmark_symbols.items()
    ):
        raise ValueError("Explicit supported market/benchmark mapping required")
    review_bytes = _read(review_bundle, limit=MAX_BUNDLE_BYTES)
    doc = json.loads(review_bytes)
    if not isinstance(doc, dict) or doc.get("schema_version") != (
        "article_issuer_review_bundle_v1"
    ) or doc.get("status") != "REVIEW_ONLY_NOT_APPROVED":
        raise ValueError("Unsupported article-issuer review bundle")
    if doc.get("approved_ticker_links_created") != 0 or (
        doc.get("point_in_time_training_eligible") is not False
    ) or doc.get("human_review_required") is not True:
        raise ValueError("Review bundle must explicitly prohibit automatic ticker approval")
    items = doc.get("items")
    if not isinstance(items, list) or len(items) > MAX_NEWS_ITEMS or (
        doc.get("items_truncated") is not False
        or type(doc.get("items_total")) is not int
        or doc["items_total"] != len(items)
    ):
        return {
            "schema_version": "news_minute_coverage_v1",
            "status": "INCOMPLETE_NEWS_REVIEW_BUNDLE",
            "candidate_rows_observed": len(items) if isinstance(items, list) else None,
            "training_dataset_eligible": False,
            "database_modified": False,
        }
    observed = _when(doc["listing_observed_at"])
    evaluation = _when(doc["evaluation_cutoff"])
    if observed > evaluation or evaluation > cutoff:
        raise ValueError("Review chronology exceeds report cutoff")

    # Source contract verifies each file hash, row schema and adjustment flag.
    minute_audit = audit_market_source(minute_manifest)
    manifest_bytes = _read(minute_manifest, limit=65536)
    manifest_doc = json.loads(manifest_bytes)
    entries = manifest_doc["sources"]
    series: dict[tuple[str, str], dict[str, object]] = {}
    for entry, info in zip(entries, minute_audit["sources"], strict=True):
        path = minute_manifest.parent / entry["filename"]
        raw = _read(path, limit=32 * 1024 * 1024)
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Minute source changed after audit")
        # The prior source audit validated every row. This second read only
        # indexes coverage; it never calls a market endpoint or computes labels.
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = row["market"], row["symbol"]
                item = series.setdefault(key, {
                    "kind": entry["source_kind"],
                    "source_status": info["status"],
                    "provider": entry["provider"],
                    "source_sha256": entry["sha256"],
                    "first_close_at": None,
                    "last_close_at": None,
                    "bars_observed_by_cutoff": 0,
                })
                closed = _when(row["close_at"])
                available = _when(row["observed_at"])
                if available > cutoff:
                    continue
                item["bars_observed_by_cutoff"] += 1
                item["first_close_at"] = (
                    min(item["first_close_at"], closed)
                    if item["first_close_at"] is not None else closed
                )
                item["last_close_at"] = (
                    max(item["last_close_at"], closed)
                    if item["last_close_at"] is not None else closed
                )
        if hashlib.sha256(_read(path, limit=32 * 1024 * 1024)).hexdigest() != (
            entry["sha256"]
        ):
            raise ValueError("Minute source changed during index")
    if _read(minute_manifest, limit=65536) != manifest_bytes:
        raise ValueError("Minute source manifest changed during index")
    if _read(review_bundle, limit=MAX_BUNDLE_BYTES) != review_bytes:
        raise ValueError("Review bundle changed during inspection")

    counts: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    seen_candidates: set[tuple[int, str, str]] = set()
    for item in items:
        if not isinstance(item, dict) or any(
            item.get(flag) is not False
            for flag in ("approved", "point_in_time_training_eligible")
        ) or item.get("human_review_required") is not True or (
            item.get("candidate_type") not in {
                "EXACT_REVIEWED_NAME", "DERIVED_ALIAS_REVIEW_HINT"
            }
        ):
            raise ValueError("Only unapproved explicit review candidates allowed")
        article_id, market, symbol = (
            item.get("article_id"), item.get("market"), item.get("symbol")
        )
        if type(article_id) is not int or article_id < 1 or (
            market not in MARKETS or not isinstance(symbol, str)
            or SYMBOL.fullmatch(symbol) is None
        ):
            raise ValueError("Invalid article/market/symbol candidate")
        key = (article_id, market, symbol)
        if key in seen_candidates:
            raise ValueError("Repeated article-market-symbol candidate")
        seen_candidates.add(key)
        received = _when(item["article_detected_at"])
        registry_seen = _when(item["registry_evidence_seen_at"])
        if not observed <= registry_seen <= received <= evaluation:
            raise ValueError("Future or historical listing/news evidence mismatch")
        benchmark = benchmark_symbols.get(market)
        stock = series.get((market, symbol))
        reference = series.get((market, benchmark)) if benchmark else None
        if benchmark is None:
            status = "BENCHMARK_NOT_CONFIGURED"
        elif benchmark == symbol:
            status = "BENCHMARK_EQUALS_STOCK"
        elif stock is None or stock["bars_observed_by_cutoff"] == 0:
            status = "STOCK_MINUTES_NOT_OBSERVED"
        elif reference is None or reference["bars_observed_by_cutoff"] == 0:
            status = "BENCHMARK_MINUTES_NOT_OBSERVED"
        elif stock["kind"] != "STOCK" or reference["kind"] != "BENCHMARK":
            status = "INSTRUMENT_KIND_MISMATCH"
        elif stock["source_status"] != "SCHEMA_COMPATIBLE_SOURCE_UNVERIFIED" or (
            reference["source_status"] != "SCHEMA_COMPATIBLE_SOURCE_UNVERIFIED"
        ):
            status = "ADJUSTMENT_PROVENANCE_REQUIRED"
        elif stock["last_close_at"] <= received or reference["last_close_at"] <= received:
            status = "NO_POST_NEWS_MINUTE_OBSERVATIONS"
        else:
            status = "POST_NEWS_STOCK_AND_BENCHMARK_OBSERVED_UNVERIFIED"
        counts[status] += 1
        rows.append({
            "article_id": article_id,
            "market": market,
            "symbol": symbol,
            "benchmark_symbol": benchmark,
            "candidate_type": item["candidate_type"],
            "article_received_at": received.isoformat(),
            "status": status,
            "stock_source_sha256": stock["source_sha256"] if stock else None,
            "benchmark_source_sha256": reference["source_sha256"] if reference else None,
            "stock_bars_observed_by_cutoff": (
                stock["bars_observed_by_cutoff"] if stock else 0
            ),
            "benchmark_bars_observed_by_cutoff": (
                reference["bars_observed_by_cutoff"] if reference else 0
            ),
            "issuer_identity_verified": False,
            "return_label_generated": False,
        })
    return {
        "schema_version": "news_minute_coverage_v1",
        "status": "REVIEW_ONLY_PRICE_COVERAGE",
        "review_bundle_sha256": hashlib.sha256(review_bytes).hexdigest(),
        "minute_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "evaluation_cutoff": cutoff.isoformat(),
        "candidates_inspected": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "items": rows,
        "news_source_authenticated": False,
        "issuer_identity_independently_verified": False,
        "minute_source_origin_authenticated": False,
        "adjusted_prices_independently_verified": False,
        "point_in_time_links_verified": False,
        "exchange_sessions_verified": False,
        "training_dataset_eligible": False,
        "market_impact_predicted": False,
        "return_labels_generated": 0,
        "database_modified": False,
        "network_requests": 0,
        "model_invoked": False,
        "note": (
            "This is source-coverage reconnaissance, NOT a real-time price "
            "connector, aligned minute-return calculation or authenticated "
            "issuer link. Review and independent provenance are required."
        ),
    }
