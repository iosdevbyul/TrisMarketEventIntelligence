"""Read-only live-news validation against a *previously captured* listing bundle.

No network, no live DB write, no synthetic link approvals, no backdating.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.event_ticker_proposals import (
    _when,
    load_reviewed_issuers,
    propose_event_tickers,
)
from tris_market_event_intelligence.issuer_alias_diagnostics import inspect_alias_coverage
from tris_market_event_intelligence.listing_snapshot_import import (
    preview_listing_snapshot,
)
from tris_market_event_intelligence.news_pipeline_evidence import inspect_news_pipeline
from tris_market_event_intelligence.news_stage_transitions import inspect_news_stage_transitions
from tris_market_event_intelligence.ticker_review_queue import (
    reconcile_ticker_review_queue,
)

MAX_PAGES = 16
PAGE_SIZE = 500
MAX_REGISTRY_BYTES = 2_000_000


def _verified_bundle(bundle: Path) -> tuple[Path, tuple, datetime, dict[str, object]]:
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("Existing real listing research bundle required")
    manifest = bundle / "manifest.json"
    registry = bundle / "review-required-registry.csv"
    if registry.is_symlink() or not registry.is_file() or (
        registry.stat().st_size > MAX_REGISTRY_BYTES
    ):
        raise ValueError("Existing bounded non-symlink listing registry required")
    snapshot_report, expected_rows = preview_listing_snapshot(manifest)
    # A registry from another capture would otherwise silently bypass source
    # provenance checks even when the manifest and files are intact.
    with registry.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        current = list(reader)
        if reader.fieldnames is None or set(reader.fieldnames) != set(expected_rows[0]):
            raise ValueError("Registry headers disagree with original snapshot")
        if current != expected_rows:
            raise ValueError("Registry differs from manifest-pinned source files")
    issuers = load_reviewed_issuers(registry)
    times = [_when(str(s["observed_at"])) for s in snapshot_report["source_reports"]]
    first_observed = min(times)
    return registry, issuers, first_observed, snapshot_report


def validate_existing_listing_news(
    bundle: Path, news_db: Path, output_dir: Path, *,
    ledger: Path | None = None,
    max_pages: int = MAX_PAGES,
    as_of: datetime | None = None,
    post_snapshot_only: bool = False,
) -> dict[str, object]:
    """Scan existing extracted events in ascending ID order, create reports only.

    Every output page is a valid exact-bundle review artifact. If a review ledger
    is supplied, its immutable decisions are reconciled only with that page's
    SHA-256, not assumed to carry over from a previous unrelated capture.
    """
    if not 1 <= max_pages <= MAX_PAGES:
        raise ValueError("max_pages must be 1..16")
    if news_db.is_symlink() or not news_db.is_file():
        raise ValueError("Existing nonsymlink news database required")
    registry, issuers, first_seen, source = _verified_bundle(bundle)
    if output_dir.is_symlink() or output_dir.exists() or (
        output_dir.parent.is_symlink() or not output_dir.parent.is_dir()
    ):
        raise ValueError("Output must be a new folder under an existing real directory")
    cutoff = as_of or datetime.now(UTC)
    if cutoff.utcoffset() is None:
        raise ValueError("Timezone-aware evaluation cutoff required")
    cutoff = cutoff.astimezone(UTC)
    if cutoff < first_seen:
        raise ValueError("Evaluation cutoff predates listing observation")

    # Load a stable report using bounded pages. Record evidence for every
    # processed event, not just candidate rows or the latest 100 news items.
    output_dir.mkdir(mode=0o700)
    try:
        cursor = 0
        counts: Counter[str] = Counter()
        before = 0
        after = 0
        proposals_count = 0
        event_count = 0
        samples: list[dict[str, object]] = []
        pages: list[dict[str, object]] = []
        pending = approved = rejected = 0
        alias_counts: Counter[str] = Counter()
        alias_total = 0
        alias_samples: list[dict[str, object]] = []
        alias_derived_count = 0
        exhausted = False
        for page_number in range(1, max_pages + 1):
            report = propose_event_tickers(
                news_db, issuers, limit=PAGE_SIZE, as_of=cutoff,
                after_event_id=cursor,
                received_since=first_seen if post_snapshot_only else None,
            )
            items = report["items"]
            if not items:
                exhausted = True
                break
            assert isinstance(items, list)
            for item in items:
                received = _when(str(item["article_detected_at"]))
                if received < first_seen:
                    before += 1
                else:
                    after += 1
                counts[str(item["status"])] += 1
                if item["proposals"] and len(samples) < 25:
                    samples.append({
                        "event_id": item["event_id"],
                        "article_detected_at": item["article_detected_at"],
                        "candidates": [
                            {"market": c["market"], "symbol": c["symbol"]}
                            for c in item["proposals"]
                        ],
                    })
            proposals_count += int(report["proposals"])
            event_count += len(items)
            page_path = output_dir / f"proposals-page-{page_number:03}.json"
            with page_path.open("x", encoding="utf-8") as output:
                json.dump(report, output, ensure_ascii=False, indent=2)
                output.write("\n")
            # This is a separate diagnostic, never a second automatic
            # matching/approval route. Investigate names excluded from the
            # exact full-name pipeline, including older articles.
            unmatched_ids = [
                int(item["event_id"]) for item in items
                if item["status"] in (
                    "NO_VERIFIED_NAME_MATCH", "FUTURE_REGISTRY_EVIDENCE",
                )
            ]
            alias_report = inspect_alias_coverage(
                news_db, issuers, unmatched_ids,
                first_seen=first_seen, as_of=cutoff,
            )
            alias_counts.update(alias_report["counts"])
            alias_total += int(alias_report.get("hints_total", 0))
            alias_derived_count = max(
                alias_derived_count, int(alias_report.get("derived_aliases_checked", 0))
            )
            for hint in alias_report["hints"]:
                if len(alias_samples) < 25:
                    alias_samples.append(hint)
            alias_path = output_dir / f"alias-diagnostics-page-{page_number:03}.json"
            with alias_path.open("x", encoding="utf-8") as output:
                json.dump(alias_report, output, ensure_ascii=False, indent=2)
                output.write("\n")
            page_record: dict[str, object] = {
                "path": str(page_path),
                "sha256": hashlib.sha256(page_path.read_bytes()).hexdigest(),
                "events": len(items),
                "candidates": report["proposals"],
                "alias_diagnostics_path": str(alias_path),
                "alias_diagnostics_sha256": hashlib.sha256(
                    alias_path.read_bytes()
                ).hexdigest(),
            }
            if ledger is not None:
                queue = reconcile_ticker_review_queue(page_path, ledger)
                pending += int(queue["pending"])
                approved += int(queue["approved_post_hoc"])
                rejected += int(queue["rejected"])
                page_record["review_queue"] = {
                    "pending": queue["pending"],
                    "approved_post_hoc": queue["approved_post_hoc"],
                    "rejected": queue["rejected"],
                }
            else:
                pending += int(report["proposals"])
            pages.append(page_record)
            next_id = int(report["next_cursor"])
            if next_id <= cursor:
                raise ValueError("Nonadvancing event cursor")
            cursor = next_id
            if not report["has_more"]:
                exhausted = True
                break
        # Include the actual collector -> article -> extracted-event evidence.
        # Distinguish a successful RSS poll, a new article, and a new event.
        pipeline = inspect_news_pipeline(
            news_db, snapshot_observed_at=first_seen, as_of=cutoff,
        )
        transitions = inspect_news_stage_transitions(
            news_db, snapshot_observed_at=first_seen, as_of=cutoff,
        )
        summary: dict[str, object] = {
            "schema_version": "listing_news_validation_v1",
            "status": (
                "REVIEW_CANDIDATES_PRESENT" if proposals_count else
                "NO_TICKER_CANDIDATES"
            ),
            "listing_manifest": str(bundle / "manifest.json"),
            "listing_registry": str(registry),
            "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
            "listing_issuer_rows": len(issuers),
            "source_reports": source["source_reports"],
            "first_listing_observed_at": first_seen.isoformat(),
            "news_db": str(news_db),
            "evaluation_cutoff": cutoff.isoformat(),
            "events_inspected": event_count,
            "post_snapshot_only": post_snapshot_only,
            "pre_observation_news_events": before,
            "post_observation_news_events": after,
            "status_counts": dict(sorted(counts.items())),
            "candidate_count": proposals_count,
            "diagnostic_derived_aliases_checked": alias_derived_count,
            "diagnostic_alias_mentions": alias_total,
            "diagnostic_alias_status_counts": dict(sorted(alias_counts.items())),
            "diagnostic_alias_samples": alias_samples,
            "diagnostic_aliases_approved": 0,
            "pending_human_review": pending,
            "approved_post_hoc": approved,
            "rejected": rejected,
            "pages": pages,
            "has_more_events": not exhausted,
            "resume_after_event_id": cursor if not exhausted else None,
            "new_news_available": after > 0,
            "news_pipeline_evidence": pipeline,
            "news_stage_transitions": transitions,
            "candidate_samples": samples,
            "report_only": True,
            "source_origin_independently_authenticated": False,
            "historical_listing_coverage_verified": False,
            "human_review_authenticated": False,
            "point_in_time_training_eligible": False,
            "database_modified": False,
            "model_invoked": False,
            "note": (
                "Pre-observation events cannot use this snapshot as listing evidence; "
                "NO_VERIFIED_NAME_MATCH also includes title/name coverage limitations. "
                "Derived legal-name aliases are diagnostic only and require review; "
                "a matching candidate is never an automatically approved link."
            ),
        }
        with (output_dir / "summary.json").open("x", encoding="utf-8") as output:
            json.dump(summary, output, ensure_ascii=False, indent=2)
            output.write("\n")
        return summary
    except Exception:
        # Only our newly created folder is deleted on failure.
        shutil.rmtree(output_dir)
        raise
