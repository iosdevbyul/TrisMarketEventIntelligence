"""Opt-in, source-pinned live US issuer directory -> review-only research artifacts.

No model calls, LaunchAgent activation, live DB writes or historical backdating.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import httpx

from tris_market_event_intelligence.event_ticker_proposals import (
    load_reviewed_issuers,
    propose_event_tickers,
)
from tris_market_event_intelligence.listing_snapshot_import import (
    MAX_SOURCE_BYTES,
    _krx_rows,
    _us_rows,
    export_snapshot_registry,
)
from tris_market_event_intelligence.ticker_review_queue import reconcile_ticker_review_queue

US_DIRECTORY = {
    "NASDAQ_DIRECTORY": (
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "nasdaqlisted.txt",
    ),
    "NYSE_DIRECTORY": (
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        "otherlisted.txt",
    ),
}
MAX_KRX_BYTES = MAX_SOURCE_BYTES


def _existing_plain_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_KRX_BYTES:
        raise ValueError("KRX reviewed CSV must be a bounded regular non-symlink file")
    content = path.read_bytes()
    if len(content) > MAX_KRX_BYTES:
        raise ValueError("KRX reviewed CSV exceeded limit during reading")
    _krx_rows(content)
    if path.read_bytes() != content:
        raise ValueError("KRX reviewed CSV changed during processing")
    return content


def _fetch_official_directory(client: httpx.Client, kind: str) -> bytes:
    url, _ = US_DIRECTORY[kind]
    # Only these two literal official HTTPS URLs are permitted.
    with client.stream("GET", url, follow_redirects=False) as response:
        response.raise_for_status()
        if response.is_redirect:
            raise ValueError("Official directory redirects are not followed")
        length = response.headers.get("content-length")
        if length is not None and int(length) > MAX_SOURCE_BYTES:
            raise ValueError("Official directory exceeds size limit")
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes(65536):
            size += len(chunk)
            if size > MAX_SOURCE_BYTES:
                raise ValueError("Official directory exceeded streaming size limit")
            chunks.append(chunk)
    raw = b"".join(chunks)
    if not raw:
        raise ValueError("Empty official directory response")
    rows, _ = _us_rows(raw, kind)
    if not rows:
        raise ValueError("No supported common-share records in official directory")
    return raw


def _exclusive_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)


def _exclusive_json(path: Path, document: dict[str, object]) -> None:
    _exclusive_bytes(
        path, (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def acquire_and_reconcile_listings(
    output_dir: Path, *, fetch_us: bool = False,
    krx_reviewed: Path | None = None,
    news_db: Path | None = None, ledger: Path | None = None,
    limit: int = 100, client: httpx.Client | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Create a new isolated research bundle; no files are overwritten.

    Fetching is explicitly opt-in. KRX is a previously reviewed operator CSV,
    never a guessed web/API endpoint. No manufactured human decisions.
    """
    if not fetch_us and krx_reviewed is None:
        raise ValueError("Select --fetch-us and/or --krx-reviewed")
    if not 1 <= limit <= 1000:
        raise ValueError("Matching limit must be 1..1000")
    when = now or datetime.now(UTC)
    if when.utcoffset() is None:
        raise ValueError("Observation clock requires timezone")
    when = when.astimezone(UTC)
    if output_dir.is_symlink() or output_dir.exists() or (
        not output_dir.parent.is_dir() or output_dir.parent.is_symlink()
    ):
        raise ValueError("Output must be a new directory inside an existing real folder")
    if news_db is not None and (news_db.is_symlink() or not news_db.is_file()):
        raise ValueError("Optional news DB must exist and not be a symlink")
    if ledger is not None and news_db is None:
        raise ValueError("Review reconciliation requires --news-db")
    krx_data = _existing_plain_file(krx_reviewed) if krx_reviewed is not None else None

    # Obtain and validate every source before creating any research artifacts.
    source_files: list[tuple[str, str, bytes, str]] = []
    if fetch_us:
        owned = client is None
        session = client or httpx.Client(timeout=20, trust_env=False, follow_redirects=False)
        try:
            for kind, (_, filename) in US_DIRECTORY.items():
                raw = _fetch_official_directory(session, kind)
                source_files.append((kind, filename, raw, "nasdaqtrader-official-https"))
        finally:
            if owned:
                session.close()
    if krx_data is not None:
        source_files.append((
            "KRX_REVIEWED_EXPORT", "krx-reviewed.csv", krx_data,
            "operator-reviewed-krx-export",
        ))
    if len(source_files) > 3:
        raise ValueError("Too many source snapshots")

    # Single timestamp is a conservative later-than-receipt bound for all
    # snapshots, not the publisher's file creation or historical listing date.
    observed = datetime.now(UTC) if now is None else when
    observed_at = observed.astimezone(UTC).isoformat()
    output_dir.mkdir(mode=0o700)
    try:
        sources: list[dict[str, str]] = []
        for kind, filename, raw, source_id in source_files:
            _exclusive_bytes(output_dir / filename, raw)
            sources.append({
                "kind": kind, "filename": filename,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "observed_at": observed_at, "source_id": source_id,
            })
        manifest = output_dir / "manifest.json"
        _exclusive_json(manifest, {
            "schema_version": "listing_snapshot_v1", "sources": sources,
        })
        registry = output_dir / "review-required-registry.csv"
        preview = export_snapshot_registry(manifest, registry, write=True)
        issuers = load_reviewed_issuers(registry)
        report: dict[str, object] = {
            "schema_version": "listing_research_workflow_v1",
            "status": "SNAPSHOTS_CAPTURED_REVIEW_REQUIRED",
            "manifest": str(manifest),
            "registry": str(registry),
            "issuer_rows": len(issuers),
            "source_reports": preview["source_reports"],
            "observed_at": observed_at,
            "news_matching_performed": False,
            "human_review_completed": False,
            "market_listing_history_verified": False,
            "point_in_time_training_eligible": False,
            "database_modified": False,
            "launchagent_modified": False,
            "model_invoked": False,
        }
        if news_db is not None:
            matches = propose_event_tickers(news_db, issuers, limit=limit)
            proposals = output_dir / "proposals.json"
            _exclusive_json(proposals, matches)
            report.update({
                "news_matching_performed": True,
                "proposals": str(proposals),
                "events_inspected": matches["events_inspected"],
                "ticker_candidates": matches["proposals"],
            })
            if ledger is not None:
                queue = reconcile_ticker_review_queue(proposals, ledger)
                report["review_queue"] = {
                    key: queue[key] for key in ("pending", "approved_post_hoc", "rejected")
                }
        _exclusive_json(output_dir / "report.json", report)
        return report
    except Exception:
        # Only this function's unique newly-created directory may be removed.
        # No external or existing file is ever removed.
        shutil.rmtree(output_dir)
        raise
