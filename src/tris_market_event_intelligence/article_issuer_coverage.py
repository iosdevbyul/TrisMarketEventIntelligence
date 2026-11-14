"""Read-only article-level listed issuer coverage after an observed snapshot.

Events-only matching misses named issuers in articles without classified
economic events. This bounded diagnostic fills that visibility gap without
promoting candidate identities to approved links or historical training data.
"""
from __future__ import annotations

import sqlite3
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.event_ticker_proposals import (
    ListedIssuer,
    _matches,
    _when,
)
from tris_market_event_intelligence.story_index import canonical_url

MAX_ARTICLES = 50_000
MAX_POST_SNAPSHOT = 1_000
MAX_SAMPLES = 25


def inspect_article_issuer_coverage(
    db_path: Path, issuers: tuple[ListedIssuer, ...], *,
    observed_at: datetime, as_of: datetime,
    max_articles: int = MAX_POST_SNAPSHOT,
) -> dict[str, object]:
    """Find headline issuer mentions for post-observation RSS articles only.

    Uses a single bounded, ascending SQLite scan; older cross-source copies
    are retained as duplicate evidence, not silently discarded. Names that
    identify more than one issuer are never emitted as ticker proposals.
    """
    if db_path.is_symlink() or not db_path.is_file():
        raise ValueError("Existing non-symlink news DB required")
    if observed_at.utcoffset() is None or as_of.utcoffset() is None:
        raise ValueError("Timezone-aware observation and cutoff required")
    start = observed_at.astimezone(UTC)
    end = as_of.astimezone(UTC)
    if start > end:
        raise ValueError("Listing observation must precede report cutoff")
    if not 1 <= max_articles <= MAX_POST_SNAPSHOT:
        raise ValueError("max_articles must be 1..1000")
    with closing(sqlite3.connect(
        db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5,
    )) as db:
        db.execute("PRAGMA query_only=ON")
        count = db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        if count > MAX_ARTICLES:
            return {
                "status": "BOUNDED_SCAN_LIMIT",
                "total_articles": count,
                "max_articles": MAX_ARTICLES,
                "database_modified": False,
                "approved_links_created": 0,
            }
        # One ordered scan across article IDs lets us identify historical
        # cross-source representatives without expanding a second query.
        rows = db.execute(
            "SELECT id, source, url, title, detected_at FROM articles ORDER BY id"
        ).fetchall()

    representative: dict[tuple[str, str], int] = {}
    inspected = 0
    deferred = 0
    counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = {}
    samples: list[dict[str, object]] = []
    for article_id, source, url, title, detected_at in rows:
        try:
            key = (canonical_url(url), title.strip().casefold())
        except (ValueError, TypeError, AttributeError):
            key = (f"invalid:{article_id}", "")
        representative.setdefault(key, article_id)
        seen = _when(detected_at)
        if not start <= seen <= end:
            continue
        if inspected >= max_articles:
            deferred += 1
            continue
        inspected += 1
        duplicates = representative[key] != article_id
        status = "CROSS_SOURCE_REPEAT" if duplicates else ""
        proposals: list[dict[str, object]] = []
        ambiguous_names: set[str] = set()
        future_evidence = False
        if not duplicates:
            matches: dict[tuple[str, str], tuple[ListedIssuer, set[str]]] = {}
            alias_owners: dict[str, set[tuple[str, str]]] = {}
            for issuer in issuers:
                names = {name for name in issuer.aliases if _matches(title, name)}
                if not names:
                    continue
                if issuer.evidence_seen_at > seen:
                    future_evidence = True
                    continue
                if not (issuer.valid_from <= seen and (
                    issuer.valid_until is None or seen < issuer.valid_until
                )):
                    continue
                issuer_key = (issuer.market, issuer.symbol)
                if issuer_key not in matches:
                    matches[issuer_key] = (issuer, set())
                matches[issuer_key][1].update(names)
                for name in names:
                    alias_owners.setdefault(name.casefold(), set()).add(issuer_key)
            ambiguous_names = {
                name for name, owners in alias_owners.items() if len(owners) > 1
            }
            for (market, symbol), (issuer, matched_names) in sorted(matches.items()):
                if any(name.casefold() in ambiguous_names for name in matched_names):
                    continue
                proposals.append({
                    "market": market,
                    "symbol": symbol,
                    "matched_names": sorted(matched_names),
                    "registry_source_id": issuer.source_id,
                    "registry_evidence_seen_at": issuer.evidence_seen_at.isoformat(),
                    "article_detected_at": seen.isoformat(),
                    "relationship": "ARTICLE_NAME_CANDIDATE_ONLY",
                    "human_review_required": True,
                    "approved": False,
                })
            status = (
                "AMBIGUOUS_ISSUER_NAME" if ambiguous_names
                else "NAME_CANDIDATES" if proposals
                else "FUTURE_LISTING_EVIDENCE" if future_evidence
                else "NO_EXACT_ISSUER_NAME"
            )
        counts[status] += 1
        counts["candidate_mentions"] += len(proposals)
        per_source = source_counts.setdefault(source, Counter())
        per_source[status] += 1
        if len(samples) < MAX_SAMPLES and (
            proposals or ambiguous_names or future_evidence
        ):
            samples.append({
                "article_id": article_id,
                "source": source,
                "detected_at": seen.isoformat(),
                "headline": title[:350],
                "representative_article_id": representative[key],
                "status": status,
                "candidates": proposals,
                "ambiguous_names": sorted(ambiguous_names),
            })
    statuses = (
        "NAME_CANDIDATES", "AMBIGUOUS_ISSUER_NAME",
        "FUTURE_LISTING_EVIDENCE", "NO_EXACT_ISSUER_NAME",
        "CROSS_SOURCE_REPEAT",
    )
    return {
        "schema_version": "article_issuer_coverage_v1",
        "status": "OBSERVED_READ_ONLY",
        "listing_observed_at": start.isoformat(),
        "evaluation_cutoff": end.isoformat(),
        "post_snapshot_articles_inspected": inspected,
        "uninspected_articles_due_to_limit": deferred,
        "has_more_articles": bool(deferred),
        "status_counts": {name: counts[name] for name in statuses},
        "candidate_mentions": counts["candidate_mentions"],
        "by_source": {
            source: {name: values[name] for name in statuses}
            for source, values in sorted(source_counts.items())
        },
        "candidate_samples": samples,
        "samples_truncated": counts["candidate_mentions"] > sum(
            len(sample["candidates"]) for sample in samples
        ),
        "issuer_name_match_is_identity_proof": False,
        "listing_source_authenticated": False,
        "historical_listing_coverage_verified": False,
        "economic_event_required": False,
        "automatic_approvals": 0,
        "approved_links_created": 0,
        "training_dataset_eligible": False,
        "market_impact_predicted": False,
        "database_modified": False,
        "model_invoked": False,
        "note": (
            "Observes only post-snapshot local RSS headlines, not full article "
            "content. Does not derive shortened aliases, infer a causal issuer "
            "relationship, or override the existing event-level review queue. "
            "Human review plus independent timestamped issuer verification "
            "are required before any link becomes training input."
        ),
    }
