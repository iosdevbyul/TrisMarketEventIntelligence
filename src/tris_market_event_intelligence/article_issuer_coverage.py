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
from tris_market_event_intelligence.issuer_alias_diagnostics import derived_alias_candidates
from tris_market_event_intelligence.story_index import canonical_url

MAX_ARTICLES = 50_000
MAX_POST_SNAPSHOT = 1_000
MAX_SAMPLES = 25
MAX_ALIAS_HINTS = 25
MAX_REVIEW_ITEMS = 1000


def inspect_article_issuer_coverage(
    db_path: Path, issuers: tuple[ListedIssuer, ...], *,
    observed_at: datetime, as_of: datetime,
    max_articles: int = MAX_POST_SNAPSHOT,
    include_review_items: bool = False,
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
        columns = {row[1] for row in db.execute("PRAGMA table_info(articles)")}
        if not {"id", "source", "url", "title", "detected_at"}.issubset(columns):
            return {
                "status": "UNAVAILABLE_ARTICLE_SCHEMA",
                "missing_columns": sorted(
                    {"id", "source", "url", "title", "detected_at"} - columns
                ),
                "database_modified": False,
                "approved_links_created": 0,
                "training_dataset_eligible": False,
            }
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

    # Legal suffix variants are hypotheses, never trusted issuer aliases.
    derived_aliases = derived_alias_candidates(issuers)
    representative: dict[tuple[str, str], int] = {}
    alias_counts: Counter[str] = Counter()
    alias_hints: list[dict[str, object]] = []
    review_items: list[dict[str, object]] = []
    review_items_total = 0
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
        # Only unmatched representative headlines receive extra hints.
        # This path never promotes duplicates or conflicting exact names.
        if status == "NO_EXACT_ISSUER_NAME":
            for alias, owners in derived_aliases.items():
                if not _matches(title, alias):
                    continue
                owner_keys = {(owner.market, owner.symbol) for owner in owners}
                if len(owner_keys) > 1:
                    hint_status = "AMBIGUOUS_DERIVED_ALIAS"
                else:
                    eligible = [
                        owner for owner in owners
                        if owner.evidence_seen_at <= seen
                        and owner.valid_from <= seen
                        and (owner.valid_until is None or seen < owner.valid_until)
                    ]
                    if eligible:
                        hint_status = "NEEDS_HUMAN_ALIAS_REVIEW"
                    elif any(owner.evidence_seen_at > seen for owner in owners):
                        hint_status = "FUTURE_ALIAS_EVIDENCE"
                    else:
                        hint_status = "INACTIVE_ALIAS"
                alias_counts[hint_status] += 1
                if len(alias_hints) < MAX_ALIAS_HINTS:
                    alias_hints.append({
                        "article_id": article_id,
                        "source": source,
                        "detected_at": seen.isoformat(),
                        "candidate_alias": alias,
                        "status": hint_status,
                        "possible_issuers": (
                            [{"market": owner.market, "symbol": owner.symbol,
                              "canonical_name": owner.canonical_name}
                             for owner in owners]
                            if hint_status == "NEEDS_HUMAN_ALIAS_REVIEW" else []
                        ),
                        "human_review_required": True,
                        "approved": False,
                        "training_dataset_eligible": False,
                    })
        # The optional complete review artifact is distinct from the
        # 25-row diagnostic samples. It never has approval authority.
        # Each record carries the immutable article ID and source URL needed
        # to reconcile later decisions to original captured news.
        if include_review_items:
            for candidate in proposals:
                review_items_total += 1
                if len(review_items) < MAX_REVIEW_ITEMS:
                    review_items.append({
                        "article_id": article_id,
                        "article_source": source,
                        "article_url": url,
                        "headline": title[:350],
                        "article_detected_at": seen.isoformat(),
                        "candidate_type": "EXACT_REVIEWED_NAME",
                        "market": candidate["market"],
                        "symbol": candidate["symbol"],
                        "matched_names": candidate["matched_names"],
                        "registry_source_id": candidate["registry_source_id"],
                        "registry_evidence_seen_at": candidate[
                            "registry_evidence_seen_at"
                        ],
                        "human_review_required": True,
                        "approved": False,
                        "point_in_time_training_eligible": False,
                    })
            # Derived legal-name hints remain a separate, explicitly
            # less trusted class and are never upgraded to exact matches.
            if status == "NO_EXACT_ISSUER_NAME":
                for hint in alias_hints:
                    if hint["article_id"] != article_id or (
                        hint["status"] != "NEEDS_HUMAN_ALIAS_REVIEW"
                    ):
                        continue
                    for issuer_hint in hint["possible_issuers"]:
                        review_items_total += 1
                        if len(review_items) < MAX_REVIEW_ITEMS:
                            review_items.append({
                                "article_id": article_id,
                                "article_source": source,
                                "article_url": url,
                                "headline": title[:350],
                                "article_detected_at": seen.isoformat(),
                                "candidate_type": "DERIVED_ALIAS_REVIEW_HINT",
                                "market": issuer_hint["market"],
                                "symbol": issuer_hint["symbol"],
                                "matched_names": [hint["candidate_alias"]],
                                "registry_source_id": None,
                                "registry_evidence_seen_at": None,
                                "human_review_required": True,
                                "approved": False,
                                "point_in_time_training_eligible": False,
                            })
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
    result: dict[str, object] = {
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
        "review_items_total": review_items_total if include_review_items else None,
        "review_items_truncated": (
            review_items_total > len(review_items) if include_review_items else None
        ),
        "derived_aliases_checked": len(derived_aliases),
        "derived_alias_hint_counts": {
            name: alias_counts[name] for name in (
                "NEEDS_HUMAN_ALIAS_REVIEW", "AMBIGUOUS_DERIVED_ALIAS",
                "FUTURE_ALIAS_EVIDENCE", "INACTIVE_ALIAS",
            )
        },
        "derived_alias_hints_total": sum(alias_counts.values()),
        "derived_alias_hints": alias_hints,
        "derived_alias_hints_truncated": sum(alias_counts.values()) > len(alias_hints),
        "derived_aliases_approved": 0,
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
            "content. Legal-suffix derived aliases are diagnostic-only hints, "
            "not approved names or identity evidence. No causal issuer "
            "relationship is inferred and existing event review is unchanged. "
            "Human review plus independent timestamped issuer verification "
            "are required before any link becomes training input."
        ),
    }
    if include_review_items:
        result["review_items"] = review_items
    return result
