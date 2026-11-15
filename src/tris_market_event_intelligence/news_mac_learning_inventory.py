"""Read-only private Intel Mac NEWS learning inventory.

Never emit raw news, identities, paths, review contents, or protected TEST gold.
No model run, network request, file creation, DB schema migration or price data.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from tris_market_event_intelligence.news_learning_readiness import (
    inspect_news_learning_readiness,
)

COHORT = frozenset({"historical_news_cohort_v1", "historical_news_cohort_v2"})
REVIEW = "news_human_importance_review_v1"
PLAN = "news_holdout_plan_v1"
MAX_SCAN_FILES = 256
MAX_JSON_BYTES = 8_000_000


def _inventory_files(directory: Path) -> tuple[dict[str, list[tuple[Path, bytes, dict]]], int]:
    """Inspect only direct children and one fixed research subfolder."""
    found: dict[str, list[tuple[Path, bytes, dict]]] = {
        "cohort": [], "review": [], "holdout": [],
    }
    invalid = 0
    candidates = list(directory.iterdir())
    if (directory / "research").is_dir() and not (directory / "research").is_symlink():
        candidates.extend((directory / "research").iterdir())
    if len(candidates) > MAX_SCAN_FILES:
        raise ValueError("Too many private files for bounded read-only inventory")
    for path in candidates:
        if path.suffix.lower() != ".json" or path.is_symlink() or not path.is_file():
            continue
        try:
            if not 0 < path.stat().st_size <= MAX_JSON_BYTES:
                invalid += 1
                continue
            raw = path.read_bytes()
            doc = json.loads(raw)
            if not isinstance(doc, dict):
                invalid += 1
                continue
            schema = doc.get("schemaVersion")
            category = ("cohort" if schema in COHORT else
                        "review" if schema == REVIEW else
                        "holdout" if schema == PLAN else None)
            if category is not None:
                found[category].append((path, raw, doc))
        except (OSError, UnicodeError, json.JSONDecodeError):
            invalid += 1
    return found, invalid


def _source_db_inventory(db_path: Path) -> dict[str, object]:
    if not db_path.is_file() or db_path.is_symlink():
        return {"status": "NOT_FOUND", "articleCount": None}
    # Connecting to an immutable ro handle cannot change WAL or database.
    uri = db_path.resolve().as_uri() + "?mode=ro&immutable=1"
    try:
        with closing(sqlite3.connect(uri, uri=True, timeout=2)) as conn:
            conn.execute("PRAGMA query_only=ON")
            present = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='articles'"
            ).fetchone()
            if not present:
                return {"status": "ARTICLES_TABLE_NOT_FOUND", "articleCount": None}
            count = conn.execute("SELECT count(*) FROM articles").fetchone()[0]
            return {"status": "READ_ONLY_SQLITE_SNAPSHOT", "articleCount": count}
    except sqlite3.Error:
        return {"status": "READ_ONLY_DB_ACCESS_FAILED", "articleCount": None}


def inspect_mac_news_learning(*, data_dir: Path | None = None) -> dict[str, object]:
    base = data_dir or (
        Path.home() / "Library/Application Support/TrisMarketEventIntelligence"
    )
    common = {
        "schemaVersion": "news_mac_learning_inventory_v1",
        "stockPricesUsed": False, "networkCalls": 0, "modelCalls": 0,
        "sourceDbModified": False, "researchArtifactsModified": False,
        "testGoldScored": False, "humanReviewerAuthenticated": False,
        "modelImprovementValidated": False, "promotionAllowed": False,
    }
    if base.is_symlink() or not base.is_dir():
        return {**common, "status": "PRIVATE_DATA_DIRECTORY_MISSING",
                "database": {"status": "NOT_FOUND", "articleCount": None},
                "artifactCounts": {"cohort": 0, "review": 0, "holdout": 0},
                "matchedResearchTriples": 0, "readiness": None,
                "nextAction": "VERIFY_PRIVATE_NEWS_COLLECTOR_STORAGE"}
    db = _source_db_inventory(base / "news.sqlite3")
    found, invalid = _inventory_files(base)
    counts = {category: len(records) for category, records in found.items()}
    # Cross-check only hashes, not label values or protected TEST samples.
    cohorts = {hashlib.sha256(raw).hexdigest(): path
               for path, raw, _ in found["cohort"]}
    reviews = {hashlib.sha256(raw).hexdigest(): (path, doc)
               for path, raw, doc in found["review"]}
    matched = []
    for plan_path, _, doc in found["holdout"]:
        cohort_sha = doc.get("cohortSha256")
        review_sha = doc.get("humanReviewSha256")
        if cohort_sha in cohorts and review_sha in reviews and (
            reviews[review_sha][1].get("cohortSha256") == cohort_sha
        ):
            matched.append((cohorts[cohort_sha], plan_path, reviews[review_sha][0]))
    issue_counts: Counter[str] = Counter()
    ready = None
    # Never silently pick a 'latest' plan. Valid triples are auditable,
    # but only one unique pinned set may be automatically preflighted.
    if len(matched) == 1:
        try:
            report = inspect_news_learning_readiness(
                cohort_path=matched[0][0], plan_path=matched[0][1],
                review_path=matched[0][2],
            )
            ready = {
                "status": report["status"],
                "trainingArticlesWithTimeGatedHumanLabels":
                    report["trainingArticlesWithTimeGatedHumanLabels"],
                "trainingLabelCounts": report["trainingLabelCounts"],
                "trainFoldResearchFeasible": report["trainFoldResearchFeasible"],
                "crossSplitLexicalReviewCandidates":
                    report["crossSplitLexicalReviewCandidates"],
                "blockingFindings": report["blockingFindings"],
                "humanGoldIndependentlyAdjudicated": False,
            }
        except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError):
            issue_counts["MATCHED_INPUT_VALIDATION_FAILED"] += 1
    elif len(matched) > 1:
        issue_counts["MULTIPLE_MATCHED_RESEARCH_SETS_REQUIRE_SELECTION"] += 1
    if not counts["cohort"]:
        next_action = "EXPORT_READ_ONLY_NEWS_COHORT"
    elif not counts["review"]:
        next_action = "PREPARE_AND_COMPLETE_HUMAN_NEWS_REVIEW"
    elif not counts["holdout"]:
        next_action = "CREATE_TIME_GATED_HOLDOUT_FROM_REVIEW"
    elif not matched:
        next_action = "RECONCILE_COHORT_REVIEW_HOLDOUT_SHA"
    elif len(matched) > 1:
        next_action = "SELECT_EXPLICIT_PINNED_RESEARCH_TRIPLE"
    elif ready is None:
        next_action = "REPAIR_INVALID_RESEARCH_INPUTS"
    elif not ready["trainFoldResearchFeasible"]:
        next_action = "COLLECT_INDEPENDENT_TRAIN_HUMAN_GOLD"
    elif ready["crossSplitLexicalReviewCandidates"]:
        next_action = "HUMAN_AUDIT_CROSS_SPLIT_NEAR_DUPLICATES"
    else:
        next_action = "AUDIT_SEMANTIC_LEAKAGE_AND_RUN_GUARDED_RESEARCH"
    return {
        **common,
        "status": ("PRELIMINARY_NEWS_RESEARCH_READY"
                   if ready is not None and not ready["blockingFindings"]
                   else "NEWS_RESEARCH_INPUTS_NEED_ATTENTION"),
        "database": db,
        "artifactCounts": counts,
        "boundedInvalidOrOversizedJsonFiles": invalid,
        "matchedResearchTriples": len(matched),
        "inventoryIssues": dict(sorted(issue_counts.items())),
        "readiness": ready,
        "nextAction": next_action,
        "actualMacInspected": True,
        "crossPageOrSemanticLeakageCleared": False,
    }
