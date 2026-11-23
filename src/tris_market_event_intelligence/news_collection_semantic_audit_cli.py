"""Read-only audit of news collection, verified original content and repeated Gemma reviews.

No stock prices, Yahoo, outcomes, trading, collector invocations, or model calls.
This checks mechanical integrity and consistency, not economic correctness.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.gemma_original_article_batch_cli import (
    verified_articles,
)
from tris_market_event_intelligence.news_impact_target_policy import target_key

DIRECTIONS = {"UP", "DOWN", "NEUTRAL", "UNCERTAIN"}


def collection_audit(db_path: Path) -> dict:
    if not db_path.is_file() or db_path.is_symlink():
        raise ValueError("Existing regular collection database is required")
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        sources = dict(db.execute(
            "SELECT source, COUNT(*) FROM articles GROUP BY source ORDER BY source"
        ).fetchall())
        statuses = dict(db.execute(
            "SELECT status, COUNT(*) FROM article_full_text GROUP BY status ORDER BY status"
        ).fetchall())
        duplicated = db.execute(
            "SELECT url, COUNT(*) AS copies, "
            "COUNT(DISTINCT source) AS source_count "
            "FROM articles WHERE url IS NOT NULL AND url != '' "
            "GROUP BY url HAVING COUNT(*) > 1 "
            "ORDER BY copies DESC, url LIMIT 100"
        ).fetchall()
        duplicate_urls = db.execute(
            "SELECT COUNT(*) FROM (SELECT url FROM articles "
            "WHERE url IS NOT NULL AND url != '' GROUP BY url HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        duplicate_extra_rows = db.execute(
            "SELECT COALESCE(SUM(copies - 1), 0) FROM ("
            "SELECT COUNT(*) AS copies FROM articles "
            "WHERE url IS NOT NULL AND url != '' "
            "GROUP BY url HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        without_status = db.execute(
            "SELECT a.id, a.source, a.title, a.url FROM articles a "
            "LEFT JOIN article_full_text f ON f.article_id = a.id "
            "WHERE f.article_id IS NULL ORDER BY a.id"
        ).fetchall()
        review_samples = db.execute(
            "SELECT a.id, a.source, a.title, f.status "
            "FROM article_full_text f JOIN articles a ON a.id = f.article_id "
            "WHERE f.status = 'REVIEW_REQUIRED' ORDER BY a.id LIMIT 100"
        ).fetchall()
        source_statuses = db.execute(
            "SELECT a.source, COALESCE(f.status, 'MISSING_STATUS'), COUNT(*) "
            "FROM articles a LEFT JOIN article_full_text f ON f.article_id = a.id "
            "GROUP BY a.source, COALESCE(f.status, 'MISSING_STATUS') "
            "ORDER BY a.source, COALESCE(f.status, 'MISSING_STATUS')"
        ).fetchall()
    _, valid = verified_articles(db_path, 1000)
    return {
        "totalArticles": sum(sources.values()),
        "bySource": sources,
        "fullTextStatuses": statuses,
        "duplicateUrls": duplicate_urls,
        "duplicateExtraArticleRows": duplicate_extra_rows,
        "duplicateUrlExamples": [
            {"url": url, "articleCount": count, "sourceCount": source_count}
            for url, count, source_count in duplicated
        ],
        "duplicateUrlExamplesLimit": 100,
        "articlesWithoutFullTextStatus": len(without_status),
        "missingStatusArticles": [
            {"articleId": article_id, "source": source, "title": title, "url": url}
            for article_id, source, title, url in without_status
        ],
        "reviewRequiredExamples": [
            {"articleId": article_id, "source": source, "title": title, "status": status}
            for article_id, source, title, status in review_samples
        ],
        "reviewRequiredExamplesLimit": 100,
        "fullTextStatusBySource": {
            source: {state: count for src, state, count in source_statuses if src == source}
            for source in sources
        },
        "completeBodyIntegrity": valid,
        "completeBodySampleLimit": 1000,
        "limitations": "Body integrity counts inspect all COMPLETE rows; selected sample is limited",
    }


def _load_run(folder: Path) -> dict[int, dict]:
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError(f"Invalid repeated review folder: {folder}")
    result: dict[int, dict] = {}
    for path in sorted(folder.glob("*.json")):
        if path.is_symlink():
            raise ValueError(f"Symlink review not accepted: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("schemaVersion") != "gemma_original_article_review_v2":
            raise ValueError(f"Unexpected Gemma review schema: {path}")
        article = report["article"]
        article_id = article["articleId"]
        if type(article_id) is not int or article_id < 1 or article_id in result:
            raise ValueError("Invalid or duplicate article ID in review run")
        if not isinstance(article.get("bodySha256"), str):
            raise TypeError("Missing original body hash")
        impacts = report["analysis"]["impacts"]
        if not isinstance(impacts, list):
            raise TypeError("Invalid impact list")
        for impact in impacts:
            if (not impact["target"].strip()
                    or impact["direction"] not in DIRECTIONS
                    or not impact["reason"].strip()
                    or not impact["evidence_quote"].strip()):
                raise ValueError("Invalid source-backed impact")
        result[article_id] = report
    return result


def repeatability_audit(db_path: Path, folders: list[Path]) -> dict:
    """Cross-run agreement, without pretending agreement means financial truth."""
    if len(folders) < 2 or len(set(map(str, folders))) != len(folders):
        raise ValueError("At least two distinct review directories are required")
    runs = [_load_run(folder) for folder in folders]
    source_articles, _ = verified_articles(db_path, 1000)
    source_hashes = {entry["articleId"]: entry["bodySha256"] for entry in source_articles}
    ids = sorted(set().union(*(set(run) for run in runs)))
    cases = []
    matched = 0
    consistent = 0
    same_counts = 0
    same_targets = 0
    alias_equivalent_target_directions = 0
    same_common_directions = 0
    stable_empty_articles = 0
    stable_nonempty_articles = 0
    mixed_empty_articles = 0
    both_nonempty_articles = 0
    event_kind_comparable_articles = 0
    matching_event_kind_articles = 0
    event_review_required_articles = 0
    no_shared_target_articles = 0
    changed_direction_articles = 0
    shared_target_pairs = 0
    changed_direction_pairs = 0
    target_jaccards = []
    for article_id in ids:
        present = [run.get(article_id) for run in runs]
        if any(report is None for report in present):
            cases.append({
                "articleId": article_id, "status": "MISSING_RUN",
                "runsPresent": [i + 1 for i, report in enumerate(present) if report],
            })
            continue
        assert all(report is not None for report in present)
        hashes = {report["article"]["bodySha256"] for report in present}
        models = {report["model"] for report in present}
        if len(hashes) != 1 or len(models) != 1:
            cases.append({
                "articleId": article_id, "status": "INCOMPARABLE_SOURCE_OR_MODEL",
            })
            continue
        if source_hashes.get(article_id) not in hashes:
            cases.append({
                "articleId": article_id, "status": "SOURCE_NOT_CURRENTLY_VERIFIED",
            })
            continue
        matched += 1
        signatures = []
        summaries = []
        for report in present:
            impacts = report["analysis"]["impacts"]
            signature = sorted((impact["target"].casefold().strip(), impact["direction"])
                               for impact in impacts)
            signatures.append(signature)
            summaries.append(report["analysis"]["summary"].strip())
        stable = all(signature == signatures[0] for signature in signatures[1:])
        all_empty = all(not signature for signature in signatures)
        all_nonempty = all(bool(signature) for signature in signatures)
        mixed_empty = not all_empty and not all_nonempty
        if stable and all_empty:
            stable_empty_articles += 1
        if stable and all_nonempty:
            stable_nonempty_articles += 1
        if mixed_empty:
            mixed_empty_articles += 1
        if all_nonempty:
            both_nonempty_articles += 1
        kinds = [report.get("event", {}).get("eventKind") for report in present]
        event_kind_comparable = all(isinstance(kind, str) for kind in kinds)
        if event_kind_comparable:
            event_kind_comparable_articles += 1
            if len(set(kinds)) == 1:
                matching_event_kind_articles += 1
        assessments = [
            report.get("eventAssessment", {}).get("status") for report in present
        ]
        if "REVIEW_REQUIRED" in assessments:
            event_review_required_articles += 1
        alias_signatures = [
            sorted((target_key(target), direction) for target, direction in signature)
            for signature in signatures
        ]
        alias_equivalent = all(signature == alias_signatures[0]
                               for signature in alias_signatures[1:])
        count_stable = all(len(signature) == len(signatures[0])
                           for signature in signatures[1:])
        target_sets = [{target for target, _direction in signature}
                       for signature in signatures]
        targets_stable = all(targets == target_sets[0] for targets in target_sets[1:])
        shared = set.intersection(*target_sets) if target_sets else set()
        direction_maps = [dict(signature) for signature in signatures]
        common_directions_stable = all(
            all(direction_map[target] == direction_maps[0][target]
                for target in shared)
            for direction_map in direction_maps[1:]
        )
        union = set.union(*target_sets) if target_sets else set()
        jaccard = len(shared) / len(union) if union else 1.0
        target_jaccards.append(jaccard)
        direction_changes = [
            {"target": target,
             "directionsByRun": [mapping[target] for mapping in direction_maps]}
            for target in sorted(shared)
            if len({mapping[target] for mapping in direction_maps}) > 1
        ]
        shared_target_pairs += len(shared)
        changed_direction_pairs += len(direction_changes)
        if not shared and union:
            no_shared_target_articles += 1
        if direction_changes:
            changed_direction_articles += 1
        if stable:
            consistent += 1
        if alias_equivalent:
            alias_equivalent_target_directions += 1
        if count_stable:
            same_counts += 1
        if targets_stable:
            same_targets += 1
        if common_directions_stable:
            same_common_directions += 1
        cases.append({
            "articleId": article_id,
            "status": "STABLE_TARGET_DIRECTIONS" if stable else "DIVERGENT_TARGET_DIRECTIONS",
            "allRunsEmpty": all_empty,
            "allRunsNonEmpty": all_nonempty,
            "mixedEmptyAndNonEmpty": mixed_empty,
            "eventKindsByRun": kinds,
            "eventKindsIdentical": (
                len(set(kinds)) == 1 if event_kind_comparable else None
            ),
            "eventReviewStatusesByRun": assessments,
            "impactDecisionStatusesByRun": [
                report.get("impactDecisionStatus") for report in present
            ],
            "impactCountsStable": count_stable,
            "targetSetsIdentical": targets_stable,
            "limitedAliasTargetDirectionsIdentical": alias_equivalent,
            "sharedTargetDirectionsIdentical": common_directions_stable,
            "sharedTargets": sorted(shared),
            "targetJaccard": jaccard,
            "sharedDirectionDisagreements": direction_changes,
            "targetsByRun": [sorted(targets) for targets in target_sets],
            "impactsByRun": [
                [{"target": target, "direction": direction} for target, direction in signature]
                for signature in signatures
            ],
            "summaryIdentical": len(set(summaries)) == 1,
            "analysisByRun": [
                {
                    "title": report["article"].get("title"),
                    "summary": report["analysis"]["summary"],
                    "limitations": report["analysis"]["limitations"],
                    "impacts": [
                        {
                            "target": impact["target"],
                            "direction": impact["direction"],
                            "reason": impact["reason"],
                            "evidenceId": impact.get("evidence_id"),
                            "evidenceQuote": impact["evidence_quote"],
                        }
                        for impact in report["analysis"]["impacts"]
                    ],
                }
                for report in present
            ],
        })
    return {
        "schemaVersion": "gemma_news_repeatability_audit_v1",
        "runCount": len(runs),
        "filesByRun": [len(run) for run in runs],
        "articleIdsSeen": len(ids),
        "comparableArticles": matched,
        "stableArticles": consistent,
        "agreement": consistent / matched if matched else None,
        "stableEmptyArticles": stable_empty_articles,
        "stableNonEmptyArticles": stable_nonempty_articles,
        "allRunsNonEmptyArticles": both_nonempty_articles,
        "mixedEmptyNonEmptyArticles": mixed_empty_articles,
        "comparableEventKindArticles": event_kind_comparable_articles,
        "matchingEventKindArticles": matching_event_kind_articles,
        "eventReviewRequiredArticles": event_review_required_articles,
        "sameImpactCountArticles": same_counts,
        "sameTargetSetArticles": same_targets,
        "limitedAliasTargetDirectionArticles": alias_equivalent_target_directions,
        "sameSharedTargetDirectionArticles": same_common_directions,
        "noSharedTargetArticles": no_shared_target_articles,
        "changedDirectionArticles": changed_direction_articles,
        "sharedTargetPairs": shared_target_pairs,
        "changedDirectionPairs": changed_direction_pairs,
        "meanTargetJaccard": (
            sum(target_jaccards) / len(target_jaccards) if target_jaccards else None
        ),
        "sharedTargetDirectionWarning": (
            "Common-target direction agreement is vacuously true when "
            "there are no shared target labels; interpret with sharedTargets"
        ),
        "cases": cases,
        "limitations": [
            "Agreement checks exact target name and direction, not economic reasoning quality",
            "Stable wrong answers remain wrong; reviewers must inspect explanations and source",
            "Partial, incomplete, or differing-source model runs are never scored as stable",
            "Historical outputs are retrospective and are not sealed predictions",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Audit output already exists; use a new filename")
    try:
        collection = collection_audit(args.db)
        report = {
            "schemaVersion": "news_collection_semantic_audit_v1",
            "collection": collection,
            "repeatability": repeatability_audit(args.db, args.run) if args.run else None,
            "priceDataAnalyzed": False,
            "databaseModified": False,
        }
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write("\n")
        summary = {
            "totalArticles": collection["totalArticles"],
            "fullTextStatuses": collection["fullTextStatuses"],
            "duplicateUrls": collection["duplicateUrls"],
            "duplicateExtraArticleRows": collection["duplicateExtraArticleRows"],
            "articlesWithoutFullTextStatus": collection["articlesWithoutFullTextStatus"],
            "integrity": collection["completeBodyIntegrity"],
            "repeatability": (
                {key: report["repeatability"][key]
                 for key in (
                    "articleIdsSeen", "comparableArticles", "stableArticles",
                    "agreement", "stableEmptyArticles", "stableNonEmptyArticles",
                    "allRunsNonEmptyArticles", "mixedEmptyNonEmptyArticles",
                    "comparableEventKindArticles", "matchingEventKindArticles",
                    "eventReviewRequiredArticles", "sameImpactCountArticles",
                    "sameTargetSetArticles",
                    "limitedAliasTargetDirectionArticles",
                    "noSharedTargetArticles", "changedDirectionArticles",
                    "sharedTargetPairs", "changedDirectionPairs",
                    "meanTargetJaccard",
                )}
                if report["repeatability"] else None
            ),
            "output": str(args.output),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (OSError, ValueError, TypeError, KeyError, sqlite3.DatabaseError) as exc:
        parser.exit(2, f"news-collection-semantic-audit: {exc}\n")


if __name__ == "__main__":
    main()
