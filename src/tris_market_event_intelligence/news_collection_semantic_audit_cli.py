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
        duplicate_urls = db.execute(
            "SELECT COUNT(*) FROM (SELECT url FROM articles "
            "WHERE url IS NOT NULL AND url != '' GROUP BY url HAVING COUNT(*) > 1)"
        ).fetchone()[0]
    _, valid = verified_articles(db_path, 1000)
    return {
        "totalArticles": sum(sources.values()),
        "bySource": sources,
        "fullTextStatuses": statuses,
        "duplicateUrls": duplicate_urls,
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
            raise ValueError("Missing original body hash")
        impacts = report["analysis"]["impacts"]
        if not isinstance(impacts, list):
            raise ValueError("Invalid impact list")
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
        if stable:
            consistent += 1
        cases.append({
            "articleId": article_id,
            "status": "STABLE_TARGET_DIRECTIONS" if stable else "DIVERGENT_TARGET_DIRECTIONS",
            "impactsByRun": [
                [{"target": target, "direction": direction} for target, direction in signature]
                for signature in signatures
            ],
            "summaryIdentical": len(set(summaries)) == 1,
        })
    return {
        "schemaVersion": "gemma_news_repeatability_audit_v1",
        "runCount": len(runs),
        "filesByRun": [len(run) for run in runs],
        "articleIdsSeen": len(ids),
        "comparableArticles": matched,
        "stableArticles": consistent,
        "agreement": consistent / matched if matched else None,
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
            "integrity": collection["completeBodyIntegrity"],
            "repeatability": (
                {key: report["repeatability"][key]
                 for key in ("articleIdsSeen", "comparableArticles", "stableArticles", "agreement")}
                if report["repeatability"] else None
            ),
            "output": str(args.output),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (OSError, ValueError, TypeError, KeyError, sqlite3.DatabaseError) as exc:
        parser.exit(2, f"news-collection-semantic-audit: {exc}\n")


if __name__ == "__main__":
    main()
