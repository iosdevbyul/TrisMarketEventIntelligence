"""Independent, human-labelled news-event quality evaluation on *collected* SQLite articles.

This never infers ground truth from the extractor. Export files are local
review artifacts and must not be uploaded to public CI.
"""
from __future__ import annotations

import csv
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

from tris_market_event_intelligence.claim_extraction import extract_claims
from tris_market_event_intelligence.multi_events import ADDITIONAL_PATTERNS, PATTERNS

CATEGORIES = frozenset(
    {category for category, _, _ in PATTERNS + ADDITIONAL_PATTERNS}
    | {"MONETARY_POLICY"}
)
FIELDS = (
    "article_id", "source", "url", "title", "capture_sha256",
    "predicted_categories", "gold_categories",
)
MIN_ACCEPTANCE_REVIEWS = 100


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    if not db_path.is_file():
        raise FileNotFoundError("No collected-news SQLite database")
    return sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)


def create_review_sheet(
    db_path: Path, output: Path, *, count: int = 100, seed: int = 20261113,
) -> dict[str, object]:
    """Stratified deterministic sampling; no benchmark rows or synthetic positives."""
    if not 1 <= count <= 1000:
        raise ValueError("Sample count must be 1..1000")
    with _connect_readonly(db_path) as db:
        rows = db.execute(
            """SELECT a.id, a.source, a.url, a.title, a.summary, c.sha256
            FROM articles a JOIN captures c ON c.id=a.capture_id
            WHERE a.source NOT LIKE 'benchmark%'
            ORDER BY a.source, a.id"""
        ).fetchall()
    groups: dict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        groups[row[1]].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected: list[tuple] = []
    names = sorted(groups)
    while len(selected) < count and names:
        for source in list(names):
            if len(selected) >= count:
                break
            if groups[source]:
                selected.append(groups[source].pop())
            if not groups[source]:
                names.remove(source)
    if not selected:
        raise ValueError("No collected articles available for evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for article_id, source, url, title, summary, digest in selected:
            predicted = {c.classification for c in extract_claims(source, title, summary)}
            writer.writerow({
                "article_id": article_id, "source": source, "url": url,
                "title": title, "capture_sha256": digest,
                "predicted_categories": "|".join(sorted(predicted)) or "NONE",
                "gold_categories": "",
            })
    return {
        "mode": "UNLABELLED_LOCAL_REVIEW_QUEUE",
        "available_articles": len(rows),
        "exported_articles": len(selected),
        "reviewed_articles": 0,
        "quality_verified": False,
    }


def _labels(value: str) -> set[str]:
    if not value or not value.strip():
        raise ValueError("Unlabelled gold_categories; assign NONE or category names")
    result = set(x.strip() for x in value.split("|"))
    if "NONE" in result and len(result) != 1:
        raise ValueError("NONE cannot be combined with event categories")
    if not result or not result.issubset(CATEGORIES | {"NONE"}):
        raise ValueError("Unrecognized category or empty label")
    return set() if result == {"NONE"} else result


def score_review_sheet(
    db_path: Path, sheet: Path, *, min_reviews: int = MIN_ACCEPTANCE_REVIEWS,
) -> dict[str, object]:
    """Compute article-level micro precision/recall against independent gold labels."""
    if min_reviews < 1:
        raise ValueError("min_reviews must be positive")
    with sheet.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Invalid review sheet columns")
        entries = list(reader)
    if not entries:
        raise ValueError("Review sheet is empty")
    ids = []
    for row in entries:
        try:
            identifier = int(row["article_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid article_id in review sheet") from exc
        if identifier < 1 or identifier in ids:
            raise ValueError("Duplicate or invalid article_id in review sheet")
        ids.append(identifier)
    with _connect_readonly(db_path) as db:
        placeholders = ",".join("?" for _ in ids)
        records = db.execute(
            f"""SELECT a.id, a.source, a.url, a.title, a.summary, c.sha256
            FROM articles a JOIN captures c ON c.id=a.capture_id
            WHERE a.id IN ({placeholders})""", ids,
        ).fetchall()
    by_id = {record[0]: record for record in records}
    if len(by_id) != len(entries):
        raise ValueError("Review sheet references missing article IDs")
    tp = fp = fn = tn = 0
    discrepancies = []
    by_source: dict[str, dict[str, int]] = defaultdict(
        lambda: {"reviewed": 0, "tp": 0, "fp": 0, "fn": 0}
    )
    for row, identifier in zip(entries, ids, strict=True):
        article_id, source, url, title, summary, digest = by_id[identifier]
        if any((
            row["source"] != source,
            row["url"] != url,
            row["title"] != title,
            row["capture_sha256"] != digest,
        )):
            raise ValueError("Source article changed since review export")
        gold = _labels(row["gold_categories"])
        predicted = {c.classification for c in extract_claims(source, title, summary)}
        # Guard against accidentally reporting a stale prefilled prediction.
        if _labels(row["predicted_categories"]) != predicted:
            raise ValueError("Extraction code changed; export a new review sheet")
        matched = len(gold & predicted)
        extra = len(predicted - gold)
        missed = len(gold - predicted)
        tp += matched
        fp += extra
        fn += missed
        tn += int(not gold and not predicted)
        by_source[source]["reviewed"] += 1
        by_source[source]["tp"] += matched
        by_source[source]["fp"] += extra
        by_source[source]["fn"] += missed
        if extra or missed:
            # No article text or copyrighted titles in summary output.
            discrepancies.append({
                "article_id": article_id,
                "missing_categories": sorted(gold - predicted),
                "extra_categories": sorted(predicted - gold),
            })
    reviewed = len(entries)
    return {
        "mode": "INDEPENDENT_COLLECTED_ARTICLE_HUMAN_GOLD",
        "reviewed_articles": reviewed,
        "required_reviews": min_reviews,
        "review_sample_requirement_met": reviewed >= min_reviews,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative_articles": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "by_source": dict(sorted(by_source.items())),
        "discrepancies": discrepancies,
        "quality_verified": reviewed >= min_reviews and (tp + fn) > 0,
        "gate_6_ready": False,
    }
