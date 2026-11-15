"""Explicit, one-shot NEWS snapshot refresh with immutable baseline comparison.

No collector run, no background scheduler, no model, no human gold, no prices.
Persist a private fresh cohort only if *new* news passed the entire audit.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.historical_news_cohort import historical_news_cohort
from tris_market_event_intelligence.news_observation_coverage import (
    audit_news_observation_coverage,
)

MAX_ARTICLES = 5000


def refresh_news_coverage(*, baseline_path: Path, source_db: Path,
                          output_path: Path, as_of: datetime) -> dict:
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("Timezone-aware as_of required")
    if (output_path.suffix != ".json" or output_path.exists()
            or output_path.is_symlink() or not output_path.parent.is_dir()
            or output_path.parent.is_symlink()):
        raise ValueError("Refresh output must be a NEW private JSON in an ordinary directory")
    if output_path.resolve() == baseline_path.resolve():
        raise ValueError("Must not replace immutable baseline")
    # Source DB is opened by the existing verified, read-only cohort exporter.
    report = historical_news_cohort(
        source_db=source_db, as_of=as_of, after_id=0, limit=MAX_ARTICLES,
    )
    if report["hasMore"] or len(report["items"]) != report["totalObservedByCutoff"]:
        raise ValueError("NEWS archive exceeds 5000-item complete snapshot limit")
    if not report["items"]:
        raise ValueError("NEWS archive is empty")
    # Reuse the existing pinned full-v2 snapshot validation and growth audit
    # without persisting an output that has not passed. Private, same-dir
    # 0600 temporary file is removed on any failure or zero new articles.
    fd, temporary_name = tempfile.mkstemp(
        prefix=".news-coverage-refresh-", suffix=".json", dir=output_path.parent,
    )
    temp = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        evidence = audit_news_observation_coverage(
            baseline_path=baseline_path, newer_path=temp,
        )
        growth = evidence["growth"]
        new_articles = growth["newArticles"]
        if new_articles == 0:
            return {
                "status": "NO_NEW_NEWS_SNAPSHOT_NOT_CREATED",
                "baselineArticleCount": evidence["baseline"]["articleCount"],
                "latestObservedArticleCount": evidence["newer"]["articleCount"],
                "growth": growth,
                "sourceOnlyChronologyWarnings":
                    evidence["newer"]["sourceOnlyChronologyWarnings"],
                "outputCreated": False,
                "humanGoldCreated": False,
                "modelCalls": 0,
                "networkCalls": 0,
                "stockPricesUsed": False,
                "productionDbModified": False,
                "promotionAllowed": False,
            }
        # Same-filesystem hard link creation is atomic and fails if target
        # already exists. Then unlink only the transient link. No overwrite.
        os.link(temp, output_path, follow_symlinks=False)
        return {
            "status": "NEW_NEWS_SNAPSHOT_VERIFIED_AND_CREATED",
            "baselineArticleCount": evidence["baseline"]["articleCount"],
            "latestObservedArticleCount": evidence["newer"]["articleCount"],
            "growth": growth,
            "sourceOnlyChronologyWarnings":
                evidence["newer"]["sourceOnlyChronologyWarnings"],
            "outputCreated": True,
            "output": str(output_path),
            "humanGoldCreated": False,
            "modelCalls": 0,
            "networkCalls": 0,
            "stockPricesUsed": False,
            "productionDbModified": False,
            "promotionAllowed": False,
        }
    finally:
        temp.unlink(missing_ok=True)
