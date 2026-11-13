"""Source-linked headline replay and separately measured stored-news coverage.

Headline replay tests lexical classification in isolation. It cannot certify
collector recall, historical page availability or real-world fact authenticity.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from tris_market_event_intelligence.claim_extraction import extract_claims
from tris_market_event_intelligence.story_index import canonical_url


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    expected: str
    found: bool
    extracted_categories: tuple[str, ...]
    claim_kind_ok: bool
    observed_in_database: bool | None = None


def load_benchmark(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Invalid benchmark schema")
    positives = data.get("positives")
    negatives = data.get("negatives")
    if not isinstance(positives, list) or not isinstance(negatives, list):
        raise ValueError("Benchmark requires positive and negative arrays")
    if len(positives) != 10 or len(negatives) < 3:
        raise ValueError("Benchmark requires 10 positive and at least 3 negative cases")
    ids = set()
    for item in positives:
        if not isinstance(item, dict) or not all(
            isinstance(item.get(k), str) and item[k] for k in
            ("id", "date", "title", "url", "category", "source")
        ):
            raise ValueError("Positive case missing required fields")
        if urlsplit(item["url"]).scheme != "https":
            raise ValueError("Source links must be HTTPS")
        if item["id"] in ids:
            raise ValueError("Duplicate case ID")
        ids.add(item["id"])
    return data


def replay(path: Path) -> dict[str, object]:
    benchmark = load_benchmark(path)
    found = 0
    results = []
    for case in benchmark["positives"]:
        # Use just the documented article title, never a fabricated article body.
        extracted = extract_claims(case["source"], case["title"], None)
        categories = tuple(dict.fromkeys(x.classification for x in extracted))
        matched = case["category"] in categories
        kind_ok = (
            not case.get("claim_kind")
            or any(
                x.classification == case["category"]
                and x.claim_kind == case["claim_kind"] for x in extracted
            )
        )
        found += int(matched and kind_ok)
        results.append(CaseResult(
            case["id"], case["category"], matched, categories, kind_ok,
        ).__dict__)
    false_positive = []
    neg_count = 0
    for item in benchmark["negatives"]:
        actual = extract_claims("benchmark_news", item["title"], None)
        if item.get("expected_claim_kind"):
            unexpected = any(
                claim.claim_kind != item["expected_claim_kind"]
                for claim in actual
            ) or not actual
        else:
            unexpected = bool(actual)
        neg_count += int(unexpected)
        if unexpected:
            false_positive.append(item["id"])
    return {
        "mode": "source_linked_headline_replay_not_live_collection",
        "as_of": benchmark["as_of"], "total": len(results),
        "passed": found, "headline_recall": found / len(results),
        "negative_cases": len(benchmark["negatives"]),
        "negative_errors": neg_count,
        "negative_error_ids": false_positive, "results": results,
    }


def _normal_title(title: str) -> str:
    return " ".join(re.findall(r"[a-zA-Z0-9가-힣]+", title.lower()))


def stored_coverage(path: Path, db_path: Path) -> dict[str, object]:
    """Audit actual SQLite articles without injecting any benchmark records."""
    bench = load_benchmark(path)
    if not db_path.exists():
        raise FileNotFoundError("No actual collected news SQLite database exists")
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT source, url, title FROM articles"
        ).fetchall()
    finally:
        db.close()
    observed = []
    for case in bench["positives"]:
        target_url = canonical_url(case["url"])
        target_title = _normal_title(case["title"])
        matches = []
        for source, url, title in rows:
            # Exclude test fixtures or manually inserted benchmark records.
            if source.startswith("benchmark"):
                continue
            try:
                same_url = canonical_url(url) == target_url
            except ValueError:
                same_url = False
            if same_url or _normal_title(title) == target_title:
                matches.append(source)
        observed.append({
            "id": case["id"], "found": bool(matches),
            "matched_sources": sorted(set(matches)),
        })
    count = sum(item["found"] for item in observed)
    return {
        "mode": "sqlite_collected_article_presence_not_live_network_verification",
        "total": len(observed), "observed": count,
        "article_coverage": count / len(observed),
        "database_articles": len(rows), "results": observed,
    }
