"""Real source-backed title replay without fictional article text."""

import json
from pathlib import Path

from tris_market_event_intelligence.collector import Article, NewsStore
from tris_market_event_intelligence.evaluation import replay, stored_coverage

CASES = Path(__file__).resolve().parents[1] / "benchmarks" / "2026-11-13-headlines.json"


def test_headline_replay_has_ten_source_links_and_reports_failures() -> None:
    report = replay(CASES)
    assert report["mode"] == "source_linked_headline_replay_not_live_collection"
    assert report["total"] == 10
    assert len(report["results"]) == 10
    assert 0 <= report["passed"] <= 10
    assert report["headline_recall"] == report["passed"] / 10
    assert all(
        case["extracted_categories"] is not None
        for case in report["results"]
    )


def test_sqlite_discovery_audit_does_not_confuse_replay_with_collection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "news.db"
    store = NewsStore(path)
    store.initialize()
    audit = stored_coverage(CASES, path)
    assert audit["observed"] == 0
    # A real collector source is required for this separate presence audit.
    bench = json.loads(CASES.read_text())
    sample = bench["positives"][0]
    article = Article(
        "gdelt_doc", sample["url"], sample["url"], sample["title"],
        None, None, "2026-11-14T00:00:00+00:00",
    )
    store.save("gdelt_doc", "https://api.gdeltproject.org/api/v2/doc/doc",
               b"independent-collected-fixture", [article], "now", None, None)
    audit = stored_coverage(CASES, path)
    assert audit["observed"] == 1
    assert audit["results"][0]["matched_sources"] == ["gdelt_doc"]
    assert audit["mode"] != replay(CASES)["mode"]
