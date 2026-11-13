"""Bounded live-network probe of the existing RSS and discovery pipelines.

No portal scraping, no AI, no article-content uploads. The probe report is metrics only.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import httpx

from tris_market_event_intelligence.claim_store import ClaimProcessor
from tris_market_event_intelligence.collector import NewsStore, OfficialNewsCollector, utc_now
from tris_market_event_intelligence.discovery import SearchDiscovery
from tris_market_event_intelligence.evaluation import stored_coverage
from tris_market_event_intelligence.source_catalog import DEFAULT_FEEDS
from tris_market_event_intelligence.story_index import StoryIndex

BENCHMARK = Path(__file__).resolve().parents[2] / "benchmarks/2026-11-13-headlines.json"
RSS_IDS = (
    "fed_monetary", "bls_ppi", "bls_latest",
    "fed_all", "bbc_business", "bbc_technology",
)


def portal_robots_preflight(client: httpx.Client) -> dict[str, dict[str, str | int]]:
    """Only read robots.txt; NEVER fetch article HTML in GitHub Actions."""
    from urllib.robotparser import RobotFileParser

    from tris_market_event_intelligence.web_scraper import PORTALS, USER_AGENT

    results: dict[str, dict[str, str | int]] = {}
    for name, (landing, _) in PORTALS.items():
        host = httpx.URL(landing).host
        try:
            response = client.get(
                f"https://{host}/robots.txt",
                headers={"User-Agent": USER_AGENT},
            )
            if response.status_code == 200 and len(response.content) <= 262144:
                rules = RobotFileParser()
                rules.parse(response.text.splitlines())
                results[name] = {
                    "status": "robots_checked",
                    "landing_allowed": int(rules.can_fetch(USER_AGENT, landing)),
                    "http_status": response.status_code,
                }
            elif response.status_code in {404, 410}:
                results[name] = {
                    "status": "not_declared_permission_unknown",
                    "http_status": response.status_code,
                }
            else:
                results[name] = {
                    "status": "blocked_or_unavailable",
                    "http_status": response.status_code,
                }
        except httpx.HTTPError as exc:
            results[name] = {
                "status": "network_error",
                "error_type": type(exc).__name__,
            }
    return results





def run_probe(client: httpx.Client | None = None) -> dict[str, object]:
    """Run real network checks, or injected mock transport for deterministic tests."""
    owned = client is None
    session = client or httpx.Client(timeout=12, trust_env=False, follow_redirects=False)
    try:
        with tempfile.TemporaryDirectory() as work:
            db_path = Path(work) / "live.sqlite3"
            store = NewsStore(db_path)
            feeds = {key: DEFAULT_FEEDS[key] for key in RSS_IDS}
            rss = OfficialNewsCollector(store, client=session, feeds=feeds).collect_once()
            # One bounded 1-day query avoids hammering the news search index.
            search = SearchDiscovery(
                store, client=session, gdelt_topics=("semiconductor",),
            ).collect_once(naver=False)
            claims = ClaimProcessor(db_path).process_pending()
            stories = StoryIndex(db_path).index_pending()
            with store.connect() as db:
                source_counts = [
                    {"source": name, "count": count}
                    for name, count in db.execute(
                        "SELECT source, COUNT(*) FROM articles GROUP BY source ORDER BY source"
                    )
                ]
                article_diagnostics = {
                    "empty_titles": db.execute(
                        "SELECT COUNT(*) FROM articles WHERE LENGTH(TRIM(title))=0"
                    ).fetchone()[0],
                    "with_summaries": db.execute(
                        "SELECT COUNT(*) FROM articles WHERE summary IS NOT NULL "
                        "AND LENGTH(TRIM(summary))>0"
                    ).fetchone()[0],
                    "articles_on_or_after_nov_13": db.execute(
                        "SELECT COUNT(*) FROM articles WHERE published_at >= ?",
                        ("2026-11-13T00:00:00",),
                    ).fetchone()[0],
                }
                claim_counts = [
                    {"category": name, "count": count}
                    for name, count in db.execute(
                        "SELECT classification, COUNT(*) FROM article_event_claims "
                        "GROUP BY classification ORDER BY classification"
                    )
                ]
                failures = [
                    {"source": name, "error_type": error.split(":", 1)[0]}
                    for name, error in db.execute(
                        "SELECT source, last_error FROM fetch_state "
                        "WHERE last_error IS NOT NULL ORDER BY source"
                    )
                ]
            coverage = stored_coverage(BENCHMARK, db_path)
            portal_policy = portal_robots_preflight(session)
            ok = sum(item["status"] == "ok" for item in rss.values())
            return {
                "checked_at": utc_now(),
                "mode": "LIVE_GITHUB_RUNNER_NOT_LOCAL_MAC",
                "rss": rss,
                "rss_success": ok,
                "rss_total": len(rss),
                "search": {key: asdict(value) for key, value in search.items()},
                "articles": store.stats()["articles"],
                "articles_per_source": source_counts,
                "article_diagnostics": article_diagnostics,
                "portal_robots_preflight": portal_policy,
                "claims": asdict(claims),
                "claims_per_category": claim_counts,
                "source_errors": failures,
                "story_groups": asdict(stories),
                "benchmark_observed": coverage["observed"],
                "benchmark_total": coverage["total"],
                "gate_6_ready": False,
                "collection_validation": (
                    "INSUFFICIENT_EVIDENCE"
                    if coverage["observed"] < coverage["total"]
                    else "BENCHMARK_PRESENT_NOT_AUTHENTICATED"
                ),
                "limitations": [
                    "Not the user's Mac or home internet",
                    "No portal HTML pages fetched; robots.txt only",
                    "Current feeds cannot guarantee historical Nov 13 coverage",
                    "Articles are only observed, not independently authenticated",
                ],
            }
    finally:
        if owned:
            session.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Bounded live collector verification")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_probe()
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    if report["rss_success"] == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
