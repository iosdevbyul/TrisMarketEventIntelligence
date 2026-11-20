"""Read-only semantic replay of the same thirty frozen historical articles.

Reads source text, never feeds market outcome labels to the LLM, and creates
an immutable per-article semantic evidence audit. No price requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from tris_market_event_intelligence.event_evidence_policy import plain_text
from tris_market_event_intelligence.llm_runtime import LLMSettings
from tris_market_event_intelligence.news_llm_factory import build_runtime
from tris_market_event_intelligence.sec_filing_exhibit_evidence import (
    configured_sec_agent,
    enrich_sec_article,
)
from tris_market_event_intelligence.semantic_source_evidence import (
    EXCERPT_SCHEMA,
    SYSTEM,
    expand_evidence_ids,
    source_excerpts,
)
from tris_market_event_intelligence.sequential_market_learning_cli import (
    load_articles,
    timestamp,
)
from tris_market_event_intelligence.source_grounded_semantic_analysis import (
    parse_semantic_response,
    semantic_summary,
    validate_semantic_interpretation,
)


def source_items(holdout_result: Path) -> tuple[list[dict], str, dict]:
    if holdout_result.is_symlink() or not holdout_result.is_file():
        raise ValueError("MISSING_OR_UNSAFE_HOLDOUT_REPORT")
    data = holdout_result.read_bytes()
    original = json.loads(data)
    if not str(original.get("schemaVersion", "")).startswith(
        "frozen_chronological_holdout_"
    ):
        raise ValueError("NOT_A_FROZEN_HOLDOUT_REPORT")
    cohort = original["cohort"]
    frozen = cohort["dateSpread"]["items"]
    prior = original["articles"]
    if len(frozen) != original["holdoutSelected"] or len(frozen) != len(prior):
        raise ValueError("FROZEN_ARTICLE_COUNT_MISMATCH")
    for item, previous in zip(frozen, prior, strict=True):
        if (item["articleId"] != previous["articleId"]
                or item["title"] != previous["title"]
                or item["source"] != previous["source"]
                or timestamp(item["publishedAtReported"])
                != timestamp(previous["releaseAt"])):
            raise ValueError("FROZEN_ARTICLE_IDENTITY_MISMATCH")
    db_path = Path(cohort["holdoutSourceDb"])
    journal = Path(cohort["trainingJournal"])
    if (db_path.is_symlink() or journal.is_symlink()
            or not db_path.is_file() or not journal.is_file()):
        raise ValueError("MISSING_OR_UNSAFE_SOURCE_DATABASE")
    with sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        setting = db.execute("SELECT settings_json FROM runs WHERE id=1").fetchone()
    if setting is None:
        raise ValueError("FROZEN_TRAINING_SETTINGS_MISSING")
    settings = json.loads(setting[0])
    if settings["sourceDb"] != str(db_path.resolve()):
        raise ValueError("FROZEN_SOURCE_DB_MISMATCH")
    cache = settings.get("sourceContentCache")
    articles, _ = load_articles(
        db_path, 5000, timestamp(cohort["inventoryAsOf"]),
        partition="all", content_cache=Path(cache) if cache else None,
    )
    by_id = {item["articleId"]: item for item in articles}
    prepared = []
    for item in frozen:
        article = by_id.get(item["articleId"])
        if article is None or (
            any(article[k] != item[k] for k in ("title", "source"))
            or article.get("sourceUrl") != item.get("sourceUrl")
            or timestamp(article["publishedAtReported"])
            != timestamp(item["publishedAtReported"])
        ):
            raise ValueError("FROZEN_SOURCE_ARTICLE_CHANGED")
        prepared.append(article)
    return prepared, hashlib.sha256(data).hexdigest(), cohort


def semantic_replay(
    *, holdout_result: Path, model: Any, output: Path,
    sec_exhibits: bool = False,
    client: httpx.Client | None = None,
) -> dict:
    if output.exists() or output.is_symlink():
        raise ValueError("SEMANTIC_REPORT_EXISTS")
    articles, original_hash, cohort = source_items(holdout_result)
    records = []
    owns_client = sec_exhibits and client is None
    http = client or (httpx.Client(timeout=20, follow_redirects=False)
                      if sec_exhibits else None)
    try:
        for article in articles:
            original_body = plain_text(article["summary"])
            analysis_article = article
            exhibit_audit = {"status": "NOT_SEC_FILING"}
            if article["source"].startswith("sec_8k_"):
                exhibit_audit = {
                    "status": "NOT_ACQUIRED",
                    "firstPublicationPITCertified": False,
                }
                if sec_exhibits and http is not None:
                    try:
                        analysis_article, exhibit_audit = enrich_sec_article(
                            article, client=http, user_agent=configured_sec_agent(),
                        )
                    except ValueError as exc:
                        exhibit_audit = {
                            "status": "UNAVAILABLE",
                            "errorCode": str(exc)[:120],
                            "firstPublicationPITCertified": False,
                        }
            source_body = plain_text(analysis_article["summary"])
            excerpts = source_excerpts(source_body)
            payload = {
                "source": article["source"], "title": article["title"],
                "sourceUrl": article["sourceUrl"],
                "publishedAtReported": article["publishedAtReported"],
                "sourceExcerpts": excerpts,
                "originalPITCertified": False,
            }
            analysis = None
            status, error_code = "INVALID_MODEL_ANALYSIS", "NO_VALID_RESULT"
            for attempt in range(2):
                if attempt:
                    payload["correction"] = (
                        "Previous response invalid: " + error_code +
                        ". Use only existing sourceExcerpts evidence IDs, "
                        "never invent a source citation or numerical fact."
                    )
                try:
                    raw = model.generate_json(
                        system=SYSTEM,
                        user=json.dumps(payload, ensure_ascii=False),
                        schema=EXCERPT_SCHEMA,
                    )
                    expanded = expand_evidence_ids(
                        parse_semantic_response(raw), excerpts,
                    )
                    analysis = validate_semantic_interpretation(
                        analysis_article, expanded,
                    )
                    status = (
                        "COMPLETE"
                        if analysis["semanticCategoryStatus"] == "MATCHES_EVENT_SOURCE_POLICY"
                        else "NEEDS_CATEGORY_REVIEW"
                    )
                    error_code = ""
                    break
                except (ValueError, TypeError, RuntimeError, httpx.HTTPError) as exc:
                    error_code = str(exc)[:150]
            records.append({
                "articleId": article["articleId"],
                "title": article["title"], "source": article["source"],
                "sourceUrl": article["sourceUrl"],
                "publishedAtReported": article["publishedAtReported"],
                "originalSourceBodySha256": hashlib.sha256(
                    original_body.encode(),
                ).hexdigest(),
                "sourceBodySha256": hashlib.sha256(source_body.encode()).hexdigest(),
                "sourceExcerptsUsed": len(excerpts),
                "officialSECExhibitEvidence": exhibit_audit,
                "status": status, "errorCode": error_code or None,
                "semanticAnalysis": analysis,
            })
    finally:
        if owns_client and http is not None:
            http.close()
    # Each output preserves the successful source analysis or the precise
    # failure code; no scored article is silently omitted.
    evaluation = semantic_summary([
        {"status": row["status"], "analysis": row["semanticAnalysis"]}
        for row in records
    ])
    result = {
        "schemaVersion": "frozen_thirty_source_grounded_semantic_v1",
        "originalHoldoutSha256": original_hash,
        "inventoryAsOf": cohort["inventoryAsOf"],
        "articleCount": len(records),
        "semanticEvidenceProtocol": "deterministic_source_excerpt_ids_v1",
        "secExhibitRetrievalAttempted": sec_exhibits,
        "priceDataAccessed": False,
        "feedbackLessonsUsed": False,
        "pointInTimeCertified": False,
        "journalModified": False,
        "summary": evaluation,
        "articles": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as dest:
        json.dump(result, dest, ensure_ascii=False, indent=2)
    return {k: v for k, v in result.items() if k != "articles"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument(
        "--sec-exhibits", action="store_true",
        help="Fetch SEC Exhibit 99.1/99.2 source text from original filing archive",
    )
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Refusing to overwrite semantic report")
    model = build_runtime(
        replace(
            LLMSettings.from_environment(explicit=True), model_id=args.model,
        ), confirm_model=True,
    )
    print(json.dumps(
        semantic_replay(
            holdout_result=args.holdout_result, model=model, output=args.output,
            sec_exhibits=args.sec_exhibits,
        ), ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
