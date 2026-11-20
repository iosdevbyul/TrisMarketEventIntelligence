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

from tris_market_event_intelligence.event_evidence_policy import plain_text
from tris_market_event_intelligence.llm_runtime import LLMSettings
from tris_market_event_intelligence.news_llm_factory import build_runtime
from tris_market_event_intelligence.sequential_market_learning_cli import (
    load_articles,
    timestamp,
)
from tris_market_event_intelligence.source_grounded_semantic_analysis import (
    SEMANTIC_SCHEMA,
    SYSTEM,
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
) -> dict:
    if output.exists() or output.is_symlink():
        raise ValueError("SEMANTIC_REPORT_EXISTS")
    articles, original_hash, cohort = source_items(holdout_result)
    records = []
    for article in articles:
        source_body = plain_text(article["summary"])
        payload = {
            "source": article["source"], "title": article["title"],
            "sourceUrl": article["sourceUrl"],
            "publishedAtReported": article["publishedAtReported"],
            "sourceBody": source_body,
            "originalPITCertified": False,
        }
        analysis = None
        status, error_code = "INVALID_MODEL_ANALYSIS", "NO_VALID_RESULT"
        for attempt in range(2):
            if attempt:
                payload["correction"] = (
                    "Previous response invalid: " + error_code +
                    ". Quotes must appear verbatim within sourceBody. "
                    "Supply only facts supported by the source text."
                )
            try:
                raw = model.generate_json(
                    system=SYSTEM,
                    user=json.dumps(payload, ensure_ascii=False),
                    schema=SEMANTIC_SCHEMA,
                )
                analysis = validate_semantic_interpretation(
                    article, parse_semantic_response(raw),
                )
                status, error_code = "COMPLETE", ""
                break
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                error_code = str(exc)[:150]
        records.append({
            "articleId": article["articleId"],
            "title": article["title"], "source": article["source"],
            "sourceUrl": article["sourceUrl"],
            "publishedAtReported": article["publishedAtReported"],
            "sourceBodySha256": hashlib.sha256(source_body.encode()).hexdigest(),
            "status": status, "errorCode": error_code or None,
            "semanticAnalysis": analysis,
        })
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
        ), ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
