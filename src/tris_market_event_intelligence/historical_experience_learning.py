"""Historical research only. Never create or simulate a forward forecast seal."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import httpx

from tris_market_event_intelligence.automatic_gemma_market_outcomes_cli import (
    evaluate_automatic_outcomes,
)
from tris_market_event_intelligence.experience_memory import digest
from tris_market_event_intelligence.gemma_original_article_batch_cli import evidence_passages
from tris_market_event_intelligence.learning_rule_generator import generate_rules
from tris_market_event_intelligence.self_reflection import SCHEMA, SYSTEM, validate_reflection


def historical_rows(report: dict) -> list[dict]:
    if (report.get("schemaVersion") != "automatic_gemma_daily_outcomes_v1"
            or report.get("forecastSealedBeforeOutcomes") is not False
            or report.get("historicalEvaluationOnly") is not True
            or report.get("marketPricesAnalyzed") is not True):
        raise ValueError("Expected real retrospective automatic ledger evaluation")
    rows = []
    for item in report["items"]:
        if item["sourceAuditStatus"] != "STORED_ORIGINAL_SHA_MATCHED":
            continue
        for horizon in item["horizons"]:
            if (horizon["dataStatus"] != "COMPLETED"
                    or horizon["verdict"] not in {"CORRECT", "INCORRECT"}):
                continue
            row = {
                **horizon, "articleId": item["articleId"], "impactIndex": item["impactIndex"],
                "model": item["model"], "direction": item["prediction"],
                "target": item["target"], "reason": item["reason"], "evidence": item["evidence"],
                "mapping": item["mapping"], "publishedAt": item["publishedAt"],
                "processedAt": item["processedAt"], "entryDay": item["entryDay"],
                "horizonSessions": horizon["sessions"],
                "predictionSourceSha256": item["predictionSourceSha256"],
                "status": "RETROSPECTIVE_UNSEALED_OBSERVATION",
                "evaluationClass": item["evaluationClass"],
                "forecastSealedBeforeOutcome": False,
            }
            rows.append(row)
    return rows


def run(
    db_path: Path, mappings: Path, prices: Path, output: Path, client: httpx.Client,
    *, model: str, limit: int, price_source: str,
) -> dict:
    if not 1 <= limit <= 100:
        raise ValueError("Expected experience limit 1..100")
    output.mkdir(parents=True, exist_ok=True)
    report = evaluate_automatic_outcomes(
        db_path, mappings, prices, model=model, price_source=price_source,
    )
    rows = historical_rows(report)
    records = []
    failures = []
    # Alternate correct and incorrect observations to avoid a one-sided sample.
    groups = [[r for r in rows if r["verdict"] == v] for v in ("CORRECT", "INCORRECT")]
    ordered = [
        group[i] for i in range(max(map(len, groups))) for group in groups if i < len(group)
    ]
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        for row in ordered:
            if len(records) >= limit:
                break
            source = db.execute(
                """SELECT f.body, f.body_sha256, f.observed_at, n.analysis_json
                   FROM article_full_text f JOIN automatic_news_analyses n
                   ON n.article_id=f.article_id AND n.body_sha256=f.body_sha256
                   WHERE f.article_id=? AND f.status='COMPLETE'
                   AND n.model=? AND n.status='COMPLETED'""",
                (row["articleId"], model),
            ).fetchone()
            try:
                if source is None:
                    raise ValueError("Original source missing")
                body = source["body"]
                if hashlib.sha256(body.encode()).hexdigest() != source["body_sha256"]:
                    raise ValueError("Original source hash mismatch")
                impact = json.loads(source["analysis_json"])["impacts"][row["impactIndex"]]
                if (impact["direction"] != row["direction"]
                        or impact["reason"] != row["reason"]
                        or impact["target"].strip() != row["target"]
                        or impact["evidence_quote"] != row["evidence"]
                        or row["evidence"] not in body):
                    raise ValueError("Original model/source mismatch")
                row["bodySha256"] = source["body_sha256"]
                key = digest(row)
                path = output / (key + ".experience.json")
                if path.exists():
                    records.append(json.loads(path.read_text(encoding="utf-8")))
                    continue
                passages = evidence_passages(body)
                response = client.post("http://127.0.0.1:11434/api/chat", json={
                    "model": model, "stream": False, "think": False, "format": SCHEMA,
                    "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 1024},
                    "messages": [
                        {"role": "system", "content": SYSTEM + (
                            " This is historical retrospective research, not a sealed forecast."
                        )},
                        {"role": "user", "content": json.dumps({
                            "outcome": row, "originalRationale": row["reason"],
                            "sourcePassages": passages,
                        }, ensure_ascii=False)},
                    ],
                })
                response.raise_for_status()
                result = response.json()
                if result.get("done_reason") == "length":
                    raise ValueError("Truncated reflection")
                reflection = validate_reflection(
                    json.loads(result["message"]["content"]),
                    {p["id"]: p["text"] for p in passages},
                    observed_at=source["observed_at"], sealed_at=row["processedAt"],
                )
                reflection.pop("evidenceAvailableAtSeal")
                reflection["sourceAvailableAtRecordedProcessing"] = True
                record = {
                    "id": key, "outcome": row, "reflection": reflection,
                    "historicalResearchOnly": True, "trainingEligible": False,
                    "promotionAllowed": False,
                }
                with path.open("x", encoding="utf-8") as stream:
                    json.dump(record, stream, ensure_ascii=False, indent=2)
                records.append(record)
            except (ValueError, KeyError, TypeError, IndexError, httpx.HTTPError) as exc:
                failures.append({"articleId": row["articleId"], "error": str(exc)[:300]})
    rules = generate_rules(
        client, model, records, cutoff=datetime.now(UTC).isoformat(), historical=True,
    )
    for rule in rules:
        path = output / (rule["ruleId"] + ".rule.json")
        if not path.exists():
            with path.open("x", encoding="utf-8") as stream:
                json.dump(rule, stream, ensure_ascii=False, indent=2)
    return {
        "status": "PARTIAL" if failures else "COMPLETED",
        "historicalScorableRows": len(rows), "experiences": len(records),
        "rules": rules, "failures": failures, "historicalResearchOnly": True,
        "prospectivePredictionAccuracy": None, "promotionAllowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("db", "mappings", "prices", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--price-source", required=True)
    args = parser.parse_args()
    with httpx.Client(timeout=300, trust_env=False) as client:
        print(json.dumps(run(
            args.db, args.mappings, args.prices, args.output_dir, client,
            model=args.model, limit=args.limit, price_source=args.price_source,
        ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
