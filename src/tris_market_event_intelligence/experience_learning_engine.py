"""Explicit, bounded research cycle over existing commitments and observed prices."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import httpx

from tris_market_event_intelligence.automatic_forecast_precommitment import ELIGIBLE
from tris_market_event_intelligence.committed_forecast_market_outcomes_cli import (
    _source_identity,
    evaluate,
)
from tris_market_event_intelligence.cross_event_validator import (
    compare_observed_pairs,
    seal_pair,
)
from tris_market_event_intelligence.experience_memory import ExperienceMemory, digest
from tris_market_event_intelligence.learning_rule_generator import generate_rules
from tris_market_event_intelligence.self_reflection import reflect


def _write_once(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def run_cycle(
    db_path: Path, prices: Path, research_dir: Path, client: httpx.Client,
    *, model: str = "gemma4:e2b", limit: int = 10,
    manifest: Path | None = None,
) -> dict:
    if not 1 <= limit <= 100:
        raise ValueError("Expected limit 1..100")
    if research_dir.is_symlink():
        raise ValueError("Research directory must not be a symlink")
    research_dir.mkdir(parents=True, exist_ok=True)
    memory_path = research_dir / "experience_memory.sqlite3"
    if memory_path.resolve() == db_path.resolve():
        raise ValueError("Research memory must be separate from production")
    memory = ExperienceMemory(memory_path)
    report = evaluate(db_path, prices, manifest)
    if report.get("status") != "COMPLETED":
        return {"status": "WAITING_FOR_COMMITMENTS", "promotionAllowed": False}
    known = {x["outcome"]["commitmentSha256"] for x in memory.records()}
    rules_dir = research_dir / "rules"
    pairs_dir = research_dir / "pairs"
    rules_dir.mkdir(exist_ok=True)
    pairs_dir.mkdir(exist_ok=True)
    rules = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(rules_dir.glob("*.json"))]
    # Freeze candidate for the cycle; no outcome is passed to the prediction function.
    active = max(rules, key=lambda rule: rule['researchCutoff']) if rules else None
    new_experiences = sealed = 0
    failures = []
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        for row in report["rows"]:
            if new_experiences >= limit:
                break
            key = row["commitmentSha256"]
            if (key in known or row["model"] != model
                    or row["status"] != "OBSERVED_UNAUTHENTICATED_PRECOMMITMENT"
                    or row.get("verdict") not in {"CORRECT", "INCORRECT"}):
                continue
            commit = db.execute(
                "SELECT * FROM automatic_forecast_commitments WHERE forecast_sha256=?",
                (key,),
            ).fetchone()
            if commit is None or _source_identity(db, commit) is not None:
                continue
            source = db.execute(
                "SELECT body FROM article_full_text WHERE article_id=?",
                (row["articleId"],),
            ).fetchone()
            try:
                reflection = reflect(
                    client, model, row, body=source["body"],
                    rationale=commit["rationale"],
                    observed_at=commit["original_body_observed_at"],
                )
                memory.record(row, report, reflection)
                known.add(key)
                new_experiences += 1
            except (ValueError, KeyError, TypeError, httpx.HTTPError) as exc:
                failures.append({"stage": "reflection", "error": type(exc).__name__})
        if active is not None:
            commitments = db.execute(
                "SELECT * FROM automatic_forecast_commitments WHERE model=? AND eligibility=? "
                "ORDER BY rowid DESC", (model, ELIGIBLE),
            )
            for commit in commitments:
                if sealed >= limit:
                    break
                # Only current publication dates can pass seal_pair's exchange clock gate.
                try:
                    published = datetime.fromisoformat(commit["publisher_stated_at"])
                    cutoff = datetime.fromisoformat(active["researchCutoff"])
                    if published <= cutoff:
                        continue
                    if abs((datetime.now(UTC).date() - published.date()).days) > 1:
                        continue
                    if _source_identity(db, commit) is not None:
                        continue
                    event = {
                        "articleId": commit["article_id"],
                        "bodySha256": commit["body_sha256"],
                        "publisherStatedAt": commit["publisher_stated_at"],
                        "bodyObservedAt": commit["original_body_observed_at"],
                        "mappingVerifiedAt": commit["mapping_verified_at"],
                        "target": commit["target"], "symbol": commit["symbol"],
                        "benchmark": commit["benchmark"], "market": commit["market"],
                        "horizonSessions": commit["horizon_sessions"],
                    }
                    name = hashlib.sha256(json.dumps(
                        [active["ruleId"], event], sort_keys=True,
                    ).encode()).hexdigest()
                    path = pairs_dir / (name + ".json")
                    if path.exists():
                        continue
                    source = db.execute(
                        "SELECT body FROM article_full_text WHERE article_id=?",
                        (event["articleId"],),
                    ).fetchone()
                    seal_pair(client, model, active, event, source["body"], path)
                    sealed += 1
                except (ValueError, KeyError, TypeError, httpx.HTTPError) as exc:
                    failures.append({"stage": "pair", "error": type(exc).__name__})
    pairs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(pairs_dir.glob("*.json"))]
    comparison = compare_observed_pairs(pairs, report)
    evaluations = research_dir / "evaluations"
    evaluations.mkdir(exist_ok=True)
    comparison_key = digest(comparison)
    evaluation_path = evaluations / (comparison_key + ".json")
    if not evaluation_path.exists():
        _write_once(evaluation_path, comparison)
    records = memory.records()
    prior_ids = set(active["trainingExperienceIds"]) if active else set()
    fresh_articles = {
        x["outcome"]["articleId"] for x in records
        if x["id"] not in prior_ids
        and (active is None or x["outcome"]["articleId"] not in active["trainingArticleIds"])
    }
    # A bounded research rotation, not a statistical acceptance threshold.
    # Wait for at least ten distinct observed articles per horizon represented
    # by the active rule before retiring it from new pair creation.
    active_metrics = [
        value for key, value in comparison["metrics"].items()
        if active is not None and key.startswith(active["ruleId"] + ":")
    ]
    ready = active is None or (
        len(active_metrics) == 2
        and all(m["distinctArticles"] >= 10 for m in active_metrics)
        and len(fresh_articles) >= 3
        and not any(
            pair["ruleId"] == active["ruleId"]
            and pair["pairSha256"] not in {
                row["pairSha256"] for row in comparison["rows"]
            }
            for pair in pairs
        )
    )
    generated = []
    if ready and (new_experiences or active is not None):
        # A new candidate receives prior comparison as descriptive feedback,
        # never as verified causal evidence or an approved production rule.
        try:
            generated = generate_rules(
                client, model, records, cutoff=datetime.now(UTC).isoformat(),
                prior_evaluation=(
                    {"ruleId": active["ruleId"], "metrics": active_metrics}
                    if active else None
                ),
            )
            for rule in generated:
                _write_once(rules_dir / (rule["ruleId"] + ".json"), rule)
        except (ValueError, KeyError, TypeError, httpx.HTTPError) as exc:
            failures.append({"stage": "rule_generation", "error": type(exc).__name__})
    return {
        "status": "PARTIAL" if failures else "CYCLE_COMPLETED",
        "newExperiences": new_experiences, "totalExperiences": len(memory.records()),
        "newRules": len(generated), "newPairs": sealed,
        "comparison": comparison, "failures": failures,
        "ruleRotationReady": ready, "freshTrainingArticles": len(fresh_articles),
        "evaluationArtifact": str(evaluation_path),
        "productionDatabaseModified": False, "promotionAllowed": False,
        "weightTrainingPerformed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--prices", required=True, type=Path)
    parser.add_argument("--price-manifest", type=Path)
    parser.add_argument("--research-dir", required=True, type=Path)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=900)
    args = parser.parse_args()
    if args.interval < 60:
        parser.error("interval must be at least 60 seconds")
    with httpx.Client(timeout=300, trust_env=False) as client:
        while True:
            result = run_cycle(
                args.db, args.prices, args.research_dir, client,
                model=args.model, limit=args.limit, manifest=args.price_manifest,
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if not args.watch:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
