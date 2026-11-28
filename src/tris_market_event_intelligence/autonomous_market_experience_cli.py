"""Bounded walk-forward historical research with resumable per-event records."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from bisect import bisect_left, bisect_right
from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from tris_market_event_intelligence.experience_memory import digest
from tris_market_event_intelligence.experience_source_evidence import story_overlap
from tris_market_event_intelligence.fresh_forecast_learning_cli import (
    load_forecasts,
    publication_rows,
    reflect,
)
from tris_market_event_intelligence.fresh_market_forecast_cli import universe
from tris_market_event_intelligence.gemma_original_article_batch_cli import (
    MODEL_PATTERN,
    evidence_passages,
    verified_articles,
)
from tris_market_event_intelligence.historical_experience_abc_v2_cli import (
    _date,
    _sealed,
    compare,
    generate_rule_v2,
    load_experiences,
    predict,
    retrieve,
)
from tris_market_event_intelligence.market_outcome_validation_cli import ZONES, _mappings, _prices
from tris_market_event_intelligence.meaning_first_resolution import substantive
from tris_market_event_intelligence.rule_critic import critique_and_revise


def read_sealed(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("recordSha256") != digest({
        k: v for k, v in record.items() if k != "recordSha256"
    }):
        raise ValueError(f"Invalid research checkpoint: {path.name}")
    return record


def eligible_memory(memories: list[dict], packet: dict) -> list[dict]:
    day = _date(packet["publishedAt"]).date()
    return [
        x for x in memories
        if x["outcome"].get("exitDay")
        and _date(x["outcome"]["publishedAt"]) < _date(packet["publishedAt"])
        and datetime.fromisoformat(x["outcome"]["exitDay"]).date() < day
    ]


def dated_articles(articles: list[dict]) -> list[dict]:
    """Reject absent, malformed, naive or inconsistent timestamps before sorting."""
    valid = []
    for article in articles:
        published = article.get("publishedAt")
        detected = article.get("detectedAt")
        if not isinstance(published, str) or not isinstance(detected, str):
            continue
        try:
            if _date(published) <= _date(detected):
                valid.append(article)
        except ValueError:
            continue
    return sorted(valid, key=lambda a: (_date(a["publishedAt"]), a["articleId"]))


def covered_instruments(packet: dict, calendars: dict[str, list[date]]) -> list[dict]:
    """Only calendar availability enters selection, never prices or return labels."""
    published = _date(packet["publishedAt"])
    covered = []
    for instrument in packet["instruments"]:
        day = published.astimezone(ZoneInfo(ZONES[instrument["market"]])).date()
        stock = calendars.get(instrument["symbol"], [])
        market = calendars.get(instrument["benchmark"], [])
        left, right = bisect_left(stock, day), bisect_right(stock, day)
        if not left or right + 4 >= len(stock):
            continue
        entry, first, fifth = stock[left - 1], stock[right], stock[right + 4]
        if (day - entry).days > 7 or (first - day).days > 7 or (fifth - day).days > 19:
            continue
        if any(bisect_left(market, d) == len(market)
               or market[bisect_left(market, d)] != d for d in (entry, first, fifth)):
            continue
        covered.append(instrument)
    return covered


def cohort(args: argparse.Namespace, originals: list[dict], memories: list[dict]) -> list[dict]:
    articles, _ = verified_articles(args.db, 10000)
    mappings = _mappings(args.mappings)
    calendars: dict[str, list[date]] = {}
    for bar in _prices(args.prices):
        calendars.setdefault(bar.symbol, []).append(bar.day)
    for days in calendars.values():
        days.sort()
        if len(set(days)) != len(days):
            raise ValueError("Duplicate price date")
    ids = {r["identity"]["articleId"] for r in originals}
    ids.update(x["outcome"]["articleId"] for x in memories)
    hashes = {r["identity"]["bodySha256"] for r in originals}
    hashes.update(x["outcome"]["bodySha256"] for x in memories)
    for previous in args.exclude_run:
        for identity in read_sealed(previous / "cohort.json")["identities"]:
            ids.add(identity["articleId"])
            hashes.add(identity["bodySha256"])
    selected = []
    with closing(sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        for article in dated_articles(articles):
            if len(selected) >= args.limit:
                break
            key, body_hash = article["articleId"], article["bodySha256"]
            if key in ids or body_hash in hashes:
                continue
            passages = [
                p for p in evidence_passages(article["body"]) if substantive(p["text"])
            ]
            if not passages:
                continue
            packet = {
                "title": article["title"], "publishedAt": article["publishedAt"],
                "sourcePassages": passages, "instruments": universe(article, mappings),
                "bodyTruncated": len(article["body"]) > 12000,
            }
            packet["instruments"] = covered_instruments(packet, calendars)
            if not packet["instruments"] or story_overlap(db, key, sorted(ids)):
                continue
            selected.append({
                "articleId": key, "bodySha256": body_hash, "model": args.model,
                "packet": packet, "version": 1,
            })
            ids.add(key)
            hashes.add(body_hash)
    return selected


def run(args: argparse.Namespace, client: httpx.Client) -> dict:
    if not 1 <= args.limit <= 100 or not MODEL_PATTERN.fullmatch(args.model):
        raise ValueError("Invalid model or limit")
    out = args.output_dir
    if out.is_symlink() or out.resolve() in {
        args.db.resolve().parent, args.forecasts.resolve(), args.experiences.resolve()
    }:
        raise ValueError("Separate research output required")
    out.mkdir(parents=True, exist_ok=True)
    originals = load_forecasts(args.forecasts)
    seed = load_experiences(args.experiences)
    config = {
        "model": args.model, "limit": args.limit, "coveragePolicy": "calendar-only-v2",
        "excludedRuns": [digest(read_sealed(p / "cohort.json")) for p in args.exclude_run],
        "pricesSha256": hashlib.sha256(args.prices.read_bytes()).hexdigest(),
        "mappingsSha256": hashlib.sha256(args.mappings.read_bytes()).hexdigest(),
        "seedDigest": digest(seed), "originalDigest": digest(originals),
    }
    manifest = out / "cohort.json"
    if manifest.exists():
        plan = read_sealed(manifest)
        if plan["config"] != config:
            raise ValueError("Research inputs changed; use a new output directory")
    else:
        plan = _sealed(manifest, {
            "config": config, "identities": cohort(args, originals, seed),
            "historicalResearchOnly": True,
        })
    print(json.dumps({
        "event": "COVERAGE_PREFLIGHT", "requested": args.limit,
        "eligibleArticles": len(plan["identities"]),
        "eligibleInstrumentOptions": sum(
            len(x["packet"]["instruments"]) for x in plan["identities"]
        ),
    }), flush=True)
    if args.preflight or not plan["identities"]:
        return {"status": "PREFLIGHT_ONLY" if plan["identities"] else "NO_PRICE_COVERAGE",
                "eligibleArticles": len(plan["identities"]), "promotionAllowed": False}
    memories = list(seed)
    artifacts = []
    rule = None
    rule_history = []
    for index, identity in enumerate(plan["identities"]):
        key = digest(identity)
        packet = identity["packet"]
        past = eligible_memory(memories, packet)
        # Candidate is fixed for ten events; only previously matured outcomes enter it.
        if index % 10 == 0:
            path = out / f"rule-{index}.json"
            if path.exists():
                record = read_sealed(path)
                if record["memoryDigest"] != digest(past):
                    raise ValueError("Rule checkpoint memory mismatch")
            else:
                candidates, errors = generate_rule_v2(
                    client, args.model, past, datetime.now(UTC).isoformat()
                )
                review = (
                    critique_and_revise(client, args.model, candidates[0], past)
                    if candidates else None
                )
                approved = next((
                    c for c in [*candidates, (review or {}).get("revisedRule")]
                    if c and c["ruleId"] == (review or {}).get("candidateForFurtherResearch")
                ), None)
                record = _sealed(path, {
                    "memoryDigest": digest(past), "rule": approved,
                    "candidateCount": len(candidates), "errors": errors, "review": review,
                })
            rule = record["rule"]
            rule_history.append(record)
        related = [
            x for x in retrieve(packet, identity["articleId"], identity["bodySha256"], past)
            if x["score"] > 0
        ]
        path = out / (key + ".abc.json")
        context = digest([past, related, rule])
        if path.exists():
            artifact = read_sealed(path)
            if artifact["contextDigest"] != context or artifact["identity"] != identity:
                raise ValueError("Forecast checkpoint context mismatch")
        else:
            arms = {
                "A": predict(client, args.model, packet),
                "B": predict(client, args.model, packet, experience=related) if related else [],
            }
            if not related:
                arms["B"] = arms["A"]
            if rule:
                arms["C"] = predict(client, args.model, packet, rule=rule)
            artifact = _sealed(path, {
                "identity": identity, "model": args.model, "arms": arms,
                "generatedAt": datetime.now(UTC).isoformat(), "contextDigest": context,
                "retrievedExperienceIds": [x["experienceId"] for x in related],
                "historicalResearchOnly": True, "promotionAllowed": False,
            })
        artifacts.append(artifact)
        # Only baseline observations enter memory, avoiding self-reinforcing selection.
        source = {
            "id": key, "identity": identity, "recordSha256": artifact["recordSha256"],
            "generatedAt": artifact["generatedAt"], "forecasts": artifact["arms"]["A"],
        }
        for row in publication_rows([source], args.prices):
            if row["horizonSessions"] != 1 or row["verdict"] not in {"CORRECT", "INCORRECT"}:
                continue
            memory_key = digest(row)
            path = out / (memory_key + ".experience.json")
            if path.exists():
                item = read_sealed(path)
                if item["outcome"] != row:
                    raise ValueError("Experience checkpoint mismatch")
            else:
                item = _sealed(path, {
                    "id": memory_key, "outcome": row,
                    "reflection": reflect(client, args.model, row, source),
                    "historicalResearchOnly": True, "promotionAllowed": False,
                    "trainingEligible": False,
                })
            memories.append(item)
        print(json.dumps({
            "event": "ARTICLE_COMPLETED", "index": index + 1,
            "articleId": identity["articleId"], "memoryCount": len(memories),
            "retrieved": len(related), "eligibleMemory": len(past), "ruleApplied": bool(rule),
        }), flush=True)
    # C is compared only within its actual active cohort, never penalized for absence.
    paired = [{**a, "arms": {k: v for k, v in a["arms"].items() if k != "C"}}
              for a in artifacts]
    report = {
        "status": "COMPLETED" if len(artifacts) == args.limit else "COHORT_EXHAUSTED",
        "articles": len(artifacts), "newExperiences": len(memories) - len(seed),
        "articlesWithRetrievedMemory": sum(bool(a["retrievedExperienceIds"]) for a in artifacts),
        "noForecastArticles": {
            arm: sum(not a["arms"][arm] for a in artifacts) for arm in ("A", "B")
        },
        "comparison": compare(paired, args.prices),
        "batches": [compare(paired[i:i + 10], args.prices)
                    for i in range(0, len(paired), 10)],
        "ruleComparison": compare([a for a in artifacts if "C" in a["arms"]], args.prices),
        "ruleHistory": rule_history,
        "historicalResearchOnly": True, "promotionAllowed": False,
        "prospectiveAccuracy": None,
        "limitation": "Retrospective memory and daily windows; not prospective or causal evidence.",
    }
    path = out / "report.json"
    _sealed(path, report)
    return {
        "status": report["status"], "articles": report["articles"],
        "newExperiences": report["newExperiences"],
        "articlesWithRetrievedMemory": report["articlesWithRetrievedMemory"],
        "noForecastArticles": report["noForecastArticles"],
        "metrics": report["comparison"]["metrics"],
        "batchMetrics": [b["metrics"] for b in report["batches"]],
        "ruleMetrics": report["ruleComparison"]["metrics"],
        "reportPath": str(path), "promotionAllowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("db", "mappings", "prices", "forecasts", "experiences", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--exclude-run", type=Path, action="append", default=[])
    parser.add_argument("--preflight", action="store_true")
    with httpx.Client(timeout=180) as client:
        print(json.dumps(run(parser.parse_args(), client), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
