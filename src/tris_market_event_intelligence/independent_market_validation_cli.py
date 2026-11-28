"""Frozen historical comparison of baseline and explicit uncertainty-aware prediction."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx

from tris_market_event_intelligence.autonomous_market_experience_cli import cohort, read_sealed
from tris_market_event_intelligence.experience_memory import digest
from tris_market_event_intelligence.fresh_forecast_learning_cli import load_forecasts, publication_rows
from tris_market_event_intelligence.fresh_market_forecast_cli import SYSTEM, validate
from tris_market_event_intelligence.gemma_original_article_batch_cli import MODEL_PATTERN
from tris_market_event_intelligence.historical_experience_abc_v2_cli import (
    _ask,
    _sealed,
    load_experiences,
    predict,
)
from tris_market_event_intelligence.prediction_failure_analysis_cli import LABELS, _matrix

POLICY = """Read the source as of publication. First identify the country, issuer,
economic fact and comparison period. Do not confuse a smaller increase with a
decrease, a trade share with an exchange rate, or a past observation with a
forecast. Select only an instrument with a concrete economic transmission
mechanism. Distinguish news relevance, uncertainty and expected price direction.
NEUTRAL means an affirmative expectation of a small price move, not merely
missing information or competing explanations. If direction is indeterminate,
emit UNCERTAIN; do not turn uncertainty into NEUTRAL. If the target is unrelated,
omit it. Do not infer an economic cause from subsequent prices. State the
dominant mechanism and counterargument using the supplied source. Predict the
net direction over the next few sessions, not just whether the news sounds good.
Do not force UP or DOWN."""


def improved_predict(client: httpx.Client, model: str, packet: dict) -> list[dict]:
    options = packet["instruments"]
    passages = packet["sourcePassages"]
    schema = {
        "type": "object", "additionalProperties": False, "required": ["forecasts"],
        "properties": {"forecasts": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["instrument_id", "direction", "evidence_id", "mechanism", "counterargument"],
            "properties": {
                "instrument_id": {"type": "string", "enum": [x["id"] for x in options]},
                "direction": {"type": "string", "enum": [*LABELS, "UNCERTAIN"]},
                "evidence_id": {"type": "string", "enum": [x["id"] for x in passages]},
                "mechanism": {"type": "string"}, "counterargument": {"type": "string"},
            },
        }}},
    }
    error = ""
    for attempt in range(2):
        try:
            answer = _ask(client, model, SYSTEM + "\n" + POLICY, {
                **packet, "previousValidationError": error,
            }, schema, seed=20261128 + attempt)
            # Reuse the existing instrument and evidence validator, preserving abstentions.
            if set(answer) != {"forecasts"} or not isinstance(answer["forecasts"], list):
                raise ValueError("Invalid improved forecast envelope")
            uncertain = {
                r["instrument_id"] for r in answer["forecasts"] if r["direction"] == "UNCERTAIN"
            }
            normalized = {"forecasts": [
                {**r, "direction": "NEUTRAL"} if r["direction"] == "UNCERTAIN" else r
                for r in answer["forecasts"]
            ]}
            result = validate(normalized, options, passages)
            for row in result:
                if row["instrument"]["id"] in uncertain:
                    row["direction"] = "UNCERTAIN"
            return result
        except (ValueError, KeyError, TypeError) as exc:
            error = str(exc)[:300]
    raise ValueError(error)


def score(artifacts: list[dict], prices: Path) -> dict:
    records = []
    for artifact in artifacts:
        identity = artifact["identity"]
        for instrument in identity["packet"]["instruments"]:
            records.append({
                "id": digest([identity["articleId"], instrument["id"]]),
                "identity": identity,
                "recordSha256": artifact["recordSha256"],
                "generatedAt": artifact["generatedAt"],
                "forecasts": [{
                    "instrument": instrument, "direction": "NEUTRAL",
                    "mechanism": "", "counterargument": "", "evidenceQuote": "",
                }],
            })
    outcomes = publication_rows(records, prices)
    lookup = {
        (a["identity"]["articleId"], f["instrument"]["symbol"], arm): f["direction"]
        for a in artifacts for arm, forecasts in a["arms"].items() for f in forecasts
    }
    result = {}
    for horizon in (1, 5):
        for actual in ("actualDirection", "relativeActualDirection"):
            eligible = [
                r for r in outcomes if r["horizonSessions"] == horizon and r[actual] in LABELS
            ]
            union = [r for r in eligible if any(
                (r["articleId"], r["symbol"], arm) in lookup for arm in ("A", "D")
            )]
            paired = [r for r in union if all(
                lookup.get((r["articleId"], r["symbol"], arm)) in LABELS for arm in ("A", "D")
            )]
            for arm in ("A", "D"):
                for scope, population in (
                    ("universe", eligible), ("union", union), ("pairedDirectional", paired)
                ):
                    selected = [
                        {**r, "prediction": lookup.get((r["articleId"], r["symbol"], arm),
                                                       "NOT_SELECTED")}
                        for r in population
                    ]
                    scored = [r for r in selected if r["prediction"] in LABELS]
                    counts = Counter(r["prediction"] for r in selected)
                    metrics = _matrix(scored, actual_key=actual)
                    result[f"{horizon}d:{actual}:{scope}:{arm}"] = {
                        **metrics, "population": len(population),
                        "decisionCounts": dict(counts),
                        "coverage": len(scored) / len(population) if population else None,
                        "allPopulationCorrectRate": (
                            metrics["correct"] / len(population) if population else None
                        ),
                        "populationBaselines": {
                            label: sum(r[actual] == label for r in population) / len(population)
                            if population else None for label in LABELS
                        },
                    }
    return {"metrics": result, "outcomes": outcomes}


def run(args: argparse.Namespace, client: httpx.Client) -> dict:
    if not 1 <= args.limit <= 30 or not MODEL_PATTERN.fullmatch(args.model):
        raise ValueError("Expected 1..30 independent validation articles")
    if not args.exclude_run:
        raise ValueError("Development runs must be excluded")
    out = args.output_dir.resolve()
    protected = [args.db.parent, args.forecasts, args.experiences, *args.exclude_run]
    if args.output_dir.is_symlink() or any(out == p.resolve() for p in protected):
        raise ValueError("Separate validation output required")
    if any(out.is_relative_to(p.resolve()) for p in protected[1:]):
        raise ValueError("Cannot write inside research inputs")
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "model": args.model, "limit": args.limit,
        "policyDigest": digest({"system": SYSTEM, "improved": POLICY}),
        "pricesSha256": hashlib.sha256(args.prices.read_bytes()).hexdigest(),
        "mappingsSha256": hashlib.sha256(args.mappings.read_bytes()).hexdigest(),
        "excludedRuns": [digest(read_sealed(p / "cohort.json")) for p in args.exclude_run],
    }
    originals = load_forecasts(args.forecasts)
    memories = load_experiences(args.experiences)
    config["originalDigest"] = digest(originals)
    config["seedDigest"] = digest(memories)
    manifest = out / "cohort.json"
    if manifest.exists():
        plan = read_sealed(manifest)
        if plan["config"] != config:
            raise ValueError("Frozen validation configuration changed")
    else:
        plan = _sealed(manifest, {
            "config": config, "identities": cohort(args, originals, memories),
            "historicalResearchOnly": True,
        })
    if args.preflight or not plan["identities"]:
        return {"status": "PREFLIGHT_ONLY", "articles": len(plan["identities"])}
    artifacts = []
    for identity in plan["identities"]:
        path = out / (digest(identity) + ".validation.json")
        if path.exists():
            artifact = read_sealed(path)
            if artifact["identity"] != identity or artifact["policyDigest"] != config["policyDigest"]:
                raise ValueError("Validation checkpoint mismatch")
        else:
            packet = identity["packet"]
            artifact = _sealed(path, {
                "identity": identity, "policyDigest": config["policyDigest"],
                "arms": {
                    "A": predict(client, args.model, packet),
                    "D": improved_predict(client, args.model, packet),
                },
                "generatedAt": datetime.now(UTC).isoformat(),
                "historicalResearchOnly": True, "promotionAllowed": False,
            })
        artifacts.append(artifact)
        print(json.dumps({"event": "VALIDATION_ARTICLE", "completed": len(artifacts),
                          "articleId": identity["articleId"]}), flush=True)
    result = {
        "status": "INDEPENDENT_HISTORICAL_COMPARISON",
        "articles": len(artifacts), "requestedArticles": args.limit,
        "cohortSha256": plan["recordSha256"],
        "historicalResearchOnly": True, "promotionAllowed": False,
        "prospectiveAccuracy": None,
        "limitations": [
            "Historical outcomes already existed; model pretraining leakage is not excluded.",
            "Date-only windows include possible pre-release movement.",
            "Lexical story deduplication is not exhaustive semantic deduplication.",
            "Unrelated targets are not independently annotated; universe rates are descriptive.",
        ],
        **score(artifacts, args.prices),
    }
    _sealed(out / "report.json", result)
    return {"status": result["status"], "articles": len(artifacts),
            "report": str(out / "report.json"), "promotionAllowed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("db", "mappings", "prices", "forecasts", "experiences", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--exclude-run", type=Path, action="append", required=True)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--preflight", action="store_true")
    with httpx.Client(timeout=180, trust_env=False) as client:
        print(json.dumps(run(parser.parse_args(), client), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
