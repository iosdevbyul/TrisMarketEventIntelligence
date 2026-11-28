"""Research on immutable fresh forecasts; never backdate a forecast commitment."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from tris_market_event_intelligence.experience_memory import digest
from tris_market_event_intelligence.fresh_market_forecast_cli import infer
from tris_market_event_intelligence.learning_rule_generator import generate_rules
from tris_market_event_intelligence.market_outcome_validation_cli import ZONES, _direction, _prices
from tris_market_event_intelligence.rule_critic import critique_and_revise
from tris_market_event_intelligence.self_reflection import (
    SYSTEM,
    reflection_schema,
    validate_reflection,
)

POLICY = "previous_civil_day_close_to_first_and_fifth_post_publication_day_close_v1"


def store(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def load_forecasts(directory: Path) -> list[dict]:
    records = []
    for path in sorted(directory.glob("*.forecast.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record["id"] != digest(record["identity"])
                or path.stem != record["id"] + ".forecast"
                or record["recordSha256"] != digest({
                    k: v for k, v in record.items() if k != "recordSha256"
                }) or record["forecastSealedBeforeOutcomes"] is not False):
            raise ValueError("Forecast integrity failure")
        records.append(record)
    return records


def publication_rows(records: list[dict], prices: Path) -> list[dict]:
    series = {}
    for bar in _prices(prices):
        by_day = series.setdefault(bar.symbol, {})
        if bar.day in by_day:
            raise ValueError("Duplicate price date")
        by_day[bar.day] = bar.adjusted_close
    rows = []
    for record in records:
        identity = record["identity"]
        packet = identity["packet"]
        published = datetime.fromisoformat(packet["publishedAt"])
        if published.utcoffset() is None:
            raise ValueError("Publication time must be timezone-aware")
        for index, forecast in enumerate(record["forecasts"]):
            instrument = forecast["instrument"]
            symbol, benchmark = instrument["symbol"], instrument["benchmark"]
            day = published.astimezone(ZoneInfo(ZONES[instrument["market"]])).date()
            stock = series.get(symbol, {})
            market = series.get(benchmark, {})
            past = sorted(d for d in stock if d < day)
            future = sorted(d for d in stock if d > day)
            # Daily closes cannot tell whether the publication-day close was
            # before or after release. Exclude it as an endpoint, not as a return.
            entry = past[-1] if past else None
            for horizon in (1, 5):
                exit_day = future[horizon - 1] if len(future) >= horizon else None
                sr = mr = None
                if entry is not None and exit_day is not None:
                    # Avoid spanning stale data or suspended instruments silently.
                    if (day - entry).days <= 7:
                        sr = stock[exit_day] / stock[entry] - 1
                        if entry in market and exit_day in market:
                            mr = market[exit_day] / market[entry] - 1
                excess = sr - mr if sr is not None and mr is not None else None
                actual = _direction(sr) if sr is not None else None
                relative = _direction(excess) if excess is not None else None
                direction = forecast["direction"]
                rows.append({
                    "forecastId": record["id"], "forecastSha256": record["recordSha256"],
                    "articleId": identity["articleId"], "bodySha256": identity["bodySha256"],
                    "impactIndex": index, "target": instrument["name"], "symbol": symbol,
                    "direction": direction, "actualDirection": actual,
                    "relativeActualDirection": relative,
                    "verdict": ("UNAVAILABLE" if actual is None else
                                "CORRECT" if actual == direction else "INCORRECT"),
                    "relativeVerdict": ("UNAVAILABLE" if relative is None else
                                        "CORRECT" if relative == direction else "INCORRECT"),
                    "horizonSessions": horizon, "entryDay": entry.isoformat() if entry else None,
                    "exitDay": exit_day.isoformat() if exit_day else None,
                    "stockReturn": sr, "benchmarkReturn": mr, "excessReturn": excess,
                    "publishedAt": packet["publishedAt"], "processedAt": record["generatedAt"],
                    "policy": POLICY, "status": "RETROSPECTIVE_UNSEALED_OBSERVATION",
                    "reason": forecast["mechanism"], "counterargument": forecast["counterargument"],
                    "evidence": forecast["evidenceQuote"],
                })
    return rows


def metrics(rows: list[dict]) -> dict:
    output = {}
    for h in (1, 5):
        cohort = [r for r in rows if r["horizonSessions"] == h]
        for kind, actual in (("verdict", "actualDirection"),
                             ("relativeVerdict", "relativeActualDirection")):
            scored = [r for r in cohort if r[kind] in {"CORRECT", "INCORRECT"}]
            counts = Counter(r[kind] for r in cohort)
            output[f"{h}d:{kind}"] = {
                "counts": dict(counts), "denominator": len(scored),
                "accuracy": counts["CORRECT"] / len(scored) if scored else None,
                "constantBaselines": {
                    direction: sum(r[actual] == direction for r in scored)
                    for direction in ("UP", "DOWN", "NEUTRAL")
                },
                "distinctArticles": len({r["articleId"] for r in scored}),
            }
    return output


def reflect(client: httpx.Client, model: str, row: dict, record: dict) -> dict:
    passages = record["identity"]["packet"]["sourcePassages"]
    response = client.post("http://127.0.0.1:11434/api/chat", json={
        "model": model, "stream": False, "think": False, "format": reflection_schema(passages),
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 1024},
        "messages": [
            {"role": "system", "content": SYSTEM + (
                " Historical daily window includes pre-release movement; do not assume causation."
            )},
            {"role": "user", "content": json.dumps({
                "outcome": row, "sourcePassages": passages,
                "originalRationale": row["reason"],
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
        observed_at=record["generatedAt"], sealed_at=record["generatedAt"],
    )
    reflection.pop("evidenceAvailableAtSeal")
    reflection["sourceInOriginalForecastPacket"] = True
    return reflection


def run(args: argparse.Namespace, client: httpx.Client) -> dict:
    if not 1 <= args.limit <= 100:
        raise ValueError("Experience limit must be 1..100")
    records = load_forecasts(args.forecasts)
    if not records:
        raise ValueError("No original forecasts")
    if args.output_dir.resolve() == args.forecasts.resolve() or args.output_dir.is_symlink():
        raise ValueError("Separate experience directory required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = publication_rows(records, args.prices)
    by_id = {r["id"]: r for r in records}
    experiences = []
    failures = []
    attempted = 0
    # Use one horizon for initial learning; never count two horizons as independent articles.
    groups = [[r for r in rows if r["verdict"] == v and r["horizonSessions"] == 1]
              for v in ("CORRECT", "INCORRECT")]
    ordered = [g[i] for i in range(max(map(len, groups))) for g in groups if i < len(g)]
    for row in ordered:
        key = digest(row)
        path = args.output_dir / (key + ".experience.json")
        try:
            if path.exists():
                experience = json.loads(path.read_text(encoding="utf-8"))
                if (experience["outcome"] != row or experience["id"] != key
                        or experience["recordSha256"] != digest({
                            k: v for k, v in experience.items() if k != "recordSha256"
                        })):
                    raise ValueError("Experience integrity failure")
            else:
                if attempted >= args.limit:
                    continue
                attempted += 1
                experience = {
                    "id": key, "outcome": row,
                    "reflection": reflect(client, args.model, row, by_id[row["forecastId"]]),
                    "historicalResearchOnly": True, "trainingEligible": False,
                    "promotionAllowed": False,
                }
                experience["recordSha256"] = digest(experience)
                store(path, experience)
            experiences.append(experience)
        except (ValueError, KeyError, TypeError, httpx.HTTPError) as exc:
            failures.append({"articleId": row["articleId"], "error": str(exc)[:300]})
    rules = []
    review = None
    try:
        rules = generate_rules(client, args.model, experiences,
                               cutoff=datetime.now(UTC).isoformat(), historical=True)
        if rules:
            review = critique_and_revise(client, args.model, rules[0], experiences)
            path = args.output_dir / (digest(review) + ".review.json")
            if not path.exists():
                store(path, review)
    except (ValueError, KeyError, TypeError, httpx.HTTPError) as exc:
        failures.append({"stage": "rule", "error": str(exc)[:300]})
    return {
        "status": "PARTIAL" if failures else "COMPLETED", "policy": POLICY,
        "policyLimitation": "Daily approximation includes pre-release price movements; not causal impact.",
        "articles": len(records), "metrics": metrics(rows), "marketRows": rows,
        "experiences": len(experiences), "newReflectionAttempts": attempted,
        "distinctExperienceArticles": len({r["outcome"]["articleId"] for r in experiences}),
        "rules": rules, "ruleReview": review, "failures": failures,
        "prospectivePredictionAccuracy": None, "promotionAllowed": False,
        "historicalResearchOnly": True, "crossEventValidated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("forecasts", "prices", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    with httpx.Client(timeout=300, trust_env=False) as client:
        result = run(args, client)
    path = args.output_dir / (digest(result) + ".report.json")
    if not path.exists():
        store(path, result)
    print(json.dumps({k: v for k, v in result.items() if k != "marketRows"}
                     | {"report": str(path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
