"""Read-only, no-model-call audit of autonomous A/B market predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from tris_market_event_intelligence.autonomous_market_experience_cli import read_sealed
from tris_market_event_intelligence.experience_memory import digest
from tris_market_event_intelligence.fresh_forecast_learning_cli import publication_rows
from tris_market_event_intelligence.market_outcome_validation_cli import NEUTRAL_THRESHOLD

LABELS = ("UP", "DOWN", "NEUTRAL")
HORIZONS = (1, 5)
OUTCOMES = ("absolute", "relative")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _matrix(rows: list[dict], *, actual_key: str) -> dict:
    matrix = {label: {prediction: 0 for prediction in LABELS} for label in LABELS}
    for row in rows:
        actual, predicted = row[actual_key], row["prediction"]
        if actual not in LABELS or predicted not in LABELS:
            raise ValueError("Invalid direction label in scored prediction")
        matrix[actual][predicted] += 1
    classes = {}
    for label in LABELS:
        tp = matrix[label][label]
        predicted_count = sum(matrix[a][label] for a in LABELS)
        actual_count = sum(matrix[label].values())
        classes[label] = {
            "support": actual_count,
            "predicted": predicted_count,
            "precision": _ratio(tp, predicted_count),
            "recall": _ratio(tp, actual_count),
        }
    present = [v["recall"] for v in classes.values() if v["recall"] is not None]
    correct = sum(matrix[label][label] for label in LABELS)
    return {
        "count": len(rows),
        "correct": correct,
        "accuracy": _ratio(correct, len(rows)),
        "balancedAccuracyPresentClasses": sum(present) / len(present) if present else None,
        "balancedAccuracyAllThreeClasses": (
            sum(v["recall"] or 0 for v in classes.values()) / len(LABELS)
        ),
        "matrixActualRowsPredictedColumns": matrix,
        "perClass": classes,
        "alwaysBaselines": {
            label: _ratio(classes[label]["support"], len(rows)) for label in LABELS
        },
    }


def _summary(rows: list[dict], group_key: str, *, actual_key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key) or "UNKNOWN")].append(row)
    return {
        name: {"count": len(group), "correct": sum(
            x["prediction"] == x[actual_key] for x in group
        ), "accuracy": _ratio(sum(
            x["prediction"] == x[actual_key] for x in group
        ), len(group))}
        for name, group in sorted(groups.items())
    }


def _safe_dir(path: Path, *, exclude: list[Path]) -> None:
    resolved = path.resolve()
    if path.is_symlink() or any(
        resolved == item.resolve() or resolved.is_relative_to(item.resolve())
        for item in exclude
    ):
        raise ValueError("Output must be outside protected input directories")
    if path.exists() and not path.is_dir():
        raise ValueError("Output path must be a directory")
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError("Use a new empty output directory to preserve prior audits")


def analyze(run_dir: Path, prices: Path) -> tuple[dict, list[dict]]:
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError("Existing non-symlink research directory required")
    if prices.is_symlink() or not prices.is_file():
        raise ValueError("Existing non-symlink price CSV required")
    cohort = read_sealed(run_dir / "cohort.json")
    config = cohort["config"]
    price_sha = _sha(prices)
    if config["pricesSha256"] != price_sha:
        raise ValueError("Price CSV differs from sealed cohort")
    identities = cohort["identities"]
    if len({i["articleId"] for i in identities}) != len(identities):
        raise ValueError("Duplicate cohort article IDs")
    detail = []
    forecast_records = []
    counts = Counter()
    for identity in identities:
        artifact = read_sealed(run_dir / (digest(identity) + ".abc.json"))
        if artifact["identity"] != identity:
            raise ValueError("Checkpoint cohort identity mismatch")
        if not {"A", "B"}.issubset(artifact["arms"]):
            raise ValueError("Missing A/B predictions")
        if artifact.get("historicalResearchOnly") is not True:
            raise ValueError("Unexpected non-historical artifact")
        for arm in ("A", "B"):
            seen = set()
            for index, forecast in enumerate(artifact["arms"][arm]):
                instrument = forecast["instrument"]
                key = (instrument["market"], instrument["symbol"])
                if key in seen:
                    raise ValueError("Duplicate arm/article/instrument prediction")
                seen.add(key)
                if forecast["direction"] not in LABELS:
                    raise ValueError("Invalid direction")
                counts[f"forecasts_{arm}"] += 1
                forecast_records.append({
                    "id": digest([identity["articleId"], arm, key, index]),
                    "recordSha256": artifact["recordSha256"],
                    "generatedAt": artifact["generatedAt"],
                    "identity": identity,
                    "forecasts": [forecast],
                    "comparisonArm": arm,
                    "retrieved": bool(artifact.get("retrievedExperienceIds")),
                })
    evaluated = publication_rows(forecast_records, prices)
    if len(evaluated) != len(forecast_records) * 2:
        raise ValueError("Evaluation cardinality mismatch")
    lookup = {}
    for source, row1, row5 in zip(
        forecast_records, evaluated[::2], evaluated[1::2], strict=True
    ):
        if (row1["horizonSessions"], row5["horizonSessions"]) != HORIZONS:
            raise ValueError("Evaluation horizon ordering mismatch")
        for row in (row1, row5):
            key = (row["articleId"], source["comparisonArm"], row["symbol"],
                   source["forecasts"][0]["instrument"]["market"], row["horizonSessions"])
            if key in lookup:
                raise ValueError("Duplicate evaluation key")
            lookup[key] = (source, row)
    for identity in identities:
        article_id = identity["articleId"]
        artifact = read_sealed(run_dir / (digest(identity) + ".abc.json"))
        instruments = {
            (fc["instrument"]["market"], fc["instrument"]["symbol"]):
            fc["instrument"]
            for arm in ("A", "B") for fc in artifact["arms"][arm]
        }
        for (market, symbol), instrument in sorted(instruments.items()):
            for horizon in HORIZONS:
                paired = all(
                    (article_id, arm, symbol, market, horizon) in lookup
                    for arm in ("A", "B")
                )
                for arm in ("A", "B"):
                    found = lookup.get((article_id, arm, symbol, market, horizon))
                    if not found:
                        continue
                    source, row = found
                    detail.append({
                        "articleId": article_id,
                        "title": identity["packet"]["title"],
                        "publishedAt": identity["packet"]["publishedAt"],
                        "market": market,
                        "symbol": symbol,
                        "target": row["target"],
                        "mappingKind": instrument.get("mappingKind"),
                        "sector": instrument.get("sector"),
                        "articleType": identity["packet"].get("articleType"),
                        "arm": arm,
                        "horizon": horizon,
                        "pairedAB": paired,
                        "retrievedExperience": source["retrieved"],
                        "prediction": row["direction"],
                        "reason": row["reason"],
                        "counterargument": row["counterargument"],
                        "evidence": row["evidence"],
                        "actualDirection": row["actualDirection"],
                        "relativeActualDirection": row["relativeActualDirection"],
                        "verdict": row["verdict"],
                        "relativeVerdict": row["relativeVerdict"],
                        "stockReturn": row["stockReturn"],
                        "benchmarkReturn": row["benchmarkReturn"],
                        "excessReturn": row["excessReturn"],
                        "entryDay": row["entryDay"],
                        "exitDay": row["exitDay"],
                        "pricePolicy": row["policy"],
                        "priceSha256": row["pricesSha256"],
                    })
    result = {
        "status": "RETROSPECTIVE_DIAGNOSTIC_ONLY",
        "historicalResearchOnly": True,
        "promotionAllowed": False,
        "prospectiveAccuracy": None,
        "articles": len(identities),
        "priceSha256": price_sha,
        "thresholdAbsoluteReturn": NEUTRAL_THRESHOLD,
        "counts": dict(counts),
        "evaluatedRows": len(detail),
        "definition": {
            "allSelected": "Only emitted forecasts; includes unpaired targets",
            "pairedAB": "Same article, market, symbol, horizon selected by both arms",
            "accuracy": "Correct / scored; missing and unavailable reported separately",
            "balancedAccuracyPresentClasses": "Mean recall for classes present in actual labels",
            "balancedAccuracyAllThreeClasses": "Mean recall over UP/DOWN/NEUTRAL; absent classes count zero",
            "dailyWindowWarning": (
                "Entry is pre-publication-day adjusted close and exits are later daily "
                "closes; intraday causality and event-only attribution cannot be inferred."
            ),
        },
        "comparisons": {},
    }
    for scope in ("allSelected", "pairedAB"):
        scoped = detail if scope == "allSelected" else [
            row for row in detail if row["pairedAB"]
        ]
        for horizon in HORIZONS:
            for arm in ("A", "B"):
                selected = [
                    row for row in scoped if row["horizon"] == horizon and row["arm"] == arm
                ]
                for outcome in OUTCOMES:
                    actual_key = ("actualDirection" if outcome == "absolute"
                                  else "relativeActualDirection")
                    scored = [x for x in selected if x[actual_key] in LABELS]
                    key = f"{scope}:{arm}:{horizon}d:{outcome}"
                    result["comparisons"][key] = {
                        **_matrix(scored, actual_key=actual_key),
                        "selected": len(selected),
                        "unavailable": len(selected) - len(scored),
                        "predictions": dict(Counter(x["prediction"] for x in selected)),
                        "actuals": dict(Counter(x[actual_key] for x in scored)),
                        "bySymbol": _summary(scored, "symbol", actual_key=actual_key),
                        "bySector": _summary(scored, "sector", actual_key=actual_key),
                        "byArticleType": _summary(
                            scored, "articleType", actual_key=actual_key
                        ),
                        "byRetrievedExperience": _summary(
                            scored, "retrievedExperience", actual_key=actual_key
                        ),
                    }
    return result, detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _safe_dir(args.output_dir, exclude=[args.run, args.prices.parent])
    report, rows = analyze(args.run, args.prices)
    (args.output_dir / "failure-analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output_dir / "prediction-details.csv").open(
        "x", newline="", encoding="utf-8"
    ) as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({
        "status": report["status"],
        "articles": report["articles"],
        "evaluatedRows": report["evaluatedRows"],
        "outputDir": str(args.output_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
