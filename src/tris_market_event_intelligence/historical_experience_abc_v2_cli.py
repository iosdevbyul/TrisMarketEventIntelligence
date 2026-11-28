"""Research-only historical A/B/C forecast comparison on existing immutable artifacts.

Experience reflections were created after the events. Chronological filtering does
NOT authenticate their availability at historical publication; this is a
retrospective counterfactual, not temporal replay or prospective validation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx

from tris_market_event_intelligence.experience_memory import digest
from tris_market_event_intelligence.fresh_forecast_learning_cli import (
    load_forecasts, publication_rows,
)
from tris_market_event_intelligence.fresh_market_forecast_cli import (
    SYSTEM, validate,
)
from tris_market_event_intelligence.gemma_original_article_batch_cli import MODEL_PATTERN
from tris_market_event_intelligence.learning_rule_generator import (
    FIELDS, SCHEMA, SYSTEM as RULE_SYSTEM, validate_aliased_rules,
)
from tris_market_event_intelligence.rule_critic import critique_and_revise


def _date(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Timezone-aware dates required")
    return result.astimezone(UTC)


def _tokens(value: str) -> set[str]:
    return {x for x in re.findall(r"[^\W_]{3,}", value.casefold()) if x}


def _article_text(packet: dict) -> str:
    return packet["title"] + " " + " ".join(
        x["text"] for x in packet["sourcePassages"][:8]
    )


def load_experiences(directory: Path) -> list[dict]:
    output = []
    for path in sorted(directory.glob("*.experience.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("recordSha256") != digest({
                k: v for k, v in row.items() if k != "recordSha256"
            }) or row.get("id") != digest(row["outcome"])
                or not row.get("historicalResearchOnly")
                or row.get("promotionAllowed") is not False):
            raise ValueError(f"Experience integrity failure: {path.name}")
        output.append(row)
    return output


def retrieve(
    packet: dict, article_id: int, body_hash: str, records: list[dict],
    *, max_items: int = 5, excluded_ids: set[int] | None = None,
) -> list[dict]:
    """Rank past-outcome experiences using only current article's source text.

    No current event realized price, validation labels, or outcome is used to rank.
    Chronology applies to the historic market outcome date; the post-hoc
    reflection creation time is NOT treated as a historic availability timestamp.
    """
    cutoff = _date(packet["publishedAt"])
    terms = _tokens(_article_text(packet))
    if not 1 <= max_items <= 12:
        raise ValueError("Invalid retrieval limit")
    ranked = []
    for item in records:
        row = item["outcome"]
        if (row["articleId"] == article_id
                or row["articleId"] in (excluded_ids or set())
                or row["bodySha256"] == body_hash
                or row.get("verdict") not in {"CORRECT", "INCORRECT"}
                or row.get("status") != "RETROSPECTIVE_UNSEALED_OBSERVATION"
                or not row.get("exitDay")
                or _date(row["publishedAt"]) >= cutoff
                or datetime.fromisoformat(row["exitDay"]).date() >= cutoff.date()):
            continue
        rationale = str(row.get("reason", ""))
        hypothesis = str(item.get("reflection", {}).get("hypothesis", ""))
        # Do not rank by correctness, realized return or holdout outcome.
        similarity = len(terms & _tokens(
            f"{row.get('target', '')} {rationale} {hypothesis}"
        ))
        ranked.append((similarity, row["exitDay"], item["id"], item))
    ranked.sort(key=lambda x: (-x[0], -int(x[1].replace("-", "")), x[2]))
    seen_articles: set[int] = set()
    result = []
    for score, _, _, item in ranked:
        row = item["outcome"]
        if row["articleId"] in seen_articles:
            continue
        seen_articles.add(row["articleId"])
        result.append({
            "experienceId": item["id"], "articleId": row["articleId"],
            "score": score, "target": row["target"],
            "forecastDirection": row["direction"], "outcomeVerdict": row["verdict"],
            "actualDirection": row["actualDirection"],
            "marketHorizon": row["horizonSessions"],
            "reflection": {
                k: str(item.get("reflection", {}).get(k, ""))[:350]
                for k in ("category", "hypothesis", "missing_evidence", "falsification")
            },
            "historicallyBackfilled": True,
        })
        if len(result) >= max_items:
            break
    return result


def _ask(client: httpx.Client, model: str, system: str, prompt: dict, schema: dict,
         *, seed: int = 20261128) -> dict:
    response = client.post("http://127.0.0.1:11434/api/chat", json={
        "model": model, "stream": False, "think": False, "format": schema,
        "options": {"temperature": 0, "seed": seed, "num_ctx": 8192,
                    "num_predict": 1536},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    })
    response.raise_for_status()
    result = response.json()
    if result.get("done_reason") == "length":
        raise ValueError("Truncated Gemma response")
    return json.loads(result["message"]["content"])


def predict(client: httpx.Client, model: str, packet: dict, *,
            experience: list[dict] | None = None, rule: dict | None = None) -> list[dict]:
    from tris_market_event_intelligence.fresh_market_forecast_cli import infer
    if experience is None and rule is None:
        return infer(client, model, packet)
    options = packet["instruments"]
    passages = packet["sourcePassages"]
    schema = {
        "type": "object", "additionalProperties": False, "required": ["forecasts"],
        "properties": {"forecasts": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["instrument_id", "direction", "evidence_id", "mechanism", "counterargument"],
            "properties": {
                "instrument_id": {"type": "string", "enum": [x["id"] for x in options]},
                "direction": {"type": "string", "enum": ["UP", "DOWN", "NEUTRAL"]},
                "evidence_id": {"type": "string", "enum": [x["id"] for x in passages]},
                "mechanism": {"type": "string"}, "counterargument": {"type": "string"},
            },
        }}},
    }
    system = SYSTEM + (
        "\nHistoric experiences are post-hoc unverified hypotheses, not contemporary "
        "reporting or proof of causal direction. Source citations must point to the "
        "CURRENT article only; prior outcomes may guide questions, not dictate labels."
    )
    context = {"historicalExperiences": experience} if experience is not None else {
        "unvalidatedResearchRule": {
            key: rule[key] for key in ("condition", "analysis_action", "exceptions")
        }
    }
    error = ""
    for attempt in range(2):
        try:
            answer = _ask(client, model, system, {
                **packet, **context, "previousValidationError": error,
            }, schema, seed=20261128 + attempt)
            return validate(answer, options, passages)
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)[:300]
    raise ValueError(error)


def generate_rule_v2(client: httpx.Client, model: str, experiences: list[dict],
                     cutoff: str) -> tuple[list[dict], list[dict]]:
    """Keep >=2 independent supporting articles, retry validation with feedback."""
    sorted_rows = sorted(experiences, key=lambda x: (x["outcome"]["articleId"], x["id"]))
    unique: dict[int, dict] = {}
    for item in sorted_rows:
        unique.setdefault(item["outcome"]["articleId"], item)
    selected = list(unique.values())[:12]
    if (len(selected) < 3 or
            {x["outcome"]["verdict"] for x in selected} != {"CORRECT", "INCORRECT"}):
        return [], [{"reason": "INSUFFICIENT_DISTINCT_MIXED_EXPERIENCES"}]
    aliases = {f"R{i}": x["id"] for i, x in enumerate(selected, 1)}
    schema = json.loads(json.dumps(SCHEMA))
    for field in FIELDS[4:]:
        schema["properties"]["rules"]["items"]["properties"][field]["items"]["enum"] = list(aliases)
    evidence = [{
        "id": alias, "articleId": x["outcome"]["articleId"],
        "target": x["outcome"]["target"], "direction": x["outcome"]["direction"],
        "verdict": x["outcome"]["verdict"],
        "reflection": x["reflection"],
    } for alias, x in zip(aliases, selected, strict=True)]
    errors = []
    for attempt in range(3):
        try:
            answer = _ask(client, model, RULE_SYSTEM, {
                "experiences": evidence,
                "previousValidationErrors": errors,
                "requirement": "Each candidate cites two DIFFERENT articleIds. "
                               "Return empty rules if support is insufficient.",
            }, schema, seed=20261130 + attempt)
            rules = validate_aliased_rules(answer, aliases, selected, cutoff)
            for rule in rules:
                rule.pop("ruleId")
                rule.update({"status": "HISTORICAL_RESEARCH_CANDIDATE",
                             "historicalResearchOnly": True})
                rule["ruleId"] = digest(rule)
            return rules, errors
        except (ValueError, KeyError, TypeError) as exc:
            errors.append({"attempt": attempt + 1, "reason": str(exc)[:300]})
    return [], errors


def _sealed(path: Path, payload: dict) -> dict:
    value = {**payload, "recordSha256": digest(payload)}
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != value:
            raise ValueError(f"Immutable research forecast conflict: {path.name}")
        return previous
    if path.is_symlink():
        raise ValueError("Symlink output prohibited")
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    return value


def compare(artifacts: list[dict], prices: Path) -> dict:
    """Evaluate all variants on the union of targets, counting every missing call."""
    from tris_market_event_intelligence.fresh_forecast_learning_cli import publication_rows
    variants = ("A", "B", "C") if any("C" in a["arms"] for a in artifacts) else ("A", "B")
    synthetic = []
    keys = set()
    for artifact in artifacts:
        identity = artifact["identity"]
        for arm, forecasts in artifact["arms"].items():
            for forecast in forecasts:
                inst = forecast["instrument"]
                key = (identity["articleId"], inst["market"], inst["symbol"])
                keys.add(key)
                synthetic.append({
                    "id": digest([key, arm]), "recordSha256": artifact["recordSha256"],
                    "generatedAt": artifact["generatedAt"], "identity": identity,
                    "forecasts": [forecast], "comparisonArm": arm,
                })
    outcomes = publication_rows(synthetic, prices) if synthetic else []
    lookups = {}
    for record, rows in zip(synthetic, [outcomes[i:i + 2] for i in range(0, len(outcomes), 2)],
                            strict=True):
        for row in rows:
            lookups[(record["identity"]["articleId"],
                     record["forecasts"][0]["instrument"]["market"],
                     row["symbol"], row["horizonSessions"], record["comparisonArm"])] = row
    results = []
    for article_id, market, symbol in sorted(keys):
        for horizon in (1, 5):
            by_arm = {arm: lookups.get((article_id, market, symbol, horizon, arm))
                      for arm in variants}
            observed = next((x for x in by_arm.values() if x is not None), None)
            for arm, row in by_arm.items():
                results.append({
                    "articleId": article_id, "market": market, "symbol": symbol,
                    "horizon": horizon, "arm": arm,
                    "selectionStatus": "SELECTED" if row else "NOT_SELECTED",
                    "direction": row["direction"] if row else None,
                    "actualDirection": observed["actualDirection"] if observed else None,
                    "relativeActualDirection": (
                        observed["relativeActualDirection"] if observed else None
                    ),
                    "verdict": row["verdict"] if row else "NOT_SELECTED",
                    "relativeVerdict": row["relativeVerdict"] if row else "NOT_SELECTED",
                })
    metrics = {}
    for horizon in (1, 5):
        for arm in variants:
            group = [x for x in results if x["horizon"] == horizon and x["arm"] == arm]
            for label in ("verdict", "relativeVerdict"):
                scored = [x for x in group if x[label] in {"CORRECT", "INCORRECT"}]
                counts = Counter(x[label] for x in group)
                metrics[f"{arm}:{horizon}d:{label}"] = {
                    "selected": sum(x["selectionStatus"] == "SELECTED" for x in group),
                    "universe": len(group), "scored": len(scored),
                    "counts": dict(counts),
                    "conditionalAccuracy": (
                        counts["CORRECT"] / len(scored) if scored else None
                    ),
                    "allUniverseCorrectRate": (
                        counts["CORRECT"] / len(group) if group else None
                    ),
                    "selectionRate": (
                        (len(group) - counts["NOT_SELECTED"]) / len(group)
                        if group else None
                    ),
                    "constantUpCorrect": sum(
                        x["actualDirection" if label == "verdict"
                          else "relativeActualDirection"] == "UP" for x in group
                    ),
                    "constantDownCorrect": sum(
                        x["actualDirection" if label == "verdict"
                          else "relativeActualDirection"] == "DOWN" for x in group
                    ),
                }
    return {"status": "RETROSPECTIVE_COMPARISON_ONLY", "metrics": metrics,
            "rows": results, "outcomeUniverse": len(keys),
            "causalEffectEstablished": False, "prospectiveAccuracy": None,
            "promotionAllowed": False}


def run(args: argparse.Namespace, client: httpx.Client) -> dict:
    if not MODEL_PATTERN.fullmatch(args.model) or not 1 <= args.limit <= 100:
        raise ValueError("Invalid model/limit")
    if args.output_dir.is_symlink() or args.output_dir.resolve() in {
            args.forecasts.resolve(), args.experiences.resolve()}:
        raise ValueError("Separate non-symlink output required")
    originals = load_forecasts(args.forecasts)
    memories = load_experiences(args.experiences)
    # Entire holdout cohort chosen solely from original identities, before seeing outcomes.
    originals.sort(key=lambda x: (x["identity"]["packet"]["publishedAt"],
                                  x["identity"]["articleId"], x["id"]))
    training_ids = {x["outcome"]["articleId"] for x in memories}
    training_hashes = {x["outcome"]["bodySha256"] for x in memories}
    training_forecast_ids = {x["outcome"]["forecastId"] for x in memories}
    candidates = [r for r in originals if
                  r["identity"]["articleId"] not in training_ids
                  and r["identity"]["bodySha256"] not in training_hashes
                  and r["id"] not in training_forecast_ids]
    # Conservative near-duplicate lexical guard using original source snapshots.
    train_texts = [_tokens(x["outcome"].get("evidence", "")) for x in memories]
    def overlaps(r: dict) -> bool:
        terms = _tokens(_article_text(r["identity"]["packet"]))
        return any(
            terms and t and len(terms & t) >= 5
            and len(terms & t) / len(terms | t) >= .65
            for t in train_texts
        )
    holdout = [r for r in candidates if not overlaps(r)][:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Rules are derived ONLY from the provided training memory, never holdout outcomes.
    rule_path = args.output_dir / "candidate-rule-v2.json"
    rules = []
    rule_errors = []
    review = None
    if rule_path.exists():
        saved = json.loads(rule_path.read_text(encoding="utf-8"))
        rules, rule_errors, review = saved["rules"], saved["errors"], saved["review"]
        if saved["trainingDigest"] != digest(memories):
            raise ValueError("Rule-training memory changed after research started")
    elif memories:
        rules, rule_errors = generate_rule_v2(
            client, args.model, memories, datetime.now(UTC).isoformat())
        if rules:
            review = critique_and_revise(client, args.model, rules[0], memories)
        _sealed(rule_path, {"rules": rules, "errors": rule_errors, "review": review,
                            "trainingDigest": digest(memories)})
    approved_rule = None
    if rules and review and review.get("candidateForFurtherResearch"):
        identifier = review["candidateForFurtherResearch"]
        for candidate in [*rules, review.get("revisedRule")]:
            if candidate and candidate["ruleId"] == identifier:
                approved_rule = candidate
    artifacts = []
    failures = []
    for item in holdout:
        identity = item["identity"]
        packet = identity["packet"]
        related = retrieve(packet, identity["articleId"], identity["bodySha256"], memories)
        path = args.output_dir / (item["id"] + ".abc.json")
        if path.exists():
            artifact = json.loads(path.read_text(encoding="utf-8"))
            if (artifact.get("recordSha256") != digest({
                    k: v for k, v in artifact.items() if k != "recordSha256"})
                    or artifact.get("identity") != identity
                    or artifact.get("model") != args.model):
                raise ValueError("Research forecast artifact integrity failure")
            artifacts.append(artifact)
            continue
        try:
            arms = {
                "A": predict(client, args.model, packet),
                "B": predict(client, args.model, packet, experience=related),
            }
            if approved_rule and identity["articleId"] not in approved_rule["trainingArticleIds"]:
                arms["C"] = predict(client, args.model, packet, rule=approved_rule)
            artifact = _sealed(path, {
                "schemaVersion": "historical_experience_abc_v2",
                "identity": identity, "model": args.model, "arms": arms,
                "retrievedExperienceIds": [x["experienceId"] for x in related],
                "ruleId": approved_rule["ruleId"] if "C" in arms else None,
                "generatedAt": datetime.now(UTC).isoformat(),
                "historicalResearchOnly": True, "prospectivelySealed": False,
                "promotionAllowed": False,
            })
            artifacts.append(artifact)
        except (ValueError, TypeError, KeyError, httpx.HTTPError) as exc:
            failures.append({"articleId": identity["articleId"], "error": str(exc)[:300]})
    # Price data is opened only after immutable A/B/C decisions have been recorded.
    evaluation = compare(artifacts, args.prices)
    report = {
        "status": "PARTIAL" if failures else "COMPLETED",
        "trainingExperiences": len(memories), "holdoutArticles": len(holdout),
        "evaluatedArticles": len(artifacts), "ruleCandidateCount": len(rules),
        "ruleValidationErrors": rule_errors, "ruleReview": review,
        "comparison": evaluation, "failures": failures,
        "policyLimitation": "Daily return includes prepublication movement. "
            "Experience reflections were generated retrospectively and were not "
            "available at the historical forecast timestamp.",
        "historicalResearchOnly": True, "promotionAllowed": False,
    }
    report_path = args.output_dir / (digest(report) + ".report.json")
    if not report_path.exists():
        _sealed(report_path, report)
    return {**{k: v for k, v in report.items() if k != "comparison"},
            "comparison": {k: v for k, v in evaluation.items() if k != "rows"},
            "reportPath": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("forecasts", "experiences", "prices", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    with httpx.Client(timeout=300, trust_env=False) as client:
        print(json.dumps(run(args, client), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
