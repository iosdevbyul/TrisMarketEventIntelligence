"""Audit saved retrospective reflections without invoking a language model."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tris_market_event_intelligence.autonomous_market_experience_cli import (
    eligible_memory,
    read_sealed,
)
from tris_market_event_intelligence.experience_memory import digest
from tris_market_event_intelligence.historical_experience_abc_v2_cli import (
    _article_text,
    _tokens,
    load_experiences,
    retrieve,
)
from tris_market_event_intelligence.prediction_failure_analysis_cli import (
    _safe_dir,
    analyze,
)
from tris_market_event_intelligence.self_reflection import CATEGORIES


def citation_flags(reflection: dict, packet: dict) -> list[str]:
    passages = {p["id"]: p["text"] for p in packet["sourcePassages"]}
    identifier = reflection.get("evidence_id")
    quote = reflection.get("evidenceQuote")
    flags = []
    if not identifier:
        flags.append("NO_CITATION")
    elif identifier not in passages or passages[identifier] != quote:
        flags.append("CITATION_MISMATCH")
    if reflection.get("category") not in CATEGORIES:
        flags.append("UNKNOWN_CATEGORY")
    for field in ("hypothesis", "missing_evidence", "falsification"):
        value = reflection.get(field)
        if not isinstance(value, str) or not value.strip():
            flags.append("MISSING_" + field.upper())
    if reflection.get("semanticEntailmentVerified") is not True:
        flags.append("SEMANTIC_SUPPORT_UNVERIFIED")
    return flags


def paired_effect(rows: list[dict]) -> dict:
    pairs = {}
    for row in rows:
        if not row["pairedAB"] or not row["retrievedExperience"]:
            continue
        key = (row["articleId"], row["market"], row["symbol"], row["horizon"])
        pairs.setdefault(key, {})[row["arm"]] = row
    result = {}
    for horizon in (1, 5):
        for verdict in ("verdict", "relativeVerdict"):
            counts = Counter()
            transitions = Counter()
            for key, pair in pairs.items():
                if key[-1] != horizon or set(pair) != {"A", "B"}:
                    continue
                a, b = pair["A"], pair["B"]
                if any(x[verdict] not in {"CORRECT", "INCORRECT"} for x in (a, b)):
                    continue
                counts[a[verdict] + "_TO_" + b[verdict]] += 1
                transitions[a["prediction"] + "_TO_" + b["prediction"]] += 1
            result[f"{horizon}d:{verdict}"] = {
                "pairs": sum(counts.values()),
                "verdictTransitions": dict(counts),
                "directionTransitions": dict(transitions),
                "netCorrectChange": (
                    counts["INCORRECT_TO_CORRECT"] - counts["CORRECT_TO_INCORRECT"]
                ),
            }
    return result


def audit(run_dir: Path, seed_dir: Path, prices: Path) -> dict:
    _, detail = analyze(run_dir, prices)
    cohort = read_sealed(run_dir / "cohort.json")
    report = read_sealed(run_dir / "report.json")
    seed = load_experiences(seed_dir)
    if digest(seed) != cohort["config"]["seedDigest"]:
        raise ValueError("Seed memory differs from sealed cohort")
    fresh = load_experiences(run_dir)
    if len(fresh) != report["newExperiences"]:
        raise ValueError("Experience count differs from sealed report")
    by_forecast = {}
    for item in fresh:
        key = item["outcome"]["forecastId"]
        by_forecast.setdefault(key, []).append(item)
    memories = list(seed)
    reflections = []
    retrievals = []
    categories = Counter()
    flags_count = Counter()
    used_ids = set()
    for identity in cohort["identities"]:
        key = digest(identity)
        packet = identity["packet"]
        artifact = read_sealed(run_dir / (key + ".abc.json"))
        past = eligible_memory(memories, packet)
        related = [
            item for item in retrieve(
                packet, identity["articleId"], identity["bodySha256"], past
            ) if item["score"] > 0
        ]
        if [x["experienceId"] for x in related] != artifact["retrievedExperienceIds"]:
            raise ValueError("Retrieved experience replay mismatch")
        by_id = {item["id"]: item for item in past}
        terms = _tokens(_article_text(packet))
        for item in related:
            original = by_id[item["experienceId"]]
            outcome = original["outcome"]
            overlap = terms & _tokens(
                f"{outcome['target']} {outcome.get('reason', '')} "
                f"{original['reflection'].get('hypothesis', '')}"
            )
            used_ids.add(item["experienceId"])
            retrievals.append({
                "articleId": identity["articleId"],
                "title": packet["title"],
                "experienceId": item["experienceId"],
                "sourceArticleId": outcome["articleId"],
                "sourceTarget": outcome["target"],
                "sourceSymbol": outcome["symbol"],
                "sourceVerdict": outcome["verdict"],
                "score": item["score"],
                "matchedTokens": sorted(overlap),
                "reflectionAsProvided": item["reflection"],
                "sameSymbolInCurrentForecast": any(
                    f["instrument"]["symbol"] == outcome["symbol"]
                    for f in artifact["arms"]["A"]
                ),
            })
        event_memories = sorted(
            by_forecast.pop(key, []), key=lambda item: item["outcome"]["impactIndex"]
        )
        for item in event_memories:
            outcome = item["outcome"]
            if (outcome["articleId"] != identity["articleId"]
                    or outcome["bodySha256"] != identity["bodySha256"]
                    or outcome["forecastSha256"] != artifact["recordSha256"]):
                raise ValueError("Experience source identity mismatch")
            reflection = item["reflection"]
            flags = citation_flags(reflection, packet)
            flags_count.update(flags)
            categories[(outcome["verdict"], reflection["category"])] += 1
            reflections.append({
                "experienceId": item["id"],
                "articleId": identity["articleId"],
                "title": packet["title"],
                "symbol": outcome["symbol"],
                "prediction": outcome["direction"],
                "actual": outcome["actualDirection"],
                "verdict": outcome["verdict"],
                "return": outcome["stockReturn"],
                "originalRationale": outcome["reason"],
                "sourcePassages": packet["sourcePassages"],
                "reflection": reflection,
                "flags": flags,
                "semanticReviewRequired": True,
            })
        memories.extend(event_memories)
    if by_forecast:
        raise ValueError("Unmatched experience forecast IDs")
    return {
        "status": "REFLECTION_DIAGNOSTIC_ONLY",
        "historicalResearchOnly": True,
        "promotionAllowed": False,
        "modelCalls": 0,
        "newExperiences": len(reflections),
        "seedExperiences": len(seed),
        "retrievalCount": len(retrievals),
        "distinctRetrievedExperiences": len(used_ids),
        "categoriesByVerdict": {
            f"{verdict}:{category}": count
            for (verdict, category), count in sorted(categories.items())
        },
        "citationAndStructureFlags": dict(flags_count),
        "pairedEffectWithRetrievedMemory": paired_effect(detail),
        "limitations": [
            "Citation identity does not establish semantic entailment or causality.",
            "Token overlap is not proof of semantic retrieval relevance.",
            "Paired retrospective changes are descriptive, not prospective evidence.",
        ],
        "reflections": reflections,
        "retrievals": retrievals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "seed", "prices", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.seed, args.prices)
    _safe_dir(args.output_dir, exclude=[args.run, args.seed, args.prices.parent])
    path = args.output_dir / "reflection-quality.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({
        k: v for k, v in result.items() if k not in {"reflections", "retrievals"}
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
