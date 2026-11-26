"""Plan mature diverse event research, seal original-date hypotheses, and compare controls.

Research only. Coverage selection uses date availability, not return values.
Historical source and mapping availability are not independently authenticated.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from zoneinfo import ZoneInfo

from tris_market_event_intelligence.collector import utc_now
from tris_market_event_intelligence.event_profile_integration_cli import (
    error_code,
    source_passages,
)
from tris_market_event_intelligence.independent_event_replay_cli import (
    REACTION_SYSTEM,
    contract,
    evaluate_row,
    targets_for,
    validate,
    write_new,
)
from tris_market_event_intelligence.llm_runtime import (
    ConfiguredLLMProvider,
    LLMCapabilities,
    LLMExecutionError,
    LLMSettings,
)
from tris_market_event_intelligence.market_outcome_validation_cli import ZONES, _mappings
from tris_market_event_intelligence.market_price_recovery_cli import _observations
from tris_market_event_intelligence.original_event_date_comparison_cli import original_anchor
from tris_market_event_intelligence.semantic_event_integration_cli import pinned_articles
from tris_market_event_intelligence.source_aware_event_cli import extract_facts
from tris_market_event_intelligence.unified_event_integration_cli import clock, digest, encode

VERSION = "mature_event_research_v1"
RULES = (
    "Distinguish an actual completed transaction from a Form 144 proposed sale notice.",
    "Strong reported growth does not establish a positive price surprise. Do not invent consensus.",
    "Explain any difference between economic effect and predicted market direction explicitly.",
    "Each horizon needs a supported forward mechanism; abstain if evidence is insufficient.",
)
# Explicit policy candidates, not automatic listing approvals.
MACRO = {
    "US": re.compile(r"\b(?:U\.S\.|United States|Federal Reserve|FOMC|BEA|BLS)\b|미국", re.IGNORECASE),
    "KR": re.compile(r"Bank of Korea|한국은행|한국|국내", re.IGNORECASE),
}
US_SOURCES = {"bea_news", "bls_cpi", "bls_ppi", "bls_latest", "bls_employment",
              "fed_all", "fed_monetary"}


def load_sealed(path, version):
    data = json.loads(path.read_text(encoding="utf-8"))
    checksum = data.pop("sealSha256")
    if data.get("version") != version or digest(encode(data)) != checksum:
        raise ValueError("Invalid sealed artifact")
    return {**data, "sealSha256": checksum}


def seal(payload):
    return {**payload, "sealSha256": digest(encode(payload))}


def dates_only(prices):
    return {symbol: tuple(r["date"] for r in rows) for symbol, rows in prices.items()}


def mature(mapping, anchor, dates):
    stock = dates.get(mapping["symbol"], ())
    bench = dates.get(mapping["benchmark"], ())
    day = anchor.astimezone(ZoneInfo(ZONES[mapping["market"]])).date().isoformat()
    if not stock or not bench:
        return False
    if stock[0] > day or bench[0] > day:
        return False
    window = [d for d in bench if d > day][:6]
    return len(window) == 6 and set(window) <= set(stock)


def candidates_for(item, mappings, facts):
    names = targets_for(item, mappings)
    if item["kind"] != "ECONOMIC_RELEASE" or not facts["indicators"]:
        return names
    # Only explicit reviewed sector proxies with matching country and broad-market name.
    country = "US" if item["source"] in US_SOURCES else (
        "KR" if item["source"].startswith("bok_") else None)
    if country is None:
        return names
    for name, mapping in mappings.items():
        if mapping["mappingKind"] != "SECTOR_ETF_PROXY":
            continue
        market_country = "KR" if mapping["market"] == "XKRX" else (
            "US" if mapping["market"] in {"XNAS", "XNYS"} else None)
        if market_country == country and MACRO[country].search(name):
            if re.search(r"market|economy|equities|증시|시장|경제", name, re.IGNORECASE):
                names.append(name)
    return sorted(set(names))[:12]


def balanced(rows, limit):
    # Round robin across source kind and instrument buckets.
    buckets = defaultdict(deque)
    for row in sorted(rows, key=lambda r: (r["anchorAt"], r["eventId"])):
        buckets[(row["kind"], row["symbol"])].append(row)
    selected = []
    while buckets and len(selected) < limit:
        for key in sorted(buckets):
            selected.append(buckets[key].popleft())
            if not buckets[key]:
                del buckets[key]
            if len(selected) == limit:
                break
    return selected


def make_plan(news_db, event_db, run_id, mappings_path, prices_path, *, limit=24,
              excluded_ids=()):
    if not 1 <= limit <= 100 or news_db.samefile(event_db):
        raise ValueError("Invalid research limit or aliased databases")
    mappings = _mappings(mappings_path)
    dates = dates_only(_observations(prices_path))
    items, excluded = pinned_articles(news_db, event_db, run_id)
    skipped = Counter()
    seen = set()
    rows = []
    for item in sorted(items, key=lambda i: i["eventId"]):
        if item["eventId"] in excluded_ids:
            skipped["PRIOR_RESEARCH_EVENT"] += 1
            continue
        key = tuple(sorted(d["bodySha256"] for d in item["documents"]))
        if key in seen:
            skipped["DUPLICATE_BODY_SET"] += 1
            continue
        seen.add(key)
        facts = extract_facts(item)
        if facts["marketRelevance"] == "LOW_PRIORITY_TITLE_CUE":
            skipped["LOW_PRIORITY_TITLE_CUE"] += 1
            continue
        anchor, field = original_anchor(item)
        if anchor is None:
            skipped["NO_VALID_DECLARED_CLOCK"] += 1
            continue
        names = candidates_for(item, mappings, facts)
        if not names:
            skipped["NO_REVIEWED_TARGET"] += 1
            continue
        names = [n for n in names if mature(mappings[n], anchor, dates)]
        if not names:
            skipped["NO_MATURE_PRICE_WINDOW"] += 1
            continue
        # One deterministic target per event, fixed before model calls.
        name = names[0]
        rows.append({"eventId": item["eventId"], "kind": item["kind"], "title": item["title"],
                     "anchorAt": anchor.isoformat(), "anchorField": field,
                     "symbol": mappings[name]["symbol"], "target": name,
                     "mapping": mappings[name], "documentHashes": list(key)})
    selected = balanced(rows, limit)
    skipped["BUDGET"] = len(rows) - len(selected)
    return seal({"version": VERSION + "_plan", "runId": run_id, "createdAt": utc_now(),
                 "mappingSha256": digest(mappings_path.read_bytes()),
                 "priceSha256": digest(prices_path.read_bytes()),
                 "admitted": len(items), "excluded": dict(excluded), "skipped": dict(skipped),
                 "eligible": len(rows), "selected": selected,
                 "selectedByKind": dict(Counter(r["kind"] for r in selected)),
                 "selectedBySymbol": dict(Counter(r["symbol"] for r in selected)),
                 "selectionUsesReturnValues": False, "trainingEligible": False})


def forecast(news_db, event_db, plan, provider, *, improved=False):
    items, excluded = pinned_articles(news_db, event_db, plan["runId"])
    index = {i["eventId"]: i for i in items}
    records = []
    system = REACTION_SYSTEM + (
        "\nResearch rules:\n" + "\n".join(RULES) if improved else "")
    calls = 0
    for row in plan["selected"]:
        item = index.get(row["eventId"])
        result = {**row, "status": "PIN_NOT_AVAILABLE"}
        if item is None or sorted(d["bodySha256"] for d in item["documents"]) != row["documentHashes"]:
            records.append(result)
            continue
        anchor, _ = original_anchor(item)
        if anchor is None or anchor.isoformat() != row["anchorAt"]:
            records.append({**result, "status": "ANCHOR_CHANGED"})
            continue
        observed = clock(item["observedAt"])
        passages = [p for p in source_passages(item)
                    if observed and clock(p["observedAt"]) and clock(p["observedAt"]) <= observed][:6]
        if not passages:
            records.append({**result, "status": "NO_PINNED_PASSAGES"})
            continue
        facts = extract_facts(item)
        schema = contract([row["target"]], passages)
        packet = {"title": item["title"], "kind": item["kind"], "anchorAt": row["anchorAt"],
                  "allowedTargets": [row["target"]], "passages": passages,
                  "documentType": facts["documentType"],
                  "indicators": facts["indicators"]}
        calls += 1
        try:
            answer = validate(provider.generate_json(system=system, user=encode(packet),
                                                     schema=schema), schema, passages)
            result.update({"status": "CANDIDATE", "answer": answer,
                           "inputSha256": digest(encode(packet)),
                           "observedAt": item["observedAt"]})
        except (LLMExecutionError, ValueError, TypeError, KeyError) as exc:
            result.update({"status": "REJECTED", "error": error_code(exc)})
        records.append(result)
    return seal({"version": VERSION + "_forecast", "planSha256": plan["sealSha256"],
                 "priceSha256": plan["priceSha256"], "runId": plan["runId"],
                 "model": provider.model_id, "systemSha256": digest(system),
                 "policy": "RULES_V1" if improved else "BASELINE",
                 "createdAt": utc_now(), "records": records, "modelCalls": calls,
                 "excluded": dict(excluded), "prospectivePrediction": False,
                 "historicalSourceVersionVerified": False, "trainingEligible": False})


def evaluate(forecast_data, prices_path):
    if digest(prices_path.read_bytes()) != forecast_data["priceSha256"]:
        raise ValueError("Price dataset differs from frozen selection")
    prices = _observations(prices_path)
    rows = []
    for record in forecast_data["records"]:
        if record["status"] != "CANDIDATE":
            continue
        mapping = record["mapping"] if record["answer"]["target"] != "NONE" else None
        for label in evaluate_row({**record, "mapping": mapping,
                                   "observedAt": record["anchorAt"]}, prices):
            rows.append({"eventId": record["eventId"], "kind": record["kind"],
                         "symbol": record["symbol"], "title": record["title"],
                         "answer": record["answer"], **label})
    scored = [r for r in rows if r.get("verdict") in {"CORRECT", "INCORRECT"}]
    observed = [r for r in rows if r["status"] == "OBSERVED_RETROSPECTIVE"]
    return seal({"version": VERSION + "_outcomes", "forecastSha256": forecast_data["sealSha256"],
                 "planSha256": forecast_data["planSha256"], "policy": forecast_data["policy"],
                 "rows": rows, "observedRows": len(observed), "scoredRows": len(scored),
                 "verdictCounts": dict(Counter(r["verdict"] for r in observed)),
                 "baselineOnScored": {d: sum(r["actualDirection"] == d for r in scored)
                                      for d in ("UP", "DOWN", "NEUTRAL")},
                 "marketReturnsAnalyzed": bool(observed), "trainingEligible": False,
                 "prospectivePrediction": False})


def compare_policies(base, improved):
    if base["planSha256"] != improved["planSha256"]:
        raise ValueError("Policy plans differ")
    if base["policy"] != "BASELINE" or improved["policy"] != "RULES_V1":
        raise ValueError("Unexpected policy pair")
    def indexed(report):
        return {(r["eventId"], r["horizonSessions"]): r for r in report["rows"]
                if r.get("verdict") in {"CORRECT", "INCORRECT"}}
    a, b = indexed(base), indexed(improved)
    shared = sorted(a.keys() & b.keys())
    for key in shared:
        if a[key]["actualDirection"] != b[key]["actualDirection"]:
            raise ValueError("Outcome labels differ")
    return {"version": VERSION + "_policy_comparison", "pairedScoredRows": len(shared),
            "baselineCorrect": sum(a[k]["verdict"] == "CORRECT" for k in shared),
            "rulesCorrect": sum(b[k]["verdict"] == "CORRECT" for k in shared),
            "baselineOnlyScored": len(a.keys() - b.keys()),
            "rulesOnlyScored": len(b.keys() - a.keys()),
            "baselineObservedRows": base["observedRows"], "rulesObservedRows": improved["observedRows"],
            "independentEventValidation": False, "automaticPromotion": False,
            "trainingEligible": False,
            "failureMemory": [{"eventId": r["eventId"], "horizon": r["horizonSessions"],
                               "prediction": r["prediction"], "actual": r["actualDirection"],
                               "stockReturn": r["stockReturn"], "excessReturn": r["excessReturn"],
                               "answer": r["answer"], "causalExplanationVerified": False}
                              for r in base["rows"] if r.get("verdict") == "INCORRECT"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    for name in ("news-db", "event-db", "mappings", "prices", "output"):
        plan.add_argument("--" + name, type=Path, required=True)
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--limit", type=int, default=24)
    plan.add_argument("--exclude-seal", type=Path)
    run = sub.add_parser("forecast")
    for name in ("news-db", "event-db", "plan", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--model", default="gemma4:e2b")
    run.add_argument("--improved", action="store_true")
    outcome = sub.add_parser("evaluate")
    for name in ("forecast", "prices", "output"):
        outcome.add_argument("--" + name, type=Path, required=True)
    comparison = sub.add_parser("compare")
    for name in ("baseline", "improved", "output"):
        comparison.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        excluded = []
        if args.exclude_seal:
            old = load_sealed(args.exclude_seal, "independent_event_replay_v1")
            excluded = [r["eventId"] for r in old["records"]]
        result = make_plan(args.news_db, args.event_db, args.run_id, args.mappings,
                           args.prices, limit=args.limit, excluded_ids=excluded)
    elif args.command == "forecast":
        provider = ConfiguredLLMProvider(LLMSettings(
            kind="ollama", model_id=args.model, endpoint="http://127.0.0.1:11434",
            capabilities=LLMCapabilities(context_window=8192, max_output_tokens=1536),
        ))
        if not provider.ensure_local_installed():
            parser.error("Requested local model not installed")
        result = forecast(args.news_db, args.event_db, load_sealed(args.plan, VERSION + "_plan"),
                          provider, improved=args.improved)
    elif args.command == "evaluate":
        result = evaluate(load_sealed(args.forecast, VERSION + "_forecast"), args.prices)
    else:
        result = compare_policies(load_sealed(args.baseline, VERSION + "_outcomes"),
                                  load_sealed(args.improved, VERSION + "_outcomes"))
    write_new(args.output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
