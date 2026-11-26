"""Source-aware event facts and a controlled, pinned relationship diagnostic."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from tris_market_event_intelligence.collector import utc_now
from tris_market_event_intelligence.event_profile_integration_cli import (
    REVIEW_SYSTEM,
    error_code,
    review_contract,
    source_passages,
    validate_review,
)
from tris_market_event_intelligence.llm_runtime import (
    ConfiguredLLMProvider,
    LLMCapabilities,
    LLMExecutionError,
    LLMSettings,
)
from tris_market_event_intelligence.semantic_event_integration_cli import pinned_articles
from tris_market_event_intelligence.unified_event_integration_cli import digest, encode

VERSION = "source_aware_event_v1"
# Explicit form semantics do not establish the underlying transaction or its impact.
FORM_TYPES = {
    "4": "INSIDER_TRANSACTION", "4/A": "INSIDER_TRANSACTION",
    "144": "PROPOSED_SALE_NOTICE", "144/A": "PROPOSED_SALE_NOTICE",
    "424B2": "PROSPECTUS", "424B5": "PROSPECTUS",
    "S-3": "REGISTRATION", "S-3ASR": "REGISTRATION",
    "8-A12B": "SECURITY_REGISTRATION", "CERT": "LISTING_CERTIFICATION",
    "10-K": "PERIODIC_REPORT", "10-Q": "PERIODIC_REPORT",
}
INDICATORS = {
    "PCE": r"\bPCE\b|personal consumption expenditures|개인소비지출",
    "CPI": r"\bCPI\b|consumer price index|소비자물가지수|소비자물가",
    "PPI": r"\bPPI\b|producer price index|생산자물가",
    "GDP": r"\bGDP\b|gross domestic product|국내총생산",
    "BUSINESS_SENTIMENT": r"\bCBSI\b|\bESI\b|기업경기조사|기업심리지수|경제심리지수",
    "CONSUMER_SENTIMENT": r"\bCCSI\b|consumer sentiment|소비자동향조사|소비자심리지수",
    "HOME_SALES": r"new home sales|신규주택판매",
    "LOAN_RATES": r"average contract interest rates|대출금리",
    "SECURITIES_FINANCING": r"statistics on securities financing transactions",
    "PAYROLL": r"nonfarm payroll|비농업\s*고용",
    "POLICY_RATE": r"discount rate meetings|federal open market committee|기준금리",
}
SCHEMA = """
CREATE TABLE IF NOT EXISTS hybrid_event_facts (
    run_id TEXT NOT NULL, event_id TEXT NOT NULL, version TEXT NOT NULL,
    input_sha256 TEXT NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY(run_id,event_id,version,input_sha256)
);
CREATE TABLE IF NOT EXISTS hybrid_event_diagnostics (
    review_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, model TEXT NOT NULL,
    processed_at TEXT NOT NULL, payload TEXT NOT NULL
);
"""
# These are log-derived controls for this exact snapshot, not a general gold benchmark.
# The positive is a duplicate original, not independent cross-source corroboration.
CONTROL_RUN = "6e47de6a4d3cc71c406659dccecc9be23fa5acd8a1966febfb35ec7690fb94b6"
CONTROLS = (
    ("article:21446", "article:21445", "SAME_ORIGINAL", "BOK identical document hashes"),
    ("article:20872", "article:22662", "UNRELATED", "NVIDIA earnings versus SpaceX approval"),
    ("article:20792", "article:22636", "UNRELATED", "FOMC minutes versus PCE release report"),
    ("article:21102", "article:22661", "UNRELATED", "Alphabet notes versus OpenAI court case"),
)
TX_PATH = re.compile(
    r"/(?:nonDerivativeTransaction|derivativeTransaction)/transactionAmounts/"
    r"transactionAcquiredDisposedCode/value:\s*([AD])\b"
)
TX_SHARES = re.compile(
    r"/(?:nonDerivativeTransaction|derivativeTransaction)/transactionAmounts/"
    r"transactionShares/value:\s*([0-9]+(?:\.[0-9]+)?)\b"
)


def anchor(doc, start, end):
    return {"bodySha256": doc["bodySha256"], "rawSha256": doc["rawSha256"],
            "observedAt": doc["observedAt"], "start": start, "end": end,
            "text": doc["text"][start:end]}


def extract_facts(item):
    result = {"kind": item["kind"], "source": item["source"], "title": item["title"],
              "issuerCandidates": item.get("issuers", []), "form": item.get("secForm"),
              "documentType": "UNKNOWN", "indicators": [], "transactionDirections": [],
              "transactionShareFields": [], "evidence": [], "marketRelevance": "UNASSESSED",
              "firstPublicVerified": False, "predictionEligible": False}
    if item["kind"] == "SEC":
        form = item.get("secForm") or ""
        result["documentType"] = FORM_TYPES.get(form, "SEC_FILING")
        if form in {"4", "4/A"}:
            for doc in item["documents"]:
                for match in TX_PATH.finditer(doc["text"]):
                    result["transactionDirections"].append({
                        "code": match[1], "meaning": "ACQUIRED" if match[1] == "A" else "DISPOSED",
                        "evidence": anchor(doc, match.start(), match.end()),
                    })
                for match in TX_SHARES.finditer(doc["text"]):
                    result["transactionShareFields"].append({
                        "value": match[1], "evidence": anchor(doc, match.start(), match.end()),
                    })
        # Share counts and direction fields are separate, never joined by proximity.
        # Disposal is not necessarily an open-market sale.
        for doc in item["documents"]:
            for match in re.finditer(r"\bItem\s+(2\.02|5\.02)\b", doc["text"], re.IGNORECASE):
                # Checkbox/cover-page references are cues, not proof of actual item disclosure.
                result["evidence"].append({"cue": "SEC_ITEM_" + match[1],
                                           **anchor(doc, match.start(), match.end())})
    elif item["kind"] == "ECONOMIC_RELEASE":
        result["documentType"] = "OFFICIAL_RELEASE"
        for key, pattern in INDICATORS.items():
            match = re.search(pattern, item["title"], re.IGNORECASE)
            if match:
                result["indicators"].append({"id": key, "titleSpan": match.group()})
    else:
        result["documentType"] = "NEWS_ARTICLE"
        if re.match(r"\s*\[(?:포토\+?|사진|화보)\]", item["title"]):
            result["marketRelevance"] = "LOW_PRIORITY_TITLE_CUE"
    return result


def controls(items, run_id):
    if run_id != CONTROL_RUN:
        return [], {"status": "NO_LABELS_FOR_SNAPSHOT"}
    index = {item["eventId"]: item for item in items}
    pairs = []
    missing = []
    for left, right, expected, basis in CONTROLS:
        if left not in index or right not in index:
            missing.append([left, right])
            continue
        a, b = index[left], index[right]
        if expected == "SAME_ORIGINAL":
            hashes_a = {(d["bodySha256"], d["rawSha256"]) for d in a["documents"]}
            hashes_b = {(d["bodySha256"], d["rawSha256"]) for d in b["documents"]}
            if not hashes_a & hashes_b:
                missing.append([left, right])
                continue
        pairs.append((a, b, expected, basis))
    return pairs, {"status": "LOG_DERIVED_DIAGNOSTIC", "missing": missing,
                   "independentPositiveControls": 0}


def assess(answer, expected):
    if expected == "SAME_ORIGINAL":
        return answer["relation"] in {"SAME_EVENT", "REPRINT"}
    return answer["relation"] == expected


def run(news_db, event_db, run_id, provider, *, execute=False, limit=4):
    if not 0 <= limit <= 20 or news_db.samefile(event_db):
        raise ValueError("Invalid budget or aliased databases")
    items, excluded = pinned_articles(news_db, event_db, run_id)
    facts = {item["eventId"]: extract_facts(item) for item in items}
    pairs, coverage = controls(items, run_id)
    report = {"version": VERSION, "runId": run_id, "admitted": len(items),
              "excluded": dict(excluded), "documentTypes": dict(Counter(
                  f["documentType"] for f in facts.values())),
              "transactionDirectionFields": dict(Counter(
                  d["meaning"] for f in facts.values() for d in f["transactionDirections"])),
              "indicatorTitleCues": dict(Counter(
                  d["id"] for f in facts.values() for d in f["indicators"])),
              "lowPriorityTitleCues": sum(f["marketRelevance"] == "LOW_PRIORITY_TITLE_CUE"
                                          for f in facts.values()),
              "controlCoverage": coverage, "modelCalls": 0, "cachedReviews": 0,
              "results": [], "predictionEligible": False, "marketReturnsAnalyzed": False}
    if not execute:
        report["controls"] = [{"left": a["title"], "right": b["title"],
                               "expected": expected, "basis": basis}
                              for a, b, expected, basis in pairs]
        return report
    with closing(sqlite3.connect(event_db, timeout=15)) as db:
        db.executescript(SCHEMA)
        with db:
            for item in items:
                key = item["eventId"]
                payload = encode(facts[key])
                pin = digest(encode([payload, item["evidence"]]))
                db.execute("INSERT OR IGNORE INTO hybrid_event_facts VALUES (?,?,?,?,?)",
                           (run_id, key, VERSION, pin, payload))
        for a, b, expected, basis in pairs:
            left, right = source_passages(a)[:4], source_passages(b)[:4]
            contract = review_contract(left, right)
            # Expected labels are intentionally omitted from the model input.
            packet = encode({"left": {"facts": facts[a["eventId"]], "passages": left},
                             "right": {"facts": facts[b["eventId"]], "passages": right}})
            rid = digest(encode([VERSION, run_id, provider.model_id, packet, REVIEW_SYSTEM, contract]))
            row = db.execute("SELECT payload FROM hybrid_event_diagnostics WHERE review_id=?",
                             (rid,)).fetchone()
            if row:
                report["cachedReviews"] += 1
                result = json.loads(row[0])
            else:
                if report["modelCalls"] >= limit:
                    continue
                report["modelCalls"] += 1
                try:
                    answer = validate_review(provider.generate_json(
                        system=REVIEW_SYSTEM, user=packet, schema=contract), left, right)
                    result = {"answer": answer, "status": "CANDIDATE"}
                except (LLMExecutionError, ValueError, TypeError, KeyError) as exc:
                    result = {"error": error_code(exc), "status": "REJECTED"}
                with db:
                    db.execute("INSERT OR IGNORE INTO hybrid_event_diagnostics VALUES (?,?,?,?,?)",
                               (rid, run_id, provider.model_id, utc_now(), encode(result)))
            result = {**result, "leftTitle": a["title"], "rightTitle": b["title"],
                      "expected": expected, "basis": basis}
            result["matchedControl"] = (
                result["status"] == "CANDIDATE" and assess(result["answer"], expected))
            report["results"].append(result)
    report["controlsMatched"] = sum(r["matchedControl"] for r in report["results"])
    report["controlsEvaluated"] = len(report["results"])
    report["factsStored"] = len(items)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news-db", type=Path, required=True)
    parser.add_argument("--event-db", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    provider = ConfiguredLLMProvider(LLMSettings(
        kind="ollama", model_id=args.model, endpoint="http://127.0.0.1:11434",
        capabilities=LLMCapabilities(context_window=8192, max_output_tokens=1024),
    ))
    if args.execute and not provider.ensure_local_installed():
        parser.error("Requested local model not installed")
    print(json.dumps(run(args.news_db, args.event_db, args.run_id, provider,
                         execute=args.execute, limit=args.limit), ensure_ascii=False))


if __name__ == "__main__":
    main()
