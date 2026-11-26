"""Evidence-pinned event chronology and existing commitment/outcome bridge."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from contextlib import closing
from pathlib import Path

from tris_market_event_intelligence.committed_forecast_market_outcomes_cli import evaluate
from tris_market_event_intelligence.event_integration_relations import MONTHS
from tris_market_event_intelligence.semantic_event_integration_cli import pinned_articles
from tris_market_event_intelligence.source_aware_event_cli import anchor
from tris_market_event_intelligence.unified_event_integration_cli import clock, digest, encode, readonly

VERSION = "event_timeline_bridge_v1"
POSITIVE = {"SAME_EVENT", "REPRINT", "FOLLOW_UP", "CORRECTION"}
DATE_EN = re.compile(r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2}),?\s+(\d{4})\b", re.I)
DATE_KR = re.compile(r"(?:(\d{4})년\s*)?(?:(\d{1,2})월\s*)?(\d{1,2})일")
ACTION = re.compile(r"reinstat|board meeting|이사회|복귀|승인|announc|발표", re.I)
REACTION = re.compile(
    r"(?:주가|거래되고|거래됐다|전 거래일|shares|stock price|trading at|증시).{0,140}"
    r"(?:\d+(?:\.\d+)?\s*%|상승|하락)|"
    r"\d+(?:\.\d+)?\s*%.{0,100}(?:상승|하락|거래되고|shares|stock)", re.I
)
TICKER_KR = re.compile(r"([가-힣A-Za-z][가-힣A-Za-z0-9 ]{0,25})\((\d{6})\)")


def event_dates(item, fact):
    text = fact["evidence"]["text"]
    result = []
    for match in DATE_EN.finditer(text):
        if ACTION.search(text[max(0, match.start() - 90):match.end() + 130]):
            result.append({"date": f"{match[3]}-{MONTHS[match[1].lower()]:02}-{int(match[2]):02}",
                           "basis": "EXPLICIT_ACTION_NEARBY", "text": match.group()})
    if item["kind"] == "NEWS":
        published = clock(item.get("publisherDeclaredAt"))
        for match in DATE_KR.finditer(text):
            if not ACTION.search(text[match.end():match.end() + 65]):
                continue
            if published is None and (not match[1] or not match[2]):
                continue
            year = int(match[1]) if match[1] else published.year
            month = int(match[2]) if match[2] else published.month
            day = int(match[3])
            if 1 <= month <= 12 and 1 <= day <= 31:
                result.append({"date": f"{year:04}-{month:02}-{day:02}",
                               "basis": "EXPLICIT_ACTION_NEARBY" if match[1] and match[2]
                               else "RELATIVE_DATE_CANDIDATE", "text": match.group()})
    return result


def source_record(item, fact):
    return {"eventId": item["eventId"], "source": item["source"], "kind": item["kind"],
            "title": item["title"], "publisherDeclaredAt": item.get("publisherDeclaredAt"),
            "secAcceptanceAt": item.get("secAcceptanceAt"),
            "observedAt": item.get("observedAt"),
            "bodyObservedAt": fact["evidence"]["observedAt"],
            "actionDateCues": event_dates(item, fact), "evidence": fact["evidence"]}


def reaction_targets(item):
    observations = []
    targets = []
    for doc in item["documents"]:
        for match in REACTION.finditer(doc["text"][:12000]):
            observations.append(anchor(doc, match.start(), match.end()))
        for match in TICKER_KR.finditer(doc["text"][:12000]):
            targets.append({"name": match[1].strip(), "symbolCandidate": match[2],
                            "marketCandidate": "KR", "role": "MENTIONED_LISTED_TARGET_CANDIDATE",
                            "mappingVerified": False,
                            "evidence": anchor(doc, max(0, match.start() - 80),
                                               min(len(doc["text"]), match.end() + 240))})
    return observations[:20], targets[:20]


def build_record(candidate, answer, left, right):
    a = source_record(left, candidate["leftFact"])
    b = source_record(right, candidate["rightFact"])
    dates_a = {d["date"] for d in a["actionDateCues"]}
    dates_b = {d["date"] for d in b["actionDateCues"]}
    conflict = bool(dates_a and dates_b and dates_a.isdisjoint(dates_b))
    observations, targets = reaction_targets(right)
    direct = [{"symbolCandidate": x["tickerCandidate"], "role": "OFFICIAL_ISSUER_CANDIDATE",
               "mappingVerified": False} for x in left.get("issuers", []) if x["tickerCandidate"]]
    return {"candidateId": candidate["candidateId"], "relation": answer["relation"],
            "sources": [a, b], "actionDateDisagreement": conflict,
            "chronologyStatus": "UNRESOLVED_DATE_CONFLICT" if conflict else "SOURCE_CLOCKS_ONLY",
            "reactionObservations": observations, "targetCandidates": direct + targets,
            "marketReactionInNews": bool(observations),
            "earlierForecastMayUseLaterNews": False,
            "firstPublicVerified": False, "predictionEligible": False,
            "trainingEligible": False}


def attach_outcomes(records, outcomes):
    # Outcome evaluation remains tied to original precommitments, never to a new event date.
    rows = outcomes.get("rows", [])
    indexed = {}
    for row in rows:
        indexed.setdefault((row["articleId"], row["bodySha256"]), []).append(row)
    for record in records:
        attached = []
        for source in record["sources"]:
            key = (int(source["eventId"].split(":")[1]), source["evidence"]["bodySha256"])
            attached.extend(indexed.get(key, []))
        record["existingForecastOutcomes"] = attached
        record["outcomeAttribution"] = "ORIGINAL_ARTICLE_ONLY_NOT_EVENT_CAUSALITY"
        record["marketBridgeStatus"] = "EXISTING_OUTCOMES_ATTACHED" if attached else "NO_MATCHING_COMMITMENT"
    return sum(len(r["existingForecastOutcomes"]) for r in records)


def run(news_db, event_db, run_id, scan_id, prices=None, manifest=None):
    if news_db.samefile(event_db):
        raise ValueError("Distinct databases required")
    items, rejected = pinned_articles(news_db, event_db, run_id)
    index = {i["eventId"]: i for i in items}
    records = []
    blocked = Counter()
    with closing(readonly(event_db)) as db:
        scan = db.execute("SELECT run_id FROM cross_source_match_runs WHERE scan_id=?", (scan_id,)).fetchone()
        if scan is None or scan[0] != run_id:
            raise ValueError("Unknown or mismatched scan")
        rows = db.execute("""
            SELECT c.candidate_id,c.payload,r.payload
            FROM cross_source_match_candidates c JOIN cross_source_match_reviews r
            ON c.candidate_id=r.candidate_id WHERE c.scan_id=?
            ORDER BY c.candidate_id,r.review_id
        """, (scan_id,))
        for cid, raw, review in rows:
            candidate, result = json.loads(raw), json.loads(review)
            if cid != digest(encode([scan_id, candidate])):
                raise ValueError("Candidate hash mismatch")
            answer = result.get("answer", {})
            if result.get("status") != "CANDIDATE" or answer.get("relation") not in POSITIVE:
                blocked["NON_POSITIVE_REVIEW"] += 1
                continue
            left, right = index.get(candidate["leftId"]), index.get(candidate["rightId"])
            if left is None or right is None:
                blocked["PIN_NOT_AVAILABLE"] += 1
                continue
            valid = True
            for item, fact in ((left, candidate["leftFact"]), (right, candidate["rightFact"])):
                evidence = fact["evidence"]
                if not any(d["bodySha256"] == evidence["bodySha256"]
                           and d["rawSha256"] == evidence["rawSha256"]
                           and d["observedAt"] == evidence["observedAt"]
                           and d["text"][evidence["start"]:evidence["end"]] == evidence["text"]
                           for d in item["documents"]):
                    valid = False
            if not valid:
                blocked["PIN_SPAN_MISMATCH"] += 1
                continue
            records.append(build_record({**candidate, "candidateId": cid}, answer, left, right))
    report = {"version": VERSION, "runId": run_id, "scanId": scan_id, "excluded": dict(rejected),
              "blocked": dict(blocked), "records": records, "linkedRecords": len(records),
              "dateConflicts": sum(r["actionDateDisagreement"] for r in records),
              "reactionBearingNews": sum(r["marketReactionInNews"] for r in records),
              "firstPublicVerified": False, "predictionEligible": False, "trainingEligible": False,
              "marketReturnsAnalyzed": False, "attachedOutcomeRows": 0}
    if prices is not None:
        outcomes = evaluate(news_db, prices, manifest)
        report["attachedOutcomeRows"] = attach_outcomes(records, outcomes)
        report["marketReturnsAnalyzed"] = any(
            row.get("status") == "OBSERVED_UNAUTHENTICATED_PRECOMMITMENT"
            for r in records for row in r["existingForecastOutcomes"])
        report["existingEvaluatorStatus"] = outcomes.get("status")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news-db", type=Path, required=True)
    parser.add_argument("--event-db", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--price-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.news_db, args.event_db, args.run_id, args.scan_id,
                 args.prices, args.price_manifest)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({**report, "records": report["records"][:10]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
