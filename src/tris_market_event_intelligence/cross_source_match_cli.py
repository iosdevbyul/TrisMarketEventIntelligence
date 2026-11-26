"""Discover cross-source event candidates from pinned source evidence, then review."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from tris_market_event_intelligence.collector import utc_now
from tris_market_event_intelligence.event_integration_relations import MONTHS
from tris_market_event_intelligence.event_profile_integration_cli import (
    REVIEW_SYSTEM,
    error_code,
    review_contract,
    validate_review,
)
from tris_market_event_intelligence.llm_runtime import (
    ConfiguredLLMProvider,
    LLMCapabilities,
    LLMExecutionError,
    LLMSettings,
)
from tris_market_event_intelligence.semantic_event_integration_cli import pinned_articles
from tris_market_event_intelligence.source_aware_event_cli import anchor, extract_facts
from tris_market_event_intelligence.unified_event_integration_cli import clock, digest, encode

VERSION = "cross_source_match_v1"
SCHEMA = """
CREATE TABLE IF NOT EXISTS cross_source_match_runs (
    scan_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, processed_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cross_source_match_candidates (
    candidate_id TEXT PRIMARY KEY, scan_id TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cross_source_match_reviews (
    review_id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL, model TEXT NOT NULL,
    processed_at TEXT NOT NULL, payload TEXT NOT NULL
);
"""
# Limited, explicit series identity; unsupported series abstain instead of guessing.
SERIES = {
    "PCE": (r"\bPCE\b|personal consumption expenditures|개인소비지출", {"bea_news"}),
    "GDP": (r"\bGDP\b|gross domestic product|국내총생산", {"bea_news"}),
    "CPI": (r"\bCPI\b|consumer price index|소비자물가", {"bls_cpi", "bls_latest"}),
    "PPI": (r"\bPPI\b|producer price index|생산자물가", {"bls_ppi", "bls_latest"}),
    "CBSI": (r"\bCBSI\b|기업심리지수|기업경기조사",
             {"bok_statistics", "bok_all"}),
    "CCSI": (r"\bCCSI\b|소비자심리지수|소비자동향조사",
             {"bok_statistics", "bok_all"}),
}
US = re.compile(r"\b(?:U\.S\.|US\b|United States\b|American\b|BEA\b)|미국|美(?=\s)", re.IGNORECASE)
KR = re.compile(r"\b(?:Bank of Korea|South Korea)\b|한국은행|한국|국내", re.IGNORECASE)
PERCENT = re.compile(r"(?<![\d.])([+-]?\d+(?:\.\d+)?)\s*(?:percent\b|%)(?!\s*(?:points?|포인트))",
                     re.IGNORECASE)
POINT = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:로|으로|기록|points?\b)", re.IGNORECASE)
YOY = re.compile(r"from (?:a |the previous )?year (?:ago)?|over the (?:past |last )?12 months|"
                 r"year.over.year|year.ago|전년\s*(?:동월|동기)?\s*대비", re.IGNORECASE)
MOM = re.compile(r"from (?:the )?(?:previous|prior) month|month.over.month|전월\s*대비", re.IGNORECASE)
ANNUAL = re.compile(r"annual(?:ized)? rate|annualized|연율", re.IGNORECASE)
CORE = re.compile(r"excluding food and energy|less food and energy|core\b|근원|"
                  r"에너지와 식료품을 제외", re.IGNORECASE)
EXPECT = re.compile(r"forecast|consensus|expected|estimate[ds]? by|예상치|전망치|예상했", re.IGNORECASE)
DOWN = re.compile(r"\b(?:decreased|declined|fell|dropped|down)\b|하락|감소", re.IGNORECASE)
UP = re.compile(r"\b(?:increased|rose|grew|up)\b|상승|증가", re.IGNORECASE)
MONTH_RE = re.compile(r"\b(" + "|".join(MONTHS) + r")\s+(\d{4})\b|"
                      r"(?:(\d{4})년\s*)?(1[0-2]|0?[1-9])월", re.IGNORECASE)
QUARTER_RE = re.compile(
    r"\b(?:Q([1-4])|([1-4])(?:st|nd|rd|th)\s+quarter|"
    r"(first|second|third|fourth)\s+quarter)[,\s]+(\d{4})\b|"
    r"(?:(\d{4})년\s*)?([1-4])분기", re.IGNORECASE)
SEC_EVENT = {
    "EARNINGS": r"announc(?:ed|ing|es) (?:its |financial )?results|quarterly results|"
                r"results for the quarter|실적\s*(?:발표|공개)|분기\s*실적",
    "EXECUTIVE_REINSTATEMENT": r"reinstated.{0,100}(?:chief executive|president|CEO)|"
                               r"(?:CEO|최고경영자|젠슨 황).{0,60}복귀|복귀.{0,60}(?:CEO|최고경영자)",
}
ISSUER = {
    "NVDA": r"\bNVIDIA\b|엔비디아", "AMD": r"\bAMD\b|Advanced Micro Devices|에이엠디",
    "AAPL": r"\bApple\b|애플", "META": r"\bMeta\b|메타",
    "MSFT": r"\bMicrosoft\b|마이크로소프트", "GOOGL": r"\bAlphabet\b|\bGoogle\b|알파벳|구글",
}


def stamp(item):
    return clock(item.get("secAcceptanceAt") or item.get("publisherDeclaredAt"))


def year_for_month(item, month):
    # Relative year is explicitly marked inferred, never a verified first-public date.
    seen = stamp(item)
    if seen is None:
        return None
    return seen.year - int(month > seen.month)


def period_cues(text, item, quarterly):
    result = set()
    for match in (QUARTER_RE if quarterly else MONTH_RE).finditer(text):
        if quarterly:
            q, digit, word, year, kr_year, kr_q = match.groups()
            quarter = int(q or digit or kr_q or (
                ("first", "second", "third", "fourth").index(word.lower()) + 1))
            month = (quarter - 1) * 3 + 1
            explicit = year or kr_year
        else:
            name, year, kr_year, kr_month = match.groups()
            month = MONTHS[name.lower()] if name else int(kr_month)
            explicit = year or kr_year
        y = int(explicit) if explicit else year_for_month(item, month)
        if y is not None:
            result.add((f"{y:04}-{month:02}", bool(explicit)))
    return result


def sentences(doc):
    # Preserve source offsets; decimal points are not sentence delimiters.
    text = doc["text"][:24000]
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+(?=[A-Z가-힣])|\n{2,}", text):
        end = match.start()
        if end > start:
            yield start, end, text[start:end]
        start = match.end()
    if start < len(text):
        yield start, len(text), text[start:]


def metric_facts(item):
    title = item["title"]
    result = []
    seen = set()
    for series, (pattern, providers) in SERIES.items():
        if item["kind"] == "ECONOMIC_RELEASE" and item["source"] not in providers:
            continue
        country = "KR" if series in {"CBSI", "CCSI"} else "US"
        if item["kind"] == "NEWS" and not (KR if country == "KR" else US).search(title):
            # Permit an explicit authority/country in the actual metric sentence below.
            title_country = False
        else:
            title_country = True
        for doc in item["documents"]:
            for start, _, sentence in sentences(doc):
                if len(sentence) > 1600 or not re.search(pattern, sentence, re.IGNORECASE):
                    continue
                if not title_country and not (KR if country == "KR" else US).search(sentence):
                    continue
                periods = period_cues(sentence, item, series == "GDP")
                if not periods:
                    periods = period_cues(title, item, series == "GDP")
                if len({p[0] for p in periods}) != 1:
                    continue
                period = sorted(periods, key=lambda p: not p[1])[0]
                # Multiple indicators in a sentence are ambiguous unless individually segmented.
                other = [k for k, (pat, _) in SERIES.items()
                         if k != series and re.search(pat, sentence, re.IGNORECASE)]
                if other:
                    continue
                variant = "CORE" if CORE.search(sentence) else "HEADLINE"
                matches = list((POINT if country == "KR" else PERCENT).finditer(sentence))
                for i, value in enumerate(matches):
                    prev = matches[i - 1].end() if i else 0
                    nxt = matches[i + 1].start() if i + 1 < len(matches) else len(sentence)
                    segment = sentence[prev:nxt]
                    # Ambiguous actual/forecast combinations abstain rather than match.
                    if EXPECT.search(segment):
                        continue
                    modes = [k for k, regex in (
                        ("YOY", YOY), ("MOM", MOM), ("ANNUALIZED", ANNUAL))
                             if regex.search(segment)]
                    if country == "KR":
                        modes = ["INDEX_LEVEL"] if len(matches) == 1 else []
                    if len(modes) != 1:
                        continue
                    # Do not infer direction from unrelated comparisons.
                    number = Decimal(value[1])
                    if country == "US":
                        down, up = bool(DOWN.search(segment)), bool(UP.search(segment))
                        if down and up or not (down or up or value[1].startswith(("+", "-"))):
                            continue
                        if down:
                            number = -abs(number)
                    key = (country, series, variant, period[0], modes[0], str(number.normalize()))
                    if key in seen:
                        continue
                    seen.add(key)
                    result.append({"key": list(key), "periodExplicit": period[1],
                                   "evidence": anchor(doc, start + prev, start + nxt)})
    return result


def sec_facts(item):
    result = []
    if item["kind"] == "SEC":
        if item.get("secForm") not in {"8-K", "8-K/A"}:
            return result
        issuers = {x["tickerCandidate"] for x in item.get("issuers", []) if x["tickerCandidate"]}
    elif item["kind"] == "NEWS":
        issuers = {key for key, pattern in ISSUER.items()
                   if re.search(pattern, item["title"], re.IGNORECASE)}
    else:
        return result
    if len(issuers) != 1:
        return result
    issuer = next(iter(issuers))
    for event, pattern in SEC_EVENT.items():
        for doc in item["documents"]:
            match = re.search(pattern, doc["text"][:12000], re.IGNORECASE)
            if not match:
                continue
            if item["kind"] == "NEWS" and not re.search(pattern, item["title"], re.IGNORECASE):
                continue
            start, end = max(0, match.start() - 220), min(len(doc["text"]), match.end() + 420)
            result.append({"key": [issuer, event], "evidence": anchor(doc, start, end)})
            break
    return result


def discover(items):
    index = defaultdict(list)
    metrics = {}
    securities = {}
    for item in items:
        facts = extract_facts(item)
        if facts["marketRelevance"] == "LOW_PRIORITY_TITLE_CUE":
            continue
        if item.get("claimKind") in {"FORECAST", "RUMOR"}:
            continue
        metrics[item["eventId"]] = metric_facts(item)
        securities[item["eventId"]] = sec_facts(item)
        if item["kind"] in {"SEC", "ECONOMIC_RELEASE"}:
            for fact in metrics[item["eventId"]] + securities[item["eventId"]]:
                index[tuple(fact["key"])].append((item, fact))
    best = {}
    for news in items:
        if news["kind"] != "NEWS":
            continue
        for fact in metrics.get(news["eventId"], []) + securities.get(news["eventId"], []):
            for official, other in index[tuple(fact["key"])]:
                if official["source"] == news["source"]:
                    continue
                if {d["rawSha256"] for d in official["documents"]} & {
                    d["rawSha256"] for d in news["documents"]
                }:
                    continue
                a, b = stamp(official), stamp(news)
                # Board archives often lack declared timestamps; do not invent them.
                if a is not None and b is not None and abs(a - b) > timedelta(days=7):
                    continue
                if len(fact["key"]) == 2 and (a is None or b is None or abs(a - b) > timedelta(days=3)):
                    continue
                score = 100 if len(fact["key"]) == 6 else 40
                score += 10 * int(bool(fact.get("periodExplicit") and other.get("periodExplicit")))
                score += 5 * int(a is not None and b is not None)
                pair = (official["eventId"], news["eventId"])
                candidate = {"leftId": pair[0], "rightId": pair[1],
                             "leftTitle": official["title"], "rightTitle": news["title"],
                             "leftUrl": official["url"], "rightUrl": news["url"],
                             "leftSource": official["source"], "rightSource": news["source"],
                             "basis": "METRIC_PERIOD_VALUE" if len(fact["key"]) == 6 else "ISSUER_EVENT_CUE",
                             "key": fact["key"], "score": score, "leftFact": other, "rightFact": fact,
                             "declaredTimesAvailable": bool(a and b),
                             "firstPublicVerified": False, "predictionEligible": False}
                if pair not in best or score > best[pair]["score"]:
                    best[pair] = candidate
    ranked = sorted(best.values(), key=lambda r: (-r["score"], r["leftId"], r["rightId"]))
    return ranked, {"metricFacts": sum(map(len, metrics.values())),
                    "securityFacts": sum(map(len, securities.values()))}


def review_passage(fact):
    return [{"id": "1", **fact["evidence"]}]


def run(news_db, event_db, run_id, provider, *, limit=10, execute=False):
    if not 0 <= limit <= 100 or news_db.samefile(event_db):
        raise ValueError("Invalid budget or aliased databases")
    items, excluded = pinned_articles(news_db, event_db, run_id)
    candidates, counts = discover(items)
    scan_id = digest(encode([VERSION, run_id, candidates]))
    report = {"version": VERSION, "runId": run_id, "scanId": scan_id,
              "admitted": len(items), "excluded": dict(excluded), **counts,
              "candidateCount": len(candidates),
              "byBasis": dict(Counter(c["basis"] for c in candidates)),
              "candidateExamples": candidates[:20], "modelCalls": 0, "cachedReviews": 0,
              "results": [], "firstPublicVerified": False, "marketReturnsAnalyzed": False,
              "predictionEligible": False}
    if not execute:
        return report
    with closing(sqlite3.connect(event_db, timeout=15)) as db:
        db.executescript(SCHEMA)
        with db:
            db.execute("INSERT OR IGNORE INTO cross_source_match_runs VALUES (?,?,?,?)",
                       (scan_id, run_id, utc_now(), encode(report)))
            for candidate in candidates:
                cid = digest(encode([scan_id, candidate]))
                db.execute("INSERT OR IGNORE INTO cross_source_match_candidates VALUES (?,?,?)",
                           (cid, scan_id, encode(candidate)))
        per_original = Counter()
        for candidate in candidates:
            cid = digest(encode([scan_id, candidate]))
            left, right = review_passage(candidate["leftFact"]), review_passage(candidate["rightFact"])
            contract = review_contract(left, right)
            packet = encode({"left": {"title": candidate["leftTitle"], "passages": left},
                             "right": {"title": candidate["rightTitle"], "passages": right}})
            rid = digest(encode([VERSION, cid, provider.model_id, REVIEW_SYSTEM, packet, contract]))
            row = db.execute("SELECT payload FROM cross_source_match_reviews WHERE review_id=?",
                             (rid,)).fetchone()
            if row:
                report["cachedReviews"] += 1
                result = json.loads(row[0])
            else:
                if report["modelCalls"] >= limit or per_original[candidate["leftId"]] >= 2:
                    continue
                report["modelCalls"] += 1
                per_original[candidate["leftId"]] += 1
                try:
                    answer = validate_review(provider.generate_json(
                        system=REVIEW_SYSTEM, user=packet, schema=contract), left, right)
                    result = {"status": "CANDIDATE", "answer": answer}
                except (LLMExecutionError, ValueError, TypeError, KeyError) as exc:
                    result = {"status": "REJECTED", "error": error_code(exc)}
                with db:
                    db.execute("INSERT OR IGNORE INTO cross_source_match_reviews VALUES (?,?,?,?,?)",
                               (rid, cid, provider.model_id, utc_now(), encode(result)))
            report["results"].append({"candidateId": cid, **candidate, **result})
    report["byRelation"] = dict(Counter(
        r.get("answer", {}).get("relation", "REJECTED") for r in report["results"]))
    report["independentHumanVerified"] = 0
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news-db", type=Path, required=True)
    parser.add_argument("--event-db", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    provider = ConfiguredLLMProvider(LLMSettings(
        kind="ollama", model_id=args.model, endpoint="http://127.0.0.1:11434",
        capabilities=LLMCapabilities(context_window=8192, max_output_tokens=1024),
    ))
    if args.execute and not provider.ensure_local_installed():
        parser.error("Requested local model not installed")
    print(json.dumps(run(args.news_db, args.event_db, args.run_id, provider,
                         limit=args.limit, execute=args.execute), ensure_ascii=False))


if __name__ == "__main__":
    main()
