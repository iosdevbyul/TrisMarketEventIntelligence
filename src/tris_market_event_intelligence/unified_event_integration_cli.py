"""Read-only source integration into an isolated, versioned event candidate ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tris_market_event_intelligence.collector import utc_now
from tris_market_event_intelligence.event_integration_relations import cross_source_edges, features
from tris_market_event_intelligence.global_economic_collection_cli import BOARDS, FEEDS
from tris_market_event_intelligence.korean_official_release_cli import BOARDS as KOREAN_BOARDS
from tris_market_event_intelligence.multi_events import extract_news_event
from tris_market_event_intelligence.news_content_quality import classify
from tris_market_event_intelligence.news_source_provider import PUBLISHERS
from tris_market_event_intelligence.sec_operational_content import canonical_index, index_identity
from tris_market_event_intelligence.story_index import canonical_url, title_fingerprint

VERSION = "unified_event_v2"
OFFICIAL = set(BOARDS) | set(FEEDS) | set(KOREAN_BOARDS) | {
    "fed_all", "fed_monetary", "bok_all", "bok_monetary", "bok_statistics",
    "bok_corrections", "bls_latest", "sec_press",
}
ISSUERS = {
    "1045810": ("NVDA", r"\bNVIDIA\b|엔비디아"),
    "1326801": ("META", r"\bMeta Platforms\b|메타 플랫폼스"),
    "1652044": ("GOOGL", r"\bAlphabet\b|알파벳"),
    "2488": ("AMD", r"\bAdvanced Micro Devices\b|\bAMD\b"),
    "320193": ("AAPL", r"\bApple Inc\b|애플"),
    "789019": ("MSFT", r"\bMicrosoft\b|마이크로소프트"),
}
CORRECTION = re.compile(r"(?i)\b(?:correction|corrected|retraction|retracted|denial|denies)\b|정정|반박|부인")
SCHEMA = """
CREATE TABLE IF NOT EXISTS integration_runs (
    run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, cutoff TEXT NOT NULL,
    report TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS event_candidates (
    run_id TEXT NOT NULL REFERENCES integration_runs(run_id),
    event_id TEXT NOT NULL, kind TEXT NOT NULL, source TEXT NOT NULL,
    category TEXT NOT NULL, quality TEXT NOT NULL, observed_at TEXT,
    payload TEXT NOT NULL, PRIMARY KEY(run_id,event_id)
);
CREATE TABLE IF NOT EXISTS event_candidate_links (
    run_id TEXT NOT NULL, left_id TEXT NOT NULL, right_id TEXT NOT NULL,
    relationship TEXT NOT NULL, evidence TEXT NOT NULL,
    PRIMARY KEY(run_id,left_id,right_id),
    FOREIGN KEY(run_id,left_id) REFERENCES event_candidates(run_id,event_id),
    FOREIGN KEY(run_id,right_id) REFERENCES event_candidates(run_id,event_id)
);
"""


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def clock(value):
    try:
        parsed = datetime.fromisoformat(value or "")
        return parsed.astimezone(UTC) if parsed.utcoffset() is not None else None
    except (ValueError, TypeError):
        return None


def readonly(path):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    db.execute("BEGIN")
    return db


def tables(db):
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def source_kind(source):
    if re.fullmatch(r"sec_(?:8k|filing)_\d{1,10}", source):
        return "SEC"
    if source in PUBLISHERS:
        return "NEWS"
    if source in OFFICIAL:
        return "ECONOMIC_RELEASE"
    return "UNKNOWN"


def article_candidates(db, cutoff, limit):
    known = tables(db)
    full = "article_full_text" in known
    story = "story_members" in known
    columns = (
        "f.status,f.observed_at,f.body,f.body_sha256,f.raw_sha256,f.raw_payload,f.sparse_pages"
        if full else "NULL,NULL,NULL,NULL,NULL,NULL,NULL"
    )
    sql = (
        "SELECT a.id,a.source,a.external_id,a.url,a.title,a.summary,a.published_at,a.detected_at,"
        "c.sha256,c.payload,c.fetched_at," + columns +
        (",s.story_id" if story else ",NULL") +
        " FROM articles a LEFT JOIN captures c ON c.id=a.capture_id" +
        (" LEFT JOIN article_full_text f ON f.article_id=a.id" if full else "") +
        (" LEFT JOIN story_members s ON s.article_id=a.id" if story else "") +
        " ORDER BY a.id DESC LIMIT ?"
    )
    capture_cache = {}
    for row in db.execute(sql, (limit,)):
        (aid, source, external, url, title, summary, published, detected,
         cap_sha, cap_raw, cap_time, status, observed, body, body_sha, raw_sha,
         raw, sparse, story_id) = tuple(row)
        seen, captured, pub, obs = map(clock, (detected, cap_time, published, observed))
        if seen is None or seen > cutoff:
            continue
        kind = source_kind(source)
        if cap_sha not in capture_cache:
            capture_cache[cap_sha] = isinstance(cap_raw, bytes) and digest(cap_raw) == cap_sha
        cap_ok = capture_cache[cap_sha] and captured is not None and captured <= cutoff
        quality = "METADATA_ONLY"
        body_valid = (
            status == "COMPLETE" and isinstance(body, str) and bool(body_sha)
            and isinstance(raw, bytes) and digest(raw) == raw_sha
            and obs is not None and seen <= obs <= cutoff
            and classify(body, url=url, title=title, digest=body_sha, sparse=sparse)
            == "TEXT_CANDIDATE"
        )
        evidence = []
        relation_text = ""
        if body_valid and cap_ok:
            quality = "TEXT_CANDIDATE"
            relation_text = body
            evidence.append({"bodySha256": body_sha, "rawSha256": raw_sha,
                             "observedAt": observed, "scope": "CAPTURED_BODY"})
        elif kind == "ECONOMIC_RELEASE" and {
            "official_release_documents", "official_release_originals"
        } <= known:
            for doc in db.execute("""
                SELECT d.url,d.sha256,d.body,d.status,d.sparse_pages,o.payload,o.observed_at
                FROM official_release_documents d JOIN official_release_originals o
                ON o.url=d.url AND o.sha256=d.sha256 WHERE d.article_id=?
                ORDER BY o.observed_at,d.url,d.sha256
            """, (aid,)):
                target, sha, text, state, pages, payload, stamp = tuple(doc)
                when = clock(stamp)
                if (when is None or when > cutoff or not isinstance(payload, bytes)
                        or digest(payload) != sha):
                    continue
                usable = state == "TEXT_EXTRACTED" and classify(text, sparse=pages) == "TEXT_CANDIDATE"
                if usable:
                    relation_text += "\n" + text[:50000]
                evidence.append({"url": target, "rawSha256": sha, "observedAt": stamp,
                                 "bodySha256": digest(text) if usable else None,
                                 "scope": "ATTACHMENT_TEXT" if usable else "RAW_DOCUMENT"})
            if cap_ok and any(e["scope"] == "ATTACHMENT_TEXT" for e in evidence):
                quality = "ATTACHMENT_TEXT_CANDIDATE"
        if not cap_ok:
            quality = "DISCOVERY_UNVERIFIED"
        if kind == "UNKNOWN":
            quality = "UNKNOWN_SOURCE"
        identity = None
        issuers = []
        form = None
        if kind == "SEC":
            try:
                identity = index_identity(canonical_index(url, source))
            except ValueError:
                pass
            if identity:
                cik, _accession = identity
                cik = str(int(cik))
                issuers = [{"cik": cik, "tickerCandidate": ISSUERS.get(cik, (None, ""))[0],
                            "basis": "SEC_ARCHIVE_IDENTITY"}]
                try:
                    info = json.loads(summary or "{}")
                    form = info.get("form") if isinstance(info, dict) else None
                except ValueError:
                    pass
            else:
                quality = "SEC_IDENTITY_UNVERIFIED"
        elif cap_ok:
            for cik, (ticker, pattern) in ISSUERS.items():
                if re.search(pattern, title, re.IGNORECASE):
                    issuers.append({"cik": cik, "tickerCandidate": ticker,
                                    "basis": "TITLE_ALIAS_CANDIDATE"})
        extracted = extract_news_event(source, title, body if body_valid else None) if cap_ok else None
        category = extracted.event.classification if extracted else (
            "SEC_FILING" if kind == "SEC" else "ECONOMIC_RELEASE" if kind == "ECONOMIC_RELEASE"
            else "UNKNOWN"
        )
        safe_url = None
        try:
            safe_url = canonical_url(url)
        except ValueError:
            pass
        stamps = [clock(e["observedAt"]) for e in evidence]
        available = max([seen, captured, *stamps]) if cap_ok and evidence else None
        yield {
            "eventId": "article:" + str(aid), "kind": kind, "source": source,
            "articleId": aid, "externalId": external, "url": url, "title": title,
            "canonicalUrl": safe_url, "storyCandidateId": story_id,
            "captureSha256": cap_sha, "captureVerified": bool(cap_ok),
            "publisherDeclaredAt": published if kind != "SEC" else None,
            "secAcceptanceAt": published if kind == "SEC" else None,
            "detectedAt": detected, "captureObservedAt": cap_time,
            "firstPublicAt": None, "observedAt": available.isoformat() if available else None,
            "declaredClockConsistent": bool(pub and pub <= seen),
            "category": category, "tags": list(extracted.tags) if extracted else [],
            "claimKind": extracted.claim_kind if extracted else "UNKNOWN",
            "quality": quality, "evidence": evidence, "issuers": issuers,
            "secIdentity": list(identity) if identity else None, "secForm": form,
            "titleFingerprint": title_fingerprint(title, published) if pub else None,
            "novelty": "CORRECTION_CUE" if CORRECTION.search(title) else "UNKNOWN",
            "integrationFeatures": features(kind, source, title, relation_text) if cap_ok else {},
            "predictionEligible": False, "trainingEligible": False,
        }


def macro_candidates(db, cutoff, limit):
    verified = {}
    for row in db.execute("""
        SELECT m.source,m.series,m.period,m.vintage_start,m.vintage_end,m.value,
               m.fetched_at,m.raw_digest,r.payload,r.fetched_at
        FROM macro_observation m LEFT JOIN macro_raw r
        ON r.source=m.source AND r.series=m.series AND r.digest=m.raw_digest
        ORDER BY m.fetched_at DESC,m.source,m.series,m.period DESC LIMIT ?
    """, (limit,)):
        source, series, period, start, end, value, fetched, sha, raw, raw_time = tuple(row)
        seen, captured = clock(fetched), clock(raw_time)
        if seen is None or seen > cutoff:
            continue
        key = (source, series, sha)
        if key not in verified:
            verified[key] = isinstance(raw, str) and digest(raw) == sha
        valid = verified[key] and captured is not None and captured <= cutoff
        yield {
            "eventId": "macro:" + digest(encode([source, series, period, start, end, value])),
            "kind": "ECONOMIC_OBSERVATION", "source": source, "series": series,
            "period": period, "vintageStart": start or None, "vintageEnd": end or None,
            "value": value, "rawSha256": sha, "rawVerified": bool(valid),
            "observedAt": max(seen, captured).isoformat() if valid else None,
            "firstPublicAt": None, "category": "ECONOMIC_INDICATOR",
            "quality": "NUMERIC_CONTEXT_ONLY" if valid else "RAW_UNVERIFIED",
            "predictionEligible": False, "trainingEligible": False,
        }


def links(events):
    """Direct bounded edges; no transitive identity or first-disclosure inference."""
    indexes = defaultdict(list)
    for item in events:
        if item["kind"] == "ECONOMIC_OBSERVATION":
            if item["quality"] == "NUMERIC_CONTEXT_ONLY":
                indexes[("SERIES_PERIOD", item["source"], item["series"], item["period"])].append(item)
            continue
        if not item["captureVerified"] or item["kind"] == "UNKNOWN":
            continue
        if item["secIdentity"]:
            indexes[("ACCESSION", *item["secIdentity"])].append(item)
        elif item["kind"] != "SEC":
            if item["canonicalUrl"]:
                indexes[("URL", item["canonicalUrl"])].append(item)
            if item["titleFingerprint"]:
                indexes[("TITLE", item["titleFingerprint"])].append(item)
            if item["storyCandidateId"] is not None:
                indexes[("STORY", item["storyCandidateId"])].append(item)
        if item["quality"] not in {"TEXT_CANDIDATE", "ATTACHMENT_TEXT_CANDIDATE"}:
            continue
        if item["claimKind"] in {"FORECAST", "RUMOR"}:
            continue
        observed = clock(item["observedAt"])
        declared = clock(item["publisherDeclaredAt"] or item["secAcceptanceAt"])
        if declared is None or observed is None or not item["declaredClockConsistent"]:
            continue
        for issuer in item["issuers"]:
            indexes[("ISSUER", issuer["cik"])].append(item)
    emitted = set()
    for left, right, relationship, basis in cross_source_edges(events):
        pair = tuple(sorted((left["eventId"], right["eventId"])))
        if pair in emitted:
            continue
        emitted.add(pair)
        yield pair[0], pair[1], relationship, encode({
            "basis": basis, "referenceId": left["eventId"], "reportId": right["eventId"],
            "economicEventVerified": False, "firstPublicVerified": False, "causalityVerified": False,
        })
    for key, group in sorted(indexes.items(), key=lambda entry: str(entry[0])):
        # Bounded direct candidate links, rather than quadratic all-pairs groups.
        ordered = sorted(group, key=lambda e: (e["observedAt"] or "", e["eventId"]))
        for i, current in enumerate(ordered):
            for prior in ordered[max(0, i - 12):i]:
                pair = tuple(sorted((prior["eventId"], current["eventId"])))
                if pair in emitted:
                    continue
                mode = key[0]
                if mode == "ISSUER":
                    if current["kind"] == prior["kind"]:
                        continue
                    if current["category"] != prior["category"] or current["category"] == "UNKNOWN":
                        continue
                    a = clock(prior["publisherDeclaredAt"] or prior["secAcceptanceAt"])
                    b = clock(current["publisherDeclaredAt"] or current["secAcceptanceAt"])
                    if a is None or b is None or abs(a - b) > timedelta(days=2):
                        continue
                relation = {
                    "ACCESSION": "SAME_FILING_IDENTITY",
                    "URL": "SAME_URL_CANDIDATE",
                    "TITLE": "REPRINT_CANDIDATE",
                    "STORY": "STORY_GROUP_CANDIDATE",
                    "SERIES_PERIOD": "OBSERVATION_VERSION_CANDIDATE",
                    "ISSUER": "CROSS_SOURCE_EVENT_CANDIDATE",
                }[mode]
                if mode == "ISSUER" and current["novelty"] == "CORRECTION_CUE":
                    relation = "CORRECTION_CANDIDATE"
                emitted.add(pair)
                yield pair[0], pair[1], relation, encode({
                    "basis": list(key), "economicEventVerified": False,
                    "firstPublicVerified": False, "causalityVerified": False,
                })


def integrate(news, macro, output, *, cutoff, article_limit=10000, macro_limit=50000):
    paths = (news, macro, output)
    if any(p.is_symlink() for p in paths) or len({p.resolve() for p in paths}) != 3:
        raise ValueError("Distinct non-symlink databases required")
    if not news.is_file() or not macro.is_file():
        raise ValueError("Both source databases must exist")
    if output.exists() and any(output.samefile(p) for p in (news, macro)):
        raise ValueError("Output must not alias source databases")
    if cutoff.utcoffset() is None or cutoff > datetime.now(UTC):
        raise ValueError("Valid timezone-aware cutoff required")
    if not 1 <= article_limit <= 50000 or not 1 <= macro_limit <= 100000:
        raise ValueError("Invalid source scan limits")
    with closing(readonly(news)) as db:
        total_articles = db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        events = list(article_candidates(db, cutoff, article_limit))
    with closing(readonly(macro)) as db:
        total_macro = db.execute("SELECT COUNT(*) FROM macro_observation").fetchone()[0]
        events.extend(macro_candidates(db, cutoff, macro_limit))
    edges = list(links(events))
    # Immutable snapshot identity; exact reruns reuse it instead of duplicating rows.
    cutoff_text = cutoff.astimezone(UTC).isoformat()
    run_id = digest(encode([VERSION, cutoff_text, events, edges]))
    report = {
        "runId": run_id, "version": VERSION, "cutoff": cutoff_text,
        "byKind": dict(Counter(e["kind"] for e in events)),
        "byQuality": dict(Counter(e["quality"] for e in events)),
        "byRelationship": dict(Counter(e[2] for e in edges)),
        "totalArticles": total_articles, "totalMacroObservations": total_macro,
        "articleScanTruncated": total_articles > article_limit,
        "macroScanTruncated": total_macro > macro_limit,
        "eventCandidates": len(events), "candidateLinks": len(edges),
        "crossSourceExamples": [
            {"left": left, "right": right, "relationship": relation, "evidence": json.loads(evidence)}
            for left, right, relation, evidence in edges
            if relation in {"SEC_ACCESSION_MENTION_CANDIDATE", "INDICATOR_REPORT_CANDIDATE",
                            "RELEASE_SERIES_PERIOD_CONTEXT", "SHARED_QUOTATION_CANDIDATE"}
        ][:20],
        "firstPublicVerified": 0, "predictionEligible": 0, "trainingEligible": 0,
        "sourceDatabasesModified": False, "modelCalls": 0, "marketReturnsAnalyzed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(output, timeout=15)) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript(SCHEMA)
        with db:
            inserted = db.execute("INSERT OR IGNORE INTO integration_runs VALUES (?,?,?,?)",
                                  (run_id, utc_now(), cutoff_text, encode(report))).rowcount
            if inserted:
                db.executemany("INSERT INTO event_candidates VALUES (?,?,?,?,?,?,?,?)", [
                    (run_id, e["eventId"], e["kind"], e["source"], e["category"],
                     e["quality"], e["observedAt"], encode(e)) for e in events
                ])
                db.executemany("INSERT INTO event_candidate_links VALUES (?,?,?,?,?)",
                               [(run_id, *e) for e in edges])
    return {**report, "reusedSnapshot": not bool(inserted)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news-db", required=True, type=Path)
    parser.add_argument("--macro-db", required=True, type=Path)
    parser.add_argument("--event-db", required=True, type=Path)
    parser.add_argument("--as-of", type=datetime.fromisoformat)
    parser.add_argument("--article-limit", type=int, default=10000)
    parser.add_argument("--macro-limit", type=int, default=50000)
    args = parser.parse_args()
    report = integrate(args.news_db, args.macro_db, args.event_db,
                       cutoff=args.as_of or datetime.now(UTC),
                       article_limit=args.article_limit, macro_limit=args.macro_limit)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
