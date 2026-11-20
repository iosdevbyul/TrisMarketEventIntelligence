"""Opt-in SEC issuer Form 8-K event acquisition with raw filing captures.

SEC acceptance is a provable filing receipt, not necessarily the earliest
issuer press-release timestamp or public market-availability clock.
Existing historical journal is never changed. SEC metadata alone cannot
be used as invented article body. Only downloaded source document text is
a substantive article candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from collections import Counter
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from pathlib import Path

import httpx

from tris_market_event_intelligence.collector import Article, NewsStore, utc_now

_CIK = re.compile(r"[1-9][0-9]{0,9}\Z")
_ACCESSION = re.compile(r"[0-9]{10}-[0-9]{2}-[0-9]{6}\Z")
_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\Z")
_ALLOWED = {"2.02": "EARNINGS_8K", "1.01": "MATERIAL_CONTRACT_8K"}
_MAX_SUBMISSIONS_BYTES = 8_000_000
_MAX_DOCUMENT_BYTES = 2_000_000
_MAX_FILINGS_PER_CIK = 30


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"script", "style"}:
            self.skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self.skip = False

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)


def source_body(payload: bytes) -> str:
    """Extract readable primary 8-K text, with no inferred financial facts."""
    if len(payload) > _MAX_DOCUMENT_BYTES:
        raise ValueError("SEC_DOCUMENT_TOO_LARGE")
    value = payload.decode("utf-8", errors="replace")
    if "<html" in value[:3000].lower() or "<body" in value[:3000].lower():
        parser = _Text()
        parser.feed(value)
        value = " ".join(parser.parts)
    value = re.sub(r"\s+", " ", html.unescape(value)).strip()
    return value[:12000]


def verified_sec_acceptance(value: str) -> datetime:
    """Only explicitly offset-aware SEC acceptance timestamps are accepted."""
    if not isinstance(value, str):
        raise ValueError("SEC_ACCEPTANCE_CLOCK_UNVERIFIED")
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("SEC_ACCEPTANCE_CLOCK_UNVERIFIED") from exc
    if ts.utcoffset() is None:
        raise ValueError("SEC_ACCEPTANCE_CLOCK_UNVERIFIED")
    return ts.astimezone(UTC)


def submission_candidates(
    payload: dict, *, cik: int, since: date, until: date,
) -> tuple[list[dict], Counter[str]]:
    if not isinstance(payload, dict) or int(payload.get("cik", -1)) != cik:
        raise ValueError("SEC_CIK_RESPONSE_MISMATCH")
    recent = (payload.get("filings") or {}).get("recent") or {}
    keys = ("accessionNumber", "form", "acceptanceDateTime", "primaryDocument", "items")
    if not all(isinstance(recent.get(k), list) for k in keys):
        raise ValueError("SEC_SUBMISSION_SCHEMA_MISSING")
    size = len(recent["accessionNumber"])
    if any(len(recent[k]) != size for k in keys):
        raise ValueError("SEC_SUBMISSION_LENGTH_MISMATCH")
    candidates = []
    excluded: Counter[str] = Counter()
    for i in range(size):
        if recent["form"][i] != "8-K":
            excluded["NOT_FORM_8K"] += 1
            continue
        items = set(re.findall(r"\d+\.\d+", str(recent["items"][i])))
        eligible = sorted(items & _ALLOWED.keys())
        if not eligible:
            excluded["NOT_EARNINGS_OR_MATERIAL_CONTRACT"] += 1
            continue
        try:
            accepted = verified_sec_acceptance(recent["acceptanceDateTime"][i])
        except ValueError:
            excluded["ACCEPTANCE_TIME_UNVERIFIED"] += 1
            continue
        if not since <= accepted.date() <= until:
            excluded["OUTSIDE_DATE_RANGE"] += 1
            continue
        accession = str(recent["accessionNumber"][i])
        filename = str(recent["primaryDocument"][i])
        if not _ACCESSION.fullmatch(accession) or (
            not _FILENAME.fullmatch(filename) or ".." in filename
        ):
            excluded["INVALID_SEC_ARCHIVE_PATH"] += 1
            continue
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/"
            f"{accession.replace('-', '')}/{filename}"
        )
        candidates.append({
            "cik": cik,
            "issuer": str(payload.get("name") or ""),
            "tickerHints": list(payload.get("tickers") or []),
            "accession": accession,
            "sourceUrl": url,
            "form": "8-K", "items": eligible,
            "eventCategory": _ALLOWED[eligible[0]],
            "acceptedAt": accepted.isoformat(),
            "timeMeaning": "SEC_FILING_ACCEPTED_NOT_FIRST_ISSUER_RELEASE",
            "issuerReleaseTimeVerified": False,
        })
    candidates.sort(key=lambda item: (item["acceptedAt"], item["accession"]))
    return candidates[:_MAX_FILINGS_PER_CIK], excluded


def collect_sec_filings(
    *, db: Path, ciks: list[int], since: date, until: date,
    user_agent: str, import_files: bool = False,
    client: httpx.Client | None = None,
) -> dict:
    """Fetch official SEC metadata, optionally ingest raw 8-K primary documents."""
    if not user_agent.strip() or "@" not in user_agent or len(user_agent) > 200:
        raise ValueError("SEC_IDENTIFYING_USER_AGENT_REQUIRED")
    if not ciks or len(ciks) > 12 or len(set(ciks)) != len(ciks):
        raise ValueError("INVALID_CIK_LIST")
    if any(not _CIK.fullmatch(str(cik)) for cik in ciks):
        raise ValueError("INVALID_CIK")
    if not since < until or (until - since).days > 730:
        raise ValueError("INVALID_DATE_WINDOW")
    if db.is_symlink():
        raise ValueError("SYMLINK_DB_NOT_ALLOWED")
    if import_files and not db.is_file():
        raise ValueError("EXISTING_ARTICLE_DATABASE_REQUIRED")
    owned = client is None
    http = client or httpx.Client(
        timeout=20, follow_redirects=False,
        headers={"User-Agent": user_agent, "Accept": "application/json"},
    )
    store = NewsStore(db) if import_files else None
    result = []
    try:
        for cik in ciks:
            endpoint = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
            response = http.get(endpoint, headers={"User-Agent": user_agent})
            response.raise_for_status()
            if len(response.content) > _MAX_SUBMISSIONS_BYTES:
                raise ValueError("SEC_SUBMISSIONS_TOO_LARGE")
            candidates, rejected = submission_candidates(
                response.json(), cik=cik, since=since, until=until,
            )
            inserted = 0
            documents_unavailable = 0
            if store is not None:
                source = f"sec_8k_{cik}"
                for row in candidates:
                    filing = http.get(row["sourceUrl"], headers={
                        "User-Agent": user_agent, "Accept": "text/html",
                    })
                    if filing.status_code != 200 or (
                        len(filing.content) > _MAX_DOCUMENT_BYTES
                    ):
                        documents_unavailable += 1
                        continue
                    body = source_body(filing.content)
                    if len(body) < 250:
                        documents_unavailable += 1
                        continue
                    fetched_at = utc_now()
                    article = Article(
                        source=source, external_id=row["accession"],
                        url=row["sourceUrl"],
                        title=(f'{row["issuer"]} SEC 8-K '
                               f'Item {",".join(row["items"])} '
                               f'{row["eventCategory"]}'),
                        summary=body,
                        published_at=row["acceptedAt"],
                        detected_at=fetched_at,
                    )
                    inserted += store.save(
                        source, row["sourceUrl"], filing.content,
                        [article], fetched_at, None, None,
                    )
            result.append({
                "cik": cik, "candidates": candidates,
                "rejections": dict(sorted(rejected.items())),
                "candidateCount": len(candidates),
                "inserted": inserted,
                "documentUnavailable": documents_unavailable,
            })
    finally:
        if owned:
            http.close()
    return {
        "schemaVersion": "official_sec_8k_event_acquisition_v1",
        "mode": "IMPORT_RAW_PRIMARY_DOCUMENTS" if import_files else "PREVIEW_METADATA_ONLY",
        "sourceDatabase": str(db),
        "dateWindow": [since.isoformat(), until.isoformat()],
        "issuers": result,
        "candidateTotal": sum(x["candidateCount"] for x in result),
        "importedTotal": sum(x["inserted"] for x in result),
        "firstIssuerReleaseCertified": False,
        "SECReceiptTimeNotFirstPublicRelease": True,
        "existingTrainingJournalModified": False,
        "marketReturnsAnalyzed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--ciks", required=True, help="Comma-separated numeric SEC CIKs")
    parser.add_argument("--since", type=date.fromisoformat, required=True)
    parser.add_argument("--until", type=date.fromisoformat, required=True)
    parser.add_argument("--import-files", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Refusing to overwrite existing SEC acquisition report")
    agent = os.environ.get("SEC_USER_AGENT", "")
    ciks = [int(v.strip()) for v in args.ciks.split(",")]
    report = collect_sec_filings(
        db=args.db, ciks=ciks, since=args.since, until=args.until,
        user_agent=agent, import_files=args.import_files,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "candidateTotal": report["candidateTotal"],
        "importedTotal": report["importedTotal"],
        "reportPath": str(args.output),
        "SECReceiptTimeNotFirstPublicRelease": True,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
