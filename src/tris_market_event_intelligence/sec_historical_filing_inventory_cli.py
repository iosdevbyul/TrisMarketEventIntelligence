"""Bounded, source-pinned SEC submissions filing inventory.

SEC company history provides *candidate filings*, not guaranteed source
publication first-seen time, issuer-news text, or causal price labels.
User-Agent is operator supplied. No trade or automatic research activation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import date
from pathlib import Path

import httpx

from tris_market_event_intelligence.historical_source_acquisition import (
    create_private_artifact,
)

CIK = "0001045810"
BASE = "https://data.sec.gov/submissions/"
HISTORY_NAME = re.compile(r"^CIK0001045810-submissions-\d{3}\.json$")
ACCESSION = re.compile(r"^\d{10}-\d{2}-\d{6}$")
FORM_ALLOWED = frozenset(("8-K", "10-Q", "10-K"))
MAX_SOURCE_BYTES = 8_000_000
MAX_HISTORY_FILES = 8
MAX_ROWS = 15_000
MAX_PAGES = 1 + MAX_HISTORY_FILES
MAX_USER_AGENT_LENGTH = 200


def _get_json(client: httpx.Client, url: str, user_agent: str) -> tuple[dict, str]:
    if not url.startswith(BASE) or (url != BASE + f"CIK{CIK}.json"
                                   and not HISTORY_NAME.fullmatch(url.removeprefix(BASE))):
        raise ValueError("Unapproved SEC JSON URL")
    with client.stream(
        "GET", url, headers={
            "User-Agent": user_agent,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        }, follow_redirects=False,
    ) as response:
        response.raise_for_status()
        if (response.headers.get("Content-Length", "0").isdigit()
                and int(response.headers["Content-Length"]) > MAX_SOURCE_BYTES):
            raise ValueError("SEC page too large")
        raw = bytearray()
        for chunk in response.iter_bytes():
            raw.extend(chunk)
            if len(raw) > MAX_SOURCE_BYTES:
                raise ValueError("SEC page too large")
    packet = json.loads(raw)
    if not isinstance(packet, dict):
        raise TypeError("SEC page must be object")
    return packet, hashlib.sha256(raw).hexdigest()


def _rows(columns: dict, *, page_id: str, page_sha: str) -> list[dict]:
    fields = (
        "accessionNumber", "filingDate", "acceptanceDateTime", "form",
        "primaryDocument", "items",
    )
    if not isinstance(columns, dict):
        raise TypeError("SEC columns missing")
    values = [columns.get(key) for key in fields]
    if (not all(isinstance(v, list) for v in values)
            or len({len(v) for v in values}) != 1
            or len(values[0]) > MAX_ROWS):
        raise ValueError("Malformed SEC parallel columns")
    out = []
    for vals in zip(*values, strict=True):
        acc, filing_date, accepted, form, primary_doc, items = vals
        if not isinstance(form, str) or form not in FORM_ALLOWED:
            continue
        if (not isinstance(acc, str) or not ACCESSION.fullmatch(acc)
                or not isinstance(filing_date, str)
                or not isinstance(accepted, str)
                or not isinstance(primary_doc, str)
                or not isinstance(items, str)):
            raise ValueError("Malformed historical filing metadata")
        day = date.fromisoformat(filing_date)
        # SEC submissions 'acceptanceDateTime' may be a timezone-less
        # representation. Store raw value, do NOT turn it into a
        # verified New York publication/market availability timestamp.
        if not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?",
            accepted,
        ):
            raise ValueError("Invalid unverified SEC acceptance clock text")
        if ("/" in primary_doc or "\\" in primary_doc
                or primary_doc.startswith(".") or not primary_doc
                or len(primary_doc) > 180):
            raise ValueError("SEC primary document contains path traversal")
        out.append({
            "accession": acc,
            "filingDate": day.isoformat(),
            "rawSecAcceptanceDateTime": accepted,
            "form": form,
            "formItems": items,
            "secPrimaryDocument": primary_doc,
            "secIndexUrl": (
                f"https://www.sec.gov/Archives/edgar/data/{int(CIK)}/"
                f"{acc.replace('-', '')}/{acc}-index.htm"
            ),
            "submissionSourcePage": page_id,
            "submissionSourcePageSha256": page_sha,
            "secAcceptanceTimeIndependentlyVerified": False,
            "publisherFirstReleaseTimeVerified": False,
            "filingExhibitFactsVerified": False,
            "eligibleAsNewsModelFeature": False,
        })
    return out


def build_sec_filing_inventory(
    *, main: dict, main_sha: str, histories: list[tuple[str, dict, str]],
    start: date, end: date,
) -> dict:
    if (end < start or (end - start).days > 4000
            or not isinstance(main, dict) or str(main.get("cik")) != str(int(CIK))
            or not isinstance(main.get("filings"), dict)):
        raise ValueError("Invalid NVIDIA filing scope or CIK")
    recent = main["filings"].get("recent")
    rows = _rows(recent, page_id=f"CIK{CIK}.json", page_sha=main_sha)
    seen_history = set()
    for name, doc, digest in histories:
        if (not HISTORY_NAME.fullmatch(name) or name in seen_history
                or len(seen_history) >= MAX_HISTORY_FILES):
            raise ValueError("Invalid or duplicate SEC history page")
        seen_history.add(name)
        rows.extend(_rows(doc, page_id=name, page_sha=digest))
    existing = {}
    for item in rows:
        acc = item["accession"]
        if acc in existing:
            if existing[acc] != item:
                raise ValueError("Conflicting SEC accession provenance")
            continue
        existing[acc] = item
    filtered = [
        item for item in existing.values()
        if start.isoformat() <= item["filingDate"] <= end.isoformat()
    ]
    filtered.sort(key=lambda item: (item["filingDate"], item["accession"]))
    forms = dict(sorted(Counter(row["form"] for row in filtered).items()))
    return {
        "schemaVersion": "sec_historical_filing_inventory_v1",
        "status": "SEC_SUBMISSIONS_CANDIDATES_REQUIRE_INDEPENDENT_TIME_AND_EXHIBIT_VERIFICATION",
        "companyCik": CIK,
        "ticker": "NVDA",
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "sources": [{"name": f"CIK{CIK}.json", "sha256": main_sha}]
            + [{"name": name, "sha256": sha} for name, _, sha in histories],
        "sourcePages": len(histories) + 1,
        "candidateFilings": len(filtered),
        "formCounts": forms,
        "originalIssuerTextDownloaded": False,
        "originalFilingExhibitsPinned": False,
        "secAcceptanceTimeIndependentlyVerified": False,
        "firstPublisherPublicationTimeVerified": False,
        "trainingReady": False,
        "prospectiveAccuracyMeasured": False,
        "modelCalls": 0,
        "productionDbModified": False,
        "filings": filtered,
    }


def inventory_from_sec(
    *, client: httpx.Client, user_agent: str, start: date, end: date,
) -> dict:
    if (not user_agent or not 12 <= len(user_agent) <= MAX_USER_AGENT_LENGTH
            or "@" not in user_agent
            or "\n" in user_agent or "\r" in user_agent):
        raise ValueError("SEC requires identifying application and contact User-Agent")
    main, sha = _get_json(client, BASE + f"CIK{CIK}.json", user_agent)
    files = main.get("filings", {}).get("files", [])
    if not isinstance(files, list) or len(files) > 100:
        raise ValueError("Invalid SEC historical page manifest")
    names = []
    for ref in files:
        if not isinstance(ref, dict):
            raise TypeError("Invalid SEC history page")
        name = ref.get("name")
        if not isinstance(name, str) or not HISTORY_NAME.fullmatch(name):
            raise ValueError("Unsafe SEC history page name")
        if ref.get("filingFrom") is None or ref.get("filingTo") is None:
            raise ValueError("Missing SEC historical coverage")
        if (date.fromisoformat(ref["filingFrom"]) <= end
                and date.fromisoformat(ref["filingTo"]) >= start):
            names.append(name)
    if len(names) > MAX_HISTORY_FILES:
        raise ValueError("SEC historical range requires too many pages")
    histories = []
    for name in names:
        # Less than 2 requests/second, far beneath SEC max 10/s.
        time.sleep(0.55)
        packet, digest = _get_json(client, BASE + name, user_agent)
        histories.append((name, packet, digest))
    return build_sec_filing_inventory(
        main=main, main_sha=sha, histories=histories, start=start, end=end,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Opt-in SEC history acquisition: filename/source metadata only"
    )
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        user_agent = os.getenv("TRIS_SEC_USER_AGENT", "")
        with httpx.Client(timeout=httpx.Timeout(30), trust_env=False) as client:
            report = inventory_from_sec(
                client=client, user_agent=user_agent,
                start=args.start, end=args.end,
            )
        fingerprint = create_private_artifact(
            args.output, (json.dumps(report, ensure_ascii=False, indent=2)
                          + "\n").encode(),
        )
        print(json.dumps({
            "status": report["status"],
            "filings": report["candidateFilings"],
            "sourcePages": report["sourcePages"],
            "byForm": report["formCounts"],
            "sha256": fingerprint,
            "secClockVerified": False,
            "trainingReady": False,
            "outputCreated": True,
        }))
    except (httpx.HTTPError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        parser.exit(2, "sec-history-inventory: INVALID_SOURCE_OR_NETWORK\n")


if __name__ == "__main__":
    main()
