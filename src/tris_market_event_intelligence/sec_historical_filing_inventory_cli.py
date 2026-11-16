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
        raise ValueError("SEC_UNAPPROVED_SOURCE_URL")
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
            raise ValueError("SEC_SOURCE_TOO_LARGE")
        raw = bytearray()
        for chunk in response.iter_bytes():
            raw.extend(chunk)
            if len(raw) > MAX_SOURCE_BYTES:
                raise ValueError("SEC_SOURCE_TOO_LARGE")
    packet = json.loads(raw)
    if not isinstance(packet, dict):
        raise TypeError("SEC_SCHEMA_PAGE_NOT_OBJECT")
    return packet, hashlib.sha256(raw).hexdigest()


def _safe_primary_document_path(value: str) -> bool:
    """Permit only bounded relative SEC paths; never use unchecked values as URLs."""
    if not value or len(value) > 240:
        return False
    if any(ord(ch) < 33 or ord(ch) == 127 for ch in value):
        return False
    if any(ch in value for ch in ("\\", "?", "#", "%", ":")):
        return False
    parts = value.split("/")
    return not value.startswith("/") and all(
        part not in ("", ".", "..") for part in parts
    )


def _rows(columns: dict, *, page_id: str, page_sha: str) -> list[dict]:
    fields = (
        "accessionNumber", "filingDate", "acceptanceDateTime", "form",
        "primaryDocument",
    )
    if not isinstance(columns, dict):
        raise TypeError("SEC_SCHEMA_COLUMNS_NOT_OBJECT")
    values = [columns.get(key) for key in fields]
    if (not all(isinstance(v, list) for v in values)
            or len({len(v) for v in values}) != 1
            or len(values[0]) > MAX_ROWS):
        raise ValueError("SEC_SCHEMA_COLUMNS")
    # Historical SEC pages can omit the 8-K items column, or use null.
    items_column = columns.get("items")
    if items_column is None:
        items_column = [""] * len(values[0])
    if not isinstance(items_column, list) or len(items_column) != len(values[0]):
        raise ValueError("SEC_SCHEMA_ITEMS")
    out = []
    for vals in zip(*values, items_column, strict=True):
        acc, filing_date, accepted, form, primary_doc, items = vals
        if not isinstance(form, str) or form not in FORM_ALLOWED:
            continue
        if (not isinstance(acc, str) or not ACCESSION.fullmatch(acc)
                or not isinstance(filing_date, str)
                or not isinstance(accepted, str)
                or not isinstance(primary_doc, str)
                or (items is not None and not isinstance(items, str))):
            raise ValueError("SEC_FILING_METADATA")
        day = date.fromisoformat(filing_date)
        # SEC submissions 'acceptanceDateTime' may be a timezone-less
        # representation. Store raw value, do NOT turn it into a
        # verified New York publication/market availability timestamp.
        if not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?",
            accepted,
        ):
            raise ValueError("SEC_ACCEPTANCE_FORMAT")
        # Candidate discovery uses the accession index, not primaryDocument.
        # Missing/unsafe document paths remain quarantined, with source SHA.
        primary_doc_safe = _safe_primary_document_path(primary_doc)
        out.append({
            "accession": acc,
            "filingDate": day.isoformat(),
            "rawSecAcceptanceDateTime": accepted,
            "form": form,
            "formItems": items or "",
            "secPrimaryDocument": primary_doc if primary_doc_safe else None,
            "secPrimaryDocumentPathStatus": (
                "SAFE_RELATIVE_PATH" if primary_doc_safe
                else "QUARANTINED_MISSING_OR_UNSAFE"
            ),
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


def _valid_company_cik(value: object) -> bool:
    """Match SEC's numeric CIK or zero-padded decimal string, never a different filer."""
    if type(value) is int:
        return value == int(CIK)
    if isinstance(value, str):
        return bool(re.fullmatch(r"\d{1,10}", value)) and int(value) == int(CIK)
    return False


def build_sec_filing_inventory(
    *, main: dict, main_sha: str, histories: list[tuple[str, dict, str]],
    start: date, end: date,
) -> dict:
    if end < start or (end - start).days > 4000:
        raise ValueError("SEC_DATE_SCOPE_INVALID")
    if not isinstance(main, dict) or not _valid_company_cik(main.get("cik")):
        raise ValueError("SEC_CIK_MISMATCH")
    if not isinstance(main.get("filings"), dict):
        raise TypeError("SEC_FILINGS_SCHEMA_INVALID")
    recent = main["filings"].get("recent")
    rows = _rows(recent, page_id=f"CIK{CIK}.json", page_sha=main_sha)
    seen_history = set()
    for name, doc, digest in histories:
        if (not HISTORY_NAME.fullmatch(name) or name in seen_history
                or len(seen_history) >= MAX_HISTORY_FILES):
            raise ValueError("SEC_HISTORY_MANIFEST")
        seen_history.add(name)
        rows.extend(_rows(doc, page_id=name, page_sha=digest))
    existing = {}
    for item in rows:
        acc = item["accession"]
        if acc in existing:
            # The SEC main and historical pages can overlap around a
            # pagination boundary. Their page-name/SHA necessarily differ.
            # Compare filing facts, not the independently pinned page origin.
            non_origin = (
                "filingDate", "rawSecAcceptanceDateTime", "form",
                "formItems", "secPrimaryDocument",
                "secPrimaryDocumentPathStatus", "secIndexUrl",
            )
            if any(existing[acc][key] != item[key] for key in non_origin):
                raise ValueError("SEC_ACCESSION_CONFLICT")
            continue
        existing[acc] = item
    filtered = [
        item for item in existing.values()
        if start.isoformat() <= item["filingDate"] <= end.isoformat()
    ]
    filtered.sort(key=lambda item: (item["filingDate"], item["accession"]))
    forms = dict(sorted(Counter(row["form"] for row in filtered).items()))
    quarantined = sum(
        row["secPrimaryDocumentPathStatus"] == "QUARANTINED_MISSING_OR_UNSAFE"
        for row in filtered
    )
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
        "quarantinedPrimaryDocumentPaths": quarantined,
        "allPrimaryDocumentsHaveSafePaths": quarantined == 0,
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
        raise ValueError("SEC_HISTORY_MANIFEST")
    names = []
    for ref in files:
        if not isinstance(ref, dict):
            raise TypeError("SEC_HISTORY_MANIFEST")
        name = ref.get("name")
        if not isinstance(name, str) or not HISTORY_NAME.fullmatch(name):
            raise ValueError("SEC_HISTORY_MANIFEST")
        if ref.get("filingFrom") is None or ref.get("filingTo") is None:
            raise ValueError("SEC_HISTORY_MANIFEST")
        if (date.fromisoformat(ref["filingFrom"]) <= end
                and date.fromisoformat(ref["filingTo"]) >= start):
            names.append(name)
    if len(names) > MAX_HISTORY_FILES:
        raise ValueError("SEC_HISTORY_PAGE_LIMIT")
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
            "quarantinedPrimaryDocumentPaths": report["quarantinedPrimaryDocumentPaths"],
            "sha256": fingerprint,
            "secClockVerified": False,
            "trainingReady": False,
            "outputCreated": True,
        }))
    except httpx.HTTPStatusError as exc:
        parser.exit(
            2, f"sec-history-inventory: SEC_HTTP_{exc.response.status_code}\\n",
        )
    except httpx.TimeoutException:
        parser.exit(2, "sec-history-inventory: SEC_TIMEOUT\\n")
    except httpx.ConnectError:
        parser.exit(2, "sec-history-inventory: SEC_CONNECTION_FAILED\\n")
    except httpx.HTTPError:
        parser.exit(2, "sec-history-inventory: SEC_TRANSPORT_FAILED\\n")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        # Expose only code-owned, fixed diagnostic categories; do not disclose
        # SEC response content, identifying User-Agent/email, or local paths.
        allowed = {
            "SEC_SCHEMA_PAGE_NOT_OBJECT", "SEC_SCHEMA_COLUMNS_NOT_OBJECT",
            "SEC_SCHEMA_COLUMNS", "SEC_SCHEMA_ITEMS", "SEC_FILING_METADATA",
            "SEC_ACCEPTANCE_FORMAT", "SEC_PRIMARY_DOCUMENT_PATH",
            "SEC_SCOPE_OR_CIK", "SEC_DATE_SCOPE_INVALID", "SEC_CIK_MISMATCH",
            "SEC_FILINGS_SCHEMA_INVALID", "SEC_HISTORY_MANIFEST", "SEC_ACCESSION_CONFLICT",
            "SEC_HISTORY_PAGE_LIMIT", "SEC_SOURCE_TOO_LARGE",
            "SEC_UNAPPROVED_SOURCE_URL",
        }
        code = str(exc) if str(exc) in allowed else "SEC_SOURCE_VALIDATION_FAILED"
        if isinstance(exc, json.JSONDecodeError):
            code = "SEC_RESPONSE_NOT_JSON"
        if str(exc) == "SEC requires identifying application and contact User-Agent":
            code = "SEC_USER_AGENT_INVALID"
        parser.exit(2, f"sec-history-inventory: {code}\\n")


if __name__ == "__main__":
    main()
