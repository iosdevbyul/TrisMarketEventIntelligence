"""Source-grounded SEC filing-body evidence for the fixed PIT research cohort.

A present-day original issuer document is useful historical *content* evidence,
not proof that market participants saw it before a historical decision cutoff.
This command never consumes price outcomes and never enables training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx

from tris_market_event_intelligence.historical_issuer_corroboration_cli import _write
from tris_market_event_intelligence.sec_publication_evidence_capture_cli import (
    _read_cohort, _validate_cohort, _verify_index,
)
from tris_market_event_intelligence.sec_source_lineage_audit_cli import (
    MAX_REPORT_BYTES, _read_json,
)

MAX_PRIMARY_BYTES = 8_000_000
MAX_INDEX_BYTES = 1_500_000
MAX_TASKS = 12
MAX_EXCERPT = 1100
ACC = re.compile(r"\d{10}-\d{2}-\d{6}\Z")
SHA = re.compile(r"[a-f0-9]{64}\Z")
SAFE_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,110}\Z")
ITEM = re.compile(r"\bItem\s+(?:1\.01|1\.02|2\.02|5\.02|7\.01|8\.01|9\.01)\b", re.I)
DOCTYPE = frozenset({"8-K", "10-Q", "10-K"})


class _IndexRows(HTMLParser):
    """Extract original EDGAR document-file table cells without interpreting JS."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_row = False
        self.in_cell = False
        self.cell_words: list[str] = []
        self.cell_links: list[str] = []
        self.cells: list[tuple[str, list[str]]] = []
        self.rows: list[list[tuple[str, list[str]]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.in_row = True
            self.cells = []
        elif tag == "td" and self.in_row:
            self.in_cell = True
            self.cell_words = []
            self.cell_links = []
        elif tag == "a" and self.in_cell:
            self.cell_links.extend(value for key, value in attrs
                                   if key == "href" and isinstance(value, str))

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self.in_cell:
            self.cells.append((
                re.sub(r"\s+", " ", " ".join(self.cell_words)).strip(),
                self.cell_links[:],
            ))
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if self.cells:
                self.rows.append(self.cells[:])
            self.in_row = False

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell_words.append(data)


def _approved_primary(url: str, acc: str) -> bool:
    part = urlsplit(url)
    prefix = f"/Archives/edgar/data/1045810/{acc.replace('-', '')}/"
    if (not ACC.fullmatch(acc) or part.scheme != "https"
            or part.netloc != "www.sec.gov" or part.query or part.fragment
            or not part.path.startswith(prefix)):
        return False
    name = part.path.removeprefix(prefix)
    return (bool(SAFE_FILENAME.fullmatch(name))
            and name.lower().endswith((".htm", ".html", ".txt"))
            and not name.lower().endswith("-index.htm"))


def _primary_url(raw_index: bytes, task: dict) -> str | None:
    if len(raw_index) > MAX_INDEX_BYTES:
        raise ValueError("SEC_INDEX_TOO_LARGE")
    parser = _IndexRows()
    parser.feed(raw_index.decode("utf-8-sig", errors="replace"))
    candidates: list[str] = []
    for row in parser.rows:
        # Document / type columns in the real SEC "Document Format Files" table
        # are not guaranteed to be fixed positions across EDGAR versions.
        # At least one standalone table cell must EXACTLY name the target form.
        if not any(cell.strip().upper() == task["form"] for cell, _ in row):
            continue
        for _, links in row:
            for href in links:
                uri = urljoin(task["secIndexUrl"], href)
                if _approved_primary(uri, task["accession"]):
                    candidates.append(uri)
    # A multi-document row is ambiguous. Do not silently select a different
    # exhibit or turn the first arbitrary HTML link into a source.
    matches = sorted(set(candidates))
    return matches[0] if len(matches) == 1 else None


def _text(raw: bytes, *, html: bool) -> str:
    if len(raw) > MAX_PRIMARY_BYTES or not raw:
        raise ValueError("SEC_PRIMARY_DOCUMENT_SIZE_INVALID")
    if not html:
        body = raw.decode("utf-8-sig", errors="replace")
    else:
        class _Visible(HTMLParser):
            def __init__(self) -> None:
                super().__init__(convert_charrefs=True)
                self.parts: list[str] = []
                self.hidden = 0

            def handle_starttag(self, tag: str, attrs: list) -> None:
                if tag in ("script", "style"):
                    self.hidden += 1

            def handle_endtag(self, tag: str) -> None:
                if tag in ("script", "style") and self.hidden:
                    self.hidden -= 1

            def handle_data(self, data: str) -> None:
                if not self.hidden:
                    self.parts.append(data)
        parser = _Visible()
        parser.feed(raw.decode("utf-8-sig", errors="replace"))
        body = " ".join(parser.parts)
    return re.sub(r"\s+", " ", body).strip()


def _source_review(raw: bytes, url: str) -> dict:
    body = _text(raw, html=not url.endswith(".txt"))
    if len(body) < 120 or "nvidia" not in body.lower():
        raise ValueError("SEC_PRIMARY_DOCUMENT_ISSUER_OR_CONTENT_INVALID")
    item_hits = list(dict.fromkeys(match.group().upper().replace(" ", "")
                                   for match in ITEM.finditer(body)))
    # Readable and quoted from actual locally pinned bytes, NEVER generated
    # issuer facts or a model summary. Limit even if source is megabytes long.
    excerpt = body[:MAX_EXCERPT]
    return {
        "visibleIssuerNamePresent": True,
        "secItemMarkersObservedInSource": item_hits[:20],
        "primarySourceVerbatimExcerpt": excerpt,
        "excerptIsOriginalSourceOnly": True,
        "extractedTextIsRetrospectiveEvidence": True,
        "historicalFirstPublicationAt": None,
        "independentContemporaneousObservedAt": None,
        "marketAvailabilityVerified": False,
        "usableAsPredecisionModelInput": False,
    }


def build_primary_evidence(
    *, cohort: dict, cohort_sha: str,
    batch: dict, batch_sha: str,
    source_dir: Path, limit: int,
    run: bool = False, client: httpx.Client | None = None,
    user_agent: str | None = None,
) -> dict:
    queue = _validate_cohort(cohort)
    if (not isinstance(cohort_sha, str) or not SHA.fullmatch(cohort_sha)
            or not isinstance(batch_sha, str) or not SHA.fullmatch(batch_sha)
            or not isinstance(limit, int) or isinstance(limit, bool)
            or not 1 <= limit <= MAX_TASKS
            or not source_dir.is_dir() or source_dir.is_symlink()):
        raise ValueError("SEC_PRIMARY_INPUTS_INVALID")
    if (batch.get("schemaVersion") != "sec_pit_original_index_capture_v1"
            or batch.get("status") !=
            "SEC_ORIGINAL_INDEX_SHA_PINNED_PUBLICATION_TIME_UNPROVEN"
            or batch.get("cohortSha256") != cohort_sha
            or batch.get("trainingReady") is not False
            or batch.get("knownFirstPublicAvailabilityCases") != 0
            or batch.get("capturedIndexCount") != len(batch.get("sourceRecords", []))):
        raise ValueError("SEC_PRIMARY_BATCH_PROVENANCE_INVALID")
    selected = queue[:min(limit, len(batch["sourceRecords"]))]
    verified = batch["sourceRecords"][:len(selected)]
    resolved, missing, reused, unresolved = [], 0, 0, 0
    for task, index_row in zip(selected, verified, strict=True):
        acc = task["accession"]
        if (index_row.get("accession") != acc
                or index_row.get("form") != task["form"]
                or index_row.get("historicalFirstPublicAvailabilityAt") is not None
                or index_row.get("eligibleForNewsModelTraining") is not False
                or index_row.get("firstIndependentHistoricalObservedAt") is not None):
            raise ValueError("SEC_PRIMARY_TASK_LINEAGE_MISMATCH")
        source_index = source_dir / f"sec-{acc}-index.html"
        if (source_index.is_symlink() or not source_index.is_file()
                or source_index.stat().st_size > MAX_INDEX_BYTES):
            raise ValueError("SEC_PRIMARY_PINNED_INDEX_MISSING")
        index_raw = source_index.read_bytes()
        if (not isinstance(index_row.get("secIndexSha256"), str)
                or hashlib.sha256(index_raw).hexdigest()
                != index_row["secIndexSha256"]):
            raise ValueError("SEC_PRIMARY_INDEX_SHA_INVALID")
        _verify_index(index_raw, task)
        url = _primary_url(index_raw, task)
        if url is None:
            unresolved += 1
            resolved.append({
                "accession": acc, "form": task["form"],
                "status": "PRIMARY_DOCUMENT_NOT_UNAMBIGUOUS_IN_ORIGINAL_INDEX",
                "primaryDocumentUrl": None,
                "primaryDocumentSha256": None,
                "historicalPublicationAtVerified": False,
                "trainingEligible": False,
            })
            continue
        original = source_dir / f"sec-{acc}-primary.source"
        if original.is_symlink():
            raise ValueError("SEC_PRIMARY_SOURCE_SYMLINK")
        present = original.exists()
        if present:
            if (not original.is_file()
                    or original.stat().st_size > MAX_PRIMARY_BYTES):
                raise ValueError("SEC_PRIMARY_CACHED_SOURCE_INVALID")
            raw = original.read_bytes()
            reused += 1
        elif run:
            if client is None or user_agent is None:
                raise ValueError("SEC_PRIMARY_NETWORK_CLIENT_REQUIRED")
            if missing:
                time.sleep(0.55)
            with client.stream("GET", url, follow_redirects=False, headers={
                "User-Agent": user_agent, "Accept": "text/html,text/plain",
            }) as response:
                response.raise_for_status()
                if response.status_code != 200:
                    raise ValueError("SEC_PRIMARY_DOWNLOAD_STATUS_INVALID")
                chunks = bytearray()
                for chunk in response.iter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > MAX_PRIMARY_BYTES:
                        raise ValueError("SEC_PRIMARY_DOWNLOAD_TOO_LARGE")
                raw = bytes(chunks)
            missing += 1
        else:
            unresolved += 1
            resolved.append({
                "accession": acc, "form": task["form"],
                "status": "PRIMARY_DOCUMENT_EXISTS_IN_INDEX_NOT_YET_CAPTURED",
                "primaryDocumentUrl": url,
                "primaryDocumentSha256": None,
                "historicalPublicationAtVerified": False,
                "trainingEligible": False,
            })
            continue
        details = _source_review(raw, url)
        digest = hashlib.sha256(raw).hexdigest()
        if run and not present:
            from tris_market_event_intelligence.historical_source_acquisition import (
                create_private_artifact,
            )
            create_private_artifact(original, raw)
        resolved.append({
            "accession": acc, "form": task["form"], "filingDate": task["filingDate"],
            "status": "PRIMARY_DOCUMENT_SHA_PINNED_RESEARCH_ONLY",
            "primaryDocumentUrl": url, "primaryDocumentSha256": digest,
            "reusedOriginalSource": present,
            **details,
            "historicalPublicationAtVerified": False,
            "trainingEligible": False,
        })
    pinned = sum(x["status"] == "PRIMARY_DOCUMENT_SHA_PINNED_RESEARCH_ONLY"
                 for x in resolved)
    return {
        "schemaVersion": "sec_original_primary_content_research_v1",
        "status": "SEC_ORIGINAL_CONTENT_WITH_UNVERIFIED_HISTORICAL_PUBLICATION",
        "cohortSha256": cohort_sha,
        "indexBatchSha256": batch_sha,
        "issuer": "NVIDIA", "ticker": "NVDA",
        "reviewedIndexCases": len(resolved),
        "primaryContentsPinned": pinned,
        "primaryContentsNew": missing,
        "primaryContentsReused": reused,
        "primaryNotYetResolved": unresolved,
        "cases": resolved,
        "originalSourceTextOnly": True,
        "primaryIndexAsContentNotAsFirstPublicationProof": True,
        "independentHistoricalPublicationTimesVerified": 0,
        "outcomeDataRead": False,
        "modelCalls": 0,
        "trainingReady": False, "backtestReady": False,
        "promotionAllowed": False, "productionDbModified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture original SEC 8-K/10-Q/10-K primary document text"
    )
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, choices=range(1, MAX_TASKS + 1),
                        default=12)
    parser.add_argument("--output", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preview", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        if args.run and args.output is None:
            raise ValueError("SEC_PRIMARY_NEW_OUTPUT_REQUIRED")
        cohort, cohort_sha = _read_cohort(args.cohort)
        batch, batch_sha = _read_json(args.batch, MAX_REPORT_BYTES)
        ua = os.getenv("TRIS_SEC_USER_AGENT", "")
        if args.run and (not 12 <= len(ua) <= 200 or "@" not in ua
                         or "\r" in ua or "\n" in ua):
            raise ValueError("SEC_PRIMARY_USER_AGENT_INVALID")
        with httpx.Client(timeout=httpx.Timeout(30), trust_env=False) as client:
            report = build_primary_evidence(
                cohort=cohort, cohort_sha=cohort_sha,
                batch=batch, batch_sha=batch_sha,
                source_dir=args.source_dir, limit=args.limit,
                run=args.run, client=client if args.run else None,
                user_agent=ua if args.run else None,
            )
        if args.run:
            _write(args.output, report)
        print(json.dumps({
            "status": report["status"],
            "reviewedIndexCases": report["reviewedIndexCases"],
            "primaryContentsPinned": report["primaryContentsPinned"],
            "newSourceDocuments": report["primaryContentsNew"],
            "reusedSourceDocuments": report["primaryContentsReused"],
            "unresolvedCases": report["primaryNotYetResolved"],
            "independentPublicTimesVerified": 0,
            "trainingReady": False, "outputCreated": args.run,
        }))
    except httpx.HTTPStatusError as exc:
        parser.exit(2, f"sec-primary-content: SEC_HTTP_{exc.response.status_code}\n")
    except httpx.HTTPError:
        parser.exit(2, "sec-primary-content: SEC_NETWORK_FAILURE\n")
    except (OSError, ValueError, TypeError, KeyError, UnicodeError,
            json.JSONDecodeError):
        parser.exit(2, "sec-primary-content: INVALID_PRIMARY_EVIDENCE\n")


if __name__ == "__main__":
    main()
