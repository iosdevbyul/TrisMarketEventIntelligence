"""Independent evidence-gate diagnostics for news chronology and event families.

Document provenance alone cannot authenticate the earliest public availability.
The gate records what an external reviewer must corroborate; it never promotes
an operator's assertion to certified causality or prospective market accuracy.
"""
from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from tris_market_event_intelligence.point_in_time_outcome import SealedForecast

FIELDS = {
    "article_id", "evidence_url", "evidence_type", "evidence_published_at",
    "captured_at", "sha256", "source_url", "first_public_at",
    "family_id", "relationship", "family_evidence_url",
}
TYPES = {"OFFICIAL_RELEASE", "PUBLISHER_METADATA", "ARCHIVE_SNAPSHOT"}
RELATIONSHIPS = {"NEW", "REPRINT", "FOLLOW_UP", "CORRECTION", "UNKNOWN"}
MAX_ROWS = 5000
MAX_SIZE = 2 * 1024 * 1024


def _stamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Evidence dates require timezone-aware timestamps")
    return parsed.astimezone(UTC)


def _url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc) and not parsed.username


@dataclass(frozen=True)
class Evidence:
    article_id: int
    evidence_url: str
    evidence_type: str
    evidence_published_at: datetime
    captured_at: datetime
    sha256: str
    source_url: str
    first_public_at: datetime
    family_id: str
    relationship: str
    family_evidence_url: str

    def __post_init__(self) -> None:
        if self.article_id < 1 or self.evidence_type not in TYPES:
            raise ValueError("Invalid evidence article or source type")
        if self.relationship not in RELATIONSHIPS or not self.family_id.strip():
            raise ValueError("Invalid event family linkage")
        if not _url(self.evidence_url) or not _url(self.source_url):
            raise ValueError("Evidence needs explicit HTTPS URLs")
        if self.family_evidence_url and not _url(self.family_evidence_url):
            raise ValueError("Family link evidence must be a valid HTTPS URL")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("Invalid lowercase source SHA256")
        if self.captured_at < self.evidence_published_at:
            raise ValueError("Evidence captured before its stated publication")
        if self.captured_at < self.first_public_at:
            raise ValueError("Evidence captured before stated original public availability")


def load_evidence(path: Path) -> dict[int, Evidence]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_SIZE:
        raise ValueError("Expected bounded existing non-symlink evidence CSV")
    original = path.read_bytes()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != FIELDS:
            raise ValueError("Evidence CSV must match the exact versioned field contract")
        items: dict[int, Evidence] = {}
        for index, row in enumerate(reader):
            if index >= MAX_ROWS:
                raise ValueError("Evidence exceeds row limit")
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed evidence CSV")
            item = Evidence(
                article_id=int(row["article_id"]),
                evidence_url=row["evidence_url"],
                evidence_type=row["evidence_type"],
                evidence_published_at=_stamp(row["evidence_published_at"]),
                captured_at=_stamp(row["captured_at"]),
                sha256=row["sha256"],
                source_url=row["source_url"],
                first_public_at=_stamp(row["first_public_at"]),
                family_id=row["family_id"],
                relationship=row["relationship"],
                family_evidence_url=row["family_evidence_url"],
            )
            if item.article_id in items:
                raise ValueError("Duplicate article evidence")
            items[item.article_id] = item
    if path.read_bytes() != original:
        raise ValueError("Evidence file modified during inspection")
    return items


def evaluate_chronology(
    forecasts: list[SealedForecast],
    evidence: dict[int, Evidence],
) -> dict[str, object]:
    """Check source clock consistency and group events without inferring causality.

    Output labels are documentary review diagnostics, NOT a certificate that
    claimed public timestamps were genuine or actually observable at cutoff.
    """
    by_article: dict[int, list[SealedForecast]] = defaultdict(list)
    for f in forecasts:
        by_article[f.event.article_id].append(f)
    rows: dict[int, dict[str, object]] = {}
    groups: dict[str, set[int]] = defaultdict(set)
    for article_id, articles in by_article.items():
        event = articles[0].event
        if any(a.event != event for a in articles):
            raise ValueError("Inconsistent article identity across forecast horizons")
        item = evidence.get(article_id)
        if item is None:
            state = "EVIDENCE_MISSING"
        elif (item.source_url != event.source_url
              or item.first_public_at != event.verified_cutoff
              or item.family_id != event.family_id
              or item.relationship != event.novelty):
            state = "EVIDENCE_CONFLICT"
        elif item.evidence_published_at > item.first_public_at:
            state = "EARLIEST_PUBLIC_TIME_UNSUPPORTED"
        elif event.novelty in {"REPRINT", "FOLLOW_UP", "CORRECTION"} and not item.family_evidence_url:
            state = "FAMILY_LINK_EVIDENCE_MISSING"
        else:
            # Plausible documentary chronology is not independently attested.
            state = "DOCUMENTARY_CONSISTENT_UNATTESTED"
        groups[event.family_id].add(article_id)
        rows[article_id] = {
            "articleId": article_id, "familyId": event.family_id,
            "relationship": event.novelty,
            "chronologyStatus": state,
            "independentlyCertified": False,
            "evidenceURL": item.evidence_url if item else None,
            "captureSHA256": item.sha256 if item else None,
            "captureAt": item.captured_at.isoformat() if item else None,
            "familyEvidenceURL": item.family_evidence_url if item else None,
        }
    family_counts = Counter({key: len(ids) for key, ids in groups.items()})
    duplicate_family_count = sum(size > 1 for size in family_counts.values())
    return {
        "articles": rows, "articleCount": len(rows), "familyCount": len(groups),
        "repeatedFamilyCount": duplicate_family_count,
        "reusedArticleForecastCount": sum(len(v) - 1 for v in by_article.values()),
        "chronologyStatusCounts": dict(sorted(Counter(
            r["chronologyStatus"] for r in rows.values()
        ).items())),
        "independentPublicTimeCertified": False,
        "independentEventFamilyCertified": False,
    }
