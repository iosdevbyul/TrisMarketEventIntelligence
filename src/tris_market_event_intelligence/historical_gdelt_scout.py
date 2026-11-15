"""Bounded opt-in scouting of archived GDELT GKG document references.

GKG batch timestamps are NOT publisher-first-seen timestamps and extracted
organization mentions are NOT verified issuer/event/price causal links.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx

from tris_market_event_intelligence.historical_shock_research import _checked_bytes

NY = ZoneInfo("America/New_York")
MAX_ZIP_BYTES = 24_000_000
MAX_DECOMPRESSED_BYTES = 100_000_000
MAX_PACKET_BYTES = 6_000_000
SLOT_OFFSETS_MINUTES = (-1440, -720, -240, -60)
ORG_NVDA = re.compile(r"(?i)\b(?:nvidia|nvidia\s+corporation)\b")


def _slots(day: date) -> list[tuple[datetime, str]]:
    opening = datetime.combine(day, time(9, 30), tzinfo=NY).astimezone(UTC)
    items = []
    for offset in SLOT_OFFSETS_MINUTES:
        target = opening + timedelta(minutes=offset)
        target = target.replace(minute=target.minute // 15 * 15, second=0, microsecond=0)
        stamp = target.strftime("%Y%m%d%H%M%S")
        items.append((
            target, f"https://data.gdeltproject.org/gdeltv2/{stamp}.gkg.csv.zip"
        ))
    return items


def _read_study(path: Path) -> tuple[dict, str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_PACKET_BYTES:
        raise ValueError("Bounded existing research packet required")
    raw = path.read_bytes()
    if len(raw) > MAX_PACKET_BYTES:
        raise ValueError("Research packet exceeds bound")
    doc = json.loads(raw)
    if (not isinstance(doc, dict) or doc.get("schemaVersion") != (
        "historical_news_macro_shock_research_v1"
    ) or doc.get("status") != "RESEARCH_PACKET_UNVERIFIED"
            or doc.get("symbol") != "NVDA"
            or doc.get("benchmarkSymbol") != "SPY"
            or doc.get("rankingIsRetrospectiveNotProspective") is not True
            or doc.get("forwardPredictionSealed") is not False):
        raise ValueError("Only NVDA-SPY retrospective research packet supported")
    rows = doc.get("topEvents")
    if not isinstance(rows, list) or len(rows) != 100:
        raise ValueError("Expected 100 original ranked shock dates")
    return doc, hashlib.sha256(raw).hexdigest()


def _fetch(client: httpx.Client, *, slot: datetime, url: str) -> dict:
    if urlsplit(url).hostname != "data.gdeltproject.org" or not url.startswith(
        "https://data.gdeltproject.org/gdeltv2/"
    ) or not url.endswith(".gkg.csv.zip"):
        raise ValueError("Invalid GDELT archive host or path")
    try:
        with client.stream("GET", url, follow_redirects=False) as response:
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            if content_length is not None and int(content_length) > MAX_ZIP_BYTES:
                raise ValueError("GDELT archive exceeds compressed size cap")
            buf = bytearray()
            for chunk in response.iter_bytes():
                buf.extend(chunk)
                if len(buf) > MAX_ZIP_BYTES:
                    raise ValueError("GDELT archive exceeds compressed size cap")
    except (httpx.HTTPError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("GDELT archive exceeds"):
            raise
        raise ValueError("GDELT_ARCHIVE_FETCH_FAILED") from exc
    hits: dict[str, dict] = {}
    total_rows = 0
    with zipfile.ZipFile(io.BytesIO(buf)) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].file_size > MAX_DECOMPRESSED_BYTES:
            raise ValueError("Unexpected GDELT archive structure or decompressed size")
        with archive.open(infos[0]) as source:
            for line in source:
                total_rows += 1
                if total_rows > 300_000 or len(line) > 1_000_000:
                    raise ValueError("GDELT archive row limit exceeded")
                fields = line.decode("utf-8", errors="replace").rstrip("\n").split("\t")
                if len(fields) < 15 or fields[2] != "1":
                    continue
                url_candidate = fields[4]
                parsed = urlsplit(url_candidate)
                if parsed.scheme not in ("http", "https") or not parsed.hostname:
                    continue
                # GKG 2.1 0-index: 13 = V1Organizations; 14 = V2Organizations.
                org_fields = fields[13] + ";" + fields[14]
                if not ORG_NVDA.search(org_fields):
                    continue
                hits.setdefault(url_candidate, {
                    "sourceUrl": url_candidate,
                    "sourceName": fields[3],
                    "gkgBatchSlotUtc": slot.isoformat(),
                    "documentDateField": fields[1],
                    "issuerVerified": False,
                    "articleTextRetrieved": False,
                    "independentCausalAttribution": False,
                })
    return {
        "gkgSlot": slot.isoformat(), "archiveSha256": hashlib.sha256(buf).hexdigest(),
        "archiveRows": total_rows, "matchingDocumentReferences": list(hits.values())[:50],
        "matchingDocumentsTotal": len(hits),
        "earliestArticleAvailabilityVerified": False,
        "archiveSlotNotPublicationTimestamp": True,
    }


def scout_historical_gdelt(
    *, study_path: Path, rank: int = 1,
    run: bool = False, client: httpx.Client | None = None,
) -> dict:
    doc, sha = _read_study(study_path)
    if type(rank) is not int or not 1 <= rank <= 100:
        raise ValueError("Shock rank must be 1..100")
    entry = doc["topEvents"][rank - 1]
    day = date.fromisoformat(entry["date"])
    slots = _slots(day)
    common = {
        "schemaVersion": "gdelt_gkg_historical_scout_v1",
        "status": "UNVERIFIED_GDELT_DOCUMENT_REFERENCES" if run else "PREVIEW_ONLY",
        "studySha256": sha, "ticker": "NVDA", "benchmark": "SPY",
        "shockRank": rank, "shockDate": day.isoformat(),
        "retrospectiveOutcomeKnown": True,
        "archivePlan": [{"slotUtc": stamp.isoformat(), "url": url}
                        for stamp, url in slots],
        "gkgDataAttribution": "The GDELT Project (https://www.gdeltproject.org/)",
        "earliestArticleAvailabilityVerified": False,
        "sourceArticleRightsVerified": False,
        "economicImpactCausallyVerified": False,
        "predictionOrTrainingPerformed": False,
        "macroDataUsed": False, "modelCalls": 0,
        "productionDbModified": False, "promotionAllowed": False,
    }
    if not run:
        return {**common, "archivesFetched": 0, "archives": []}
    owns = client is None
    session = client or httpx.Client(timeout=httpx.Timeout(45), trust_env=False)
    try:
        archives = [_fetch(session, slot=slot, url=url) for slot, url in slots]
    finally:
        if owns:
            session.close()
    return {**common, "archivesFetched": len(archives), "archives": archives}
