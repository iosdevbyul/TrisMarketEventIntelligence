"""Rank multiple bounded, SHA-pinned SEC passages before any model inference.

v1/v2 immutable; v3 only on explicit opt-in. Offset units are positions in
normalized visible source text, NOT raw HTML bytes or historical publication.
Rank labels describe retrieval relevance, never verified semantic correctness.
"""
from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path

from tris_market_event_intelligence.sec_primary_content_research_cli import _text
from tris_market_event_intelligence.sec_revenue_context_guard import (
    REFERENT_UNRESOLVED,
    revenue_context_status,
)
from tris_market_event_intelligence.sec_semantic_evidence_gate import (
    EVENT,
    FINANCIAL,
    INSUFFICIENT,
    MAGNITUDE,
    RISK_DISCLOSURE,
    RISK_SUBJECT,
    source_evidence_status,
)

VERSION = "sec_ranked_multi_passage_v3"
MAX_VISIBLE = 2_000_000
MAX_PASSAGES = 3
MAX_WINDOW = 880
MIN_DISTINCT_DISTANCE = 140
CAUSE = re.compile(r"(?i)\b(?:because|due to|resulted in|as a result of)\b")
SENTENCE = re.compile(r".+?(?:[.!?]+(?=\s|$)|$)", re.DOTALL)
PAGE_NOISE = re.compile(r"(?i)(?:table of contents|index to|form 10-k|"
                        r"securities and exchange commission|pursuant to section)")


def _cut_window(visible: str, start: int, end: int) -> tuple[int, int, str]:
    """Include preceding and following explanations with source-exact offsets."""
    left = max(0, start - 300)
    right = min(len(visible), end + 180)
    if right - left > MAX_WINDOW:
        right = min(right, left + MAX_WINDOW)
    if left > 0:
        space = visible.find(" ", left, start)
        if space >= 0:
            left = space + 1
    if right < len(visible):
        space = visible.rfind(" ", end, right)
        if space > end:
            right = space
    if left > start or right < end:
        raise ValueError("SEC_MULTI_CONTEXT_TRUNCATED_FOCUS")
    window = visible[left:right]
    if len(window) > MAX_WINDOW or not window:
        raise ValueError("SEC_MULTI_CONTEXT_WINDOW_INVALID")
    return left, right, window


def select_ranked_passages(case: dict, raw: bytes) -> list[dict]:
    """Rank distinct issuer-fact passages from one checked original SEC file.

    Ranking uses facts in the focused sentence, not company-name metadata in
    neighboring text. It never regards a standalone share as causal evidence.
    """
    visible = _text(raw, html=not case["sourceUrl"].endswith(".txt"))
    if not visible or len(visible) > MAX_VISIBLE:
        raise ValueError("SEC_MULTI_VISIBLE_SIZE_INVALID")
    visible_sha = hashlib.sha256(visible.encode("utf-8")).hexdigest()
    candidates: list[tuple[int, int, dict]] = []
    for match in SENTENCE.finditer(visible):
        focus = match.group().strip()
        if not 48 <= len(focus) <= 420:
            continue
        if PAGE_NOISE.search(focus) or source_evidence_status(focus) == INSUFFICIENT:
            continue
        context_status = revenue_context_status(focus)
        if context_status == REFERENT_UNRESOLVED:
            continue
        is_event = EVENT.search(focus) is not None
        is_financial = FINANCIAL.search(focus) is not None and MAGNITUDE.search(focus) is not None
        is_risk = RISK_DISCLOSURE.search(focus) is not None and RISK_SUBJECT.search(focus) is not None
        if not (is_event or is_financial or is_risk):
            continue
        score = 5 if is_event else 4 if is_financial else 2
        # A revenue share with a referent is descriptive, not an issuer event.
        if context_status is not None:
            score = 1
        start = match.start() + len(match.group()) - len(match.group().lstrip())
        end = match.end() - len(match.group()) + len(match.group().rstrip())
        # Defensive exact offsets. Do not normalize the source after selecting.
        if visible[start:end] != focus:
            raise ValueError("SEC_MULTI_FOCUS_OFFSET_MISMATCH")
        left, right, window = _cut_window(visible, start, end)
        row = {
            "version": VERSION,
            "evidenceId": f"sec-{case['accession']}-ranked-{start}",
            "originalSourceSha256": case["sourceSha256"],
            "visibleTextSha256": visible_sha,
            "visibleStart": left,
            "visibleEnd": right,
            "spanSha256": hashlib.sha256(window.encode("utf-8")).hexdigest(),
            "focusStart": start,
            "focusEnd": end,
            "focusSha256": hashlib.sha256(focus.encode("utf-8")).hexdigest(),
            "rankScore": score,
            "rankReason": (
                "ISSUER_ACTION" if is_event else
                "QUANTIFIED_ISSUER_FACT" if is_financial else
                "ISSUER_RISK_DISCLOSURE"
            ),
            "contextStatus": context_status,
            "focusText": focus,
            "text": window,
        }
        candidates.append((-score, start, row))
    candidates.sort(key=lambda item: (item[0], item[1]))
    picked: list[dict] = []
    for _, start, row in candidates:
        if any(abs(start - p["focusStart"]) < MIN_DISTINCT_DISTANCE for p in picked):
            continue
        picked.append(row)
        if len(picked) == MAX_PASSAGES:
            break
    return picked


def apply_ranked_passages(prepared: dict, source_dir: Path) -> dict:
    """Derive a new model-only evidence set from each immutable source."""
    if not source_dir.is_dir() or source_dir.is_symlink():
        raise ValueError("SEC_MULTI_SOURCE_DIR_INVALID")
    selected = copy.deepcopy(prepared)
    selected_count = 0
    for case in selected["cases"]:
        path = source_dir / f"sec-{case['accession']}-primary.source"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 8_000_000:
            raise ValueError("SEC_MULTI_ORIGINAL_SOURCE_REQUIRED")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != case["sourceSha256"]:
            raise ValueError("SEC_SEMANTIC_SOURCE_SHA_MISMATCH")
        passages = select_ranked_passages(case, raw)
        # Require two *distinct* independently material facts. If not, keep
        # original metadata excerpt so existing insufficiency gate can abstain.
        if len(passages) < 2:
            continue
        case["sourcePassages"] = [
            {key: value for key, value in row.items() if key not in {"focusText", "text"}}
            for row in passages
        ]
        case["evidence"] = [
            {
                "id": row["evidenceId"], "text": row["text"],
                "focusText": row["focusText"],
                "secItemMarkers": [],
                "retrospectiveSourceOnly": True,
            }
            for row in passages
        ]
        selected_count += 1
    selected["multiPassageVersion"] = VERSION
    selected["multiPassageCasesSelected"] = selected_count
    return selected


def reverify_ranked_passages(
    case: dict, raw: bytes, claimed: list[dict],
) -> dict:
    """Reject reordered, omitted, shifted, inflated or fabricated passages."""
    if hashlib.sha256(raw).hexdigest() != case["sourceSha256"]:
        raise ValueError("SEC_SEMANTIC_SOURCE_SHA_MISMATCH")
    passages = select_ranked_passages(case, raw)
    if len(passages) < 2:
        raise ValueError("SEC_MULTI_PASSAGES_NOT_AVAILABLE")
    witnesses = [
        {key: value for key, value in row.items() if key not in {"focusText", "text"}}
        for row in passages
    ]
    if claimed != witnesses:
        raise ValueError("SEC_MULTI_PASSAGES_PROVENANCE_INVALID")
    verified = dict(case)
    verified["sourcePassages"] = witnesses
    verified["evidence"] = [
        {
            "id": row["evidenceId"], "text": row["text"],
            "focusText": row["focusText"],
            "secItemMarkers": [], "retrospectiveSourceOnly": True,
        }
        for row in passages
    ]
    return verified
