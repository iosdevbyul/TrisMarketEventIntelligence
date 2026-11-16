"""Deterministic substantive span extraction from locally SHA-pinned SEC originals.

Offsets are character positions in the *normalized visible extracted text*,
NOT raw HTML byte offsets or historical market publication timestamps.
"""
from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path

from tris_market_event_intelligence.sec_primary_content_research_cli import _text
from tris_market_event_intelligence.sec_semantic_evidence_gate import (
    INSUFFICIENT,
    source_evidence_status,
)

VERSION = "sec_visible_body_span_v1"
MAX_VISIBLE_CHARS = 2_000_000
MAX_SPAN_CHARS = 1100


def select_body_span(case: dict, raw: bytes) -> dict | None:
    """Choose the first independently substantive sentence after front matter.

    A selection is not proof of event causation. Quotes still need exact
    verification against this particular selected span by downstream code.
    """
    url = case["sourceUrl"]
    visible = _text(raw, html=not url.endswith(".txt"))
    if len(visible) > MAX_VISIBLE_CHARS:
        raise ValueError("SEC_BODY_EXTRACTED_TEXT_TOO_LARGE")
    # The shared text extractor normalizes whitespace deterministically.
    # Split on sentence punctuation; selected text must be a source substring.
    for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", visible):
        start, end = match.span()
        segment = match.group().strip()
        if len(segment) < 45 or len(segment) > MAX_SPAN_CHARS:
            continue
        if source_evidence_status(segment) == INSUFFICIENT:
            continue
        start += len(match.group()) - len(match.group().lstrip())
        end -= len(match.group()) - len(match.group().rstrip())
        assert visible[start:end] == segment
        return {
            "version": VERSION,
            "originalSourceSha256": case["sourceSha256"],
            "visibleTextSha256": hashlib.sha256(visible.encode("utf-8")).hexdigest(),
            "visibleStart": start,
            "visibleEnd": end,
            "spanSha256": hashlib.sha256(segment.encode("utf-8")).hexdigest(),
            "text": segment,
        }
    return None


def apply_body_spans(prepared: dict, source_dir: Path) -> dict:
    """Replace evidence only in a transient research input, not source packets."""
    if not source_dir.is_dir() or source_dir.is_symlink():
        raise ValueError("SEC_BODY_SOURCE_DIR_INVALID")
    selected = copy.deepcopy(prepared)
    selected_count = 0
    for case in selected["cases"]:
        path = source_dir / f"sec-{case['accession']}-primary.source"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 8_000_000:
            raise ValueError("SEC_BODY_ORIGINAL_SOURCE_REQUIRED")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != case["sourceSha256"]:
            raise ValueError("SEC_SEMANTIC_SOURCE_SHA_MISMATCH")
        span = select_body_span(case, raw)
        if span is None:
            # Keep original metadata-only evidence; existing gate abstains.
            continue
        case["sourceSpan"] = {key: value for key, value in span.items() if key != "text"}
        case["evidence"] = [{
            "id": f"sec-{case['accession']}-body-{span['visibleStart']}",
            "text": span["text"],
            "secItemMarkers": [],
            "retrospectiveSourceOnly": True,
        }]
        selected_count += 1
    selected["bodySpanSelectionVersion"] = VERSION
    selected["bodySpanCasesSelected"] = selected_count
    return selected


def reverify_body_span(case: dict, raw: bytes, claimed: dict) -> dict:
    """Reject an altered source span, its offsets, or an invented citation."""
    span = select_body_span(case, raw)
    if span is None:
        raise ValueError("SEC_BODY_SELECTED_SPAN_MISSING")
    expected = {key: value for key, value in span.items() if key != "text"}
    if claimed != expected:
        raise ValueError("SEC_BODY_SPAN_PROVENANCE_INVALID")
    verified = dict(case)
    verified["sourceSpan"] = expected
    verified["evidence"] = [{
        "id": f"sec-{case['accession']}-body-{span['visibleStart']}",
        "text": span["text"],
        "secItemMarkers": [],
        "retrospectiveSourceOnly": True,
    }]
    return verified
