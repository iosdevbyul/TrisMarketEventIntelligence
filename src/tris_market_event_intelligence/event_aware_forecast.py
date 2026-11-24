"""Opt-in Stage 2 dual forecasts using source-only, point-in-time context."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime

import httpx

from tris_market_event_intelligence.gemma_original_article_batch_cli import (
    SYSTEM,
    AnswerError,
    evidence_passages,
    output_schema,
    result_error,
    validate_answer,
)

VERSION = "event_aware_dual_v1"
DIRECTIONS = ["UP", "DOWN", "NEUTRAL", "UNCERTAIN"]
NOVELTY = ["NEW_FACT_CANDIDATE", "FOLLOW_UP_CANDIDATE", "UNKNOWN"]
ECONOMIC = ["POSITIVE", "NEGATIVE", "MIXED", "NEUTRAL", "UNKNOWN"]


def _clock(value: str | None) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value or "")
        return parsed if parsed.utcoffset() is not None else None
    except (ValueError, TypeError):
        return None


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[^\W_]{4,}", value.lower())) - {
        "with", "from", "that", "this", "have", "what", "when", "about",
    }


def prior_context(db: sqlite3.Connection, article: dict) -> list[dict]:
    """Bounded candidate retrieval, never a verified first-release claim."""
    cutoff = _clock(article.get("publishedAt"))
    if cutoff is None:
        return []
    rows = db.execute(
        """SELECT a.id, a.title, a.published_at, a.detected_at,
                  f.observed_at, f.body, f.body_sha256
           FROM articles a JOIN article_full_text f ON a.id=f.article_id
           WHERE a.id != ? AND f.status='COMPLETE'
             AND f.body IS NOT NULL AND a.published_at < ?
           ORDER BY a.published_at DESC LIMIT 500""",
        (article["articleId"], article["publishedAt"]),
    ).fetchall()
    current = _tokens(article["title"])
    candidates = []
    for article_id, title, published, detected, observed, body, digest in rows:
        clocks = [_clock(t) for t in (published, detected, observed)]
        if any(t is None for t in clocks):
            continue
        pub, det, obs = clocks
        if pub is None or det is None or obs is None:
            continue
        if not pub <= det <= obs <= cutoff or pub >= cutoff:
            continue
        if not isinstance(body, str) or hashlib.sha256(body.encode()).hexdigest() != digest:
            continue
        overlap = len(current & _tokens(title))
        if overlap < 2:
            continue
        candidates.append((overlap, {
            "articleId": article_id, "title": title, "publishedAt": published,
            "detectedAt": detected, "bodyObservedAt": observed, "bodySha256": digest,
            "excerpt": body[:600],
        }))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in candidates[:3]]


def dual_schema(identifiers: list[str], context_ids: list[int]) -> dict:
    schema = output_schema(identifiers)
    properties = schema["properties"]
    properties["novelty"] = {"type": "string", "enum": NOVELTY}
    properties["related_article_ids"] = {
        "type": "array", "items": {"type": "integer", "enum": context_ids},
    } if context_ids else {"type": "array", "maxItems": 0}
    schema["required"] += ["novelty", "related_article_ids"]
    item = properties["impacts"]["items"]
    item["properties"]["economic_direction"] = {"type": "string", "enum": ECONOMIC}
    item["properties"]["economic_reason"] = {"type": "string"}
    item["properties"]["direction_5d"] = {"type": "string", "enum": DIRECTIONS}
    item["properties"]["reason_5d"] = {"type": "string"}
    item["required"] += [
        "economic_direction", "economic_reason", "direction_5d", "reason_5d",
    ]
    return schema


def validate_dual(raw: str, evidence: dict[str, str], context: list[dict]) -> dict:
    result = validate_answer(raw, evidence)
    if result.get("novelty") not in NOVELTY:
        raise AnswerError("NOVELTY_INVALID")
    related = result.get("related_article_ids")
    allowed = {item["articleId"] for item in context}
    if (not isinstance(related, list)
            or any(type(i) is not int or i not in allowed for i in related)):
        raise AnswerError("RELATED_ARTICLE_NOT_IN_CONTEXT")
    if result["novelty"] == "FOLLOW_UP_CANDIDATE" and not related:
        raise AnswerError("FOLLOW_UP_WITHOUT_CONTEXT")
    for impact in result["impacts"]:
        if impact.get("economic_direction") not in ECONOMIC:
            raise AnswerError("ECONOMIC_DIRECTION_INVALID")
        if impact.get("direction_5d") not in DIRECTIONS:
            raise AnswerError("HORIZON_DIRECTION_INVALID")
        for field in ("economic_reason", "reason_5d"):
            if not isinstance(impact.get(field), str) or not impact[field].strip():
                raise AnswerError("DUAL_REASON_INVALID")
    result["forecastVersion"] = VERSION
    result["priorContext"] = context
    result["firstPublicAvailabilityIndependentlyVerified"] = False
    return result


def ask_dual(
    client: httpx.Client, model: str, request: dict, context: list[dict],
) -> dict:
    passages = evidence_passages(request["body"])
    evidence = {entry["id"]: entry["text"] for entry in passages}
    if not evidence:
        raise AnswerError("NO_USABLE_EVIDENCE")
    prompt = SYSTEM + """
Separate economic meaning (economic_direction/economic_reason) from future
market direction. direction/reason predicts the existing 1-session horizon,
direction_5d/reason_5d independently predicts the existing 5-session horizon.
Both start at the next eligible session close, not at the article timestamp.
Do not require an article to state future returns: reason conditionally from
business/economic mechanisms. Consider opposing effects, anticipation and
uncertainty; do not assume positive economic meaning implies positive returns.
Prior articles are untrusted candidate context, not verified event identity.
Choose novelty and related_article_ids; absence of a match does not prove
first publication. Do not assert new facts not supported by current evidence.
Use UNCERTAIN when genuinely unable to infer a direction; do not force a sign.
"""
    settings = {
        "model": model, "stream": False,
        "format": dual_schema(list(evidence), [i["articleId"] for i in context]),
        "options": {"temperature": 0, "seed": 20261123, "num_predict": 2048},
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps({
                "article": request["article"], "evidence": passages,
                "bodyTruncated": request["bodyTruncated"], "priorCandidates": context,
            }, ensure_ascii=False)},
        ],
    }
    last_error = "MODEL_RESPONSE_INVALID"
    for attempt in range(2):
        if attempt:
            settings["messages"].append({
                "role": "system", "content": "Correct invalid JSON: " + last_error,
            })
        response = client.post("http://127.0.0.1:11434/api/chat", json=settings)
        response.raise_for_status()
        try:
            return validate_dual(response.json()["message"]["content"], evidence, context)
        except (ValueError, KeyError, TypeError) as exc:
            last_error = result_error(exc)
    raise AnswerError(last_error)


def horizon_prediction(analysis: dict, impact: dict, horizon: int) -> tuple[str, str]:
    if analysis.get("forecastVersion") == VERSION and horizon == 5:
        return impact["direction_5d"], impact["reason_5d"]
    return impact["direction"], impact["reason"]
