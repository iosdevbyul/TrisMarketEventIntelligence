"""Human-adjudicated semantic quality evaluation, kept outside production DBs.

Gold rows are signed by neither the app nor an external authority: the reviewer
and label are human-provided metadata, not independently authenticated truth.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from tris_market_event_intelligence.news_pilot_review_packet import NewsPilotReviewPacket

SCHEMA = "news_semantic_gold_v1"
DECISIONS = frozenset({"SUPPORTED", "CONTRADICTED", "UNRELATED", "INSUFFICIENT_CONTEXT"})
ERROR_TYPES = frozenset({
    "ACTOR_MISMATCH", "POLARITY_REVERSAL", "UNRELATED_EVIDENCE",
    "CONDITION_OMITTED", "NEGATION_OMITTED", "TIME_MISMATCH",
    "UNSUPPORTED_CAUSALITY", "OTHER",
})


def load_gold(path: Path) -> list[dict[str, object]]:
    """Require a small, explicit, non-symlink human-labelled JSON input."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("Existing non-symlink human gold file required")
    if path.stat().st_size > 1_000_000:
        raise ValueError("Human gold file exceeds size limit")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"schemaVersion", "judgments"}:
        raise ValueError("Invalid gold document")
    if data["schemaVersion"] != SCHEMA or not isinstance(data["judgments"], list):
        raise ValueError("Unsupported human gold schema")
    if not 1 <= len(data["judgments"]) <= 500:
        raise ValueError("Gold requires 1..500 judgments")
    seen: set[int] = set()
    for row in data["judgments"]:
        if not isinstance(row, dict) or set(row) != {
            "journalEvidenceId", "signalSha256", "sourceEvidenceSha256",
            "reviewer", "reviewedAt", "decision", "errorTypes", "reason",
        }:
            raise ValueError("Incomplete gold judgment")
        evidence_id = row["journalEvidenceId"]
        if type(evidence_id) is not int or evidence_id < 1 or evidence_id in seen:
            raise ValueError("Invalid or duplicate gold evidence ID")
        seen.add(evidence_id)
        if row["decision"] not in DECISIONS:
            raise ValueError("Unsupported human verdict")
        if not all(
            isinstance(row[key], str) and len(row[key]) == 64
            and all(char in "0123456789abcdef" for char in row[key])
            for key in ("signalSha256", "sourceEvidenceSha256")
        ):
            raise ValueError("Invalid immutable provenance hashes")
        if (
            not isinstance(row["reviewer"], str)
            or not 1 <= len(row["reviewer"].strip()) <= 100
            or not isinstance(row["reason"], str)
            or not 20 <= len(row["reason"].strip()) <= 2000
        ):
            raise ValueError("Human reviewer and substantive reason required")
        if not isinstance(row["reviewedAt"], str):
            raise ValueError("Human review time required")
        try:
            recorded = datetime.fromisoformat(row["reviewedAt"])
        except ValueError as exc:
            raise ValueError("Invalid review timestamp") from exc
        if recorded.tzinfo is None or recorded.utcoffset() is None:
            raise ValueError("Timezone-aware review timestamp required")
        types = row["errorTypes"]
        if not isinstance(types, list) or len(types) > 8 or any(
            not isinstance(code, str) or code not in ERROR_TYPES for code in types
        ) or len(set(types)) != len(types):
            raise ValueError("Invalid semantic error types")
        if row["decision"] == "SUPPORTED" and types:
            raise ValueError("Supported judgment cannot assert semantic errors")
        if row["decision"] in {"CONTRADICTED", "UNRELATED"} and not types:
            raise ValueError("Semantic errors require a categorized cause")
    return data["judgments"]


def evaluate_gold(
    *, source_db: Path, journal_db: Path, gold_path: Path,
) -> dict[str, object]:
    """Evaluate review flags against *human supplied* judgments only.

    No negatives are generated from missing labels or unanswered questions.
    A flag is a triage suggestion, not a semantic verdict.
    """
    labels = load_gold(gold_path)
    reader = NewsPilotReviewPacket(source_db=source_db, journal_db=journal_db)
    counts: Counter[str] = Counter({decision: 0 for decision in sorted(DECISIONS)})
    cells: Counter[str] = Counter({
        "truePositive": 0, "falsePositive": 0,
        "trueNegative": 0, "falseNegative": 0, "unscored": 0,
    })
    types: Counter[str] = Counter()
    items = []
    for row in labels:
        packet = reader.build(evidence_id=row["journalEvidenceId"])
        provenance = packet["provenance"]
        if (
            row["signalSha256"] != provenance["signalSha256"]
            or row["sourceEvidenceSha256"] != provenance["sourceEvidenceSha256"]
        ):
            raise ValueError("Gold label no longer matches immutable source")
        verdict = row["decision"]
        counts[verdict] += 1
        types.update(row["errorTypes"])
        codes = packet["semanticContextReview"]["reviewCodes"]
        flag = bool(codes)
        if verdict == "INSUFFICIENT_CONTEXT":
            cell = "unscored"
        elif verdict == "SUPPORTED":
            cell = "falsePositive" if flag else "trueNegative"
        else:
            cell = "truePositive" if flag else "falseNegative"
        cells[cell] += 1
        items.append({
            "journalEvidenceId": row["journalEvidenceId"],
            "humanVerdict": verdict,
            "reviewer": row["reviewer"],
            "semanticErrorTypes": row["errorTypes"],
            "triageReviewCodes": codes,
            "triageFlagged": flag,
            "comparison": cell,
        })

    scored = sum(cells[x] for x in (
        "truePositive", "falsePositive", "trueNegative", "falseNegative",
    ))
    tp, fp, fn = (cells[k] for k in ("truePositive", "falsePositive", "falseNegative"))
    # Ten examples make reporting less misleading, not statistically significant.
    precision = tp / (tp + fp) if scored >= 10 and tp + fp else None
    recall = tp / (tp + fn) if scored >= 10 and tp + fn else None
    return {
        "schemaVersion": "news_semantic_gold_evaluation_v1",
        "status": "DESCRIPTIVE_METRICS" if scored >= 10 else "INSUFFICIENT_GOLD",
        "totalHumanJudgments": len(labels),
        "scoredJudgments": scored,
        "humanVerdicts": dict(sorted(counts.items())),
        "errorTypes": dict(sorted(types.items())),
        "confusion": dict(sorted(cells.items())),
        "triagePrecision": precision,
        "triageRecall": recall,
        "precisionRecallStatisticallyValidated": False,
        "semanticallyVerified": False,
        "independentSourceVerification": False,
        "humanReviewerAuthenticated": False,
        "modelAccuracyVerified": False,
        "items": items,
        "sourceDbModified": False,
        "journalModified": False,
        "published": False,
    }
