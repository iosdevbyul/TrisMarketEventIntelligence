"""Bounded, source-pinned real NEWS hypothesis pilot; never creates ground truth.

Runs against the existing 48-candidate private packet, not an invented
random selection and not a full-cohort train/validation accuracy claim.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import httpx
from pydantic import ValidationError

from tris_market_event_intelligence.llm_runtime import ConfiguredLLMProvider
from tris_market_event_intelligence.news_importance_review import _cohort
from tris_market_event_intelligence.news_materiality_local import (
    NewsModelGenerationError,
    _analyze_one,
    _local_model_ready,
)
from tris_market_event_intelligence.news_offline_review_page import (
    _verify_pinned_sample,
)

# Whitelisted schema diagnostics only; never record a validation error's input.
_ALLOWED_FIELDS = frozenset({
    "schema_version", "article_id", "text_sha256", "entity",
    "event_summary", "fact_status", "potential_magnitude", "direction",
    "evidence_excerpt", "conditional_mechanism", "verification_needed",
    "magnitude_basis", "market_prices_used", "price_prediction",
    "independent_human_gold",
})


def _schema_failure_field(exc: ValidationError) -> str:
    issues = exc.errors(include_input=False, include_context=False)
    for issue in issues:
        for member in issue.get("loc", ()):
            if isinstance(member, str) and member in _ALLOWED_FIELDS:
                return member
    return "root"


def _failure_code(exc: Exception) -> str:
    """Use only fixed diagnostic codes, never raw model/source/error text."""
    if isinstance(exc, NewsModelGenerationError):
        return exc.code
    if isinstance(exc, httpx.TimeoutException):
        return "MODEL_TIMEOUT"
    if isinstance(exc, httpx.HTTPStatusError):
        return "MODEL_HTTP_ERROR"
    if isinstance(exc, httpx.HTTPError):
        return "MODEL_TRANSPORT_ERROR"
    if isinstance(exc, ValidationError):
        errors = exc.errors(include_input=False, include_context=False)
        if errors and errors[0].get("type") == "json_invalid":
            return "MODEL_JSON_INVALID"
        return "MODEL_SCHEMA_INVALID"
    if isinstance(exc, json.JSONDecodeError):
        return "MODEL_RESPONSE_JSON_INVALID"
    if isinstance(exc, ValueError):
        return {
            "Model JSON exceeds bounded size": "MODEL_JSON_TOO_LARGE",
            "Model article provenance mismatch": "ARTICLE_PROVENANCE_MISMATCH",
            "Model invented evidence or entity not present in source":
                "SOURCE_EVIDENCE_MISMATCH",
            "Unassessed source must not assert a direction":
                "ABSTENTION_DIRECTION_MISMATCH",
            "Assessed magnitude lacks an identified mechanism":
                "MATERIALITY_MECHANISM_NOT_IDENTIFIED",
        }.get(str(exc), "MODEL_CONTENT_INVALID")
    return "MODEL_RESPONSE_INVALID"


MAX_PILOT = 3
ALLOWED_MODELS = frozenset({"gemma4:e2b", "gemma2:latest"})


def audit_local_news_pilot(*, cohort_path: Path, sample_path: Path,
                           model_id: str = "gemma4:e2b", run: bool = False,
                           limit: int = 2,
                           client: httpx.Client | None = None,
                           runtime: ConfiguredLLMProvider | None = None) -> dict:
    """Validated candidate selection; model contract outcomes are not correctness."""
    if runtime is None and model_id not in ALLOWED_MODELS:
        raise ValueError("Pilot needs explicitly approved local model ID")
    if runtime is not None and runtime.model_id != model_id:
        raise ValueError("Explicit pilot model ID differs from injected model")
    if type(limit) is not int or not 1 <= limit <= MAX_PILOT:
        raise ValueError("News pilot must select 1..3 candidates")
    cohort_raw, cohort = _cohort(cohort_path)
    sample = _verify_pinned_sample(cohort_path, sample_path)
    originals = {row["articleId"]: row for row in cohort["items"]}
    selected = []
    for representative in sample["items"][:limit]:
        original = originals[representative["articleId"]]
        if original["textSha256"] != representative["textSha256"]:
            raise ValueError("Candidate source fingerprint mismatch")
        selected.append(original)
    common = {
        "schemaVersion": "news_local_evidence_pilot_v1",
        "cohortSha256": hashlib.sha256(cohort_raw).hexdigest(),
        "sampleSha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        "modelId": model_id,
        "selectedArticles": len(selected),
        "totalOriginalArticles": len(cohort["items"]),
        "sampleCandidates": len(sample["items"]),
        "completeCohortInferred": False,
        "independentEventsVerified": False,
        "humanGoldUsed": False,
        "independentNewsAccuracyMeasured": False,
        "modelImprovementValidated": False,
        "promotionAllowed": False,
        "stockPricesUsed": False,
        "productionDbModified": False,
        "modelDownloadedOrInstalled": False,
        "outputIsUnverifiedResearch": True,
    }
    if not run:
        return {**common, "status": "PREVIEW_ONLY",
                "modelCalls": 0, "items": [], "counts": {}}
    owned = client is None and runtime is None
    session = client or (httpx.Client(
        timeout=httpx.Timeout(120), trust_env=False,
    ) if runtime is None else None)
    entries = []
    counts: Counter[str] = Counter()
    try:
        installed = (runtime.ensure_local_installed()
                     if runtime is not None and runtime.settings.kind == "ollama"
                     else True if runtime is not None
                     else _local_model_ready(session, model_id))
        if not installed:
            return {**common, "status": "LOCAL_MODEL_NOT_INSTALLED",
                    "modelCalls": 0, "items": [], "counts": {}}
        for article in selected:
            try:
                hypothesis = _analyze_one(
                    session, article, model_id, structured=True, runtime=runtime,
                    runtime_json_schema=(runtime.settings.capabilities.json_schema
                                         if runtime is not None else True),
                )
            except (httpx.HTTPError, ValueError, KeyError, TypeError,
                    ValidationError, json.JSONDecodeError) as exc:
                reason_code = _failure_code(exc)
                schema_field = (_schema_failure_field(exc)
                                if isinstance(exc, ValidationError) else None)
                code = "REJECTED_OR_MODEL_RUNTIME_FAILED"
                row = {
                    "articleId": article["articleId"],
                    "textSha256": article["textSha256"],
                    "status": code, "failureCode": reason_code,
                    "schemaFailureField": schema_field,
                    "contractAccepted": False, "hypothesis": None,
                }
            else:
                code = ("ACCEPTED_ABSTAINED" if hypothesis.potential_magnitude
                        == "UNASSESSED" else "ACCEPTED_UNVERIFIED_HYPOTHESIS")
                row = {
                    "articleId": article["articleId"],
                    "textSha256": article["textSha256"],
                    "status": code, "contractAccepted": True,
                    "hypothesis": hypothesis.model_dump(),
                }
            entries.append(row)
            counts[code] += 1
            if not row["contractAccepted"]:
                counts[reason_code] += 1
                if schema_field is not None:
                    counts["SCHEMA_FIELD_" + schema_field.upper()] += 1
    except (httpx.HTTPError, ValueError, TypeError, KeyError,
            json.JSONDecodeError):
        return {**common, "status": "LOCAL_MODEL_PREFLIGHT_FAILED",
                "modelCalls": 0, "items": [], "counts": {}}
    finally:
        if owned and session is not None:
            session.close()
    return {
        **common,
        "status": "UNVERIFIED_LOCAL_NEWS_PILOT_RECORDED",
        "modelCalls": len(entries),
        "counts": dict(sorted(counts.items())),
        "items": entries,
        "contractAcceptanceNotSemanticAccuracy": True,
        "modelClaimsRequireIndependentVerification": True,
    }
