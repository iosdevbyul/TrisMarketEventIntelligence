"""Opt-in, read-only side-by-side NEWS hypotheses with two pinned Ollama models.

Same sourced news events, same provider prompt and validator. This is NOT
a model-accuracy benchmark, human gold evaluation, or stock-price forecast.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from tris_market_event_intelligence.impact import (
    PROMPT_VERSION,
    EventInput,
    ImpactAnalysisError,
    ImpactResult,
    validate_result,
)
from tris_market_event_intelligence.news_two_stage_pilot import select_news_events

MODELS = ("gemma2:latest", "gemma4:e4b")
MAX_EVENTS = 3


class NewsDirectionAnalyzer(Protocol):
    model_id: str

    def analyze(self, event: EventInput) -> ImpactResult: ...


def _validate_models(models: tuple[str, str]) -> None:
    if type(models) is not tuple or len(models) != 2 or (
        models[0] != "gemma2:latest"
        or models[1] not in {
            "gemma4:e2b", "gemma4:e4b", "gemma4:12b",
            "gemma4:26b", "gemma4:31b",
        }
    ):
        raise ValueError("Expected original Gemma 2 and one explicitly sized Gemma 4")


def compare_gemma_news(
    *, db_path: Path, models: tuple[str, str] = MODELS,
    analyzers: tuple[NewsDirectionAnalyzer, NewsDirectionAnalyzer] | None = None,
    limit: int = 2, clock: Callable[[], float] = time.monotonic,
    generic_models: bool = False,
) -> dict[str, object]:
    """Compare schema/evidence acceptance and latency on identical sourced events.

    This intentionally never scores semantic correctness or self-judged
    intelligence. Each event's original news text stays out of the report.
    """
    if generic_models:
        if (type(models) is not tuple or len(models) != 2
                or not all(isinstance(model, str) and model.strip()
                           and len(model) <= 120 for model in models)
                or models[0] == models[1]):
            raise ValueError("Two distinct explicit model IDs are required")
    else:
        _validate_models(models)
    if type(limit) is not int or not 1 <= limit <= MAX_EVENTS:
        raise ValueError("Maximum 1..3 NEWS-only events per controlled pilot")
    if analyzers is not None and (
        len(analyzers) != 2
        or any(analyzer.model_id != requested
               for analyzer, requested in zip(analyzers, models, strict=True))
    ):
        raise ValueError("Each analyzer must use the exact requested model ID")
    # Selecting with existing code only, against read-only archived news DB.
    events = select_news_events(db_path, limit=limit, model_id=models[0])
    entries: list[dict[str, object]] = []
    counts: dict[str, Counter[str]] = {name: Counter() for name in models}
    for event in events:
        fingerprint = hashlib.sha256(json.dumps({
            "eventId": event.event_id,
            "evidence": event.evidence,
            "status": event.status,
            "classification": event.classification,
            "sourceUrl": event.source_url,
        }, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        row: dict[str, object] = {
            "eventId": event.event_id,
            "newsInputSha256": fingerprint,
            "modelRuns": [],
        }
        if analyzers is None:
            for name in models:
                row["modelRuns"].append({
                    "modelId": name, "status": "PREVIEW_NO_MODEL_CALL",
                })
                counts[name]["PREVIEW_NO_MODEL_CALL"] += 1
        else:
            for analyzer, name in zip(analyzers, models, strict=True):
                started = clock()
                try:
                    result = validate_result(event, analyzer.analyze(event))
                    accepted = len(result.impacts)
                    code = "ACCEPTED_EVIDENCE_BOUND"
                    if accepted == 0:
                        code = "ACCEPTED_NO_HYPOTHESIS"
                    model_row: dict[str, object] = {
                        "modelId": name, "status": code,
                        "validatedHypotheses": accepted,
                        "exactSourceEvidenceValidated": True,
                    }
                except ImpactAnalysisError as exc:
                    model_row = {
                        "modelId": name, "status": "REJECTED",
                        "failureCode": exc.code,
                        "exactSourceEvidenceValidated": False,
                    }
                except Exception:  # noqa: BLE001
                    model_row = {
                        "modelId": name, "status": "MODEL_RUNTIME_FAILED",
                        "failureCode": "MODEL_OR_NETWORK_ERROR",
                        "exactSourceEvidenceValidated": False,
                    }
                model_row["elapsedMs"] = round(max(0.0, (clock() - started) * 1000), 3)
                row["modelRuns"].append(model_row)
                counts[name][model_row["status"]] += 1
        entries.append(row)
    return {
        "schemaVersion": ("news_llm_side_by_side_v1" if generic_models
                          else "gemma4_news_side_by_side_v1"),
        "status": "PREVIEW_ONLY" if analyzers is None else (
            "NO_ELIGIBLE_NEWS" if not events else "COMPARISON_RECORDED_UNGRADED"
        ),
        "comparisonKind": "SAME_NEWS_INPUT_CONTRACT_AND_RUNTIME_NOT_GOLD_ACCURACY",
        "models": list(models),
        "promptVersion": PROMPT_VERSION,
        "selectedNewsEvents": len(events),
        "maxNewsEvents": limit,
        "modelCallCount": 0 if analyzers is None else len(events) * 2,
        "countsByModel": {key: dict(sorted(value.items())) for key, value in counts.items()},
        "items": entries,
        "sourceTextPersisted": False,
        "humanGoldUsed": False,
        "independentNewsAccuracyMeasured": False,
        "newsContextUnderstandingVerified": False,
        "candidateImprovementVerified": False,
        "productionModelChanged": False,
        "productionDbModified": False,
        "modelDownloadedOrInstalled": False,
        "promotionAllowed": False,
        "marketPricesUsed": False,
        "note": (
            "Contract acceptance and elapsed time cannot determine which model "
            "understands the news better. Independent human review is required."
        ),
    }


def compare_news_models(
    *, db_path: Path, models: tuple[str, str],
    analyzers: tuple[NewsDirectionAnalyzer, NewsDirectionAnalyzer] | None = None,
    limit: int = 2, clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Generic pair using the original same-news selector and evidence validator."""
    return compare_gemma_news(
        db_path=db_path, models=models, analyzers=analyzers,
        limit=limit, clock=clock, generic_models=True,
    )
