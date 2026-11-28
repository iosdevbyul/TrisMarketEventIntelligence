"""Choose bounded research actions from recorded results; never promote a rule."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tris_market_event_intelligence.experience_memory import digest


def fingerprint(rule: dict) -> str:
    return digest({
        key: " ".join(rule[key].lower().split())
        for key in ("condition", "analysis_action", "exceptions", "hypothesis")
    })


def rejected_history(directory: Path) -> list[dict]:
    """Read existing review artifacts, including those from earlier versions."""
    history = {}
    for path in sorted(directory.glob("*.review.json")):
        if path.is_symlink():
            raise ValueError("Unexpected review symlink")
        review = json.loads(path.read_text(encoding="utf-8"))
        if path.name != digest(review) + ".review.json":
            raise ValueError("Review artifact digest mismatch")
        initial_id = review["originalRuleId"]
        original = directory / (initial_id + ".rule.json")
        if not original.is_file() or original.is_symlink():
            continue
        rule = json.loads(original.read_text(encoding="utf-8"))
        if digest({k: v for k, v in rule.items() if k != "ruleId"}) != initial_id:
            raise ValueError("Rule artifact digest mismatch")
        for candidate, assessment in (
            (rule, review["initialReview"]),
            (review.get("revisedRule"), review.get("revisionReview")),
        ):
            if not candidate or not assessment:
                continue
            if assessment["effectiveVerdict"] == "PASS_CANDIDATE":
                continue
            key = fingerprint(candidate)
            history[key] = {
                "fingerprint": key,
                "condition": candidate["condition"],
                "analysis_action": candidate["analysis_action"],
                "exceptions": candidate["exceptions"],
                "hypothesis": candidate["hypothesis"],
                "reason": assessment["reason"],
                "flags": assessment["deterministicFlags"],
            }
    return list(history.values())


def coverage_diagnostics(report: dict, rows: list[dict]) -> dict:
    ledger = report["automaticLedgerCoverage"]
    targets: Counter[str] = Counter()
    for item in report["items"]:
        if item["status"] == "UNMAPPED":
            targets[item["target"]] += 1
    return {
        "sourceArticles": ledger["sourceArticleCount"],
        "completedCurrentBodyModelRows": ledger["completedCurrentBodyModelRows"],
        "selectedCompletedRows": ledger["selectedCompletedRows"],
        "ledgerTruncated": ledger["truncated"],
        "articlesInspected": report["articlesInspected"],
        "impactsInspected": report["impactsInspected"],
        "statusCounts": report["statusCounts"],
        "unmappedTargets": dict(targets.most_common(25)),
        "scorableRows": len(rows),
        "distinctScorableArticles": len({r["articleId"] for r in rows}),
        "dailyMetrics": report["metrics"],
    }


def choose_strategy(history: list[dict], records: list[dict]) -> str:
    if len({r["outcome"]["articleId"] for r in records}) < 3:
        return "ACQUIRE_MORE_EXPERIENCES"
    if history:
        return "EXPLORE_ALTERNATIVE_HYPOTHESIS"
    return "GENERATE_INITIAL_HYPOTHESIS"
