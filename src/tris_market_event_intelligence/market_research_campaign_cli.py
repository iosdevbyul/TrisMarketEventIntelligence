"""Single-entry, resumable research diagnosis and bounded historical comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import httpx

from tris_market_event_intelligence.autonomous_market_experience_cli import read_sealed
from tris_market_event_intelligence.experience_memory import digest
from tris_market_event_intelligence.historical_experience_abc_v2_cli import (
    _sealed,
    load_experiences,
)
from tris_market_event_intelligence.independent_market_validation_cli import run as validate_run
from tris_market_event_intelligence.market_research_diagnostics_cli import run as diagnose


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace, client: httpx.Client) -> dict:
    root = args.data_dir.resolve()
    development = root / "autonomous-market-experience-v2"
    previous = root / "autonomous-market-experience-v1"
    seed = root / "fresh-forecast-learning-v1"
    forecasts = root / "fresh-market-forecast-v1"
    mapping = root / "semantic-market-resolution-v2"
    prices = mapping / "expanded-adjusted-daily.csv"
    out = args.output_dir.resolve()
    if args.output_dir.is_symlink() or out == root or any(
        out.is_relative_to(p) for p in (development, previous, seed, forecasts, mapping)
    ):
        raise ValueError("Output must be separate from existing research inputs")
    if not 1 <= args.limit <= 30:
        raise ValueError("Validation limit must be 1..30")
    config = {
        "version": 1,
        "developmentSha256": read_sealed(development / "report.json")["recordSha256"],
        "cohortSha256": read_sealed(development / "cohort.json")["recordSha256"],
        "previousCohortSha256": read_sealed(previous / "cohort.json")["recordSha256"],
        "seedDigest": digest(load_experiences(seed)),
        "pricesSha256": file_hash(prices),
        "limit": args.limit,
        "model": args.model,
    }
    out.mkdir(parents=True, exist_ok=True)
    _sealed(out / "campaign.json", config)
    diagnostic_dir = out / "diagnostics"
    diagnostic_path = diagnostic_dir / "research-diagnostics.json"
    receipt_path = out / "diagnostics-receipt.json"
    if receipt_path.exists():
        receipt = read_sealed(receipt_path)
        if receipt["reportSha256"] != file_hash(diagnostic_path):
            raise ValueError("Diagnostic report digest mismatch")
    else:
        # Never silently trust an unreceipted report left by an interrupted write.
        if diagnostic_path.exists():
            raise ValueError("Unreceipted diagnostic report; preserve it and use a new output")
        diagnose(development, seed, prices, diagnostic_dir)
        _sealed(receipt_path, {"reportSha256": file_hash(diagnostic_path)})
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    print(json.dumps({"event": "DIAGNOSTICS_COMPLETED",
                      "report": str(diagnostic_path)}), flush=True)
    validation = validate_run(argparse.Namespace(
        db=root / "news.sqlite3",
        mappings=mapping / "reviewed-alias-mappings.csv",
        prices=prices,
        forecasts=forecasts,
        experiences=seed,
        exclude_run=[previous, development],
        output_dir=out / "validation",
        model=args.model,
        limit=args.limit,
        preflight=args.diagnostics_only,
    ), client)
    report = {
        "status": validation["status"],
        "diagnosticReport": str(diagnostic_path),
        "diagnosticSha256": file_hash(diagnostic_path),
        "developmentArticles": diagnostic["prediction"]["articles"],
        "experienceCount": diagnostic["reflection"]["newExperiences"],
        "ruleDiagnosis": diagnostic["rules"]["statusCounts"],
        "validation": validation,
        "historicalResearchOnly": True,
        "promotionAllowed": False,
        "prospectiveAccuracy": None,
    }
    if validation["status"] == "INDEPENDENT_HISTORICAL_COMPARISON":
        result = read_sealed(out / "validation" / "report.json")
        report["validationMetrics"] = result["metrics"]
        report["validationSha256"] = result["recordSha256"]
        _sealed(out / "final-research-report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--diagnostics-only", action="store_true")
    with httpx.Client(timeout=180, trust_env=False) as client:
        print(json.dumps(run(parser.parse_args(), client), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
