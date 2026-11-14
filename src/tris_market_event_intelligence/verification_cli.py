"""Run local lint/tests and optionally a bounded live Ollama regression."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

from tris_market_event_intelligence.impact import PROMPT_VERSION
from tris_market_event_intelligence.impact_pilot_report import parse_event_ids


def _run(args: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=root, capture_output=True, text=True, check=False)
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr, flush=True)
    return result


def _read_json(args: list[str], root: Path) -> dict[str, object]:
    result = _run(args, root)
    if result.returncode:
        raise RuntimeError(f"Command failed with exit {result.returncode}: {args[0]}")
    try:
        record = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Command returned invalid JSON") from exc
    if not isinstance(record, dict):
        raise TypeError("Command returned non-object JSON")
    return record


def _update(root: Path) -> None:
    branch = _run(["git", "branch", "--show-current"], root)
    if branch.returncode or branch.stdout.strip() != "main":
        raise RuntimeError("--update requires main branch; no branch switch performed")
    status = _run(["git", "status", "--porcelain"], root)
    if status.returncode or status.stdout.strip():
        raise RuntimeError("--update requires a clean working tree; no reset performed")
    if _run(["git", "pull", "--ff-only", "origin", "main"], root).returncode:
        raise RuntimeError("Fast-forward pull failed; no force or reset was attempted")


def _ollama_ready(model: str) -> None:
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            response = client.get("http://127.0.0.1:11434/api/tags")
            response.raise_for_status()
            names = {item["name"] for item in response.json()["models"]}
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Ollama unavailable: {type(exc).__name__}") from exc
    if model not in names:
        raise RuntimeError("Requested local model is not installed")


def verify(
    root: Path, *, update: bool = False, live: tuple[int, ...] | None = None,
    model: str = "gemma2:latest", output_dir: Path | None = None,
) -> dict[str, object]:
    if not model.strip():
        raise ValueError("Model name is required")
    if live is not None:
        parse_event_ids(",".join(str(item) for item in live))
    result: dict[str, object] = {
        "stage": "6/8 automation; 4/8 market impact",
        "prompt_version": PROMPT_VERSION, "model_id": model,
        "offline_tests_passed": False, "live_requested": live is not None,
        "quality_verified": False,
    }
    if update:
        _update(root)
    for name, cmd in [
        ("install", [sys.executable, "-m", "pip", "install", "-e", ".[dev]"]),
        ("lint", [sys.executable, "-m", "ruff", "check", "."]),
        ("tests", [sys.executable, "-m", "pytest", "-q"]),
    ]:
        print(f"[verify] {name}", flush=True)
        child = _run(cmd, root)
        result[f"{name}_exit_code"] = child.returncode
        if child.returncode:
            result["status"] = f"{name}_failed"
            return _save(result, output_dir)
    result["offline_tests_passed"] = True
    if live is not None:
        print("[verify] checking Ollama", flush=True)
        _ollama_ready(model)
        selected = ",".join(map(str, live))
        os.environ["MARKET_EVENT_LLM_MODEL"] = model
        print(f"[verify] analyzing explicit events: {selected}", flush=True)
        result["analysis_stats"] = _read_json([
            sys.executable, "-m", "tris_market_event_intelligence.impact_cli",
            "--provider", "local", "--event-ids", selected,
            "--batch-size", str(len(live)), "--timeout-seconds", "300",
        ], root)
        result["selected_event_ids"] = list(live)
        for key, version in [
            ("current_outcomes", PROMPT_VERSION),
            ("historical_v7_outcomes", "market-impact-v7"),
        ]:
            result[key] = _read_json([
                sys.executable, "-m",
                "tris_market_event_intelligence.impact_pilot_report_cli",
                "--event-ids", selected, "--model-id", model,
                "--prompt-version", version,
            ], root)
        stats = result["analysis_stats"]
        result["status"] = (
            "live_completed_with_failures"
            if isinstance(stats, dict) and stats.get("failed", 0) else "ok"
        )
    else:
        result["status"] = "ok"
    return _save(result, output_dir)


def _save(result: dict[str, object], output_dir: Path | None) -> dict[str, object]:
    folder = output_dir or (
        Path.home() / "Library" / "Application Support"
        / "TrisMarketEventIntelligence" / "verification-reports"
    )
    folder.mkdir(parents=True, exist_ok=True)
    date = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = folder / f"verify-{date}.json"
    with path.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(f"[verify] {result['status']}; JSON: {path}", flush=True)
    return {**result, "report_path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command Mac verification")
    parser.add_argument("--update", action="store_true",
                        help="Fast-forward clean main only")
    parser.add_argument("--live", metavar="ID[,ID]", default=None,
                        help="Explicitly analyze 1..3 selected event IDs")
    parser.add_argument("--model", default=os.getenv("MARKET_EVENT_LLM_MODEL", "gemma2:latest"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    try:
        selected = parse_event_ids(args.live) if args.live is not None else None
        result = verify(
            Path(__file__).resolve().parents[2], update=args.update,
            live=selected, model=args.model, output_dir=args.output_dir,
        )
    except (ValueError, RuntimeError, TypeError, OSError) as exc:
        parser.exit(2, f"[verify] {exc}\n")
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
