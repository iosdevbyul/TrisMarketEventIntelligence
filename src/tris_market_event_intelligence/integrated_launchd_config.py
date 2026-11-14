"""Explicit opt-in macOS integrated news pipeline schedule; no activation.

This is a *replacement* for the earlier RSS-only LaunchAgent, not a
second concurrently scheduled collector. No launchctl calls occur here.
"""
from __future__ import annotations

import argparse
import os
import plistlib
import re
import sys
from pathlib import Path

from tris_market_event_intelligence.launchd_config import LABEL as LEGACY_LABEL
from tris_market_event_intelligence.operations import OPERATIONAL_SOURCES

INTEGRATED_LABEL = "com.tris.market-event-intelligence-integrated"
MIN_INTERVAL = 300
MAX_INTERVAL = 86400
_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")


def integrated_launchd_configuration(
    python_path: Path, *, db_path: Path, status_path: Path,
    interval_seconds: int = 900,
    sources: tuple[str, ...] = OPERATIONAL_SOURCES,
    enable_local_analysis: bool = False,
    confirmed_background_inference: bool = False,
    model_name: str | None = None,
) -> dict[str, object]:
    """Safe plist content; no filesystem changes, no service activation."""
    if type(interval_seconds) is not int or not MIN_INTERVAL <= interval_seconds <= MAX_INTERVAL:
        raise ValueError("Interval must be 300..86400 seconds")
    if any(not path.is_absolute() for path in (python_path, db_path, status_path)):
        raise ValueError("All paths must be absolute")
    if any(path.is_symlink() for path in (db_path, status_path)):
        raise ValueError("Symlinked DB or status paths are not supported")
    if not sources or len(set(sources)) != len(sources) or set(sources) - set(OPERATIONAL_SOURCES):
        raise ValueError("Select unique existing operational RSS source IDs")
    if enable_local_analysis:
        if not confirmed_background_inference:
            raise ValueError("Local model requires explicit background inference confirmation")
        if model_name is None or not _MODEL.fullmatch(model_name):
            raise ValueError("Specify a safe installed local model identifier")
    elif confirmed_background_inference or model_name is not None:
        raise ValueError("Model settings require explicit local analysis opt-in")
    args = [
        str(python_path), "-m", "tris_market_event_intelligence.integrated_pipeline_cli",
        "--db", str(db_path), "--status", str(status_path),
        "--sources", ",".join(sources),
    ]
    if enable_local_analysis:
        args.append("--enable-local-analysis")
    logs = Path.home() / "Library" / "Logs" / "TrisMarketEventIntelligence"
    config: dict[str, object] = {
        "Label": INTEGRATED_LABEL,
        "ProgramArguments": args,
        "StartInterval": interval_seconds,
        "RunAtLoad": False,
        "StandardOutPath": str(logs / "integrated-stdout.log"),
        "StandardErrorPath": str(logs / "integrated-stderr.log"),
        "ProcessType": "Background",
    }
    if enable_local_analysis:
        # No API token, URL or raw prompt in the plist; loopback-only Ollama.
        config["EnvironmentVariables"] = {"MARKET_EVENT_LLM_MODEL": model_name}
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview opt-in integrated macOS news schedule; does not activate it",
    )
    base = Path.home() / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    agents = Path.home() / "Library" / "LaunchAgents"
    parser.add_argument("--db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--status", type=Path, default=base / "integrated-last-run.json")
    parser.add_argument("--interval", type=int, default=900)
    parser.add_argument("--sources", default=",".join(OPERATIONAL_SOURCES))
    parser.add_argument("--enable-local-analysis", action="store_true")
    parser.add_argument("--confirm-background-inference", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    try:
        config = integrated_launchd_configuration(
            Path(sys.executable).resolve(),
            db_path=args.db.expanduser().absolute(),
            status_path=args.status.expanduser().absolute(),
            interval_seconds=args.interval,
            sources=tuple(s.strip() for s in args.sources.split(",")),
            enable_local_analysis=args.enable_local_analysis,
            confirmed_background_inference=args.confirm_background_inference,
            model_name=args.model,
        )
        old_agent = agents / (LEGACY_LABEL + ".plist")
        destination = agents / (INTEGRATED_LABEL + ".plist")
        if args.write:
            # Presence of legacy configuration might mean an active collector.
            # Never permit two scheduled cycles against the same RSS sources.
            if old_agent.exists() or old_agent.is_symlink():
                raise ValueError("Legacy collector plist exists; manually boot it out and remove it first")
            agents.mkdir(parents=True, exist_ok=True)
            # 'x' avoids overwriting existing operator configuration.
            with destination.open("xb") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(plistlib.dumps(config))
            print(f"Created but NOT activated: {destination}")
        else:
            print(plistlib.dumps(config).decode())
        print("Explicit operator launchctl activation is required; never run alongside old RSS job.")
    except (ValueError, FileExistsError, OSError) as exc:
        parser.exit(2, f"integrated-launchd: {type(exc).__name__}\n")


if __name__ == "__main__":
    main()
