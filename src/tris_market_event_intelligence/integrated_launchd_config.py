"""Explicit opt-in macOS integrated news pipeline schedule; no activation.

This is a *replacement* for the earlier RSS-only LaunchAgent, not a
second concurrently scheduled collector. No launchctl calls occur here.
"""
from __future__ import annotations

import argparse
import os
import plistlib
import re
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.launchd_config import LABEL as LEGACY_LABEL
from tris_market_event_intelligence.listing_news_validation import _verified_bundle
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
    enable_v3_cutover_catchup: bool = False,
    listing_bundle: Path | None = None,
    review_output_root: Path | None = None,
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
    if enable_v3_cutover_catchup:
        if enable_local_analysis or confirmed_background_inference or model_name is not None:
            raise ValueError("Scheduled v3 may not activate a local model")
        if listing_bundle is None or review_output_root is None:
            raise ValueError("Scheduled v3 requires explicit pinned listing inputs")
        if any(not p.is_absolute() or p.is_symlink()
               for p in (listing_bundle, review_output_root)):
            raise ValueError("Scheduled v3 requires absolute non-symlink review paths")
    elif listing_bundle is not None or review_output_root is not None:
        raise ValueError("Listing review inputs require explicit v3 opt-in")
    args = [
        str(python_path), "-m", "tris_market_event_intelligence.integrated_pipeline_cli",
        "--db", str(db_path), "--status", str(status_path),
        "--sources", ",".join(sources),
    ]
    if enable_local_analysis:
        args.append("--enable-local-analysis")
    if enable_v3_cutover_catchup:
        assert listing_bundle is not None and review_output_root is not None
        args.extend([
            "--enable-v3-cutover-catchup", "--enable-ticker-review",
            "--listing-bundle", str(listing_bundle),
            "--review-output-root", str(review_output_root),
        ])
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


def verified_v3_activation(db_path: Path, *, now: datetime | None = None) -> dict[str, object]:
    """Read-only preflight for an already activated v3 cutover marker."""
    stamp = now or datetime.now(UTC)
    if stamp.utcoffset() is None:
        raise ValueError("Timezone-aware preflight time required")
    if db_path.is_symlink() or not db_path.is_file():
        raise ValueError("Existing non-symlink operational SQLite required")
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        row = db.execute(
            """SELECT 1 FROM sqlite_master WHERE type='table'
               AND name='event_extractor_rollout'"""
        ).fetchone()
        if row is None:
            raise ValueError("Explicit v3 cutover must be activated manually first")
        marker = db.execute(
            """SELECT activated_at, article_id_floor FROM event_extractor_rollout
               WHERE extractor_version='multi-v3'"""
        ).fetchone()
        if marker is None:
            raise ValueError("Existing multi-v3 activation required")
        last_id = int(db.execute(
            "SELECT COALESCE(MAX(id), 0) FROM articles"
        ).fetchone()[0])
    try:
        activated_at = datetime.fromisoformat(marker[0])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid v3 activation timestamp") from exc
    if (
        activated_at.utcoffset() is None
        or activated_at.astimezone(UTC) > stamp.astimezone(UTC)
        or type(marker[1]) is not int
        or not 0 <= marker[1] <= last_id
    ):
        raise ValueError("Invalid v3 activation fence")
    return {
        "activated_at": activated_at.astimezone(UTC).isoformat(),
        "article_id_floor": marker[1],
        "read_only": True,
    }


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
    parser.add_argument(
        "--enable-v3-cutover-catchup", action="store_true",
        help="Prepare v3 instead of v2; requires existing manual activation",
    )
    parser.add_argument("--listing-bundle", type=Path)
    parser.add_argument("--review-output-root", type=Path)
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
            enable_v3_cutover_catchup=args.enable_v3_cutover_catchup,
            listing_bundle=args.listing_bundle.expanduser().absolute()
            if args.listing_bundle else None,
            review_output_root=args.review_output_root.expanduser().absolute()
            if args.review_output_root else None,
        )
        old_agent = agents / (LEGACY_LABEL + ".plist")
        destination = agents / (INTEGRATED_LABEL + ".plist")
        if args.write and args.enable_v3_cutover_catchup:
            # Fail closed without a persisted manual cutover and pinned
            # issuer evidence. No new SQLite activation, approval or replay.
            assert args.listing_bundle is not None
            assert args.review_output_root is not None
            bundle = args.listing_bundle.expanduser().absolute()
            review = args.review_output_root.expanduser().absolute()
            _verified_bundle(bundle)
            if not review.is_dir() or review.is_symlink():
                raise ValueError("Existing non-symlink private review folder required")
            verified_v3_activation(args.db.expanduser().absolute())
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
