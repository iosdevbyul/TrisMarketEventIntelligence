"""Read-only macOS LaunchAgent cutover preflight.

No launchctl bootstrap/bootout, no plist write, no DB changes, no Ollama.
Only reports allowlisted service presence and safe configuration properties.
"""
from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.integrated_launchd_config import INTEGRATED_LABEL
from tris_market_event_intelligence.launchd_config import LABEL as LEGACY_LABEL

MAX_PLIST_BYTES = 65_536
CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _read_plist(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_PLIST_BYTES:
        raise ValueError("Missing, symlinked or oversized LaunchAgent plist")
    with path.open("rb") as handle:
        data = plistlib.load(handle)
    if not isinstance(data, dict):
        raise ValueError("LaunchAgent plist must be a dictionary")
    return data


def _default_runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=8, check=False,
    )


def _service_state(
    label: str, *, uid: int, runner: CommandRunner,
) -> str:
    # Launchctl's exit 113 convention is not a stable product contract.
    # A nonzero result could be missing OR a permissions/runtime failure:
    # do not assume that an unsuccessful print proves an unloaded agent.
    try:
        result = runner(["launchctl", "print", f"gui/{uid}/{label}"])
    except (OSError, subprocess.TimeoutExpired):
        return "UNKNOWN"
    return "LOADED" if result.returncode == 0 else "UNKNOWN"


def audit_launchagent_cutover(
    *, home: Path, uid: int, runner: CommandRunner = _default_runner,
    observed_at: datetime | None = None,
) -> dict[str, object]:
    """Fail closed unless absence of old agent is independently confirmed.

    Due to launchctl's nonzero ambiguity, the preflight never returns
    'safe to activate' just from one missing plist or print failure.
    """
    clock = observed_at or datetime.now(UTC)
    if clock.utcoffset() is None or uid < 0:
        raise ValueError("Timezone-aware time and nonnegative user id required")
    agents = home / "Library" / "LaunchAgents"
    names = {"legacy": LEGACY_LABEL, "integrated": INTEGRATED_LABEL}
    services: dict[str, dict[str, object]] = {}
    for key, label in names.items():
        path = agents / f"{label}.plist"
        exists = path.exists() or path.is_symlink()
        item: dict[str, object] = {
            "label": label, "plist_present": exists,
            "launchctl_state": _service_state(label, uid=uid, runner=runner),
        }
        if exists:
            try:
                config = _read_plist(path)
                args = config.get("ProgramArguments")
                item["plist_status"] = "VALID" if (
                    config.get("Label") == label
                    and isinstance(args, list)
                    and len(args) >= 3
                    and isinstance(args[2], str)
                ) else "INVALID"
                if item["plist_status"] == "VALID":
                    expected = (
                        "tris_market_event_intelligence.cycle_cli"
                        if key == "legacy"
                        else "tris_market_event_intelligence.integrated_pipeline_cli"
                    )
                    item["expected_entrypoint"] = args[2] == expected
                    item["run_at_load"] = config.get("RunAtLoad")
                    item["interval_seconds"] = config.get("StartInterval")
                    item["local_analysis_configured"] = (
                        "--enable-local-analysis" in args
                        if key == "integrated" else False
                    )
                    item["has_environment_values"] = bool(config.get("EnvironmentVariables"))
            except (OSError, ValueError, TypeError, plistlib.InvalidFileException):
                item["plist_status"] = "INVALID"
        else:
            item["plist_status"] = "ABSENT"
        services[key] = item

    legacy = services["legacy"]
    integrated = services["integrated"]
    blockers: list[str] = []
    if legacy["plist_present"]:
        blockers.append("LEGACY_PLIST_STILL_PRESENT")
    if legacy["launchctl_state"] == "LOADED":
        blockers.append("LEGACY_AGENT_LOADED")
    if integrated["launchctl_state"] == "LOADED":
        blockers.append("INTEGRATED_AGENT_ALREADY_LOADED")
    if integrated["plist_present"]:
        blockers.append("INTEGRATED_CONFIG_ALREADY_PRESENT")
    for item, key in ((legacy, "LEGACY"), (integrated, "INTEGRATED")):
        if item["launchctl_state"] == "UNKNOWN":
            blockers.append(key + "_SERVICE_STATE_UNCONFIRMED")
        if item["plist_status"] == "INVALID":
            blockers.append(key + "_CONFIG_INVALID")
        if item.get("expected_entrypoint") is False:
            blockers.append(key + "_ENTRYPOINT_MISMATCH")
    return {
        "schema_version": "integrated_agent_cutover_preflight_v1",
        "as_of_utc": clock.astimezone(UTC).isoformat(),
        "status": "BLOCKED" if blockers else "NEEDS_OPERATOR_APPROVAL",
        "blockers": sorted(set(blockers)),
        "services": services,
        "safe_to_auto_activate": False,
        "changes_applied": False,
        "launchctl_commands_used": "print_only",
        "db_accessed": False,
        "ollama_contacted": False,
        "note": (
            "An unreadable or unregistered launchctl service cannot prove it is unloaded. "
            "Never auto-bootout the old collector or auto-bootstrap integrated analysis. "
            "Inspect launchctl and decide cutover explicitly on the owner's Mac."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only macOS LaunchAgent integration preflight",
    )
    args = parser.parse_args()
    try:
        report = audit_launchagent_cutover(home=Path.home(), uid=os.getuid())
    except (ValueError, OSError) as exc:
        parser.exit(2, f"cutover-preflight: {type(exc).__name__}\n")
    print(json.dumps(report, ensure_ascii=False))
    if report["status"] != "NEEDS_OPERATOR_APPROVAL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
