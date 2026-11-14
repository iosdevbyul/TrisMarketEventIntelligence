"""LaunchAgent cutover preflight never alters a real Mac."""
from __future__ import annotations

import plistlib
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from tris_market_event_intelligence.integrated_cutover_preflight import (
    audit_launchagent_cutover,
    main,
)
from tris_market_event_intelligence.integrated_launchd_config import INTEGRATED_LABEL
from tris_market_event_intelligence.launchd_config import LABEL as LEGACY_LABEL


class MacServiceProbe:
    def __init__(self, *, loaded: tuple[str, ...] = (), ambiguous: tuple[str, ...] = ()):
        self.loaded = loaded
        self.ambiguous = ambiguous
        self.commands: list[list[str]] = []

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(argv)
        assert argv[:2] == ["launchctl", "print"]
        label = argv[2].split("/")[-1]
        if label in self.loaded:
            return subprocess.CompletedProcess(argv, 0, stdout="confidential config", stderr="")
        if label in self.ambiguous:
            return subprocess.CompletedProcess(argv, 3, stdout="", stderr="Permission denied")
        return subprocess.CompletedProcess(
            argv, 113, stdout="", stderr=f'Could not find service "{label}" in domain',
        )


def write_plist(home: Path, label: str, *, wrong_entrypoint: bool = False) -> None:
    path = home / "Library" / "LaunchAgents" / (label + ".plist")
    path.parent.mkdir(parents=True, exist_ok=True)
    module = (
        "tris_market_event_intelligence.cycle_cli"
        if label == LEGACY_LABEL else "tris_market_event_intelligence.integrated_pipeline_cli"
    )
    if wrong_entrypoint:
        module = "untrusted.module"
    path.write_bytes(plistlib.dumps({
        "Label": label,
        "ProgramArguments": ["/usr/bin/python3", "-m", module],
        "RunAtLoad": False, "StartInterval": 900,
    }))


def audit(home: Path, probe: MacServiceProbe) -> dict[str, object]:
    result = audit_launchagent_cutover(
        home=home, uid=501, runner=probe,
        observed_at=datetime(2026, 11, 14, 13, 47, tzinfo=UTC),
    )
    assert result["changes_applied"] is False
    assert result["safe_to_auto_activate"] is False
    assert result["launchctl_commands_used"] == "print_only"
    assert result["db_accessed"] is False
    assert result["ollama_contacted"] is False
    assert len(probe.commands) == 2
    return result


def test_absent_agent_plists_and_confirmed_not_loaded_needs_operator_approval(
    tmp_path: Path,
) -> None:
    probe = MacServiceProbe()
    result = audit(tmp_path, probe)
    assert result["status"] == "NEEDS_OPERATOR_APPROVAL"
    assert result["blockers"] == []
    assert result["services"]["legacy"]["launchctl_state"] == "NOT_LOADED"
    assert all(command[2].startswith("gui/501/") for command in probe.commands)
    assert not (tmp_path / "Library").exists()


def test_existing_legacy_agent_configuration_blocks_cutover(tmp_path: Path) -> None:
    write_plist(tmp_path, LEGACY_LABEL)
    probe = MacServiceProbe()
    result = audit(tmp_path, probe)
    assert result["status"] == "BLOCKED"
    assert "LEGACY_PLIST_STILL_PRESENT" in result["blockers"]
    assert result["services"]["legacy"]["expected_entrypoint"] is True


def test_loaded_agent_blocks_even_when_plist_is_missing(tmp_path: Path) -> None:
    probe = MacServiceProbe(loaded=(LEGACY_LABEL,))
    result = audit(tmp_path, probe)
    assert "LEGACY_AGENT_LOADED" in result["blockers"]
    assert result["services"]["legacy"]["plist_present"] is False


def test_integrated_agent_already_loaded_or_configured_is_not_overwritten(
    tmp_path: Path,
) -> None:
    write_plist(tmp_path, INTEGRATED_LABEL)
    probe = MacServiceProbe(loaded=(INTEGRATED_LABEL,))
    result = audit(tmp_path, probe)
    assert "INTEGRATED_AGENT_ALREADY_LOADED" in result["blockers"]
    assert "INTEGRATED_CONFIG_ALREADY_PRESENT" in result["blockers"]


def test_ambiguous_launchctl_failure_is_never_assumed_safe(tmp_path: Path) -> None:
    probe = MacServiceProbe(ambiguous=(LEGACY_LABEL,))
    result = audit(tmp_path, probe)
    assert "LEGACY_SERVICE_STATE_UNCONFIRMED" in result["blockers"]
    assert result["status"] == "BLOCKED"


def test_invalid_or_wrong_module_plist_blocks_before_any_activation(tmp_path: Path) -> None:
    write_plist(tmp_path, LEGACY_LABEL, wrong_entrypoint=True)
    result = audit(tmp_path, MacServiceProbe())
    assert "LEGACY_ENTRYPOINT_MISMATCH" in result["blockers"]
    path = tmp_path / "Library" / "LaunchAgents" / (INTEGRATED_LABEL + ".plist")
    path.write_text("not a plist")
    invalid = audit(tmp_path, MacServiceProbe())
    assert "INTEGRATED_CONFIG_INVALID" in invalid["blockers"]


def test_symlinked_plist_is_invalid_without_following_target(tmp_path: Path) -> None:
    target = tmp_path / "other.plist"
    target.write_bytes(plistlib.dumps({"Label": LEGACY_LABEL}))
    agents = tmp_path / "Library" / "LaunchAgents"
    agents.mkdir()
    (agents / (LEGACY_LABEL + ".plist")).symlink_to(target)
    result = audit(tmp_path, MacServiceProbe())
    assert "LEGACY_CONFIG_INVALID" in result["blockers"]


def test_preflight_cli_prints_no_live_confidential_launchctl_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        patch.object(Path, "home", return_value=tmp_path),
        patch("tris_market_event_intelligence.integrated_cutover_preflight.os.getuid",
              return_value=501),
        patch("tris_market_event_intelligence.integrated_cutover_preflight._default_runner",
              side_effect=MacServiceProbe(ambiguous=(LEGACY_LABEL,))),
        patch.object(sys, "argv", ["market-event-cutover-preflight"]),
        pytest.raises(SystemExit) as exc,
    ):
        main()
    assert exc.value.code == 2
    output = capsys.readouterr().out
    assert "Permission denied" not in output
    assert "BLOCKED" in output
