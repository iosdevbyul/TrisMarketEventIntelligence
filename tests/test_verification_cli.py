"""Offline regression of the single-command Mac verification workflow."""
import json
import subprocess
from pathlib import Path

import pytest

from tris_market_event_intelligence import verification_cli as cli


def completed(args: list[str], stdout: str = "", code: int = 0):
    return subprocess.CompletedProcess(args=args, returncode=code, stdout=stdout, stderr="")


def test_offline_mode_never_contacts_ollama_or_runs_inference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    invoked = []

    def run(args: list[str], root: Path):
        invoked.append(args)
        return completed(args, "All checks passed")

    monkeypatch.setattr(cli, "_run", run)
    monkeypatch.setattr(
        cli, "_ollama_ready",
        lambda model: pytest.fail("Ollama should not be contacted"),
    )
    monkeypatch.setattr(
        cli, "_read_json",
        lambda args, root: pytest.fail("Inference should not run"),
    )
    result = cli.verify(tmp_path, output_dir=tmp_path / "reports")
    assert result["status"] == "ok"
    assert result["offline_tests_passed"] is True
    assert result["live_requested"] is False
    assert len(invoked) == 3
    assert not any("impact_cli" in " ".join(args) for args in invoked)
    assert json.loads(Path(result["report_path"]).read_text())["quality_verified"] is False


def test_live_mode_runs_only_explicit_ids_and_compares_v7_v8(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    seen = []
    monkeypatch.setattr(cli, "_run", lambda args, root: completed(args))
    monkeypatch.setattr(cli, "_ollama_ready", lambda model: seen.append(("ready", model)))

    def read_json(args: list[str], root: Path):
        seen.append(tuple(args))
        if "impact_cli" in args:
            return {"processed": 1, "impacts": 0, "failed": 0}
        return {"status": "ok", "quality_verified": False, "items": []}

    monkeypatch.setattr(cli, "_read_json", read_json)
    result = cli.verify(
        tmp_path, live=(13,), output_dir=tmp_path / "reports", model="gemma2:latest"
    )
    assert result["status"] == "ok"
    assert result["analysis_stats"]["processed"] == 1
    assert result["selected_event_ids"] == [13]
    commands = [c for c in seen if isinstance(c, tuple) and len(c) > 2]
    assert len(commands) == 3
    assert commands[0][commands[0].index("--event-ids") + 1] == "13"
    assert "--batch-size" in commands[0]
    versions = [c[c.index("--prompt-version") + 1] for c in commands[1:]]
    assert versions == ["market-impact-v8", "market-impact-v7"]
    assert Path(result["report_path"]).is_file()


def test_failed_offline_test_stops_before_inference_and_saves_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls = []

    def run(args: list[str], root: Path):
        calls.append(args)
        if "pytest" in args:
            return completed(args, code=1)
        return completed(args)

    monkeypatch.setattr(cli, "_run", run)
    monkeypatch.setattr(
        cli, "_ollama_ready",
        lambda model: pytest.fail("Do not invoke Ollama after failed test"),
    )
    result = cli.verify(tmp_path, live=(13,), output_dir=tmp_path / "reports")
    assert result["status"] == "tests_failed"
    assert result["offline_tests_passed"] is False
    assert len(calls) == 3
    assert json.loads(Path(result["report_path"]).read_text())["tests_exit_code"] == 1


def test_update_refuses_dirty_tree_without_pull(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls = []

    def run(args: list[str], root: Path):
        calls.append(args)
        if args[:3] == ["git", "branch", "--show-current"]:
            return completed(args, "main\n")
        if args[:2] == ["git", "status"]:
            return completed(args, " M impact.py\n")
        pytest.fail("No pull allowed from dirty Git tree")

    monkeypatch.setattr(cli, "_run", run)
    with pytest.raises(RuntimeError, match="clean working tree"):
        cli.verify(tmp_path, update=True, output_dir=tmp_path / "reports")
    assert len(calls) == 2


def test_invalid_ids_fail_before_any_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cli, "_run", lambda args, root: pytest.fail("No commands allowed"),
    )
    with pytest.raises(ValueError):
        cli.verify(tmp_path, live=(13, 13), output_dir=tmp_path / "reports")
    with pytest.raises(ValueError):
        cli.verify(tmp_path, live=(1, 2, 3, 4), output_dir=tmp_path / "reports")
