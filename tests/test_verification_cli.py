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
        if "tris_market_event_intelligence.impact_cli" in args:
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


def test_live_next_selects_diverse_new_events_and_creates_private_review(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls = []
    monkeypatch.setenv("MARKET_EVENT_DB", str(tmp_path / "data.sqlite"))
    monkeypatch.setattr(cli, "_run", lambda args, root: completed(args))
    monkeypatch.setattr(cli, "_ollama_ready", lambda model: calls.append(("ready", model)))

    def preview(db: Path, *, model_id: str, limit: int, new_only: bool):
        assert new_only
        assert limit == 3
        return {"status": "ok", "candidates": [
            {"event_id": 8}, {"event_id": 14}, {"event_id": 19},
        ]}

    monkeypatch.setattr(cli, "pilot_candidates", preview)
    monkeypatch.setattr(cli, "cohort_review", lambda db, **kw: {
        "status": "ok", "items": [{"event_id": 8, "original_evidence": "private source"}],
    })

    def read_json(args: list[str], root: Path):
        calls.append(tuple(args))
        if "tris_market_event_intelligence.impact_cli" in args:
            return {"processed": 3, "impacts": 1, "failed": 0}
        return {"status": "ok", "items": [], "quality_verified": False}

    monkeypatch.setattr(cli, "_read_json", read_json)
    result = cli.verify(
        tmp_path, live_next=3, output_dir=tmp_path / "reports",
    )
    assert result["status"] == "ok"
    assert result["selected_event_ids"] == [8, 14, 19]
    assert result["analysis_stats"]["processed"] == 3
    assert result["live_requested"] is True
    cmd = next(c for c in calls if "tris_market_event_intelligence.impact_cli" in c)
    assert cmd[cmd.index("--event-ids") + 1] == "8,14,19"
    review = Path(result["new_cohort_review_path"])
    assert review.is_file()
    assert "private source" in review.read_text()
    assert "private source" not in Path(result["report_path"]).read_text()


def test_live_next_without_new_events_never_invokes_ollama(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("MARKET_EVENT_DB", str(tmp_path / "data.sqlite"))
    monkeypatch.setattr(cli, "_run", lambda args, root: completed(args))
    monkeypatch.setattr(
        cli, "pilot_candidates", lambda db, **kw: {
            "status": "ok", "candidates": [],
        },
    )
    monkeypatch.setattr(
        cli, "_ollama_ready",
        lambda model: pytest.fail("No model call without eligible events"),
    )
    result = cli.verify(tmp_path, live_next=3, output_dir=tmp_path / "reports")
    assert result["status"] == "ok"
    assert "analysis_stats" not in result
    assert result["live_requested"] is True


def test_live_next_rejects_out_of_range_and_mixed_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cli, "_run", lambda args, root: pytest.fail("No commands allowed"),
    )
    for count in [0, 4]:
        with pytest.raises(ValueError, match="live-next"):
            cli.verify(tmp_path, live_next=count)
    with pytest.raises(ValueError, match="either"):
        cli.verify(tmp_path, live=(13,), live_next=3)
