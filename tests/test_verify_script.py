"""Ensure the shell launcher updates source before starting the Python process."""
import os
import subprocess
from pathlib import Path


def _setup(tmp_path: Path, *, dirty: bool = False) -> tuple[Path, dict[str, str], Path]:
    root = tmp_path / "project"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    original = Path(__file__).resolve().parents[1] / "scripts" / "verify.sh"
    launcher = scripts / "verify.sh"
    launcher.write_text(original.read_text(encoding="utf-8"), encoding="utf-8")
    bin_path = tmp_path / "fakebin"
    bin_path.mkdir()
    calls = tmp_path / "calls.log"
    git = bin_path / "git"
    git.write_text(
        "#!/bin/sh\n"
        "echo git:$* >> \"$CALLS\"\n"
        "if [ \"$1\" = branch ]; then echo main; fi\n"
        + ("if [ \"$1\" = status ]; then echo \" M file.py\"; fi\n" if dirty else ""),
        encoding="utf-8",
    )
    python = bin_path / "python"
    python.write_text(
        "#!/bin/sh\n" "echo python:$* >> \"$CALLS\"\n",
        encoding="utf-8",
    )
    git.chmod(0o755)
    python.chmod(0o755)
    env = {**os.environ, "PATH": str(bin_path) + ":" + os.environ["PATH"],
           "CALLS": str(calls)}
    return launcher, env, calls


def test_pull_happens_before_python_and_update_flag_is_consumed(tmp_path: Path) -> None:
    script, env, calls = _setup(tmp_path)
    result = subprocess.run(
        ["bash", str(script), "--update", "--live", "13"], env=env,
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert calls.read_text().splitlines() == [
        "git:branch --show-current",
        "git:status --porcelain",
        "git:pull --ff-only origin main",
        "python:-c import tris_market_event_intelligence.verification_cli",
        "python:-m tris_market_event_intelligence.verification_cli --live 13",
    ]


def test_dirty_tree_refuses_pull_and_python(tmp_path: Path) -> None:
    script, env, calls = _setup(tmp_path, dirty=True)
    result = subprocess.run(
        ["bash", str(script), "--update"], env=env,
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "clean working tree" in result.stderr
    assert calls.read_text().splitlines() == [
        "git:branch --show-current", "git:status --porcelain",
    ]


def test_no_update_does_not_call_git(tmp_path: Path) -> None:
    script, env, calls = _setup(tmp_path)
    result = subprocess.run(["bash", str(script)], env=env, check=False)
    assert result.returncode == 0
    assert calls.read_text().splitlines() == [
        "python:-c import tris_market_event_intelligence.verification_cli",
        "python:-m tris_market_event_intelligence.verification_cli",
    ]


def test_update_only_runs_python_without_extra_arguments(tmp_path: Path) -> None:
    script, env, calls = _setup(tmp_path)
    result = subprocess.run(
        ["bash", str(script), "--update"], env=env,
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert calls.read_text().splitlines() == [
        "git:branch --show-current",
        "git:status --porcelain",
        "git:pull --ff-only origin main",
        "python:-c import tris_market_event_intelligence.verification_cli",
        "python:-m tris_market_event_intelligence.verification_cli",
    ]


def test_no_arguments_avoids_empty_array_expansion_in_nounset_mode(tmp_path: Path) -> None:
    script, env, calls = _setup(tmp_path)
    result = subprocess.run(
        ["bash", "-u", str(script)], env=env,
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "unbound variable" not in result.stderr
    assert calls.read_text().splitlines() == [
        "python:-c import tris_market_event_intelligence.verification_cli",
        "python:-m tris_market_event_intelligence.verification_cli",
    ]


def test_tmux_base_python_falls_back_to_named_conda_env(tmp_path: Path) -> None:
    script, env, calls = _setup(tmp_path)
    bin_path = tmp_path / "fakebin"
    python = bin_path / "python"
    python.write_text(
        "#!/bin/sh\n"
        "echo python:$* >> \"$CALLS\"\n"
        "if [ \"$1\" = -c ]; then exit 1; fi\n",
        encoding="utf-8",
    )
    conda = bin_path / "conda"
    conda.write_text(
        "#!/bin/sh\n"
        "echo conda:$* >> \"$CALLS\"\n",
        encoding="utf-8",
    )
    conda.chmod(0o755)
    result = subprocess.run(
        ["bash", str(script), "--update"], env=env,
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert calls.read_text().splitlines() == [
        "git:branch --show-current",
        "git:status --porcelain",
        "git:pull --ff-only origin main",
        "python:-c import tris_market_event_intelligence.verification_cli",
        ("conda:run -n tris-market-event python -c "
         "import tris_market_event_intelligence.verification_cli"),
        ("conda:run --no-capture-output -n tris-market-event python "
         "-m tris_market_event_intelligence.verification_cli"),
    ]


def test_no_environment_has_actionable_message(tmp_path: Path) -> None:
    script, env, _calls = _setup(tmp_path)
    bin_path = tmp_path / "fakebin"
    python = bin_path / "python"
    python.write_text(
        "#!/bin/sh\n"
        "echo python:$* >> \"$CALLS\"\n"
        "if [ \"$1\" = -c ]; then exit 1; fi\n",
        encoding="utf-8",
    )
    # Explicitly mask conda even on a host with it installed.
    conda = bin_path / "conda"
    conda.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    conda.chmod(0o755)
    result = subprocess.run(
        ["bash", str(script)], env=env,
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "Python project environment not found" in result.stderr
    assert "conda activate tris-market-event" in result.stderr
