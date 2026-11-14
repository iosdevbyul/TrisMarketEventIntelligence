"""Smoke-test the user's existing Mac shell command with empty argument arrays.

The sandbox uses a fake Python CLI so tests never access real RSS, live DB,
or private user data. Bash 3.2's set -u behavior is covered by the compatible
${array[@]+"${array[@]}"} idiom in the actual executable source.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate-listings.sh"


@pytest.fixture
def shell_project(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(SCRIPT, scripts / "validate-listings.sh")
    (scripts / "verify.sh").write_text(
        "#!/usr/bin/env bash\n"
        'printf "verify-ran\\n" >> "$HOME/verify-log"\n',
        encoding="utf-8",
    )
    home = tmp_path / "user-home"
    root = home / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    captures = root / "listing-research"
    bundle = captures / "observed-20261114T155440Z-1"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    (root / "news.sqlite3").write_text("fixture—not a real SQLite file", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Running python -c probes import; running python -m just logs exact args.
    fake_python = bin_dir / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "-c" ]]; then exit 0; fi\n'
        'printf "%s\\n" "$@" > "$HOME/cli-args"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = dict(os.environ)
    env.update({
        "HOME": str(home),
        "PATH": str(bin_dir) + os.pathsep + env["PATH"],
    })
    return project, env


@pytest.mark.parametrize("args,expected", [
    (
        ["--integrated-new-news"],
        "tris_market_event_intelligence.integrated_pipeline_cli",
    ),
    (
        ["--update", "--integrated-new-news"],
        "tris_market_event_intelligence.integrated_pipeline_cli",
    ),
    (
        ["--update"],
        "tris_market_event_intelligence.listing_news_validation_cli",
    ),
    (
        ["--update", "--post-snapshot-only"],
        "tris_market_event_intelligence.listing_news_validation_cli",
    ),
    (
        ["--materialize-new-events"],
        "tris_market_event_intelligence.post_snapshot_materialization_cli",
    ),
])
def test_mac_cli_reexec_and_empty_arrays(
    shell_project: tuple[Path, dict[str, str]],
    args: list[str],
    expected: str,
) -> None:
    project, env = shell_project
    result = subprocess.run(
        ["bash", "scripts/validate-listings.sh", *args],
        cwd=project, env=env, text=True, capture_output=True,
        timeout=15, check=False,
    )
    assert result.returncode == 0, result.stderr
    captured = (
        Path(env["HOME"]) / "cli-args"
    ).read_text(encoding="utf-8").splitlines()
    assert captured[:2] == ["-m", expected]
    assert captured.count("--integrated-new-news") == 0
    if "--integrated-new-news" in args:
        assert "--enable-ticker-review" in captured
        assert "--listing-bundle" in captured
        assert "--db" in captured
    elif "--materialize-new-events" in args:
        assert "--apply" in captured
    else:
        assert "--output-dir" in captured
    assert "--update" not in captured
    update_count = (Path(env["HOME"]) / "verify-log")
    assert update_count.read_text().count("verify-ran") == 1 if "--update" in args else (
        not update_count.exists()
    )


def test_conflicting_write_modes_rejected_before_python(
    shell_project: tuple[Path, dict[str, str]],
) -> None:
    project, env = shell_project
    proc = subprocess.run(
        [
            "bash", "scripts/validate-listings.sh",
            "--integrated-new-news", "--materialize-new-events",
        ],
        cwd=project, env=env, text=True, capture_output=True,
        timeout=15, check=False,
    )
    assert proc.returncode == 2
    assert "Choose only one write workflow" in proc.stderr
    assert not (Path(env["HOME"]) / "cli-args").exists()
