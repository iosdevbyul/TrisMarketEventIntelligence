"""Generate independent collection LaunchAgents; never activate automatically."""
from __future__ import annotations

import argparse
import os
import plistlib
import sys
from pathlib import Path

from tris_market_event_intelligence.unified_collection_cli import INTERVALS

PREFIX = "com.tris.market-event.collection."


def configurations(python: Path, repo: Path, news: Path, macro: Path, output: Path,
                   *, agent: str, ciks: tuple[int, ...] = ()) -> dict:
    if any(not p.is_absolute() for p in (python, repo, news, macro, output)):
        raise ValueError("Absolute paths required")
    if "@" not in agent:
        raise ValueError("SEC identifying contact required")
    if news == macro or any(not 1 <= cik <= 9999999999 for cik in ciks):
        raise ValueError("Invalid database or issuer scope")
    result = {}
    for job in INTERVALS:
        label = PREFIX + job
        arguments = [str(python), "-m", "tris_market_event_intelligence.unified_collection_cli",
                     "--news-db", str(news), "--macro-db", str(macro),
                     "--output-dir", str(output), "--job", job, "--watch"]
        environment = {"PYTHONPATH": str(repo / "src")}
        if job == "sec":
            environment["MARKET_EVENT_SEC_USER_AGENT"] = agent
            for cik in ciks:
                arguments.extend(("--cik", str(cik)))
        result[label] = {
            "Label": label, "ProgramArguments": arguments, "WorkingDirectory": str(repo),
            "EnvironmentVariables": environment, "RunAtLoad": True, "KeepAlive": True,
            "ThrottleInterval": 60, "ProcessType": "Background",
            "StandardOutPath": str(output / (job + ".stdout.log")),
            "StandardErrorPath": str(output / (job + ".stderr.log")),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--news-db", type=Path, required=True)
    parser.add_argument("--macro-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cik", type=int, action="append")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output_dir.resolve()
    if not (repo / "src/tris_market_event_intelligence/unified_collection_cli.py").is_file():
        parser.error("Repository source missing")
    if not args.news_db.is_file() or args.news_db.is_symlink():
        parser.error("Existing non-symlink news DB required")
    try:
        configs = configurations(Path(sys.executable).resolve(), repo,
                                 args.news_db.resolve(), args.macro_db.resolve(), output,
                                 agent=os.environ.get("MARKET_EVENT_SEC_USER_AGENT", ""),
                                 ciks=tuple(args.cik or ()))
    except ValueError as exc:
        parser.error(str(exc))
    target = Path.home() / "Library/LaunchAgents"
    if args.write:
        target.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        # Do not overwrite a previously configured or loaded service.
        paths = [target / (label + ".plist") for label in configs]
        if any(p.exists() or p.is_symlink() for p in paths):
            parser.error("Existing LaunchAgent config; inspect before replacing")
        for label, config in configs.items():
            path = target / (label + ".plist")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                plistlib.dump(config, stream)
    for label in configs:
        print(label)
    print("Configuration written; NOT loaded" if args.write else "Preview only; nothing written")


if __name__ == "__main__":
    main()
