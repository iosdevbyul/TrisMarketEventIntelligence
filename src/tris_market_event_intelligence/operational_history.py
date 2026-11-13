"""Private append-only per-cycle ledger; independent of the article database."""

from __future__ import annotations

import json
import os
from pathlib import Path


def history_path_for(status_path: Path) -> Path:
    return status_path.with_name("run-history.jsonl")


def record_cycle(path: Path, outcome: dict[str, object]) -> None:
    """Append safe metadata only; the caller must hold the cycle flock."""
    sources = outcome.get("source_results", {})
    if not isinstance(sources, dict):
        sources = {}
    entry = {
        "status": outcome["status"],
        "started_at": outcome["started_at"],
        "finished_at": outcome["finished_at"],
        "successful_sources": outcome["successful_sources"],
        "total_sources": outcome["total_sources"],
        "new_articles": outcome["new_articles"],
        "new_claims": outcome["new_claims"],
        "indexed_articles": outcome["indexed_articles"],
        "error_code": outcome.get("error_code"),
        "source_statuses": {
            name: result.get("status", "unknown")
            for name, result in sources.items()
            if isinstance(name, str) and isinstance(result, dict)
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
