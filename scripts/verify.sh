#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Pull before Python imports modules so newly fetched code is used immediately.
args=()
update=0
for arg in "$@"; do
  if [[ "$arg" == "--update" ]]; then
    update=1
  else
    args+=("$arg")
  fi
done

if [[ "$update" -eq 1 ]]; then
  if [[ "$(git branch --show-current)" != "main" ]]; then
    echo "[verify] --update requires the main branch" >&2
    exit 2
  fi
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "[verify] --update requires a clean working tree" >&2
    exit 2
  fi
  git pull --ff-only origin main
fi

exec python -m tris_market_event_intelligence.verification_cli "${args[@]}"
