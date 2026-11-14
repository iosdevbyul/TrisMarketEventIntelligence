#!/usr/bin/env bash
# Inspect the owner's *existing* private news database; no fetch, Ollama or writes.
set -euo pipefail
cd "$(dirname "$0")/.."

update=0
args=()
for arg in "$@"; do
  if [[ "$arg" == "--update" ]]; then
    update=1
  else
    args+=("$arg")
  fi
done

if [[ "$update" -eq 1 ]]; then
  if [[ "$(git branch --show-current)" != "main" ]]; then
    echo "[operational-audit] --update requires main" >&2
    exit 2
  fi
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "[operational-audit] --update requires a clean working tree" >&2
    exit 2
  fi
  git pull --ff-only origin main
fi

module="tris_market_event_intelligence.operational_evidence_cli"
runner=()
if command -v python >/dev/null 2>&1 && python -c "import ${module}" >/dev/null 2>&1; then
  runner=(python)
elif [[ -x ".venv/bin/python" ]] && \
    .venv/bin/python -c "import ${module}" >/dev/null 2>&1; then
  runner=(.venv/bin/python)
elif command -v conda >/dev/null 2>&1 && \
    conda run -n tris-market-event python -c "import ${module}" >/dev/null 2>&1; then
  runner=(conda run --no-capture-output -n tris-market-event python)
else
  echo "[operational-audit] No installed project Python found." >&2
  echo "[operational-audit] Activate tris-market-event or install the project." >&2
  exit 2
fi

if [[ "${#args[@]}" -eq 0 ]]; then
  exec "${runner[@]}" -m "$module"
fi
exec "${runner[@]}" -m "$module" "${args[@]}"
