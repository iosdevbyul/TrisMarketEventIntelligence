#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Pull before Python imports modules so newly fetched code is used immediately.
args=()
argument_count=0
update=0
for arg in "$@"; do
  if [[ "$arg" == "--update" ]]; then
    update=1
  else
    args+=("$arg")
    argument_count=$((argument_count + 1))
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

# Resolve the Python that actually has this project installed. tmux often
# starts a fresh shell in Conda (base) instead of the project environment.
# Check the selected interpreter after the git update, never before it.
verify_module="tris_market_event_intelligence.verification_cli"
runner=()
if command -v python >/dev/null 2>&1 && \
   python -c "import ${verify_module}" >/dev/null 2>&1; then
  runner=(python)
elif [[ -x ".venv/bin/python" ]] && \
     .venv/bin/python -c "import ${verify_module}" >/dev/null 2>&1; then
  runner=(.venv/bin/python)
elif command -v conda >/dev/null 2>&1 && \
     conda run -n tris-market-event python -c "import ${verify_module}" >/dev/null 2>&1; then
  runner=(conda run --no-capture-output -n tris-market-event python)
else
  echo "[verify] Python project environment not found." >&2
  echo "[verify] Activate the tris-market-event Conda environment or install the project." >&2
  echo "[verify] Example: conda activate tris-market-event" >&2
  exit 2
fi

if [[ "$argument_count" -eq 0 ]]; then
  # macOS Bash 3.2 fails on an empty array expansion with `set -u`.
  exec "${runner[@]}" -m "$verify_module"
else
  exec "${runner[@]}" -m "$verify_module" "${args[@]}"
fi
