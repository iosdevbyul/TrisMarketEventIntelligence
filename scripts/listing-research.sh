#!/usr/bin/env bash
# Single-command, opt-in listing source acquisition and review research.
set -euo pipefail
cd "$(dirname "$0")/.."

update=0
params=()
for arg in "$@"; do
  if [[ "$arg" == "--update" ]]; then
    update=1
  else
    params+=("$arg")
  fi
done

# The existing verifier handles clean-main pull, correct Python environment,
# editable installation, lint and all tests, without activating collection.
if [[ "$update" -eq 1 ]]; then
  bash scripts/verify.sh --update
fi

module="tris_market_event_intelligence.listing_research_workflow_cli"
runner=()
if command -v python >/dev/null 2>&1 && \
   python -c "import ${module}" >/dev/null 2>&1; then
  runner=(python)
elif [[ -x ".venv/bin/python" ]] && \
     .venv/bin/python -c "import ${module}" >/dev/null 2>&1; then
  runner=(.venv/bin/python)
elif command -v conda >/dev/null 2>&1 && \
     conda run -n tris-market-event python -c "import ${module}" >/dev/null 2>&1; then
  runner=(conda run --no-capture-output -n tris-market-event python)
else
  echo "[listing-research] Project environment missing; use --update on clean main." >&2
  exit 2
fi

output_set=0
news_set=0
for arg in "${params[@]}"; do
  case "$arg" in
    --output-dir) output_set=1 ;;
    --news-db) news_set=1 ;;
  esac
done

root="$HOME/Library/Application Support/TrisMarketEventIntelligence"
folder="$root/listing-research"
if [[ ! -d "$folder" ]]; then
  mkdir -p "$folder"
  chmod 700 "$folder"
fi
if [[ -L "$folder" ]]; then
  echo "[listing-research] Refusing symlinked private output folder" >&2
  exit 2
fi
if [[ "$output_set" -eq 0 ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
  params+=(--output-dir "$folder/observed-$stamp")
fi
if [[ "$news_set" -eq 0 && -f "$root/news.sqlite3" ]]; then
  params+=(--news-db "$root/news.sqlite3")
fi
exec "${runner[@]}" -m "$module" "${params[@]}"
