#!/usr/bin/env bash
# Inspect previously captured real US/KRX listing files against local news.
# Read-only by default; --materialize-new-events explicitly allows bounded event writes.
# No HTTP, model or LaunchAgent changes.
set -euo pipefail
cd "$(dirname "$0")/.."

update=0
materialize=0
integrated=0
params=()
reentry_args=()
bundle_set=0
news_set=0
output_set=0
for arg in "$@"; do
  if [[ "$arg" == "--update" ]]; then
    update=1
  else
    reentry_args+=("$arg")
  fi
  if [[ "$arg" == "--update" ]]; then
    :
  elif [[ "$arg" == "--materialize-new-events" ]]; then
    materialize=1
  elif [[ "$arg" == "--integrated-new-news" ]]; then
    integrated=1
  else
    params+=("$arg")
  fi
  case "$arg" in
    --bundle) bundle_set=1 ;;
    --news-db) news_set=1 ;;
    --output-dir) output_set=1 ;;
  esac
done
if [[ "$integrated" -eq 1 && "$materialize" -eq 1 ]]; then
  echo "[listing-validation] Choose only one write workflow" >&2
  exit 2
fi
if [[ "$update" -eq 1 ]]; then
  bash scripts/verify.sh --update
  # Bash continues executing the OLD parsed script after git pull. Re-enter
  # the freshly updated file with the original argument order, minus --update.
  # exec prevents a recursive update and preserves the final exit status.
  exec bash scripts/validate-listings.sh "${reentry_args[@]}"
fi

module="tris_market_event_intelligence.listing_news_validation_cli"
if [[ "$materialize" -eq 1 ]]; then
  module="tris_market_event_intelligence.post_snapshot_materialization_cli"
  params+=(--apply)
fi
if [[ "$integrated" -eq 1 ]]; then
  module="tris_market_event_intelligence.integrated_pipeline_cli"
fi
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
  echo "[listing-validation] Project environment missing; use --update on clean main." >&2
  exit 2
fi

root="$HOME/Library/Application Support/TrisMarketEventIntelligence"
folder="$root/listing-research"
if [[ ! -d "$folder" || -L "$folder" ]]; then
  echo "[listing-validation] Existing private listing-research folder required" >&2
  exit 2
fi
if [[ "$bundle_set" -eq 0 ]]; then
  newest=""
  # Folder names carry UTC timestamp followed by PID; lexical order
  # selects the latest actual capture, not an analysis result directory.
  for item in "$folder"/observed-*; do
    if [[ -d "$item" && ! -L "$item" && -f "$item/manifest.json" ]]; then
      newest="$item"
    fi
  done
  if [[ -z "$newest" ]]; then
    echo "[listing-validation] No captured listing bundle; run listing-research.sh." >&2
    exit 2
  fi
  if [[ "$integrated" -eq 0 ]]; then
    params+=(--bundle "$newest")
  fi
fi
if [[ "$news_set" -eq 0 ]]; then
  if [[ ! -f "$root/news.sqlite3" || -L "$root/news.sqlite3" ]]; then
    echo "[listing-validation] Existing private news.sqlite3 required" >&2
    exit 2
  fi
  if [[ "$integrated" -eq 0 ]]; then
    params+=(--news-db "$root/news.sqlite3")
  fi
fi
if [[ "$integrated" -eq 1 ]]; then
  # Explicitly integrated workflow: makes RSS network requests and may append
  # new articles/events. The preexisting scheduled LaunchAgent is untouched.
  if [[ "$bundle_set" -ne 0 || "$news_set" -ne 0 || "$output_set" -ne 0 ]]; then
    echo "[listing-validation] Integrated mode uses default pinned bundle and news DB" >&2
    exit 2
  fi
  if [[ ! -d "$newest" || ! -f "$root/news.sqlite3" ]]; then
    echo "[listing-validation] Pinned capture and existing news DB required" >&2
    exit 2
  fi
  exec "${runner[@]}" -m "$module" \
    --enable-ticker-review --listing-bundle "$newest" \
    --review-output-root "$folder" --db "$root/news.sqlite3" "${params[@]}"
fi
if [[ "$output_set" -eq 0 ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
  params+=(--output-dir "$folder/validation-$stamp")
fi
exec "${runner[@]}" -m "$module" "${params[@]}"
