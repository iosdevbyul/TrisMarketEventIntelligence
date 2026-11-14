#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  echo "Usage:"
  echo "  bash scripts/news-learning-review.sh [--update] cohort --as-of ISO_TIME --output FILE [--limit N]"
  echo "  bash scripts/news-learning-review.sh [--update] prepare --cohort FILE --output FILE"
  echo "  bash scripts/news-learning-review.sh [--update] score --cohort FILE --score HUMAN_REVIEW_JSON --output FILE"
}

if [[ "${1:-}" == "--update" ]]; then
  if [[ "$(git branch --show-current)" != "main" ]] || [[ -n "$(git status --porcelain)" ]]; then
    echo "[news-learning-review] --update requires clean main checkout" >&2
    exit 2
  fi
  git pull --ff-only origin main
  shift
fi

action="${1:-}"
if [[ "$#" -eq 0 ]]; then
  usage >&2
  exit 2
fi
shift

case "$action" in
  cohort)
    module="tris_market_event_intelligence.historical_news_cohort_cli"
    ;;
  prepare)
    module="tris_market_event_intelligence.news_importance_review_cli"
    set -- --prepare "$@"
    ;;
  score)
    module="tris_market_event_intelligence.news_importance_review_cli"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

# Python is resolved *after* optional fast-forward main update, just as in
# scripts/verify.sh. No background work, uploads or market-price access.
runner=()
if command -v python >/dev/null 2>&1 && python -c "import ${module}" >/dev/null 2>&1; then
  runner=(python)
elif [[ -x ".venv/bin/python" ]] && .venv/bin/python -c "import ${module}" >/dev/null 2>&1; then
  runner=(.venv/bin/python)
elif command -v conda >/dev/null 2>&1 && conda run -n tris-market-event python -c "import ${module}" >/dev/null 2>&1; then
  runner=(conda run --no-capture-output -n tris-market-event python)
else
  echo "[news-learning-review] Project Python environment not found." >&2
  echo "[news-learning-review] Activate tris-market-event or install the project." >&2
  exit 2
fi

exec "${runner[@]}" -m "$module" "$@"
