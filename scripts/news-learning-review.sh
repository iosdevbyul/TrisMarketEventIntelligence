#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  echo "Usage:"
  echo "  bash scripts/news-learning-review.sh [--update] cohort --as-of ISO_TIME --output FILE [--limit N]"
  echo "  bash scripts/news-learning-review.sh [--update] prepare --cohort FILE --output FILE"
  echo "  bash scripts/news-learning-review.sh [--update] score --cohort FILE --score HUMAN_REVIEW_JSON --output FILE"
  echo "  bash scripts/news-learning-review.sh [--update] evaluate --cohort FILE --review HUMAN_REVIEW_JSON --predictions MODEL_JSON --output FILE"
  echo "  bash scripts/news-learning-review.sh [--update] holdout --cohort FILE [--review HUMAN_REVIEW_JSON] --train-end ISO --validation-end ISO --embargo-hours N --output FILE"
  echo "  bash scripts/news-learning-review.sh [--update] test-usage --ledger FILE --init|--audit|--reserve PURPOSE ..."
  echo "  bash scripts/news-learning-review.sh [--update] near-duplicates --cohort FILE --holdout FILE --output FILE"
  echo "  bash scripts/news-learning-review.sh [--update] cross-page --cohort PAGE1 --holdout PLAN1 --cohort PAGE2 --holdout PLAN2 --output FILE"
  echo "  bash scripts/news-learning-review.sh [--update] research --once --cohort FILE --holdout FILE --review FILE --journal SQLITE"
  echo "  bash scripts/news-learning-review.sh research --audit --journal SQLITE"
  echo "  bash scripts/news-learning-review.sh [--update] research-loop --run --cohort FILE --holdout FILE --review FILE --journal SQLITE [--stop-file FILE]"
  echo "  bash scripts/news-learning-review.sh research-loop --stop STOP_FILE"
  echo "  bash scripts/news-learning-review.sh [--update] candidate --export --cohort FILE --holdout FILE --review FILE --journal SQLITE --experiment-sha256 SHA --artifact PRIVATE_JSON"
  echo "  bash scripts/news-learning-review.sh candidate --verify --cohort FILE --holdout FILE --review FILE --journal SQLITE --artifact PRIVATE_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] compare-candidates --candidate FILE --candidate FILE --cohort FILE --holdout FILE --review FILE --journal SQLITE --output PRIVATE_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] supervise --run --cohort FILE --holdout FILE --review FILE --journal SQLITE --status-file FILE --lock-file FILE"
  echo "  bash scripts/news-learning-review.sh supervise --status --status-file FILE"
  echo "  bash scripts/news-learning-review.sh [--update] guard-research --run --cohort FILE --holdout FILE --review FILE --journal SQLITE --status-file FILE --lock-file FILE"
  echo "  bash scripts/news-learning-review.sh guard-research --status --status-file FILE"
  echo "  bash scripts/news-learning-review.sh [--update] gemma-compare --preview|--preflight|--run [--gemma4 gemma4:e4b] [--limit 2] [--output FILE]"
  echo "  bash scripts/news-learning-review.sh [--update] gemma-blind --run --packet FILE --mapping SECRET_FILE [--gemma4 gemma4:e4b] [--limit 2]"
  echo "  bash scripts/news-learning-review.sh gemma-blind --prepare --packet FILE --output HUMAN_TEMPLATE_JSON"
  echo "  bash scripts/news-learning-review.sh gemma-blind --audit --packet FILE --review HUMAN_JSON --reviewer ID --output AUDIT_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] gemma-reveal --packet FILE --mapping SECRET_FILE --review HUMAN_JSON --reviewer ID [--review HUMAN_JSON --reviewer ID] --output PRIVATE_JSON"
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
  evaluate)
    module="tris_market_event_intelligence.news_importance_evaluation_cli"
    ;;
  holdout)
    module="tris_market_event_intelligence.news_holdout_plan_cli"
    ;;
  test-usage)
    module="tris_market_event_intelligence.news_test_usage_ledger_cli"
    ;;
  near-duplicates)
    module="tris_market_event_intelligence.news_near_duplicate_audit_cli"
    ;;
  cross-page)
    module="tris_market_event_intelligence.news_cross_page_holdout_audit_cli"
    ;;
  research)
    module="tris_market_event_intelligence.news_research_worker_cli"
    ;;
  research-loop)
    module="tris_market_event_intelligence.news_research_loop_cli"
    ;;
  candidate)
    module="tris_market_event_intelligence.news_candidate_artifact_cli"
    ;;
  compare-candidates)
    module="tris_market_event_intelligence.news_candidate_comparison_cli"
    ;;
  supervise)
    module="tris_market_event_intelligence.news_research_supervisor_cli"
    ;;
  guard-research)
    module="tris_market_event_intelligence.news_research_process_guard_cli"
    ;;
  gemma-compare)
    module="tris_market_event_intelligence.gemma4_news_comparison_cli"
    ;;
  gemma-blind)
    module="tris_market_event_intelligence.gemma4_blind_review_cli"
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
