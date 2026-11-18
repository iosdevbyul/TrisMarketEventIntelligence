#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  echo "Usage:"
  echo "  bash scripts/news-learning-review.sh historical-semantic-score --benchmark REFERENCE_JSON --faa SAVED_FAA --fda SAVED_FDA [--output NEW_SCORE_JSON] [zero model calls]"
  echo "  bash scripts/news-learning-review.sh historical-staged-transfer --reference-chain FDA_JSON --reference-run SAVED_FDA_STAGED_JSON --reference-tracks FDA_QUESTIONS --challenge-chain FAA_JSON --challenge-run SAVED_FAA_STAGED_JSON --challenge-tracks FAA_QUESTIONS --output NEW_PRIVATE_COMPARISON_JSON [zero model calls]"
  echo "  bash scripts/news-learning-review.sh historical-event-chain --chain research/faa-7379max-january-2024-event-chain.json --as-of YYYY-MM-DD --preview|--run [--model gemma4:e2b --reasoning-mode staged-compact --output NEW_JSON]"
  echo "  bash scripts/news-learning-review.sh historical-event-chain --chain research/faa-7379max-january-2024-event-chain.json --as-of YYYY-MM-DD --preview|--run [--model gemma4:e2b --output NEW_JSON]"
  echo "  bash scripts/news-learning-review.sh historical-event-chain --chain FAA_CHAIN_JSON --audit-existing SAVED_REPLAY_JSON --output NEW_PRIVATE_AUDIT_JSON [zero model calls]"
  echo "  bash scripts/news-learning-review.sh [--update] historical-impact-test --url NEWS_URL --url NEWS_URL [--url NEWS_URL] --case-id CASE_ID --as-of ISO_TIME --horizon 1 --run --source-output NEW_SOURCE_JSON --output NEW_RESULT_JSON [--model gemma4:e2b]"
  echo "  bash scripts/news-learning-review.sh [--update] historical-impact-test --input PRIVATE_NEWS_JSON --prepare|--run [--model gemma4:e2b --output NEW_PRIVATE_RESULT_JSON]"
  echo "  bash scripts/news-learning-review.sh [--update] sec-primary-semantic --primary SAVED_PRIMARY_JSON --source-dir PRIVATE_DIR --prepare|--run [--model gemma4:e2b --limit 3 --start-index 1 --body-spans|--multi-passages] --output NEW_PRIVATE_JSON"\n  echo "  bash scripts/news-learning-review.sh [--update] sec-semantic-review --primary SAVED_PRIMARY_JSON --source-dir PRIVATE_DIR --report SEMANTIC_JSON [--report ANOTHER_JSON] --output NEW_PRIVATE_REVIEW_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] sec-claim-review --primary SAVED_PRIMARY_JSON --source-dir PRIVATE_DIR --report SEMANTIC_JSON [--report ANOTHER_JSON] --output NEW_PRIVATE_CLAIMS_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] sec-candidate-replay --primary ORIGINAL_JSON --source-dir PRIVATE_DIR --baseline-report ORIGINAL_SEMANTIC_JSON --start-index 0 --limit 1 --prepare|--run [--body-spans|--multi-passages --model gemma4:e2b --output NEW_PRIVATE_JSON]"
  echo "  bash scripts/news-learning-review.sh [--update] sec-semantic-compare --primary SAVED_PRIMARY --source-dir PRIVATE_DIR --baseline-report ORIGINAL_JSON --candidate-report NEW_JSON --output NEW_PRIVATE_COMPARISON_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] sec-semantic-quality --primary SAVED_PRIMARY_JSON --source-dir PRIVATE_DIR --report SEMANTIC_JSON [--report ANOTHER_JSON] --review PRIVATE_REVIEW_JSON --output NEW_PRIVATE_AUDIT_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] sec-primary-content --cohort PIT_JSON --batch INDEX_BATCH_JSON --source-dir PRIVATE_DIR --limit 12 --preview|--run --output NEW_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] sec-source-lineage --cohort PRIVATE_PIT_JSON --study TEN_CASE_JSON --batch BATCH01_JSON --source-dir PRIVATE_DIR [--witnesses HISTORICAL_WITNESS_JSON] --preview|--output NEW_PRIVATE_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] sec-publication-capture --cohort PRIVATE_JSON --source-dir PRIVATE_DIR --limit 12 --preview|--run --output NEW_PRIVATE_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] sec-pit-cohort --inventory PRIVATE_JSON --study PRIVATE_JSON --preview|--output NEW_PRIVATE_JSON [--max-per-year-form 3]"
  echo "  bash scripts/news-learning-review.sh [--update] sec-multiyear-study --review research/nvda-multiyear-sec-eight-k-official-review-2018-2026.json --inventory PRIVATE_SEC_INVENTORY --stock-csv NVDA_CSV --benchmark-csv SPY_CSV --raw-dir PRIVATE_DIR --preview|--run [--output NEW_PRIVATE_JSON]"
  echo "  TRIS_SEC_USER_AGENT=\"YourApp Contact@example.com\" bash scripts/news-learning-review.sh [--update] sec-history-inventory --start 2016-11-15 --end 2026-11-13 --output NEW_PRIVATE_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] historical-sec-universe --stock-csv EXISTING_NVDA --benchmark-csv EXISTING_SPY --event EXISTING_SEC_PRICE_PACKET [--event ANOTHER_VERIFIED_PACKET] --preview|--output NEW_PRIVATE_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] sec-coverage-reconcile --inventory PRIVATE_SEC_INVENTORY_JSON --stock-csv NVDA_CSV --benchmark-csv SPY_CSV --event PRIVATE_VERIFIED_SEC_EVENT_JSON --preview|--output NEW_PRIVATE_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] historical-sec-event --corroboration PRIVATE_JSON --sec research/nvda-2018-11-15-sec-8k-acceptance.json --stock-csv PRIVATE_NVDA_CSV --benchmark-csv PRIVATE_SPY_CSV --output NEW_PRIVATE_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] historical-issuer-corroborate --review PRIVATE_STORY_REVIEW_JSON --issuer research/nvda-2018-11-15-issuer-primary-evidence.json --output NEW_PRIVATE_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] gdelt-replay --study PRIVATE_SHOCKS_JSON --data-dir PRIVATE_DIR --start-rank 2 --count 2 --preview|--run"
  echo "  bash scripts/news-learning-review.sh [--update] historical-price-outcomes --symbol NVDA --benchmark-symbol SPY --stock-csv FILE --benchmark-csv FILE --news-cohort FILE [--macro-csv FILE] --preview|--output PRIVATE_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] historical-independent --cohort research/independent-historical-news-2023.json --preview|--run|--evaluate [--model gemma4:e2b] [--research PRIVATE_JSON] [--output NEW_PRIVATE_JSON]"
  echo "  bash scripts/news-learning-review.sh [--update] historical-learning-compare --baseline-research FILE --baseline-audit FILE [--candidate-research FILE --candidate-audit FILE --reviewed-lessons HUMAN_APPROVED_FILE] --output NEW_PRIVATE_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] historical-human-review --prepare --audit AUDIT_JSON --output NEW_REVIEW_TEMPLATE_JSON"
  echo "  bash scripts/news-learning-review.sh historical-human-review --finalize --audit AUDIT_JSON --review EDITED_JSON --reviewer PERSON --confirm-human-review --output NEW_LESSONS_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] historical-quality --research GEMMA_RESEARCH_JSON --output NEW_PRIVATE_AUDIT_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] historical-multi --scout PRIVATE_GDELT_JSON --preview|--run [--model gemma4:e2b] [--reviewed-lessons HUMAN_REVIEWED_JSON] --output NEW_PRIVATE_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] historical-story --scout PRIVATE_GDELT_JSON --prepare|--run [--model gemma4:e2b --limit 12] --output NEW_PRIVATE_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] gdelt-scout --study PRIVATE_SHOCKS_JSON [--rank 1] --preview|--run [--output NEW_PRIVATE_JSON]"
  echo "  bash scripts/news-learning-review.sh [--update] historical-acquire tiingo --symbol NVDA --start YYYY-MM-DD --end YYYY-MM-DD --output NEW_CSV"
  echo "  bash scripts/news-learning-review.sh [--update] historical-acquire fred --series-id CPIAUCSL --start YYYY-MM-DD --end YYYY-MM-DD --output NEW_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] historical-shocks --symbol NVDA --benchmark-symbol SPY --stock-csv FILE --benchmark-csv FILE --news-cohort FILE [--macro-csv FILE] [--top 100 --controls 100] [--preview|--output PRIVATE_JSON]"
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
  echo "  bash scripts/news-learning-review.sh [--update] guard-research --run --research-mode train-cv --max-iterations 18 --cohort FILE --holdout FILE --review FILE --journal SQLITE --status-file FILE --lock-file FILE"
  echo "  bash scripts/news-learning-review.sh [--update] gemma-compare --preview|--preflight|--run [--gemma4 gemma4:e4b] [--limit 2] [--output FILE]"
  echo "  bash scripts/news-learning-review.sh [--update] model-compare --preview|--run --model-a ID --model-b ID [--kind-a ollama|openai-compatible] [--kind-b ollama|openai-compatible] [--limit 2] [--output FILE]"
  echo "  bash scripts/news-learning-review.sh [--update] gemma-blind --run --packet FILE --mapping SECRET_FILE [--gemma4 gemma4:e4b] [--limit 2]"
  echo "  bash scripts/news-learning-review.sh gemma-blind --prepare --packet FILE --output HUMAN_TEMPLATE_JSON"
  echo "  bash scripts/news-learning-review.sh gemma-blind --audit --packet FILE --review HUMAN_JSON --reviewer ID --output AUDIT_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] gemma-reveal --packet FILE --mapping SECRET_FILE --review HUMAN_JSON --reviewer ID [--review HUMAN_JSON --reviewer ID] --output PRIVATE_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] gemma-mac-preflight [--gemma4 gemma4:e4b] [--output PRIVATE_JSON]"
  echo "  bash scripts/news-learning-review.sh [--update] train-cv --once|--batch 18 --cohort FILE --holdout FILE --review FILE --journal SQLITE"
  echo "  bash scripts/news-learning-review.sh train-cv --audit --journal SQLITE"
  echo "  bash scripts/news-learning-review.sh [--update] research-policy --batch 12 --cohort FILE --holdout FILE --review FILE --journal SQLITE"
  echo "  bash scripts/news-learning-review.sh research-policy --audit --journal SQLITE"
  echo "  bash scripts/news-learning-review.sh [--update] learning-readiness --cohort FILE --holdout FILE --review FILE [--manifest FILE --journal SQLITE --usage-ledger SQLITE]"
  echo "  bash scripts/news-learning-review.sh [--update] mac-inventory [--data-dir PRIVATE_DIR]"
  echo "  bash scripts/news-learning-review.sh [--update] coverage --cohort ORIGINAL_JSON [--newer LATER_JSON]"
  echo "  bash scripts/news-learning-review.sh [--update] coverage-refresh --baseline ORIGINAL_JSON --output NEW_JSON [--as-of ISO]"
  echo "  bash scripts/news-learning-review.sh [--update] evidence-pilot --preview|--run --cohort FILE --sample FILE --model gemma4:e2b [--start-index 0] [--limit 2] [--output NEW_JSON]"
  echo "  bash scripts/news-learning-review.sh [--update] review-sample --cohort FILE --max-selected 48 --output PRIVATE_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] review-page --cohort FILE --sample PRIVATE_JSON --output PRIVATE_HTML"
  echo "  bash scripts/news-learning-review.sh [--update] review-bridge --audit --cohort FILE --sample HUMAN_EDITED --importance-template FILE --materiality-template FILE"
  echo "  bash scripts/news-learning-review.sh review-bridge --apply --cohort FILE --sample HUMAN_EDITED --importance-template FILE --materiality-template FILE --importance-output NEW_JSON --materiality-output NEW_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] materiality --prepare --cohort FILE --output HUMAN_TEMPLATE_JSON"
  echo "  bash scripts/news-learning-review.sh materiality --evaluate --cohort FILE --holdout FILE --review HUMAN_JSON --predictions MODEL_JSON --output EVALUATION_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] candidate-lock --lock --cohort FILE --holdout FILE --review FILE --journal SQLITE --manifest PRIVATE_JSON"
  echo "  bash scripts/news-learning-review.sh candidate-lock --audit --manifest PRIVATE_JSON [--usage-ledger SQLITE]"
  echo "  bash scripts/news-learning-review.sh [--update] materiality-local --preview --cohort FILE [--limit 10]"
  echo "  bash scripts/news-learning-review.sh materiality-local --run --cohort FILE --model gemma2:latest --limit N --output PRIVATE_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] materiality-transition --prepare --cohort FILE --holdout FILE --report MODEL_REPORT --output BLANK_REVIEW"
  echo "  bash scripts/news-learning-review.sh materiality-transition --evaluate --cohort FILE --holdout FILE --report MODEL_REPORT --review HUMAN_REVIEW --output AUDIT_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] materiality-pairs --prepare --cohort FILE --holdout FILE --output BLIND_REVIEW"
  echo "  bash scripts/news-learning-review.sh materiality-pairs --evaluate --cohort FILE --holdout FILE --review HUMAN_REVIEW --report MODEL_REPORT --output AUDIT_JSON"
  echo "  bash scripts/news-learning-review.sh [--update] guard-research --run --research-mode policy --max-iterations 12 --cohort FILE --holdout FILE --review FILE --journal SQLITE --status-file FILE --lock-file FILE"
}

if [[ "${1:-}" == "--update" ]]; then
  if [[ "$(git branch --show-current)" != "main" ]] || [[ -n "$(git status --porcelain)" ]]; then
    echo "[news-learning-review] --update requires clean main checkout" >&2
    exit 2
  fi
  git pull --ff-only origin main
  shift
  # Bash has parsed the previous version already; restart from the updated
  # script so newly added subcommands work in the same invocation.
  exec bash scripts/news-learning-review.sh "$@"
fi

action="${1:-}"
if [[ "$#" -eq 0 ]]; then
  usage >&2
  exit 2
fi
shift

case "$action" in
  sec-primary-semantic)
    module="tris_market_event_intelligence.sec_primary_semantic_research_cli"
    ;;
  sec-semantic-review)
    module="tris_market_event_intelligence.sec_primary_semantic_review_cli"
    ;;
  sec-claim-review)
    module="tris_market_event_intelligence.sec_claim_evidence_review_cli"
    ;;
  sec-candidate-replay)
    module="tris_market_event_intelligence.sec_semantic_candidate_replay_cli"
    ;;
  sec-semantic-compare)
    module="tris_market_event_intelligence.sec_semantic_paired_comparison_cli"
    ;;
  sec-semantic-quality)
    module="tris_market_event_intelligence.sec_primary_semantic_quality_cli"
    ;;
  sec-primary-content)
    module="tris_market_event_intelligence.sec_primary_content_research_cli"
    ;;
  sec-source-lineage)
    module="tris_market_event_intelligence.sec_source_lineage_audit_cli"
    ;;
  sec-publication-capture)
    module="tris_market_event_intelligence.sec_publication_evidence_capture_cli"
    ;;
  sec-pit-cohort)
    module="tris_market_event_intelligence.sec_pit_availability_cohort_cli"
    ;;
  sec-multiyear-study)
    module="tris_market_event_intelligence.sec_multiyear_price_study_cli"
    ;;
  sec-coverage-reconcile)
    module="tris_market_event_intelligence.sec_coverage_reconciliation_cli"
    ;;
  sec-history-inventory)
    module="tris_market_event_intelligence.sec_historical_filing_inventory_cli"
    ;;
  historical-sec-universe)
    module="tris_market_event_intelligence.historical_sec_event_universe_cli"
    ;;
  historical-sec-event)
    module="tris_market_event_intelligence.historical_sec_price_event_cli"
    ;;
  historical-issuer-corroborate)
    module="tris_market_event_intelligence.historical_issuer_corroboration_cli"
    ;;
  gdelt-replay)
    module="tris_market_event_intelligence.historical_gdelt_replay_cli"
    ;;
  historical-price-outcomes)
    module="tris_market_event_intelligence.historical_price_outcome_cohort_cli"
    ;;
  historical-impact-test)
    module="tris_market_event_intelligence.historical_news_impact_test_cli"
    ;;
  historical-reasoning-study)
    module="tris_market_event_intelligence.historical_reasoning_study_cli"
    ;;
  historical-official-research)
    module="tris_market_event_intelligence.historical_official_fact_reasoning_cli"
    ;;
  historical-cross-domain)
    module="tris_market_event_intelligence.historical_cross_domain_official_study_cli"
    ;;
  historical-event-chain)
    module="tris_market_event_intelligence.historical_official_event_chain_cli"
    ;;
  historical-semantic-score)
    module="tris_market_event_intelligence.historical_semantic_quality_cli"
    ;;
  historical-semantic-compare)
    module="tris_market_event_intelligence.historical_semantic_regression_cli"
    ;;
  historical-staged-transfer)
    module="tris_market_event_intelligence.historical_cross_event_staged_transfer_cli"
    ;;
  historical-cross-event)
    module="tris_market_event_intelligence.historical_cross_event_validation_cli"
    ;;
  historical-independent)
    module="tris_market_event_intelligence.historical_independent_news_cli"
    ;;
  historical-learning-compare)
    module="tris_market_event_intelligence.historical_learning_comparison_cli"
    ;;
  historical-human-review)
    module="tris_market_event_intelligence.historical_human_review_cli"
    ;;
  historical-quality)
    module="tris_market_event_intelligence.historical_research_quality_cli"
    ;;
  historical-multi)
    module="tris_market_event_intelligence.historical_multi_hypothesis_cli"
    ;;
  historical-story)
    module="tris_market_event_intelligence.historical_story_review_cli"
    ;;
  gdelt-scout)
    module="tris_market_event_intelligence.historical_gdelt_scout_cli"
    ;;
  historical-acquire)
    module="tris_market_event_intelligence.historical_source_acquisition_cli"
    ;;
  historical-shocks)
    module="tris_market_event_intelligence.historical_shock_research_cli"
    ;;
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
  model-compare)
    module="tris_market_event_intelligence.news_model_comparison_cli"
    ;;
  gemma-compare)
    module="tris_market_event_intelligence.gemma4_news_comparison_cli"
    ;;
  gemma-blind)
    module="tris_market_event_intelligence.gemma4_blind_review_cli"
    ;;
  gemma-reveal)
    module="tris_market_event_intelligence.gemma4_blind_reveal_cli"
    ;;
  gemma-mac-preflight)
    module="tris_market_event_intelligence.gemma4_mac_capacity_cli"
    ;;
  candidate-lock)
    module="tris_market_event_intelligence.news_train_candidate_lock_cli"
    ;;
  materiality-pairs)
    module="tris_market_event_intelligence.news_materiality_preregistered_pairs_cli"
    ;;
  materiality-transition)
    module="tris_market_event_intelligence.news_materiality_transition_review_cli"
    ;;
  materiality-local)
    module="tris_market_event_intelligence.news_materiality_local_cli"
    ;;
  materiality)
    module="tris_market_event_intelligence.news_materiality_review_cli"
    ;;
  review-page)
    module="tris_market_event_intelligence.news_offline_review_page_cli"
    ;;
  review-bridge)
    module="tris_market_event_intelligence.news_review_sample_bridge_cli"
    ;;
  review-sample)
    module="tris_market_event_intelligence.news_review_sample_cli"
    ;;
  evidence-pilot)
    module="tris_market_event_intelligence.news_local_evidence_pilot_cli"
    ;;
  coverage-refresh)
    module="tris_market_event_intelligence.news_coverage_refresh_cli"
    ;;
  coverage)
    module="tris_market_event_intelligence.news_observation_coverage_cli"
    ;;
  mac-inventory)
    module="tris_market_event_intelligence.news_mac_learning_inventory_cli"
    ;;
  learning-readiness)
    module="tris_market_event_intelligence.news_learning_readiness_cli"
    ;;
  research-policy)
    module="tris_market_event_intelligence.news_research_policy_cli"
    ;;
  train-cv)
    module="tris_market_event_intelligence.news_train_cv_exploration_cli"
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
