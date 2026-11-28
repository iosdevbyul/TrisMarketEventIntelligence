# Semantic event integration — opt-in review

PR #284 / Market Learning 1–2. v2 actual data: 43,211 records, 22 macro context edges, zero SEC/news and release/news edges. This separate module retrieves bilingual cross-source candidates, reopens pinned source bytes, asks existing local Ollama runtime for a structured relation, validates exact quotations and persists an independent review ledger. No automatic promotion, forecast, trading, collection changes or verified first publication. Model labels are candidates; quote validation is not semantic correctness. Retrieval aliases are deliberately limited; missing dates require stronger overlap. Current source version drift is rejected, not reconstructed. Preview is free of model calls; execute is opt-in, bounded to 1–200 calls per invocation. Cached rejected results are retained; no automatic retry. Different integration snapshots are distinct.

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
WORK="$DATA/event-integration-pr284"
PYTHONPATH="$WORK/src" python -m tris_market_event_intelligence.semantic_event_integration_cli \
  --news-db "$DATA/news.sqlite3" --event-db "$DATA/unified-events.sqlite3" \
  --run-id 6e47de6a4d3cc71c406659dccecc9be23fa5acd8a1966febfb35ec7690fb94b6 \
  --model gemma4:e2b --limit 20
```

Add --execute only after inspecting preview; actual model run and human evidence review are acceptance prerequisites. No claims about independent accuracy from fixture tests.
