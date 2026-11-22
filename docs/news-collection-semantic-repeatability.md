# News collection and Gemma4 semantic repeatability pilot

**Scope:** RSS source collection, original article extraction, integrity, source-backed
market impact hypotheses and repeat-run consistency. **Excluded:** prices,
historical returns, stock outcome evaluation, strategy or trading, and existing
Stage 6 price-validation artifacts.

This is a separate research branch from the price-outcome PR. The operator
chooses whether to run live collection and local Ollama. Existing production
database and historical prediction files remain unchanged.

## 1. Safe offline collection inventory

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
python -m tris_market_event_intelligence.news_collection_semantic_audit_cli \\
  --db "$DATA/news.sqlite3" \\
  --output "$DATA/news-collection-integrity-v1.json"
```

Read the report's `collection.bySource`, `fullTextStatuses`,
`duplicateUrls`, and `completeBodyIntegrity`. Complete-body eligibility
requires original-text SHA-256 match and no sparse pages. These are **data
quality checks**, not evidence of correct semantic interpretation.

## 2. Optional isolated fresh collection smoke test

The following commands use a new, disposable research database; do **not**
point this test at the live `news.sqlite3` source. Network requests and RSS
publication content can change between invocations. Check the list of sources
first and run each invocation intentionally:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
PILOT="$DATA/news-repeat-collection-pilot.sqlite3"
python -m tris_market_event_intelligence.collect_cli --list-sources
python -m tris_market_event_intelligence.collect_cli \\
  --once --fulltext --db "$PILOT"
python -m tris_market_event_intelligence.news_collection_semantic_audit_cli \\
  --db "$PILOT" \\
  --output "$DATA/news-repeat-collection-audit-first.json"
python -m tris_market_event_intelligence.collect_cli \\
  --once --fulltext --db "$PILOT"
python -m tris_market_event_intelligence.news_collection_semantic_audit_cli \\
  --db "$PILOT" \\
  --output "$DATA/news-repeat-collection-audit-second.json"
```

Compare both reports: per-source article counts, duplicate URLs, complete-body
status, and invalid body hashes. It is **not** appropriate to demand identical
article counts for two live feed pulls—new articles might genuinely arrive.

## 3. Repeat identical saved original articles with Gemma4

Use the stable production source database in read-only mode. Use separate
create-only output directories so the batch CLI does not skip already analyzed
articles and the model is actually called in both runs.

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
python -m tris_market_event_intelligence.gemma_original_article_batch_cli \\
  --db "$DATA/news.sqlite3" \\
  --output-dir "$DATA/news-semantic-repeat-run-1" \\
  --model gemma4:e2b --limit 20
python -m tris_market_event_intelligence.gemma_original_article_batch_cli \\
  --db "$DATA/news.sqlite3" \\
  --output-dir "$DATA/news-semantic-repeat-run-2" \\
  --model gemma4:e2b --limit 20
python -m tris_market_event_intelligence.news_collection_semantic_audit_cli \\
  --db "$DATA/news.sqlite3" \\
  --run "$DATA/news-semantic-repeat-run-1" \\
  --run "$DATA/news-semantic-repeat-run-2" \\
  --output "$DATA/news-semantic-repeatability-v1.json"
```

Inspect `repeatability.cases`: each article lists its target/direction in
each run. An article is only comparable if its source hash and model match and
its original text remains verified in the database. Missing or changed articles
are explicitly excluded from the agreement denominator.

**Interpretation:** High run agreement means target and direction *stability*,
not that the impacts are true. Low agreement means nondeterminism or model
instability needing investigation. Human review should independently assess
relevance, causal mechanism, whether a sector extrapolation is justified,
whether `NEUTRAL` vs `UNCERTAIN` is calibrated and whether supporting text
actually entails the proposed directional conclusion.

## Fix: sentence boundary bug

The Gemma evidence splitter formerly used escaped backslashes in a raw regex,
so `\\s` and `\\n` were read literally instead of identifying sentence whitespace
and newlines. The corrected splitter keeps exact original evidence strings but
improves coherence of model-facing evidence passages. Existing output JSONs
must be preserved; new outputs belong in separate directories. A source quote
being exact does not mean the model's inference from it is valid.
