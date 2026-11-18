# TrisMarketEventIntelligence — news and economic indicator market-pressure research

## Product purpose

This module interprets **news and economic-indicator evidence** and produces
conditional POSITIVE / NEGATIVE / MIXED / UNCERTAIN market or sector
**pressure hypotheses** with concrete transmission mechanisms, source
quotes and limitations. It does **not** forecast stock prices or returns,
claim that the market actually moved, or issue portfolio orders. This is
the purpose of TrisMarketEventIntelligence, separate from any consumer's
quantitative stock-return model.

## Generalized architecture

The `market_evidence_pressure` path accepts any number of dated NEWS and/or
ECONOMIC_INDICATOR inputs per event packet (bounded to 24 for safe model
context). It is not specialized to FAA, FDA, FTC, EPA, or legal/regulatory
actions.

- Source inputs: ID, publication date, origin, URL, quoted factual
  evidence, verification flag. Indicators also supply an indicator name,
  and optionally period/value/expectation.
- Strict `asOfDateExclusive`: only prior-date sources are provided to
  the model; excluded future IDs are recorded separately.
- The model uses the **existing** `ImpactResult` schema and
  `impact.SYSTEM_PROMPT`, rather than a newly invented direction scale.
  The output can describe opposite pressures in different markets/sectors.
- Every impact quotation must be a substring of cutoff-visible evidence;
  its `supportingSourceIds` are deterministically attached from input.
  Entities must also appear in visible source excerpts.
- If any source in the visible packet is unverified, all conditional
  hypotheses are HIGH uncertainty. No unstated economic surprises,
  company names, quantities or realized price returns are supplied.
- Model conclusions remain explicitly provisional; official page history
  and exact publication-time snapshots are **not** independently verified.
- The existing `mechanism_quality_code` and `rationale_quality_code`
  checks remain active. A schema-conformant answer is not by itself
  proof of economically correct reasoning.

This module reuses the existing impact semantics. It does **not** yet
feed automated API/cursor delivery, economic data ingestion, or normalized
event deduplication; those integrations require verification against the
running collectors and source adapters.

## Example packet format

```json
{
  "eventId": "example-macro-research",
  "subject": "Economic indicators and industry cost pressures",
  "asOfDateExclusive": "2026-11-01",
  "evidences": [
    {
      "sourceId": "economic-release-a",
      "kind": "ECONOMIC_INDICATOR",
      "publishedDate": "2026-10-29",
      "sourceName": "Official economic source",
      "sourceUrl": "https://www.bls.gov/news.release/cpi.toc.htm",
      "excerpt": "Illustrative official-release excerpt containing a measured economic value.",
      "verification": "REPORTED_UNVERIFIED",
      "indicatorName": "Example price index",
      "indicatorValue": null,
      "indicatorPeriod": null,
      "expectation": null
    },
    {
      "sourceId": "industry-report-b",
      "kind": "NEWS",
      "publishedDate": "2026-10-30",
      "sourceName": "News source",
      "sourceUrl": "https://www.example.org/industry",
      "excerpt": "Illustrative article excerpt describing a possible change in corporate costs.",
      "verification": "REPORTED_UNVERIFIED"
    }
  ]
}
```

The above is **synthetic schema demonstration**, not a real release,
verified observation, recommendation, or gold answer. Replace it with
actual supplied source quotes, dates and source IDs when testing real
news or economic indicators.

## Local use

Save a packet JSON at a private file path and first preview cutoff scope:

```bash
bash scripts/news-learning-review.sh market-evidence-pressure \
  --packet /path/to/private-evidence-packet.json --preview
```

For a real local Gemma4 analysis, once the packet's sources have been
reviewed and the model is available:

```bash
bash scripts/news-learning-review.sh market-evidence-pressure \
  --packet /path/to/private-evidence-packet.json \
  --run --model gemma4:e2b \
  --output /path/to/new-private-market-pressure-result.json
```

The output is create-only, and the model is called exactly once.
No model installation or implicit source downloads take place.

## Existing news.sqlite3 adapter (read-only)

The new `market_evidence_db_bridge` reads **actual local collected
articles** from the `articles` table and economic/monetary events from
`economic_events`. This is a read-only selection step, not another
collector. RSS headlines and summaries are unverified evidence and never
treated as full original article bodies. A source missing an explicit
timezone-aware publication timestamp is rejected instead of silently
substituting the detection time.

Choose IDs already present in the existing local database, and preview
before invoking Gemma4:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh market-evidence-pressure \
  --db "$DATA/news.sqlite3" --event-id investigation-001 \
  --subject "News and macroeconomic context" --as-of 2026-11-18 \
  --article-id 101 --economic-event-id 25 --preview
```

The IDs above are **illustrative**, not known to exist on a user's machine.
For a real run, use the same arguments with `--run --model gemma4:e2b
--output "$DATA/new-qualitative-pressure.json"`. The output is create-only.
Selecting relevant IDs is still manual and is not yet an automated event
clustering/deduplication or reliable economic-series ingestion workflow.

## What still needs validation

1. Automatic event-cluster selection across the existing collector and a
   time-verified dedicated economic-series source, eliminating manual ID selection.
2. Market/sector target resolution and transparent aggregation across
   multiple articles or indicators (with potentially conflicting signals).
3. A frozen independent benchmark of the **direction and mechanism** of
   market pressure, not just JSON compliance or regulator decision labels.
4. Empirical model comparisons and review of source truth, market
   expectations, ambiguity and whether a story was priced in.
5. Existing REST/API integration and operational regression validation.

The regulator-event first-sight pilot remains an auxiliary experiment
only; it is **not** the TrisMarketEventIntelligence product pipeline.
