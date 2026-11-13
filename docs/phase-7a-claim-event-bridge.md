# Phase 7-A — collected claims to impact event bridge

**Scope:** TrisMarketEventIntelligence only. DonghakStockVision remains unchanged.
This opt-in bridge consumes the live `article_event_claims` table populated by
the 30-minute RSS collector and creates provenance-preserving `economic_events`
which the existing `ImpactProcessor` and API already understand.

Every claim gets its own event (including several claims in one article), with
a durable `claim_event_links` mapping and atomic per-claim SQLite transaction.
Claim status, category, evidence, source URL, published/detected times and the
original capture SHA256 remain unchanged. The bridge does not verify facts,
promote claims to VERIFIED, call Ollama or run trades. Forecast/rumor/denial
claims remain identifiable and the existing `is_eligible` rules independently
exclude unsuitable AI inference inputs.

## Opt-in on the existing Mac

```sh
git pull --ff-only origin main
python -m pip install -e .
market-event-bridge --batch-size 100
curl -sS 'http://127.0.0.1:8100/api/v1/events?after_id=0&limit=5'
```

Repeated `market-event-bridge` is idempotent: `processed: 0` on a second
pass with no new claims. On a large database invoke the command again until
`processed` is less than 100, then inspect the results. It does not change
the current 30-minute LaunchAgent or automatically start Ollama.

The previous article-level event extractor remains intact; bridge events have
an explicit `claim-bridge-v1:<claim-id>` extractor version and a dedicated
claim/event link table. This makes the provenance distinction visible.
Do not treat the rule-based `VERIFIED` status as independent documentary
verification. Market impacts, if later produced, are conditional hypotheses.
Phase 7-B will verify the actual available Intel Mac Ollama models, memory
constraints, run bounded local inference and assess latency/errors. Phase 7-F
will specify a completed-analysis contract for the Donghak Codex.
