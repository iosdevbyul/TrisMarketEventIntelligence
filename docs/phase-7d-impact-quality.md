# Phase 7-D — Market impact rationale quality gate

Phase 7-B validated the local Ollama protocol and persistence path, **not** factual
or causal correctness. One real Gemma 2 v6 result had a dangling Korean clause
("...지속되고"), despite passing the prior 10-character minimum.

Version `market-impact-v7` adds a narrow, deterministic pre-persistence
`RATIONALE_INCOMPLETE` rejection for recognizably dangling Korean and English
connectives. The same gate is applied after provider parsing and immediately
before SQLite storage, including for alternative providers. A rejected response
is never edited, padded, checkpointed, or saved as an impact. Local Ollama may
make at most its existing one bounded regeneration attempt. Existing v6 runs,
impacts and errors are retained unchanged under their versioned keys.

**Limits:** This gate cannot determine whether a quoted news excerpt is true,
whether the quote supports a causal mechanism, or whether POSITIVE/NEGATIVE/MIXED
is justified. A literal evidence match is not a semantic entailment check.
Accepting a rationale does not certify analysis quality. Reported and
UNVERIFIED events continue to require HIGH uncertainty, and `impacts: []`
remains an appropriate answer when evidence is inadequate. Do not process the
whole backlog based on offline CI.

## Mac verification

```sh
git pull --ff-only origin main
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_impact_quality.py tests/test_impact.py tests/test_ollama_contract_recovery.py
python -m pytest -q
export MARKET_EVENT_LLM_MODEL=gemma2:latest
market-event-analyze --list-local-models
market-event-analyze --provider local --batch-size 1 --timeout-seconds 300
market-event-analyze --show-errors
curl -sS 'http://127.0.0.1:8100/api/v1/analysis-runs?after_id=0&limit=10'
curl -sS 'http://127.0.0.1:8100/api/v1/impacts?after_id=0&limit=10'
```

Inspect the rationale, source excerpt, and reasoning direction manually, and
compare the new `market-impact-v7` record to the preserved v6 record. A
successful model request is not proof of market impact or source truth.
