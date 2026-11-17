# Source-scoped historical reasoning (experimental v1)

This opt-in mode is separate from saved `staged` and `staged-compact`
contracts. No older retrospective output is rewritten. It reduces the
chance that one long constrained JSON string clips the final explanation.

## Pipeline

1. Generate and validate cutoff-visible source-anchored change facts using
   the existing evidence-first model draft.
2. Generate exactly one short model explanation and limitation for each
   newly visible official source. Each generated record must reference a
   claim from **that exact source**, end in punctuation, and pass the
   no-market-outcome check. Any failure stops the run; never substitute
   curator text for model-authored output.
3. Assemble the source prose verbatim in dated source order, without a
   separate long free-form narrative generation call. Retain all individual
   claims and diagnostics for human review.
4. Independently infer the two question axes from date-cutoff-visible
   sources and the *question text only*, not editorial lifecycle outcomes.
   The legacy model question status remains unchanged.
5. Separately diagnose token-limit termination, invalid JSON, suspected
   JSON Schema string boundary, a normal provider stop, and unavailable
   metadata. Normal stop **never** proves semantic completion.

The two question axes have different meanings:

- `regulatoryActionProgress`: NOT_OBSERVED, ANNOUNCED, AUTHORIZED,
  IMPLEMENTING, COMPLETED, UNKNOWN. The status concerns what the
  regulator actually announced or what implementation is evidenced.
- `requestedEvidenceStatus`: NOT_OBSERVED, PARTIAL, DIRECTLY_OBSERVED.
  This describes whether the *exact documents or answer requested*
  are visible, not how far the regulatory process has progressed.

For FAA 2024-01-24, authorization of the 737-9 MAX inspection/maintenance
process is not proof of completion of inspections on each plane. Increased
Boeing production oversight is not a finished audit. Separate 737-900ER
visual-inspection recommendations must not become a 737-9 MAX approval.
The same distinction applies to FDA authorization/approval versus missing
underlying manufacturer filings or internal review records.

## Real-model evaluation

On a Mac with existing Ollama `gemma4:e2b` installed, run this **once**
with a new output path; do not overwrite earlier staged-compact evidence:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh historical-event-chain \
  --chain research/faa-7379max-january-2024-event-chain.json \
  --question-tracks research/faa-2024-question-lifecycle.json \
  --sequence 2024-01-13 2024-01-25 \
  --reasoning-mode staged-composed \
  --run --model gemma4:e2b \
  --output "$DATA/faa-evidence-composed-v1.json"
```

Then compare the two preserved runs without asserting equivalence of model
effort, since composed mode makes **1 + N new sources + 1 question-axes**
calls per cutoff, versus 2 in staged-compact. Compare source coverage,
complete prose, contradicting agency/product scope, missing documents,
reported token consumption and false status changes. Generated prose and
question axes are structurally reviewed, **not** semantically verified.
Date-only release chronology does not establish historical publication-time
snapshots, point-in-time verification, economic effects or causality.

Both the cutoff-visible official evidence and model-generated text must be
reviewed by a human before treating explanations as factual.
