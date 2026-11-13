# Phase 7-D — independent human market-impact review

Phase 7-B's on-device E2E validated transport and storage only. The observed
v6 Korean rationale ended in "지속되고"; v7 correctly rejected an incomplete
new response. This does not establish the correctness of either direction
or causality.

This stage provides an **offline, read-only, explicitly human-labelled**
quality review workflow. It does not call Ollama, reprocess news, modify
existing stored results, or invent reference labels. The export selects
distinct events first, then additional hypotheses if needed; the Mac
currently has only one confirmed stored v6 impact, so **one is not a
multi-news evaluation**. Increase the corpus only with independently collected
and assessed real articles.

## On your Mac

Run in the existing Python environment from the project directory:

\`\`\`sh
git switch main
git pull --ff-only origin main
python -m pip install -e ".[dev]"
python -m pytest -q
market-event-impact-audit export --count 30 --output ~/impact-review-local.csv
\`\`\`

Open the **local** CSV without uploading it to any public service. Each
row contains the saved rationale, literal original event evidence, source URL,
capture digest, and the versioned model hypothesis. Five label columns must
be supplied by a human reviewer:

- source_support: SUPPORTED, UNSUPPORTED, INDETERMINATE
- causal_reasoning: JUSTIFIED, UNJUSTIFIED, INDETERMINATE
- direction_reasoning: JUSTIFIED, UNJUSTIFIED, INDETERMINATE
- uncertainty_language: CONDITIONAL, OVERSTATED, INDETERMINATE
- sentence_completion: COMPLETE, INCOMPLETE, INDETERMINATE

Do not copy the model's claims into the labels without independent review.
Check the source document as appropriate; a literal match is not a fact
check. Complete all five fields for each reviewed row, or leave all five
blank to mark it pending. Preserve every other column. Review notes are
optional and remain local.

\`\`\`sh
market-event-impact-audit score --sheet ~/impact-review-local.csv
\`\`\`

The scorer verifies each reviewed record against the original database using
a SHA-256 fingerprint of source + immutable model fields. Edited/stale data
are rejected. Output contains review counts and per-dimension label
frequencies, but never declares the model market-accurate or factuality-
verified. Existing v6, v7 analysis rows and failure records are preserved.

Do **not** make a large automatic model run from this review workflow.
