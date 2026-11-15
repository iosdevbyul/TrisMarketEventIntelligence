# Learning-08 — After-review reveal and descriptive agreement

The existing PR #133 creates a private blinded A/B NEWS review packet and a
**separate** sealed mapping to the original Gemma 2 and Gemma 4 model IDs.
This step unblinds only after **all** submitted human review files pass the
existing full, exact packet-hash, event-order and rubric audit.

## What it compares

For each of 1–10 named reviewers it counts model preference, including
TIE / BOTH_INADEQUATE / UNDECIDED. Per-model evidenceAccuracy,
contextAndMechanism and uncertaintyHonesty ratings remain separate ordinal
count distributions (SUPPORTED / PARTIAL / UNSUPPORTED / UNASSESSABLE).

When at least two reviewers grade the **same** items, pairwise exact
preference agreement is reported. Cohen's kappa is calculated for exactly
two reviewers only; if all judgments have degenerate marginal prevalence
(e.g., both always choose A), kappa is unavailable rather than fabricated.
With only one reviewer, agreement is not estimable. Neither percentage nor
kappa estimates actual news intelligence accuracy or independent adjudication.

The new create-only, owner-only private summary omits source article text,
model-generated rationales and raw reviewer explanations. It contains
pinned packet/mapping/review hashes, reviewer-entered IDs, model name,
counts and explicit `modelPromotionAllowed=false`.

**Important:** reviewer identifiers are merely supplied strings and are
not identity-verified. Source events may be inaccurate. Small-sample
(1–3 news events) qualitative agreement is not generalization proof.
The sealed model assignment is only protected by separate private files
and a hash; this is NOT cryptographic encryption or authentication.

## Existing Mac shell workflow (only once there are real completed reviews)

```sh
bash scripts/news-learning-review.sh --update gemma-reveal \
  --packet /private/blind-news-packet.json \
  --mapping /private/blind-news-mapping.secret.json \
  --review /private/review-by-human-1.json --reviewer human-1 \
  --review /private/review-by-human-2.json --reviewer human-2 \
  --output /private/new-gemma-reveal-report.json
```

The mapping must **not** be shared with reviewers before they finish.
Each review must have `humanReviewCompleted=true` and the complete
original NEWS evidence rubric, with substantive reviewer reasoning.
Missing/tampered files or duplicate reviewer names cause failure;
no partial unblind report is produced.

This tool never calls Ollama, downloads weights, writes production SQLite,
autonomously promotes a model, restarts the Mac collector or accesses
stock prices, OHLCV, trading volume and returns. Actual Gemma 4 model
inference, blind human reviews, resource fit and independent news
quality improvements **remain unverified**. Continue to use the current
production model until real human-blind evidence is reviewed.
