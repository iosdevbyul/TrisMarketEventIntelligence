# Learning-08 — Gemma 2 vs Gemma 4 blinded human NEWS review

The first PR #132 compared only strict output-contract acceptance and latency.
This phase adds an **explicit human assessment packet**. Neither structural
acceptance nor fluent language establishes better context understanding.

## Procedure

The tool uses the same existing archived NEWS event and exactly the same
prompt/version/source-validation path with locally installed `gemma2:latest`
and one explicitly sized Gemma 4 tag. No price, OHLCV, return, trade-volume
or DonghakStockVision records are read.

For each original news event, the two outputs are randomly assigned to
blinded slots `A` and `B`. The private packet contains original source
context and both candidates' evidence quotes, contextual mechanisms and
uncertainty (or a sanitized failure code). **It deliberately omits model IDs.**
A separate private sealed mapping stores the event-to-model assignment and
the hash of the exact packet. Never show that mapping to the reviewer until
their blind grading is finalized. Source and model-produced text in the
packet are **untrusted** and may themselves mention a model's name; the
software cannot guarantee complete psychological blinding.

Reviewers must assess both answers, independently checking the reported
source context. Required ratings are `SUPPORTED`, `PARTIAL`,
`UNSUPPORTED` or `UNASSESSABLE` for (a) original news attribution,
(b) exact evidence accuracy, (c) context and conditional mechanism,
and (d) uncertainty honesty. They must select `A`, `B`, `TIE`,
`BOTH_INADEQUATE`, or `UNDECIDED`, and provide a substantive reason.
Missing grades cannot be filled by a model. The audit checks original
packet hash, article IDs/input hashes, exact rubric and explicit completion,
but **does not certify that the reviewer was independent or correct**.

## Mac commands (only after Gemma 4 is intentionally installed and benchmarked)

Existing Mac shell workflow handles all operations:

```sh
bash scripts/news-learning-review.sh --update gemma-blind --run \
  --gemma4 gemma4:e4b --limit 2 \
  --packet /private/new-news-blind-packet.json \
  --mapping /private/new-news-blind-assignment.secret.json

bash scripts/news-learning-review.sh gemma-blind --prepare \
  --packet /private/new-news-blind-packet.json \
  --output /private/new-human-review-template.json

# Human independently edits only the review template and sets
# humanReviewCompleted=true when each item has complete ratings/reasons.
bash scripts/news-learning-review.sh gemma-blind --audit \
  --packet /private/new-news-blind-packet.json \
  --review /private/completed-human-review.json \
  --reviewer external-editor-1 \
  --output /private/new-blind-review-audit.json
```

Each file is a new private create-only path (owner-only mode) to avoid
unintentional overwrite. The exact model mapping is never included in
the reviewer packet, review template or audit result. Keep it separate
from human reviewers and all source material outside public GitHub/CI.

`--run` requires both Ollama models to be **already installed**.
There is no pull, download, deployment, production DB write, model
promotion, LaunchAgent activation or autonomous 24/7 learning.
The private model's actual RAM use and inference quality have **not**
been observed by mocked CI.

## Current limits

A maximum of three NEWS events prevents accidental heavy local
inference. This is a manual qualitative pilot, **not** a statistically
adequate model evaluation. It does not sample a locked TEST cohort,
establish source truth, resolve cross-provider news duplication, perform
blinded multi-reviewer adjudication, or authorize any model promotion.

Separate reveal and aggregates should follow **only after** complete
independent human reviews and source-integrity audits. Until then,
the existing Gemma 2 production model remains unchanged.
