# Learning-05: actual two-news semantic review and guarded abstention

2026-11-15: reviewed the Mac original immutable 482-article RSS cohort
and original Gemma4:e2b output from `news-evidence-pilot-completion-20261115.json`.
All source article IDs, text hashes and full cohort SHA matched. Both
model hypotheses passed the old Pydantic/literal-quote contract,
**but neither has independently established semantic correctness**.

- Article **336**, BBC Business, Microsoft service restoration: source
  reports that *Microsoft says* restoration continues. The model assigns
  `CONFIRMED / MODERATE / UNCERTAIN` but its mechanism merely repeats
  ongoing restoration and does not describe the affected organization,
  service interruption effects, duration, cost or related conditional
  causal channel. Attribution should not automatically be treated as
  independent confirmation.
- Article **410**, BBC Technology, `Tech Now`: source says presenter
  Ammie Sekhon *explores* technology to find and remove counterfeits.
  Presenter name appearing in the RSS sentence does not establish that
  the presenter is an economically impacted entity. The model outputs
  `ALLEGED / LOW / UNCERTAIN` even though no allegation or concrete
  consequence appears in the short source. Critically, its mechanism is
  `None identified based on the summary.` A materiality band with an
  explicitly missing mechanism is self-contradictory. The defensible
  output is `UNASSESSED / NOT_ASSESSED / UNCERTAIN` if no impact can
  be sourced.

## Narrow prevention

The existing **schema**, SQLite/API and research cohorts remain unchanged.
One minimal deterministic post-validator refuses assessed magnitude when
the model's own mechanism starts with an explicit no/none/not-known
statement. It does **not** manufacture a mechanism, infer business
impacts, transform human gold, or pretend a semantic grade was measured.
A legitimate conditional mechanism still passes.

The existing opt-in **structured local NEWS pilot** prompt now
explicitly distinguishes presenters from impacted entities, attributed
claims from independent confirmation, and absent evidence from a LOW
impact grade. This does not alter other live impact, pressure or
whole-cohort production default prompts. Other unsupported inference
errors (for example a nonempty but vague mechanism) can still pass and
require separate semantic review.

A failed item is accounted under
`MATERIALITY_MECHANISM_NOT_IDENTIFIED` and
`REJECTED_OR_MODEL_RUNTIME_FAILED` in future pilot reports,
not counted as a successful hypothesis or a human label. Preserve the
previous 2/2 contract-passed result unchanged as historical evidence.

**Outstanding:** historical source semantic correctness and independent
event adjudication, appropriate abstention rates, broader 48-candidate
coverage across diverse dates, repeat stability, and real Mac validation
with newly generated outputs. Do not promote or activate a model.
No price/return/volume/OHLCV data, investment execution or TEST scoring.

## Actual same-news recheck after PR #156

The operator ran the real local `gemma4:e2b` against the same two pinned
news articles and supplied the new create-only private output
`news-evidence-pilot-semantic-recheck-20261115.json`.
Both still passed the **contract**, but the model did **not** demonstrate
semantic correctness. The report explicitly flags independent accuracy and
model improvement as unverified.

- Article 336: entity Microsoft; `CONFIRMED / MODERATE / UNCERTAIN`.
  The new mechanism now describes possible user dissatisfaction, data-loss
  concerns, and reputational effects. These are clearly potential outcomes,
  **not independently observed impacts**. `CONFIRMED` still risks treating
  an attributed Microsoft restoration update as externally verified.
- Article 410: `REPORTED / LOW / UNCERTAIN`, entity remains presenter
  Ammie Sekhon. The model's mechanism is **exactly**
  `No concrete conditional organizational mechanism can be inferred from
  the sourced event.` The old validator only recognized shorter
  `No mechanism...` prefixes; it incorrectly accepted this explicit
  admission alongside the assessed LOW magnitude.

This follow-up expands that **same deterministic missing-mechanism guard**
to optionally qualified no/none statements such as `No concrete conditional
organizational mechanism` and `No specific economic impact mechanism`.
It refuses the contradictory assessed magnitude with the existing safe
`MATERIALITY_MECHANISM_NOT_IDENTIFIED` code. It does not turn the model's
claim into a human Gold label, silently rewrite LOW to UNASSESSED, or
assert that subjective news importance is solved. The existing plain
`UNASSESSED/NOT_ASSESSED/UNCERTAIN` contract still accepts valid abstention.

The earlier results are kept unchanged. Broader sample diversity and
semantic quality require actual separate research evidence, not a 2/2
technical acceptance rate. No operational scheduler, production DB,
TEST/holdout, investment price or auto-promotion changes.
