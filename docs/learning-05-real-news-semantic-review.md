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
