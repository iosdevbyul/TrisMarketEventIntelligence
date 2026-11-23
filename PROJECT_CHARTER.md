# TrisMarketEventIntelligence — Project Charter and Development Guardrails
Status: Proposed governance baseline (2026-11-23). Applies to future work after review/merge.

## Mission
The project's primary goal is to test and improve point-in-time news/economic-event forecasts against **actual subsequent market prices** for stocks, sectors, indexes and ETFs. News interpretation, source verification, and causal reasoning support this goal; none can substitute for observed-market evaluation. This project's research is independent of DonghakStockVision. Prior README statements excluding stock prices and market evaluation from this project conflict with this charter and must be reconciled in a separate, reviewed documentation change before claiming documentation consistency.

## Non-negotiable boundaries
1. Produce two distinct immutable predictions: (A) economic meaning / impact mechanism, (B) subsequent market-direction forecast for explicitly precommitted targets and horizons. Both allow uncertainty/abstention. Positive economic meaning need not imply positive subsequent return.
2. For every event record original publication, first verifiable public availability, detection, processing, market cutoff, source provenance, event family and revision. Never treat detection as original publication without evidence.
3. Differentiate new disclosure, repeated article, follow-up with new facts, correction/denial and unknown linkage; report uncertainty rather than fabricating event identity. Split historical cohorts by event family and time to avoid duplicate leakage.
4. Precommit forecast, target, window, benchmark, price-adjustment convention, market session and model/rule/data versions **before** accessing post-cutoff outcomes. Prediction and audit records are append-only; no hindsight edits.
5. A Python market validator computes observed absolute and benchmark-relative returns, directional agreement, source quality and data gaps using verifiable price timestamps and calendars. It reports observed moves, never asserts causal effects merely because the forecast matched.
6. Gemma4 or a replaceable reasoning provider proposes economic interpretations, discrepancy hypotheses, and testable candidate rules; AI-written explanations are **UNVERIFIED** until supported independently. Verified source facts are not automatically verified causal attributions.
7. Research engine tests candidate rules using chronological train / rule-selection validation / sealed untouched holdout partitions. All baseline/learned comparisons use identical eligible event-family targets, forecasts, horizons and denominators, with class-imbalance and sample-size reporting. Do not promote rules from hindsight narrative, a single case or an in-sample win.
8. Evaluation must show prediction coverage/abstention, confusion matrix, balanced accuracy / macro F1, per-horizon absolute and relative market metrics, paired baseline comparison, uncertainty, subgroup and regime results. A learning claim requires demonstrated **out-of-sample** improvement; more articles, stored rules, or prettier explanations are not proof.
9. No autonomous investment execution, no automatic trading, no unapproved 24/7 activation, and no treating fabricated/synthetic prices as real observations.
10. Preserve existing source/news/Gemma/temporal replay/legacy runs and research evidence. Before new code, inventory existing functionality including draft PR #252; reuse rather than rebuild. The 2026-11-23 roadmap numbering is distinct from legacy "Phase" card numbering.

## Authority and change control
- The six roadmap stages below are the approved *sequence for planning*, not an assertion of implementation completeness.
- Work outside a stage that is not essential to unblock its acceptance criteria goes to a later backlog. Do not silently redirect into general semantic QA, source taxonomy, API polish, or infrastructure work.
- A genuine blocker may be fixed only to the minimum extent needed to unblock the stage, with the scope and evidence recorded on the card/PR.
- Changing mission, phase order, prediction targets/horizons, evidence standards or learning success threshold requires explicit user approval with a recorded change decision.
- One active implementation workstream unless explicitly authorized; prefer a complete reviewable feature PR over many tiny PRs, but retain risk-based CI and tests.
- Trello tracks stages and acceptance evidence, GitHub PR tracks implementation, this charter defines mission. All PRs must reference the relevant Trello phase, scope, non-goals, evidence and remaining blockers.
- Manager/reviewer must reject "complete" when observed-market validation, precommitted metrics or test evidence is missing. Do not claim AI learned merely because rule memory was stored.

## Roadmap and stage gates
1. **Market outcome contract**: Define earliest availability, article-before/after price anchors, session boundaries, repeated reports and event family, market/sector benchmarks, horizons, ambiguity/abstention, causal non-attribution and data-quality statuses. Gate: deterministic examples (14:00 reversal; yesterday's repeat/rebound; after-close; overlapping releases) and signed scoring specification.
2. **Dual forecasts**: Separate semantic meaning from forecast direction/horizon; capture novelty, context, candidates, evidence, model and rule versions, time cutoff and immutable predictions. Gate: opposing semantic/market directions supported and no outcome leakage.
3. **Price/benchmark validator**: Connect timestamp-verified real minute/hour/day prices where available, adjustments, trading calendars; compute absolute/relative returns and directional matches; fail closed on missing data. Gate: historical and real-source audited samples, no causal label inferred.
4. **AI discrepancy research**: Generate competing explanations of mismatches and evidence needs, validate checkable factual predicates, mark unsupported causality UNVERIFIED. Gate: repeat-news and overlapping-event cases do not become spurious learning gold labels.
5. **Learning experiments**: Candidate rules, event-family and chronological isolation, fixed baseline/learned paired holdout, rule acceptance/rejection and versioning. Gate: honest report of effect incl. no improvement and statistical uncertainty; no automatic promotion without predeclared standard.
6. **Learning dashboard**: Surface per-event prediction, observed absolute/relative return, verification status, rule history, A/B experiment, coverage/uncertainty and improvement versus baseline. Gate: UI/API exposes actual data and distinguishes examples from observed results.

## Mandatory review questions
- Which roadmap stage and acceptance criterion is this PR satisfying?
- Which actual source/price observations and original timestamps support the result?
- What was knowable at prediction cutoff, what was only known afterward?
- Are semantic quality, forecast correctness, and *causal attribution* reported separately?
- Did the learned method outperform an unchanged baseline on independent events, or is improvement still unproven?
- Were unrelated changes deferred, existing work preserved, CI checked, and remaining limitations reported?

## Existing work and transition
- Preserve and review draft PR #252: https://github.com/iosdevbyul/TrisMarketEventIntelligence/pull/252 . Its earlier experiments and safeguards are candidate assets, **not** evidence that all six roadmap stages or learning performance have been completed.
- Older Trello work remains historical/backlog until mapped to this new roadmap. Never mark it Done solely because a new card refers to it.
- Update README/docs ownership contradiction via a separate reviewed follow-up, unless incorporated explicitly into the governance PR before merge.
