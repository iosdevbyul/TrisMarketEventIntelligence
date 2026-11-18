# Historical event understanding: semantic quality benchmark

## Objective

Measure whether a language model understands the **state of an unfolding
historical official-source event**, not whether it emits valid JSON. This
evaluation is independent of DonghakStockVision and makes no stock-price,
market-impact, or causal-completion assertion.

The first **development benchmark** uses eight question/cutoff observations
across FAA January 2024 and FDA April–October 2023. Ground truth here is **retrospective
curator judgment over dated official-release paraphrases**, not original
publication-time web captures. The benchmark is deliberately excluded from
all model prompts and only read by the offline scorer. Because FAA/FDA were
used repeatedly to guide design changes, they are NOT unseen holdouts.

## Dimensions

- **Source coverage**: every newly visible curated claim is explained.
- **Required evidence recall**: specific dated official references that must
  inform the question are present in `supportingEvidence`.
- **Regulatory decision accuracy**: model chooses ANNOUNCED, AUTHORIZED,
  RESTRICTED, or NOT_OBSERVED within a reviewer-approved range.
- **Requested evidence status**: missing, partial or direct documentation
  must be distinguished from merely related regulatory decisions.
- **Completion safety**: announcement, permission and restriction must not
  be misrepresented as completed implementation or audit findings.
- **Future data leakage**: official references unavailable at the cutoff
  must not be selected, even if known from retrospective curation.
- **Explainability**: model-screened NOT_RELEVANT decisions are preserved
  for omission reviews; a selected claim does not validate its own relevance.

The scorer reports individual failures alongside macro accuracy. Its
`evaluationSplitIsFrozenHoldout` trust gate deliberately fails on these
FAA/FDA development cases even if every scored answer matches. A score of
100% on eight observations does NOT establish generalization.

## Important adjudication case

The FDA October 2023 manufacturer filing-and-review-record question is
currently a **reference adjudication candidate**. The September mRNA and
October Novavax authorization decisions are relevant *intermediate actions*,
but neither release provides the original manufacturer filing receipts or
agency internal reviews. A `NOT_OBSERVED` status can be defensible when
interpreted as *direct document availability*, whereas `PARTIAL` is
defensible when interpreted as *all relevant progress evidence*.

The provisional reference expects PARTIAL because the former question axis
was defined to track intermediate action as well as completeness. Independent
review should resolve this ambiguity, ideally by a future schema explicitly
distinguishing direct-document availability from indirect evidence progress.
Do not train or tune the model to this disputed label as though it were
independent ground truth.

## Run against already-saved evidence (no Ollama calls)

```bash
git pull --ff-only origin feature/evidence-first-composed-reasoning
bash scripts/score-composed-evidence.sh
```

This reads `faa-evidence-composed-v7.json` and
`fda-evidence-composed-v4.json` from the existing private application
support directory and writes a new, non-overwriting score report there.
To score a different run, pass FAA path, FDA path, and new output path
as the three positional arguments.

Use the same benchmark file for comparisons. Preserve existing run artifacts
and never inject answer-key transitions into model inputs.

## Research acceptance gate

The implemented gate is intentionally strict:

| Requirement | Minimum |
| --- | --- |
| Complete newly visible source coverage | 100% |
| Required official-reference recall | 100% |
| Regulatory decision classification | 95% |
| Requested evidence status classification | 95% |
| Unproven completion safety | 100% |
| Critical errors including future leakage | 0 |
| Independent event families | At least 5 |
| Question/cutoff judgments | At least 30 |
| Independent reference adjudication | Required |
| Evaluation split | Previously untouched, frozen holdout |
| Historical publication-time snapshot verification | Required |

A benchmark can be structurally green while the **research trust gate stays
red**. This is correct and expected with only two retrospectively curated
event families. A passing score never authorizes automated trading,
unreviewed external claims, or an assertion of verified economic causality.

## Next empirical work

1. Grade the saved 2026-11-18 FAA/FDA runs and freeze that first baseline.
2. Resolve disputed scoring definitions with independent reviewer evidence,
   and version the reference rather than silently editing a key.
3. Introduce at least three additional independent official-source event
   families, using genuinely historical, publication-time verifiable records.
   Reserve these as frozen, previously untouched evaluation data, not as
   further iterative prompt-tuning examples.
4. Run a frozen holdout comparison with the same prompt/model settings,
   measuring error types rather than only aggregate success.
5. Only after thresholds, source integrity, historical PIT, independent
   adjudication and repeatability are verified should an analyst describe
   Gemma4 as sufficiently reliable for the **measured** event question class.

## Baseline measurement and next two-axis reasoning candidate

The user's offline score on FAA v7 / FDA v4 gave source coverage and required
reference recall 100%, decision classification 7/8 (87.5%), evidence status
7/8 (87.5%), completion-safety 8/8 (100%) and two reference mismatches.
Trust remained NOT_YET_TRUSTWORTHY, as intended. These are **development**
cases, and the reference has not received independent adjudication.

The FDA manufacturer-filings mismatch is diagnostically important: publicly
announced authorizations are intermediate progress but **not the original
manufacturer filing receipts or the detailed agency-review records**.
Future model answers now separately classify `directRecordAvailability`
(NOT_OBSERVED / PARTIAL / DIRECTLY_OBSERVED / UNKNOWN) and
`indirectRegulatoryProgress` (NOT_OBSERVED / OBSERVED / UNKNOWN), alongside
the existing `requestedEvidenceStatus`. The system preserves old scorer
results on FAA v7/FDA v4; new axes must be evaluated prospectively and cannot
retroactively prove that the old output was correct. Contradictions are
flagged for human review, never silently rewritten.

The next prospective model artifacts will be FAA v8 / FDA v5 using the same
existing `scripts/verify-composed-evidence.sh`; even if this development
comparison improves, **it is not evidence of untouched-holdout
performance**. Source integrity and publication-time availability remain
unverified and must be established before any research-trust declaration.
