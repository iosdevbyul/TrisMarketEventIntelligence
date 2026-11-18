# Phase 3 + Phase 4: integrated independent-event first-pass pilot

## Scope

This implementation deliberately combines official-source **intake
(Phase 3)** and **first-pass Gemma4 cutoff analysis (Phase 4)** in one
command. Its first two completely new diagnostic event families are the
FTC noncompete rule (2023 proposal, April 2024 final-rule announcement)
and EPA Lead and Copper Rule Improvements (November 2023 proposal,
October 2024 final-rule announcement).

The source packet was written separately from a reference answer key,
and has **no gold labels, future cutoff disclosures or curated model
answers**. The model settings are identical across both new event
families and frozen at `first-sight-v1-no-gold-labels`. This pilot
captures eight question/cutoff judgments (two questions times two
cutoffs times two events). It does NOT count as the required five
independent, untouched event families or thirty adjudicated judgments.

## What is and is not authenticated

The acquisition step retrieves each **currently published** official
HTML page, verifies exact short source excerpts against its visible
page text, and records capture UTC time, HTTPS source URL, agency/source
IDs and SHA-256 of the received HTML bytes. It rejects redirects,
unexpected hosts, missing excerpts, oversized responses and unexpected
media types. The model input contains **only excerpts explicitly
selected in the versioned packet**, never the entire mutable webpage.
For example, the FTC final-rule page now includes subsequent August
2024 court enforcement news; this must not be admitted into April 2024
cutoff prompts.

A current official page is NOT an independently archived page snapshot
from 2023 or 2024, so the following remain **FALSE** throughout:
`historicalPITVerified`, `historicalSnapshotVerified`,
`generalizationVerified`, `trustEligible`. Agency source dates are
only **date-granular candidate dates**, not verified publication
timestamps. No financial or market/stock outcomes are inferred.

The current EPA final-rule HTML displays an October 7, 2024 metadata
date but refers to issuance on October 8 in its body; the pilot treats
October 8 as the dated described action and only makes it cutoff-visible
on October 9. This does NOT reconcile publisher clock provenance.
Before formal historical PIT certification, original archived captures
and the actual issuance/publication timing must be independently
verified, preferably using regulator original attachments and dated
Federal Register records. This pilot must never be relabeled a
historically verified holdout based on a current-page checksum.

## End-to-end execution

```bash
git pull --ff-only origin feature/evidence-first-composed-reasoning
bash scripts/run-independent-first-sight.sh
```

Phase 3: verifies four current official pages and writes a create-only,
private source-capture ledger named
`independent-first-sight-current-source-capture-v1.json`.

Phase 4: requires the unchanged packet hash, verified quote ledger and
installed local `gemma4:e2b` before making **eight model calls**. The
private create-only result is
`independent-first-sight-gemma4-firstpass-v1.json`. Each assessment
includes original model classification, input-bound primary evidence,
raw implementation state, safe published state and explicit unknowns.
All eight complete assessments must succeed before a result is saved.
This is an actual first-pass inference **only when the local script
executes successfully**, not when the GitHub unit tests pass.

A repeated command must NOT overwrite the immutable first pass. If
new data or model code is evaluated later, create a separately
versioned packet and first-pass artifact, and clearly mark subsequent
runs as tuned development studies rather than untouched tests.

## Acceptance and follow-on

First-run success requires four current agency pages to pass short
excerpt checks, strict date exclusion (no future claims at early
cutoff), eight parsable responses, input-bound evidence indexes, all
results saved and zero source/PIT overclaims. Failure to retrieve an
official source or match its quote is a **blocker**, not permission to
invent or substitute content.

A meaningful reliability conclusion requires separate work: historical
publication-time page captures or authenticated dated filings,
independently reviewed answer labels, at least five untouched event
families, 30+ question-cutoff observations, predeclared thresholds and
zero unsupported completions on raw model results. Never use Phase 4
pilot results to invent a retrospective gold answer.

This pilot is deliberately **not** wired to the FAA/FDA v2 scoring
reference because those thresholds and labels were repeatedly tuned to
those development cases.
