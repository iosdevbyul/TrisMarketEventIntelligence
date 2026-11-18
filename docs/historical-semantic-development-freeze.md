# Historical semantic reasoning: FAA/FDA development freeze

## Observed comparative baseline (2026-11-18)

Saved real-model, cutoff-scoped Gemma4 runs all completed, but **no original
historical page-snapshot PIT is independently verified**. These figures are
retrospective, provisional development-case ratings; the same FAA/FDA cases
have repeatedly informed model design. They cannot establish generalization.

| Measure | FAA v8 / FDA v5 | FAA v9 / FDA v6 | FAA v10 / FDA v7 |
|---|---:|---:|---:|
| Required claim recall | 100% | 100% | 100% |
| Primary evidence accuracy | 75% | 75% | 75% |
| Decision classification accuracy | 87.5% | 75% | 50% |
| Requested evidence accuracy | 87.5% | 100% | 87.5% |
| Direct original record accuracy | 100% | 100% | 87.5% |
| Indirect regulatory progress accuracy | 100% | 100% | 100% |
| Raw unsupported completion claim rate | 12.5% | 12.5% | 12.5% |
| Published completion safety | 100% | 100% | 100% |
| Benchmark error count | 5 | 5 | 9 |

The 100% published completion safety is a quarantine-layer property: a
model-reported COMPLETED can be mapped to UNKNOWN without making the model
reason correctly. Repeatedly fixing prompts for these same examples can
inflate the development benchmark while hurting unseen events.

## Research decision

**Freeze FAA and FDA as development/regression cases.** Do not claim that
v10/v7 is superior to v9/v6, or continue optimizing prompt wording to
these two outcomes. Keep the historical official-source composition and
evidence-state quality checks as prototype capabilities; keep PR #250 Draft.

The model's unresolved issues include:
- Selection of old procedural notices despite newer decisions that answer
  current-state questions, and the inverse on other cutoffs.
- Regulatory ANNOUNCED / RESTRICTED / AUTHORIZED / UNKNOWN confusion.
- Unproven final implementation completion inferred from authorization.
- Narrow editorial claim reference coverage versus actual historical PIT.

## Offline regression comparison

```bash
git pull --ff-only origin feature/evidence-first-composed-reasoning
bash scripts/compare-composed-evidence.sh
```

This compares **saved** v9/v6 and v10/v7 v2 score files without Ollama calls,
rejects mismatched schemas or case populations, and writes a new report.
Even `NO_MEASURED_REGRESSION` does not mean the model is reliable.

## Next acceptance experiment

Prepare at least five **new, untouched event families** with dated, independently
retrievable and publication-time authenticated regulator/official-source
snapshots. Define 30+ cutoff/question judgments on at least five
previously untouched independent families. FAA/FDA cannot count as unseen
holdout. A frozen answer key must never enter a model prompt or iterative
tuning. Keep separate development and holdout result files, and require
independent reviewers to adjudicate ambiguous question scopes and answer
labels before calculating the formal trust gate.

Use the same staged model configuration and frozen evaluation protocol on
unseen data. If any holdout result triggers unsupported future claims,
unverified implementation, critical evidence omissions, or repeated
regulatory decision mismatches, the research-trust gate remains closed.
