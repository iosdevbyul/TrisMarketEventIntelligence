# Historical News Market Impact Test — 실제 사용자 테스트

## 목적과 범위

이 기능은 **과거에 발행된 뉴스 여러 건을 주어진 분석 시점(asOf)까지만 읽고**
다음의 완성된 답안을 한 번에 생성한다.

1. 각 기사를 사건으로 묶고 최초 보도·새로운 사실·후속 조치·반응·정정을 구별한다.
2. 관련 기사 사이에 FOLLOW_UP / NEW_FACT / CORROBORATES / CONTRADICTS / REACTION 관계를 기록한다.
3. 영향을 받을 기업과 섹터(직접·간접)를 식별하고 조건부 전달 메커니즘을 설명한다.
4. 기업·섹터별 POSITIVE / NEGATIVE / MIXED / UNCERTAIN 방향, LOW / MODERATE / HIGH 영향 강도,
   예측 시간 범위(1·3·5 거래일), **가정에 따른 초과수익률 범위(%)** 를 기록한다.
5. 주장마다 입력된 뉴스의 **실제 인용문 ID**를 연결하고 모델이 인용문을 만들 수 없게 한다.
6. **실제 주가·사후 수익률을 입력받지 않고** 별도의 새 비공개 파일에 답안을 봉인한다.

이번 기능은 가상 시나리오의 **미보정(UNCALIBRATED) 시장 반응 가설**이다.
현재 숫자 오차나 확률이 과거 자료로 보정되었다는 근거는 없다. 기사 내용과 다른 주가
움직임이 있을 수 있고, 인과관계가 입증되지 않았으므로 금융거래 판단에 사용하지 않는다.
주가 정답과 비교하는 실험은 별도 단계에서 진행한다.

기존 자동 뉴스 수집, SEC 원본 SHA 검증, 기타 과거 연구는 건드리지 않는다.
이 테스트는 별도의 명시적 실행 경로로 제공한다.

## 실제 테스트 시작

기존 Mac 프로젝트 루트에서:

    DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
    cp examples/historical-news-impact-test.example.json "$DATA/my-historical-news-case01.json"

복사한 JSON에 **실제로 테스트할 과거 기사 2~6건**을 입력한다.
예제 기사와 EXDV 회사는 **모두 가상**이며 성능 평가용으로 사용하면 안 된다.

필수 입력:

- schemaVersion: historical_news_impact_test_v1
- caseId: 실험을 구분할 ID
- asOf: 기사 분석 컷오프. 예: 2023-05-25T14:00:00-04:00
- horizonTradingDays: 1, 3, 또는 5 거래일
- articles: articleId, publishedAt (시간대 포함), source, title, body를 포함하는 2~6개 기사
- knownTickers: 선택 사항. ticker/companyName/sector 사전 (명칭·티커 혼동을 피하고 싶으면 제공)

**주의:** 모든 기사 publishedAt은 asOf보다 늦으면 안 된다. 기사는 당시 공개됐던 원문만
입력하고, 이후 주가·정답·수익률이나 사후 해설을 절대로 포함하지 않는다.
시간은 사용자가 입력한 값으로만 검증되므로, 독립적으로 출판 시각까지 입증된 것은 아니다.

### 1. 기사 형식만 점검 (모델 호출 없음)

    bash scripts/news-learning-review.sh --update historical-impact-test \
      --input "$DATA/my-historical-news-case01.json" --prepare

BLIND_NEWS_IMPACT_READY가 출력되어야 하며, articleCount와 quoteAnchorCount,
newsInputSha256를 확인한다. 이 단계는 모델을 부르지 않고 파일도 생성하지 않는다.

### 2. Gemma4 실제 분석 (한 번만 실행)

    bash scripts/news-learning-review.sh --update historical-impact-test \
      --input "$DATA/my-historical-news-case01.json" \
      --run --model gemma4:e2b \
      --output "$DATA/my-historical-news-impact-answer01.json"

결과가 성공하면 같은 이름으로 두 번 실행할 수 없다. **다음 실험은 새 파일명**을 사용한다.
원본 기사 파일을 수정하면 newsInputSha256이 바뀌므로 동일 실험의 후속 평가로
취급해서는 안 된다. 결과 파일에는 원본 기사의 해시, 사건 관계, 추정 방향·범위,
sourceQuoteId, 실제 원문 인용과 검토 상태가 들어 있다.

    python -m json.tool "$DATA/my-historical-news-impact-answer01.json"

미확인 모델 티커는 UNVERIFIED_MODEL_PROPOSAL로 표시한다. 해당 티커가 진짜 상장회사에
속하는지는 별도로 검증해야 한다. 입력한 knownTickers와 충돌하면 결과는 거부된다.

### 3. 실제 정답과 비교

**위 2단계의 답안을 먼저 보관한 후에만** 지정한 시장/종목의 가격을 조회한다.
모델은 실제 가격을 보지 않고 그 답안을 변경할 수 없어야 한다. 검증 시에는
첫 공개시각, 거래소·시간대, 정규장/장외, 다음 거래일 기준, 분할·배당 조정,
시장·섹터 대비 초과수익률을 동일한 규칙으로 계산한다.

- 방향 일치 여부: 상승/하락/중립 판정 (MIXED/UNCERTAIN은 별도 관리)
- 크기 오차: 실제 시장·섹터 대비 초과수익률이 예상 구간 내에 있는지
- 직접/간접 영향 기업과 섹터의 정확성
- 출처 인용과 뉴스 시점의 사실관계, 최초 보도와 후속보도의 차이
- 실적 발표나 거시 이벤트 등 동시에 발생한 다른 영향 가능성

이 단계의 실현 결과는 아직 자동 기록하지 않는다. 시장의 관측된 가격이 뉴스의
인과 효과를 증명하지 않는다는 점을 명확히 구별한다. 반복 사용 전에는 별도
학습용/검증용/최종 봉인용 사건 분리를 권장한다.

## 안전 장치

입력/결과는 엄격한 JSON 계약이며 기사 인용문은 입력 원문에 존재해야 한다.
기사 시간은 분석 시점 이하여야 한다. 관련 없는 보도를 강제로 연결하면 안 된다.
방향이 불확실하면 영향 수치 범위를 빈값(null)으로 기록한다.
양/음의 수치 범위는 -15%~+15%로 제한되며 **보정된 통계적 구간이 아니다**.
가설의 근거·주체·티커·방향이 계약을 위반하면 모델 응답을 조용히 수정하지 않고 실패한다.
이때 프로그램은 기존 파일을 덮어쓰지 않으며 학습, 운영 배포, 거래 실행이 없다.

중요한 제한: 현재의 출처 인용·숫자 범위 검증은 모델 설명의 의미적 참을 보증하지 않는다.
또한 knownTickers에 없는 모델 제안 티커는 확정된 종목 식별이 아니다.


## URL-only original articles — preferred blind historical test

Use only links to independent reporting on ONE incident within a narrow
publication window. Never hand-author a summary or publication timestamp.
The collector accepts multiple --url values, pulls the article paragraphs
from the real publisher HTTPS page, reads its declared publication clock,
and passes the extracted news directly to the model.

    DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
    bash scripts/news-learning-review.sh --update historical-impact-test \
      --url "https://techcrunch.com/2026/11/08/nvidia-says-its-investigating-ceo-jensen-huangs-relationship-with-possible-chinese-spy/" \
      --url "https://www.axios.com/2026/11/08/nvidia-ceo-jensen-huang-admits-affair-board-to-launch-investigation" \
      --url "https://www.washingtonpost.com/technology/2026/11/08/read-jensen-huangs-letter-nvidia-about-his-affair-with-suspected-chinese-spy/" \
      --case-id "NVDA-SUNDAY-20261108" \
      --as-of "2026-11-08T23:59:00-05:00" --horizon 1 \
      --run --model gemma4:e2b \
      --source-output "$DATA/nvda-sunday-source-original-v1.json" \
      --output "$DATA/nvda-sunday-blind-answer-v1.json"

Cutoff means November 8 evening in New York; it is not a claim that the
sources were available at the exact cutoff. Every article needs machine-
readable publisher publication metadata and substantive HTML article body.
Paywalls, unreadable pages, missing times and post-cutoff revisions must FAIL;
there is no manual backfill and no invented time. Sources may subsequently
edit a webpage without publishing dateModified, so *independent historical
page-version validity is not proven*. No market prices or outcomes are loaded.

The collected source is saved first in a create-only private JSON; if model
generation fails, use --input pointing to this exact snapshot for retry
with a new result filename. This avoids a second network fetch.

A malformed, reversed or cross-group Gemini event-link proposal is excluded
from the verified eventLinks list and recorded in rejectedEventLinks with
an explicit rejection reason; a source-grounded prediction need not fail
solely because of an invalid nonessential connection. Event clustering
remains an unverified model interpretation, not proof of causal relation.

The article text is bounded to a per-article character budget; the system
does not claim this snapshot preserves the entire raw HTML page. Any article
not fully accessible in public HTML must be replaced with another genuinely
independent publisher article, not a fabricated paraphrase.

## Post-cutoff source revision: two distinct test modes

NEWS_URL_REVISED_AFTER_CUTOFF is an intentional point-in-time safety rejection.
The error now identifies article ordinal and host, allowing precise diagnosis.
The original strict historical mode still rejects pages modified after as-of.

For a qualitative smoke test of actual, publisher-extracted, potentially edited
current articles, add --exploratory-current-pages to the URL-only command.
This does not use hand-authored summaries or post-event stock prices; however,
the source may now contain edits not available at the chosen historical cutoff.
The result explicitly records sourceIntegrityStatus, revisedSourceArticleIds,
per-article sourceModifiedAt and historicalBenchmarkEligible=false.
Such an answer MUST NOT be graded as an unbiased historical stock prediction.

For a valid blind historical market-impact backtest, obtain immutable original
versions verifiably published by the chosen cutoff (e.g. archived snapshots).
Never raise the as-of cutoff just to suppress a revision error and then claim
the response was a prediction made before the market response.

## Publisher HTML body extraction

Original publisher HTML can place the story in an article tag, a Wordpress post-content div, a story-body/content div, or main; the collector now scans those content containers while filtering navigation, recommendation, paywall, advertisement and sidebar blocks. A populated NewsArticle.articleBody embedded by the publisher in JSON-LD is also accepted as a source-body fallback. Summaries or structured-data descriptions are NOT substituted for the article. If the substantive body cannot be recovered, the operation still fails with NEWS_URL_ORIGINAL_BODY_NOT_AVAILABLE and the failing article index and publisher domain. Live publisher access is validated by the user's real URL test, not mocked CI HTML alone.

## Publisher access failures in exploratory mode

If one publisher returns HTTP 403, 404, 429, or blocks article-body extraction, URL-only exploratory mode now records its ordinal, host and reason in skippedSources and continues with other genuinely extracted articles. This works ONLY when at least two complete publisher articles remain; strict historical mode still fails closed at the first inaccessible source. A partial cohort is never claimed to be a statistically valid historical benchmark. No article text is fabricated or replaced with a human summary.

## Diagnostic stage markers (model/article data never logged)

Failures now provide STAGE and a diagnostic DETAIL. STAGE=ARTICLE_COLLECTION means the publisher URL fetch/HTML phase. STAGE=SOURCE_INPUT_VALIDATION means a real source body passed extraction but did not match the bounded case schema; schema errors show only field paths and Pydantic error categories, never article contents. STAGE=MODEL_INITIALIZATION means that Ollama/provider did not start. STAGE=MODEL_AND_ANSWER_VALIDATION means generation or validation of Gemma's JSON answer failed. STAGE=ANSWER_OUTPUT_WRITE is the create-only result persistence step. The URL runner preserves an immutable source snapshot before model generation; when a source snapshot exists, retry directly via --input PRIVATE_SOURCE_JSON --run --model gemma4:e2b --output NEW_RESULT_JSON without refetching or constructing hand-authored summaries. Do not use a reused or post-cutoff-revised snapshot for blind historical accuracy scoring.

## Bounded Gemma event-chain and analysis contract regeneration

Real local-model responses can contain duplicate event stages even when the source articles are sound. On exactly one recoverable **model-authored** validation problem (duplicate/missing stage coverage, incompatible grouping, forged source quote IDs, impact ticker/sector identity or sign/range violations, and schema-only structural failures), the runner makes **at most one** extra Ollama call with the same articles, cutoff and quote anchors plus a failure-code or field-path correction instruction. Python never silently removes duplicates, assigns article stages, alters ticker identities, edits magnitude intervals or invents model conclusions. The second result must satisfy **all original validators**; if not, execution fails closed. Output reports modelCalls and modelContractRetryReason, and rangeSchemaRetryUsed is true only for range-schema regeneration. This does not read or introduce future returns and does not make unarchived or revised live pages eligible for backtested historical accuracy.

## Recover from Gemma numeric range schema violations

When Gemma4 returns an otherwise valid research JSON with only an invalid `impacts[N].conditionalMovePctRange` (e.g., reversed percent endpoints or range width above 10 percentage points), the CLI makes at most one additional LLM call. The model receives the same extracted news and evidence anchors plus explicit sign/range schema constraints; **Python never clips, swaps, fills, or silently changes a forecast**. Other JSON/schema faults still fail closed. The saved analysis reports `modelCalls` and `rangeSchemaRetryUsed`; an exhausted retry remains a failure, not a successful prediction. Preserve the existing create-only source snapshot, do not fetch another article or reveal future stock returns during retry.

## Persistent duplicate event-stage proposals (honest partial result)

A real Gemma4 answer may repeatedly emit two or more eventStages for the same source article even after an explicit instruction to return exactly one. Previously the whole article impact analysis failed, consuming extra model calls without producing a report. Now the validator leaves **all** duplicated stage proposals unaccepted, records each proposal with `DUPLICATE_OR_AMBIGUOUS_STAGE` in `rejectedEventStages`, and identifies affected source article IDs in `unresolvedStageArticleIds`. Unknown article IDs are also rejected. No stage is chosen or fabricated by Python, and other model-proposed source-backed findings still undergo all existing target/quote/magnitude checks. Any rejected or unresolved stage forces `eventStageCoverageComplete=false`, `status=NEWS_EVENT_STAGES_INCOMPLETE_REQUIRES_REVIEW` and `historicalBenchmarkEligible=false`, and preserves the usual accuracy/training/promotion restrictions. This is explicitly a **partial analysis**, never a successful historical reconstruction or proven event chronology. Good stages and source-grounded impact hypotheses remain reviewable. Source article bodies and future prices are not changed or refetched, and repeated duplicate-stage proposals do not trigger redundant expensive model regeneration.

## Impact proposal validation as an auditable partial result

A valid JSON response from Gemma4 can use an unsupported sector taxonomy (for example INFORMATION_TECH rather than the project’s strict enum), an unverified ticker, missing source evidence or inconsistent sign/range. Such an impact is **not silently reclassified, clipped, or reported as a valid result**. The validator now checks *every* proposed impact independently: valid proposals retain the previous strict guards; invalid ones retain their original model-authored fields in `rejectedImpacts` along with `impactIndex` and a precise `NEWS_TEST_*` reason. A rejected proposal forces `impactValidationComplete=false`, `status=NEWS_PARTIAL_ANALYSIS_REQUIRES_REVIEW`, `historicalBenchmarkEligible=false`, and the existing false accuracy/training/promotion gates. Even if all impacts are rejected, the report is partial and does not misrepresent that as informed abstention. Stage uncertainty remains independently quarantined. Event-group coverage and structural output schema remain strict; this change does not introduce a new category mapping, fabricated entity/evidence, or additional model invocations. Users must assess the actual raw proposal and source before deciding whether any taxonomy extension is warranted.

## Issue #222: source-anchored news understanding (v6)

The regular `--run` command now invokes a bounded **two-stage** model flow:
(1) `historical_article_understanding.extract_claims` identifies source-anchored
FACT / ALLEGATION / DENIAL / OPINION / FORECAST items and their individual
INITIAL_REPORT / NEW_FACT / OFFICIAL_ACTION / REACTION / CORRECTION /
INDEPENDENT phases; (2) the existing event/impact reviewer consumes those
model-generated claims **alongside the original articles and exact quote
anchors**. The LLM provider interface is unchanged, and the first output is
validated against the original per-article quote membership. It is a model
interpretation, **not independent confirmation of the claim's truth**.

Event stages can now reference `quoteId`. Different quotes from the same
article may have distinct stages; duplicate quote IDs, cross-article quote
references, and unknown article IDs are quarantined instead of being silently
accepted. Legacy article-only stages are retained for compatibility but
multiple unanchored stages for one article remain ambiguous and incomplete.

Qualitative POSITIVE/NEGATIVE impact hypotheses may explicitly omit numeric
`conditionalMovePctRange` (null). If the model cannot justify a ticker, a
source-named STOCK proposal without a ticker is retained in
`unverifiedCompanyCandidates`, **not** in validated `impacts`. An unknown
sector label is not silently mapped into the taxonomy. Revised-source inputs
remain ineligible for strict point-in-time backtesting, and no market prices,
realized returns, model training, or trade execution enter this path.

A successful response in this version is NOT evidence that Gemma4 resolved
the actual historical case: inspect `eventGroups`, source-backed
`eventStages`, `rejectedEventStages`, `rejectedImpacts`,
`unverifiedCompanyCandidates`, and `articleUnderstanding` against
the privately pinned original documents. `modelCalls` includes the
additional extraction pass. The compatibility switch
`--legacy-single-pass` skips extraction for comparing previous runs;
it is not the default pipeline. If using the two revised publisher pages
from the exploratory NVDA cohort, do not score them as a blind benchmark.

## v10 review remediation: evidence-grounded target identification (v7)

In response to the real privately pinned NVDA exploratory v10 run,
`--run` now has **three distinct Gemma4 calls** by default:

1. Article quote extraction and FACT / ALLEGATION / DENIAL distinction.
2. Bounded company / sector target identification with source-quote checks.
3. Event reconstruction, evidence-linked stage chronology and conditional
   company/sector effects using the validated target-selection allowlist.

Only explicitly supplied `knownTickers` mappings verify company-to-ticker
identity. A company with a source-backed name but no verified ticker remains
an unverified candidate in `targetSelection.companies`; it must **not** be
promoted into a validated STOCK effect. Independently selected sectors may
still receive qualitative SECTOR impacts when source evidence justifies them.
Neither a TECH nor a SEMICONDUCTORS sector label can impersonate a STOCK.

A deterministic text-claim screen quarantines proposals and narrative fields
that assert observed stock declines, sell-offs or index moves without a
cutoff-valid independent market-outcome verification path. The check is
deliberately conservative, is NOT general semantic understanding, and is
not sufficient by itself to establish truth. Rejected source targets,
market assertions and raw impact proposals remain reviewable. No price feed
is accessed. If evidence is too weak, an explicit abstention is preferable to
invented identity or percentage returns. The compatibility option
`--legacy-single-pass` continues to skip both preliminary calls.

Use a new create-only output filename for each live test. The two revised
publisher pages in the current NVDA exploratory case are still NOT eligible
for strict historical replay or stock-prediction-accuracy evaluation. CI
validates code contracts; only a real local `gemma4:e2b` execution can
demonstrate whether the revised prompts improve reasoning performance.

## Claim Verification for qualitative event impact explanations

When the main event/impact pass produces **no accepted impact hypotheses**
and grounded targets exist, one bounded, independent qualitative recovery
call may run. It returns per-company and per-sector explanatory hypotheses,
never realized prices. Company hypotheses are preserved separately from
verified stock impacts, including when the ticker is unknown.

The recovery response must split its claims into:

1. `sourceClaims`: atomic quotations with exact `quoteId` and a verbatim
   `sourceExcerpt` substring of that original quote, plus explicit
   `REPORTED_FACT`, `ALLEGATION`, `DENIAL` or `OPINION` label.
2. `causalClaim`: a conditional trigger, named economic transmission channel,
   conditional potential effect, and counter-hypothesis.

A deterministic Claim Verification gate checks original source excerpt
membership, quote allowlist, duplicate evidence, obvious allegation-as-fact
wording, referenced company identity, and unconditioned economic outcomes.
Unsupported observed market outcomes cannot be upgraded into explanations.
Rejected company/sector hypotheses remain auditable with machine-readable
claim checks, while independently acceptable hypotheses survive. The original
model proposals and original rejection reasons are not erased.

`companyImpactHypotheses` are **not** verified traded-stock impacts. A
surviving recovered sector impact appears in `impacts` only after both
claim verification and the existing sector/direction/quote validation pass.
`impactClaimChecks` records all recovery verdicts, including rejected ones,
and `impactRecoveryUsed` / `impactRecoveryError` disclose the extra pass.

**Trust boundary:** exact quotation is mechanically verified, but a quoted
excerpt does NOT independently prove that a publisher's story is true or that
a downstream economic conclusion is causally established. All explanations
retain `semanticallyVerified=false`, `causalityProven=false`, market outcome
verification false and `humanReviewRequired=true`. The original two
post-cutoff-revised publisher pages remain exploratory, never a strict
historical accuracy benchmark. No stock prices or later outcomes enter the
model or Claim Verification.

## Universal Claim Verification gate (v12 live finding)

The pinned v12 local Gemma4 run showed a semantic bypass: one syntactically
accepted TECH sector proposal caused the recovery-only verifier to skip
verification altogether (`modelCalls=3`, `impactRecoveryUsed=false`,
`impactClaimChecks=[]`). The existing evidenceQuoteIds proved only that
the quotes existed, not that the proposed mechanism was supported by them.

In normal `--run` mode, all initially accepted impacts now require a
separate, bounded atomic-claim explanation pass, even when the first impact
pass produced one or more apparently valid records. Claims which fail the
source-excerpt and conditional-mechanism checks are excluded from final
`impacts` with `NEWS_TEST_CLAIM_VERIFICATION_MISSING` and remain visible
in `initialAcceptedImpacts` and `rejectedImpacts`. Independently
substantiated sector explanations replace the original unverified sector
proposal; explicit verified issuer mappings can preserve stock identities
only when that issuer receives an accepted atomic company explanation.

`verifyAllImpacts=true` and `impactClaimChecks` expose the verification
path; `impactRecoveryUsed` now indicates the additional pass for **both**
initially accepted and initially rejected impacts. Existing programmatic
legacy behavior is available only when verification is deliberately disabled;
the CLI `--legacy-single-pass` option retains its historical compatibility
behavior. No new market prices or post-cutoff results are read.

Mechanical source-excerpt membership is still NOT semantic entailment,
publisher truth, or demonstrated financial causation. Accepted explanations
remain uncalibrated, human-review-required scenarios. The post-cutoff revised
NVDA documents remain exploratory rather than eligible historical replay.

## v13 acceptance: price-contaminated exact quotes and issuer-only sector claims

The real exploratory v13 Gemma4 report contained two apparently accepted
sector hypotheses and one Nvidia company hypothesis. Crucially, source quote
`a2:q7` explicitly reported a 3.1% Nvidia stock drop and S&P 500 pressure,
and the literal quote matcher had marked it `SOURCE_QUOTE_MATCHED_CONDITIONAL`
without recognizing the **price observation**. This is not an eligible
as-of event-impact input simply because the scraped article contains it.

All model passes now receive **in-memory redacted market-outcome sentences**
and an allowlist of quote anchors with observed price/index movements
quarantined. The original private source files and article hashes are never
modified; original quote IDs remain stable (blocked IDs are absent, never
renumbered). The model cannot cite a blocked quote as as-of evidence.
`quarantinedMarketOutcomeQuotes` reports article ID, quote ID, and reason.
`eligibleSourceQuoteCount` discloses available evidence. If even one
source article has no eligible quotes, the analysis fails closed before
any generation. The atomic claim verifier independently rejects such
verbatim snippets even when provided directly. This is a conservative
price-claim guard rather than proof that other source text existed at
the historical cutoff.

A company-specific story is not automatically broad industry evidence.
A sector recovery claim citing only issuer-specific quotations is
retained in `sectorImpactHypotheses` with
`sectorEvidenceScope=ISSUER_ONLY_NO_SECTOR_WIDE_SOURCE`, never counted as
a validated sector `impact`. Explicit cross-firm/industry context in
the cited source excerpts is required before proposing an industry-wide
conditional scenario as an eligible sector impact. This lexical cue is
not a semantic entailment judgment; even eligible conditional scenarios
retain `semanticallyVerified=false`, `causalityProven=false`, and
`humanReviewRequired=true`. Quarantined proposals remain accessible in
`rejectedCompanyImpactHypotheses` for review.

**The 2026-11-08 revised-source NVDA case is exploratory only.**
`sourceQuotesExistAtAsOfVerified=false` makes the missing historical
snapshot proof explicit. No future price data or realization labels may
enter scoring, calibration, promotion, or training from this case.

## v14: layout-safe evidence matching and unsupported sector expansion

The real pinned Gemma4 v14 report correctly quarantined the post-cutoff
price observation `a2:q7`; there were no accepted final stock/sector impacts.
Its source-anchored Nvidia company hypothesis survived as explicitly
`semanticallyVerified=false`, `causalityProven=false`. One semiconductor
recovery was rejected as `EXCERPT_NOT_IN_SOURCE` because a publisher
inserted invisible formatting characters in the archived input body
(e.g. before `October` and `had`), but the model reproduced the same
readable words without those layout marks.

Atomic quote matching now accepts **only** exact substrings or matching after
removal of a narrowly enumerated set of invisible layout characters
(U+200B, U+200C, U+200D, U+2060, U+FEFF) and whitespace collapse. It
never case-folds, translates, changes punctuation, changes numbers,
reorders tokens, or fills gaps. Every accepted match records its method
(`EXACT_SUBSTRING` or `LAYOUT_NORMALIZED_SUBSTRING`) and retains
`originalSourceQuote` verbatim for audit. The source-quote price
quarantine still runs against the original quote; a modified or
price-bearing quote cannot enter the factual-evidence set.

Sector scope now requires both an explicit industry/multi-firm source
context and an exposure/cost/demand/regulatory implication **in the same
quoted passage**; merely mentioning an industry, a supplier, or Nvidia's
executive issue cannot establish industry-wide effects. Plausible but
unsupported sector propositions remain in `sectorImpactHypotheses`
and the rejected-claim audit, not in validated `impacts`.

These checks reduce both format-induced false negatives and unsupported
sector extrapolation; neither establishes publisher truth, semantic
entailment, real-market causality, or historical as-of availability
for post-cutoff-revised sources.

## Multi-event atomic reasoning study

A separate command `historical-reasoning-study` implements a complete research
analysis of multiple distinct source cases, one bounded model call per case.
It does not modify the legacy `historical-impact-test` output contract.

1. **Atomic Claim Decomposition**: Model outputs separate, provenance-bound
   FACT, ALLEGATION, DENIAL, OPINION and FORECAST excerpts. Multiple claims
   from the same quote are valid when their excerpts match the source.
   Unsupported or allegation-as-fact proposals are quarantined.
2. **Event Reconstruction**: The model groups atomic claim IDs into
   event nodes and describes chronological follow-up, correction, and
   reactions. The validator checks claim references, distinct membership,
   cross-event links and chronological ordering; relationships are never
   considered proven causal history.
3. **Economic Mechanism Reasoning**: Each source-named company or
   evidence-scoped sector hypothesis cites accepted claim IDs, describes
   a conditional trigger, named economic channel, conditional consequence
   and counter-hypothesis. No output is treated as confirmed price impact.
4. **Cross-Event Comparison**: Compares claim counts, eligible hypotheses,
   and quarantined proposals across at least two distinct case IDs. This
   is structural coverage only, not proof of generalization or predictive
   historical accuracy.

Command (provide two different immutable cases):

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh historical-reasoning-study \
  --input "$DATA/first-case-original.json" \
  --input "$DATA/second-distinct-case-original.json" \
  --model gemma4:e2b \
  --output "$DATA/multi-event-reasoning-v1.json"
```

Create-only output safety, source quote integrity, price-result redaction
and source hash provenance are retained. Offline tests cover three
**synthetic** incident domains (manufacturing, energy and transport), plus
unknown atomic claims and ungrounded sector extrapolation. Synthetic
fixtures are not genuine past episodes. Real Cross-Event Validation needs
independently timestamped *actual historical* publisher snapshots from
different events and separate assessments of outcomes; this module
correctly emits `generalizationVerified=false` and
`outcomeAccuracyVerified=false`. It neither trades, trains, nor
auto-promotes any hypothesis.

## Three real official issuer events: first substantive cross-event research

The existing `research/independent-historical-news-2023.json` is an
already-curated and previously verified official-primary issuer cohort:
Microsoft FY23 Q4 (2023-07-25), Alphabet Q2 2023 (2023-07-25,
SEC Exhibit 99.1), and AMD Q2 2023 (2023-08-01). The three official
source links and 9 distinct business facts are reused rather than
inventing a new dataset, scraping a market-return summary, or repeating
the NVDA example. Use the separate `historical-official-research`
path for these **three actual historical announcements**.

This mode does not pretend that the stored curator fact statements
are verbatim portions of independently archived publisher HTML; each
fact interpretation records `CURATED_PARAPHRASE_NOT_VERBATIM`,
the official `sourceId`, the exact curated original statement, and
`historicalAsOfVerified=false`. It neither fabricates two articles
out of one official release nor claims intraday timing verified.

After validating the existing official cohort, Gemma4 receives each
release's three curated facts, source ID and date in isolation (without
market prices), performs typed fact vs. forward-guidance assessment,
relates facts WITHIN that disclosure, and proposes conditional issuer
business-mechanism explanations with a competing counter-hypothesis
and specifically named missing evidence. Deterministic gates quarantine
invented IDs, misclassified future guidance, unconditioned conclusions
and stated historical market movements. The response keeps accepted
and rejected content separately.

The cohort comparison contrasts the *types* of economic transmission
channels across Microsoft / Alphabet / AMD, source-disjoint case IDs,
coverage and rejected claims. The comparison does **not** prove
predictive accuracy, generalization or causal market effects; it retains
`generalizationVerified=false` and `pointInTimeReplayVerified=false`.
The known official announcement date alone is insufficient to reconstruct
the exact time-of-day historical information set.

Prepare with no Ollama calls:

```bash
bash scripts/news-learning-review.sh historical-official-research \
  --cohort research/independent-historical-news-2023.json --preview
```

Run on the Mac with installed Gemma4 using a unique create-only output
path (3 model calls, one for each real disclosure):

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh historical-official-research \
  --cohort research/independent-historical-news-2023.json \
  --model gemma4:e2b --run \
  --output "$DATA/official-three-case-reasoning-v1.json"
```

The outcome is an auditable **real-case source-fact research** artifact.
It is NOT a verified `historical-reasoning-study` original-news replay
and it should not replace future acquisition of timestamped historical
publisher snapshots. Synthetic manufacturing, energy and transport
cases remain regression tests only. Official sources and original cohort
are read-only; results are created in a new private file, never auto-train,
auto-promote, request price data or execute orders.

## Real three-case semantic-quality audit after the first Gemma4 run

A live Microsoft/Alphabet/AMD research run classified **all five**
conditional business mechanisms as `REGULATION`, even though the cited
facts covered Microsoft and Google Cloud growth, Alphabet Cloud
profitability, AMD operating loss and MI300 product guidance; no cited
official source fact established regulatory measures. Its schema gate
reported zero rejected mechanisms. Alphabet proposals also introduced
unsupported quantitative thresholds (10%, 15% and USD 400m/350m), and
one Microsoft fact interpretation called the cloud business
`high-margin` although the quoted curated fact did not support this
assertion. This is a structural pass but an economic reasoning failure.

The research pipeline now keeps the *original model-authored channel
unchanged* and quarantines any `REGULATION` transmission lacking
both primary-source regulation evidence and actual regulation-related
mechanism language. Numeric thresholds absent from cited source facts
are isolated as `UNSOURCED_NUMERIC_THRESHOLD` rather than silently
promoted into evidence. Unsupported superlatives such as
`high-margin` on factual interpretations receive human-review flags
without rewriting historical source statements. These are conservative
lexical audit checks, **not** proof of semantic entailment; accepted
hypotheses remain human-review required. Python never relabels
`REGULATION` as `REVENUE` or another category by guessing.

To audit the previously generated **real Gemma4 v1** result without
rerunning the model, use the new create-only audit mode:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh historical-official-research \
  --cohort research/independent-historical-news-2023.json \
  --audit-existing "$DATA/official-three-case-reasoning-v1.json" \
  --output "$DATA/official-three-case-reasoning-audit-v2.json"
```

The cohort SHA, distinct case IDs, official source IDs and original
curated fact statements are rechecked before any reinterpretation.
The prior v1 JSON is **never modified**. The new audit artifact
contains original rejected proposals and independent quality gate
reasons, with `auditModelCalls=0`; its cross-event comparison excludes
unsupported channels. These new quality checks also run on all future
`--run` outputs. Neither the v1 report nor this audit establishes
historical PIT validity, source semantic truth or market causation.

## Real Gemma4 v3: preserve scenario meaning, not only individual IF tokens

The actual local three-case v3 run generated nine factual interpretations
and **four research mechanisms**, rejected solely for conditional syntax:
two nominal antecedents (Microsoft Azure adoption, Alphabet Cloud growth)
and two declarative *counter-hypotheses* (AMD's semiconductor cycle or
MI300 bottlenecks). All four selected plausible business channels
(`REVENUE` or `SUPPLY`), in contrast to v1's five incorrect
`REGULATION` selections. This demonstrates improvement in topic/channel
selection, **not** proof of economic truth.

Old code rejected if the *individual* trigger/counter field omitted a
token such as `if`, even when the effect itself said `If ... could`
or the alternative was stored in the explicitly hypothetical
`counterHypothesis` slot. The new scenario reviewer checks the
complete trigger/effect/alternative structure. It labels nominal triggers
as `IMPLICIT_TRIGGER_CONDITION_SCOPED_BY_EFFECT` **only** when the
effect independently contains an antecedent and uncertainty; it preserves
declarative counter-scenarios as
`DECLARATIVE_COUNTER_AS_UNVERIFIED_ALTERNATIVE`. Both stay unverified
model assumptions requiring human review. Neither changes the original
proposed text or treats the counter-scenario as a historical result.

Hard rejections remain for unconditional outcomes, observed share-price
results, missing source IDs, unsupported numerics and ungrounded regulation.
The quality gate still runs independently after scenario-shape review.

Crucially, an already saved v3 output can be re-audited without rerunning
Gemma4, producing a new create-only artifact:

```bash
DATA="$HOME/Library/Application Support/TrisMarketEventIntelligence"
bash scripts/news-learning-review.sh historical-official-research \
  --cohort research/independent-historical-news-2023.json \
  --audit-existing "$DATA/official-three-case-reasoning-v3.json" \
  --output "$DATA/official-three-case-reasoning-v3-scenario-audit-v4.json"
```

Previously rejected scenarios are *not silently reclassified* as validated
economics. Accepted source-linked, non-hallucinated former proposals appear
in a separate `recoveredResearchCandidates` collection with original
rejection reasons, immutable source text, scenario warnings and
`semanticAccuracyVerified=false`/`causalityProven=false`.
The original `rejectedEconomicMechanisms` log remains intact. The
cross-case report counts these research candidates separately from accepted
mechanisms; it does not claim historical or causal generalization.
