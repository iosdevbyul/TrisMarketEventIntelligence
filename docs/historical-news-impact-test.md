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
