"""Render a self-contained, offline human NEWS review page from a pinned packet.

The private HTML is locally generated with no third-party assets, services,
or predictions. Exported edits must pass the existing Python review bridge.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

from tris_market_event_intelligence.news_importance_review import _cohort, _read
from tris_market_event_intelligence.news_review_sample import prepare_news_review_sample
from tris_market_event_intelligence.news_review_sample_bridge import EDITABLE

MAX_HTML_BYTES = 1_200_000
POLICY = ("default-src 'none'; script-src 'unsafe-inline'; "
          "style-src 'unsafe-inline'; connect-src 'none'; "
          "img-src 'none'; font-src 'none'; object-src 'none'; "
          "base-uri 'none'; form-action 'none'")

STYLE = """
:root{font:16px -apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif;
color:#172333;background:#f1f4f9}
*{box-sizing:border-box}body{margin:0 auto;max-width:1050px;padding:28px 20px 65px}
header{border-bottom:1px solid #b6c2d3;padding-bottom:18px;margin-bottom:18px}
h1{font-size:1.65rem;margin:0 0 8px}h2{font-size:1.17rem;margin:10px 0}
p{line-height:1.55} .muted{color:#526277;font-size:.92rem}
.panel{background:white;border:1px solid #cbd5e1;border-radius:12px;
padding:22px;margin:14px 0}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
button,.btn{font:inherit;background:#1d4c87;color:white;border:0;
border-radius:7px;padding:10px 15px;cursor:pointer}
button.secondary{background:#e6ecf5;color:#1d3451}
button:focus-visible,input:focus-visible,select:focus-visible,
textarea:focus-visible,a:focus-visible{outline:3px solid #2a74c9}
button:disabled{opacity:.45;cursor:not-allowed}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:15px}
label{display:block;font-weight:600;margin:10px 0 5px}
input,select,textarea{font:inherit;width:100%;padding:10px;border:1px solid #9caec0;
border-radius:7px;background:white;color:#172333}
textarea{min-height:130px;resize:vertical}
pre{white-space:pre-wrap;overflow-wrap:anywhere}
#headline{overflow-wrap:anywhere;font-size:1.12rem}
a{color:#1254a0;overflow-wrap:anywhere}
#message{font-weight:600;white-space:pre-wrap} .warning{color:#9c321e}
.kbd{font-variant-numeric:tabular-nums}
footer{color:#526277;font-size:.85rem;margin-top:20px}
@media(prefers-color-scheme:dark){
:root{background:#111827;color:#e2e8f0}
.panel{background:#1f2937;border-color:#485568}
.muted,footer{color:#adb9c7}
input,select,textarea{background:#1a2433;color:#fff;border-color:#62718a}
button.secondary{background:#39465d;color:#fff}
a{color:#92beff}
}
"""

JS = r"""
'use strict';
const pinned = JSON.parse(document.getElementById('packet-data').textContent);
const originalRows = structuredClone(pinned.items);
const rows = structuredClone(pinned.items);
const humanKeys = ['importance','potentialMagnitude','sourceAlignment',
  'sameStoryConfirmedByHuman','reviewer','reviewedAt','reason'];
const choice = {
 importance:['','IMPORTANT','UNIMPORTANT','UNCERTAIN'],
 potentialMagnitude:['','LOW','MODERATE','HIGH','CRITICAL','UNASSESSABLE'],
 sourceAlignment:['','SUPPORTED','CONTRADICTED','UNRELATED','INSUFFICIENT_CONTEXT'],
 sameStoryConfirmedByHuman:['','true','false']
};
let position = 0;
const el = id => document.getElementById(id);
const status = (message, error=false) => {
 el('message').textContent = message;
 el('message').className = error ? 'warning' : 'muted';
};
function displayChoice(value) {
 return value === null ? '' : String(value);
}
function assignChoice(field, value) {
 return field === 'sameStoryConfirmedByHuman'
   ? (value === '' ? null : value === 'true')
   : (value || null);
}
function fillSelect(id, options) {
 const control = el(id);
 for (const option of options) {
  const node = document.createElement('option');
  node.value = option;
  node.textContent = option || '미평가 / 보류';
  control.appendChild(node);
 }
}
for (const [id, opts] of Object.entries(choice)) fillSelect(id, opts);
const reviewerKey = 'reviewer';
const fields = ['importance','sourceAlignment','potentialMagnitude',
 'sameStoryConfirmedByHuman','reviewer','reason'];
function isReviewed(row) {
 return row.importance !== null || row.potentialMagnitude !== null
  || row.sameStoryConfirmedByHuman !== null;
}
function render() {
 const row = rows[position];
 el('counter').textContent = (position+1) + ' / ' + rows.length;
 el('completed').textContent = rows.filter(isReviewed).length + '건에 판단 입력';
 el('headline').textContent = row.title;
 el('summary').textContent = row.summary || '요약 없음 — 원문 확인 권장';
 el('source').textContent = row.source + ' / 최초 관측 ' + row.firstObservedAt;
 el('link').href = row.sourceUrl;
 el('link').textContent = '원문 열기 (' + row.sourceUrl + ')';
 el('unit').textContent = '반복 보도 후보 ID: ' +
   row.candidateUnitArticleIds.join(', ') +
   ' — 동일 사건임이 확인된 것은 아닙니다';
 for (const field of fields) {
  el(field).value = displayChoice(row[field]);
 }
 el('reason').value = row.reason || '';
 el('previous').disabled = position === 0;
 el('next').disabled = position === rows.length-1;
 status('입력한 판단은 이 페이지 메모리에만 있습니다. JSON 내보내기로 직접 저장하세요.');
}
function validateRow(row) {
 const hasImportance = row.importance !== null || row.sourceAlignment !== null;
 const hasMagnitude = row.potentialMagnitude !== null;
 const hasStory = row.sameStoryConfirmedByHuman !== null;
 if (hasImportance && (!row.importance || !row.sourceAlignment))
  throw Error('중요도와 출처 정합성은 함께 입력해야 합니다');
 const hasAny = hasImportance || hasMagnitude || hasStory;
 if (!hasAny) {
  if (row.reason || row.reviewedAt || row.reviewer)
   throw Error('미평가 기사에는 평가자 정보와 근거를 남길 수 없습니다');
  return;
 }
 if (!row.reviewer || row.reviewer.trim().length > 100)
  throw Error('사람 평가자 식별자를 1~100자로 입력하세요');
 if (!row.reason || row.reason.trim().length < 20 || row.reason.trim().length > 2000)
  throw Error('판단 근거를 20~2000자로 직접 작성하세요');
 if (!row.reviewedAt || Date.parse(row.reviewedAt) <
     Date.parse(row.firstObservedAt))
  throw Error('기사 최초 관측 이후의 평가 시각이 필요합니다');
}
function saveCurrent() {
 const row = rows[position];
 for (const field of fields) {
  const value = el(field).value;
  row[field] = field === 'reason' || field === 'reviewer'
    ? (value.trim() || null) : assignChoice(field, value);
 }
 const hasAny = isReviewed(row);
 if (hasAny && !row.reviewedAt) row.reviewedAt = new Date().toISOString();
 if (!hasAny) row.reviewedAt = null;
 validateRow(row);
 return true;
}
function navigate(delta) {
 try { saveCurrent(); position = Math.max(0,Math.min(rows.length-1,position+delta)); render(); }
 catch (error) { status(error.message, true); }
}
el('previous').addEventListener('click', () => navigate(-1));
el('next').addEventListener('click', () => navigate(1));
el('save').addEventListener('click', () => {
 try { saveCurrent(); render(); } catch (error) { status(error.message,true); }
});
el('clear').addEventListener('click', () => {
 if (!confirm('현재 기사의 판단만 지울까요?')) return;
 for (const key of humanKeys) rows[position][key] = null;
 render();
});
el('export').addEventListener('click', () => {
 try {
  saveCurrent();
  for (const row of rows) validateRow(row);
  const output = structuredClone(pinned);
  output.items = rows;
  const blob = new Blob([JSON.stringify(output,null,2)+'\n'],
    {type:'application/json;charset=utf-8'});
  const href = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = href;
  a.download = 'news-review-human-edited.json';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(href), 5000);
  status('JSON 다운로드를 요청했습니다. 실제 저장 여부를 확인하고 Python bridge --audit로 검증하세요.');
 } catch (error) { status(error.message,true); }
});
el('import').addEventListener('change', async e => {
 const file = e.target.files[0];
 if (!file) return;
 try {
  const imported = JSON.parse(await file.text());
  if (!imported || !Array.isArray(imported.items) ||
      imported.items.length !== originalRows.length ||
      JSON.stringify(Object.fromEntries(Object.entries(imported)
        .filter(([k])=>k!=='items'))) !==
      JSON.stringify(Object.fromEntries(Object.entries(pinned)
        .filter(([k])=>k!=='items'))))
   throw Error('다른 코호트 또는 표본 파일은 불러올 수 없습니다');
  for (let i=0; i<rows.length; i++) {
   const actual = imported.items[i], expected = originalRows[i];
   if (!actual || Object.keys(actual).sort().join('|') !==
       Object.keys(expected).sort().join('|'))
    throw Error('표본 필드가 변경되었습니다');
   for (const field of Object.keys(expected)) {
    if (!humanKeys.includes(field) &&
        JSON.stringify(actual[field]) !== JSON.stringify(expected[field]))
     throw Error('원본 기사 정보가 바뀌었습니다');
   }
   validateRow(actual);
  }
  for (let i=0; i<rows.length; i++) rows[i] = imported.items[i];
  position = 0;
  render();
  status('사람 평가 JSON을 불러왔습니다. 새로 내보내기 전까지 로컬 메모리에만 유지됩니다.');
 } catch (error) { status('불러오기 실패: '+error.message,true); }
 e.target.value='';
});
window.addEventListener('beforeunload', e => {
 if (rows.some(isReviewed)) { e.preventDefault(); }
});
render();
"""


def _verify_pinned_sample(cohort_path: Path, sample_path: Path) -> dict:
    raw, _ = _cohort(cohort_path)
    current = json.loads(_read(sample_path))
    if not isinstance(current, dict) or (
        current.get("schemaVersion") != "news_human_review_candidate_sample_v1"
    ) or current.get("cohortSha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError("Sample and immutable source cohort do not match")
    rows = current.get("items")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
        raise ValueError("Complete bounded human candidate sample required")
    original = prepare_news_review_sample(
        cohort_path=cohort_path, max_selected=len(rows),
    )
    if set(current) != set(original) or any(
        current[k] != value for k, value in original.items() if k != "items"
    ):
        raise ValueError("Sample selection or immutable metadata was changed")
    for row, expected in zip(rows, original["items"], strict=True):
        if not isinstance(row, dict) or set(row) != set(expected) or any(
            row[k] != value for k, value in expected.items() if k not in EDITABLE
        ):
            raise ValueError("Sample article provenance changed")
    return current


def render_offline_review(*, cohort_path: Path, sample_path: Path) -> str:
    """Generate offline HTML; browser must explicitly export human edits."""
    packet = _verify_pinned_sample(cohort_path, sample_path)
    payload = json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
    # Prevent user-supplied article titles from ending the JSON <script> tag.
    payload = (payload.replace("&", "\\u0026").replace("<", "\\u003c")
               .replace(">", "\\u003e").replace("\u2028", "\\u2028")
               .replace("\u2029", "\\u2029"))
    page = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{html.escape(POLICY, quote=True)}">
<title>NEWS Human Review · Offline</title><style>{STYLE}</style></head>
<body>
<header><h1>뉴스 중요도 · 잠재적 파급력 검토</h1>
<p class="muted">오프라인 로컬 평가. AI 정답 생성 없음. 원문 링크는 클릭할 때만 외부 사이트로 이동합니다.</p>
<div class="controls"><strong id="counter"></strong>
<span id="completed" class="kbd"></span>
<label for="import">이전 평가 JSON 불러오기</label>
<input id="import" type="file" accept=".json,application/json" style="max-width:250px"></div>
</header>
<section class="panel"><h2 id="headline"></h2><p id="summary"></p>
<p class="muted" id="source"></p>
<p><a id="link" href="#" rel="noopener noreferrer" target="_blank"></a></p>
<p class="muted" id="unit"></p></section>
<section class="panel">
<h2>사람의 직접 평가</h2>
<div class="grid"><div><label for="importance">뉴스 중요도</label><select id="importance"></select></div>
<div><label for="sourceAlignment">기사 출처·내용 정합성</label>
<select id="sourceAlignment"></select></div>
<div><label for="potentialMagnitude">기업·조직의 잠재적 파급력</label>
<select id="potentialMagnitude"></select></div>
<div><label for="sameStoryConfirmedByHuman">보도 후보가 동일 사건인가?</label>
<select id="sameStoryConfirmedByHuman"></select></div></div>
<label for="reviewer">평가자 식별자 (검토한 사람)</label>
<input id="reviewer" maxlength="100" autocomplete="off" placeholder="예: reviewer-01">
<label for="reason">기사 내용에 근거한 판단 이유 (20~2000자)</label>
<textarea id="reason" maxlength="2000"
placeholder="중요도 또는 잠재적 조직 영향에 대한 구체적 근거를 직접 작성하세요."></textarea>
</section>
<div class="controls">
<button id="previous" class="secondary">이전</button>
<button id="save" class="secondary">현재 판단 확인</button>
<button id="next" class="secondary">다음</button>
<button id="clear" class="secondary">현재 판단 지우기</button>
<button id="export">검토 JSON 내보내기</button></div>
<p id="message" role="status"></p>
<footer>판단은 브라우저 메모리에만 있습니다. 자동 저장되지 않습니다.
JSON 내보내기 후 기존 review-bridge --audit에서 검증하세요.
사람 판단은 외부 검증 전까지 독립적인 사실 정답으로 간주하지 않습니다.</footer>
<script id="packet-data" type="application/json">{payload}</script>
<script>{JS}</script>
</body></html>"""
    if len(page.encode()) > MAX_HTML_BYTES:
        raise ValueError("Offline review page exceeds private size bound")
    return page
