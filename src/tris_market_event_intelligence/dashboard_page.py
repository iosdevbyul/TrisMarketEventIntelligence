"""Self-contained, local-only monitoring interface."""
PAGE = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TrisMarket · News Intelligence</title>
<style>
:root{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0b101a;color:#e8edf5;font-size:14px}
*{box-sizing:border-box}body{margin:0}button,input,select{font:inherit}button{cursor:pointer}
.app{display:grid;grid-template-columns:236px minmax(0,1fr);min-height:100vh}
aside{border-right:1px solid #202b3b;background:#0d1421;padding:27px 18px;display:flex;flex-direction:column;gap:28px}
.brand{display:flex;align-items:center;gap:11px;font-weight:750;font-size:17px;letter-spacing:-.5px;padding:0 8px}
.mark{background:#d7ef7d;color:#1a2917;width:34px;height:34px;border-radius:9px;display:grid;place-items:center;font-weight:900}
.menu-label{font-size:10px;color:#79869b;text-transform:uppercase;letter-spacing:1.6px;padding:0 11px;margin-bottom:12px}
.nav{display:flex;align-items:center;gap:10px;border-radius:9px;padding:12px;color:#8d9aae}
.nav.active{color:#e6f4be;background:#1f2b26;border:1px solid #3b4f36}
.side-note{margin-top:auto;border-top:1px solid #283244;padding:18px 9px;font-size:12px;line-height:1.7;color:#8795a8}
main{padding:30px clamp(22px,3.5vw,58px) 70px;max-width:1700px;width:100%;margin:auto}
.top{display:flex;justify-content:space-between;align-items:flex-start;gap:18px;flex-wrap:wrap;margin-bottom:31px}
.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:2px;color:#d4e98d;font-weight:700}
h1{font-size:30px;font-weight:740;letter-spacing:-1.2px;margin:9px 0}h2{font-size:16px;letter-spacing:-.3px;margin:0}
.sub{color:#8d9bb0;font-size:13px;margin:0;line-height:1.65}
.tools{display:flex;gap:12px;align-items:center} .pill{border:1px solid #304056;border-radius:100px;padding:9px 14px;color:#abb8c8;font-size:12px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#d9ab60;margin-right:7px}.dot.ok{background:#bce578}.dot.bad{background:#ff857e}
.primary{border:1px solid #adce72;color:#142014;background:#d7ef7d;padding:11px 16px;border-radius:9px;font-weight:700}
.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin-bottom:28px}
.metric{border:1px solid #253246;background:#121c2b;border-radius:13px;padding:20px 22px;min-height:129px}
.metric-label{font-size:12px;color:#9babbb;margin-bottom:17px}.metric-num{font-size:30px;font-weight:740;letter-spacing:-1px}.metric-foot{font-size:11px;margin-top:8px;color:#748398}
.dashboard{display:grid;grid-template-columns:minmax(0,1fr) 306px;gap:22px;align-items:start}
.panel{background:#121b29;border:1px solid #253246;border-radius:13px;overflow:hidden}
.panel-head{display:flex;justify-content:space-between;align-items:center;gap:14px;padding:22px 23px;border-bottom:1px solid #263244;flex-wrap:wrap}
.filters{display:flex;gap:9px;flex-wrap:wrap;align-items:center}
.search,select{background:#0e1725;border:1px solid #324057;border-radius:8px;color:#d4deeb;padding:9px 11px;min-height:38px}
.search{width:220px}select{max-width:160px}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;text-align:left}th{font-size:10px;text-transform:uppercase;letter-spacing:1.1px;color:#8391a5;font-weight:650;background:#101926;padding:13px 22px;white-space:nowrap}
td{padding:17px 22px;border-top:1px solid #222d3e;vertical-align:middle;font-size:12px;color:#aebaca}
tr.data-row{cursor:pointer}tr.data-row:hover{background:#1a2738}tr.data-row.selected{background:#202e3d}
.article-title{color:#e5ebf4;font-size:13px;font-weight:650;max-width:380px;line-height:1.5;margin-bottom:6px}
.meta{font-size:11px;color:#7d8ca0}.tag{display:inline-flex;border:1px solid #36475b;border-radius:5px;padding:5px 7px;color:#b7c4d4;font-size:11px;white-space:nowrap}
.tag.positive{color:#bde69a;border-color:#456345;background:#223628}.tag.negative{color:#ff9d9f;border-color:#6c4146;background:#37242b}
.tag.pending{color:#e6c486;border-color:#655336;background:#30291f}.tag.muted{color:#96a4b9}
.right{display:flex;flex-direction:column;gap:19px}.right .panel{padding:20px}
.section-title{font-size:11px;letter-spacing:1.2px;text-transform:uppercase;color:#91a0b3;font-weight:700;margin-bottom:17px}
.source{display:flex;justify-content:space-between;align-items:center;padding:13px 0;border-top:1px solid #273246;gap:14px}
.source-name{font-size:12px;font-weight:650;overflow-wrap:anywhere}.source-meta{font-size:11px;color:#8c9aae;margin-top:5px}
.status{font-size:11px;color:#bade88;white-space:nowrap}.status.bad{color:#ff9992}.status.neutral{color:#9aa7b8}
.detail{line-height:1.65}.detail h3{font-size:16px;margin:4px 0 14px;line-height:1.5}
.detail p{color:#b0bccd;font-size:12px}.detail dl{display:grid;grid-template-columns:105px 1fr;gap:11px;font-size:12px}
.detail dt{color:#8495ac}.detail dd{margin:0;overflow-wrap:anywhere}.detail a{color:#d7ef7d}
.empty{padding:40px 24px;color:#97a6ba;text-align:center;line-height:1.7}
.note{padding:17px 22px;border-top:1px solid #293449;color:#8191a6;font-size:11px;line-height:1.65}
.err{margin:14px 0;padding:13px;background:#38242a;border:1px solid #7b454d;border-radius:8px;color:#ffb2b2}
@media(max-width:1190px){.dashboard{grid-template-columns:1fr}.right{display:grid;grid-template-columns:1fr 1fr}}
@media(max-width:800px){.app{grid-template-columns:1fr}aside{display:none}main{padding:25px 15px}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.right{grid-template-columns:1fr}.search{width:155px}th,td{padding:14px}}
</style></head><body>
<div class="app">
<aside><div class="brand"><div class="mark">T</div> TrisMarket</div>
<div><div class="menu-label">Intelligence</div><div class="nav active">◫ &nbsp; News Monitor</div>
<div class="nav">◇ &nbsp; Event Analysis</div><div class="nav">▤ &nbsp; Source Health</div></div>
<div class="side-note">LOCAL RESEARCH WORKSPACE<br>Read-only · localhost<br><br>뉴스를 수집하거나 AI 분석을 실행하지 않습니다. 데이터베이스에 기록된 결과만 표시합니다.</div>
</aside>
<main>
<header class="top"><div><div class="eyebrow">Market Event Intelligence / Observability</div>
<h1>News Intelligence Monitor</h1><p class="sub">뉴스 수집부터 이벤트 추출, 시장 영향 판단까지 실시간으로 관찰합니다.</p></div>
<div class="tools"><span class="pill" id="connection"><span class="dot"></span>연결 확인 중</span><button class="primary" id="refresh">↻ &nbsp; 새로고침</button></div></header>
<div class="cards">
<div class="metric"><div class="metric-label">수집된 전체 기사</div><div class="metric-num" id="articlesCount">—</div><div class="metric-foot">SQLite 원본 기사</div></div>
<div class="metric"><div class="metric-label">추출된 이벤트</div><div class="metric-num" id="eventsCount">—</div><div class="metric-foot">경제·시장 이벤트 레코드</div></div>
<div class="metric"><div class="metric-label">시장 영향 분석</div><div class="metric-num" id="impactsCount">—</div><div class="metric-foot">방향·섹터 판단 기록</div></div>
<div class="metric"><div class="metric-label">자동 Gemma 분석 완료 / 실패</div><div class="metric-num" id="runsCount">—</div><div class="metric-foot" id="runsFoot">분석 처리 현황 불러오는 중</div></div>
</div>
<div id="error" hidden class="err"></div>
<div class="dashboard">
<section class="panel"><div class="panel-head"><div><h2>기사 처리 흐름</h2><p class="sub" id="lastObserved">최근 수집 시각 확인 중</p></div>
<div class="filters"><input class="search" id="search" placeholder="기사 제목 검색" aria-label="기사 제목 검색">
<select id="sourceFilter" aria-label="출처 필터"><option value="">모든 출처</option></select>
<select id="stageFilter" aria-label="분석 상태 필터"><option value="">전체 상태</option><option value="pending">미추출</option><option value="extracted">이벤트 추출</option><option value="analyzed">AI 분석 완료</option><option value="review">AI 판단 불일치·검토 필요</option><option value="failed">AI 분석 실패</option><option value="noevent">이벤트 없음</option></select></div></div>
<div class="table-wrap"><table><thead><tr><th>원본 뉴스 / 출처</th><th>수집 시각</th><th>이벤트 처리</th><th>AI / 시장 영향</th></tr></thead><tbody id="articleRows"><tr><td colspan="4" class="empty">불러오는 중...</td></tr></tbody></table></div>
<div class="note" id="tableNote">최근 기사 최대 300건을 보여줍니다. 수집 시각은 기사 발행 시각과 다를 수 있습니다.</div>
</section>
<div class="right">
<section class="panel"><div class="section-title">Source Health · 출처별 수집</div><div id="sourceRows" class="sub">불러오는 중...</div></section>
<section class="panel"><div class="section-title">Article Inspector · 기사 상세</div><div class="detail" id="detail"><p>왼쪽 기사 목록에서 한 건을 선택하면 수집·이벤트 추출·시장 영향 분석 내용을 확인할 수 있습니다.</p></div></section>
<section class="panel"><div class="section-title">데이터 해석 주의사항</div><p class="sub">시장 영향 분석 결과는 예측 기록이며, 실제 주가 검증이나 적중률을 의미하지 않습니다.</p><p class="sub" style="margin-top:12px">자동 Gemma 분석은 SQLite에 기록된 결과만 표시합니다. 기존 독립 배치 JSON은 포함되지 않습니다.</p></section>
</div></div>
</main></div>
<script>
'use strict';
const $=id=>document.getElementById(id);
let current=null,selected=null;
const n=v=>v==null?'—':Number(v).toLocaleString('ko-KR');
const date=v=>{if(!v)return '기록 없음';const d=new Date(v);return isNaN(d)?String(v):d.toLocaleString('ko-KR',{dateStyle:'short',timeStyle:'short'});};
function elt(tag,text,cls){const e=document.createElement(tag);if(text!=null)e.textContent=String(text);if(cls)e.className=cls;return e;}
function state(a){if(a.automaticStatus==='COMPLETED'&&a.automaticVerificationStatus==='REVIEW_REQUIRED')return 'review';if(a.automaticStatus==='COMPLETED')return 'analyzed';if(a.automaticStatus==='FAILED')return 'failed';if(a.impactCount>0)return 'analyzed';if(a.eventCount>0)return 'extracted';if(a.extractionCount>0)return 'noevent';return 'pending';}
function stageLabel(a){const s=state(a);return s==='review'?'AI 판단 불일치 · 검토 필요':s==='analyzed'?(a.automaticVerificationStatus==='REPEATABLE_UNREVIEWED'?'Gemma 반복 일치 · 사실 미검증':'Gemma 분석 완료 · 사실 미검증'):s==='failed'?'AI 분석 실패':s==='extracted'?'이벤트 추출':s==='noevent'?'추출 완료 · 이벤트 없음':'미추출 / 확인 필요';}
function tag(text,cls){return elt('span',text,'tag '+(cls||''));}
function renderArticles(){
const term=$('search').value.toLocaleLowerCase();const stage=$('stageFilter').value;
const rows=current.articles.filter(a=>(!term||(a.title||'').toLocaleLowerCase().includes(term)||(a.source||'').toLocaleLowerCase().includes(term))&&(!stage||state(a)===stage));
const body=$('articleRows');body.replaceChildren();
if(!rows.length){const tr=elt('tr');const td=elt('td','조건에 맞는 기사가 없습니다.','empty');td.colSpan=4;tr.append(td);body.append(tr);return;}
for(const a of rows){
const tr=elt('tr',null,'data-row'+(selected===a.id?' selected':''));
const td1=elt('td');td1.append(elt('div',a.title,'article-title'),elt('div',a.source+' · #'+a.id,'meta'));
const td2=elt('td',date(a.detectedAt));
const td3=elt('td');td3.append(tag(stageLabel(a),state(a)==='analyzed'?'positive':state(a)==='pending'?'pending':'muted'));
const td4=elt('td');
if(a.automaticStatus==='COMPLETED'){let data=null;try{data=JSON.parse(a.automaticAnalysis||'null');}catch(_){};td4.append(tag('Gemma · '+(data?.impacts?.length??0)+'개 영향',a.automaticVerificationStatus==='REVIEW_REQUIRED'?'negative':'muted'));}
else if(a.automaticStatus==='FAILED')td4.append(tag('분석 실패: '+(a.automaticError||'UNKNOWN'),'negative'));
else if(a.impactCount>0)td4.append(tag(a.latestDirection||'방향 미지정',/UP|POSITIVE|BULL|RISE/i.test(a.latestDirection||'')?'positive':/DOWN|NEGATIVE|BEAR|FALL/i.test(a.latestDirection||'')?'negative':'muted'));
else td4.append(tag(a.latestRunOutcome||'기록 없음','muted'));
tr.append(td1,td2,td3,td4);tr.onclick=()=>{selected=a.id;renderArticles();renderDetail(a);};body.append(tr);
}}
function addPair(dl,k,v){dl.append(elt('dt',k),elt('dd',v==null||v===''?'기록 없음':v));}
function renderDetail(a){
const box=$('detail');box.replaceChildren();box.append(elt('h3',a.title),tag(stageLabel(a),state(a)==='analyzed'?'positive':'muted'));
const dl=elt('dl');addPair(dl,'출처',a.source);addPair(dl,'기사 ID',a.id);addPair(dl,'발행',date(a.publishedAt));addPair(dl,'최초 수집',date(a.detectedAt));addPair(dl,'원문 상태',a.fullTextStatus);addPair(dl,'추출 횟수',a.extractionCount);addPair(dl,'이벤트 수',a.eventCount);addPair(dl,'이벤트 분류',a.eventClassification);addPair(dl,'이벤트 상태',a.eventStatus);addPair(dl,'클레임 수',a.claimCount);addPair(dl,'영향 분석 수',a.impactCount);addPair(dl,'최신 모델',a.latestModel);addPair(dl,'영향 섹터',a.latestSector);addPair(dl,'방향 예측',a.latestDirection);addPair(dl,'분석 실행 결과',a.latestRunOutcome);addPair(dl,'자동 AI 분석 상태',a.automaticStatus);addPair(dl,'AI 반복 검토 상태',a.automaticVerificationStatus);addPair(dl,'자동 AI 모델',a.automaticModel);addPair(dl,'AI 실패 코드',a.automaticError);box.append(dl);
if(a.automaticAnalysis){try{
const analysis=JSON.parse(a.automaticAnalysis);box.append(elt('div','Gemma 원문 분석','section-title'));
box.append(elt('p',analysis.summary||'요약 없음'));
if(Array.isArray(analysis.impacts)){for(const impact of analysis.impacts){
const p=elt('p');p.append(tag(impact.direction||'UNCERTAIN',impact.direction==='UP'?'positive':impact.direction==='DOWN'?'negative':'muted'));
p.append(document.createTextNode(' '+(impact.target||'대상 미지정')+' — '+(impact.reason||'')));
box.append(p);}}
if(analysis.limitations)box.append(elt('p','한계: '+analysis.limitations));
}catch(_){box.append(elt('p','저장된 AI 결과를 읽을 수 없습니다.'));}}
if(a.summary){box.append(elt('div','기사 요약','section-title'),elt('p',a.summary));}
if(a.latestRationale){box.append(elt('div','시장 영향 분석 근거','section-title'),elt('p',a.latestRationale));}
if(/^https?:\/\//i.test(a.url||'')){const link=elt('a','원문 기사 열기');link.href=a.url;link.target='_blank';link.rel='noopener noreferrer';box.append(link);}
}
function renderSources(){
const box=$('sourceRows');box.replaceChildren();
if(!current.sources.length){box.append(elt('p','출처 상태 기록이 없습니다.'));return;}
for(const s of current.sources){const row=elt('div',null,'source');const left=elt('div');left.append(elt('div',s.source,'source-name'),elt('div',n(s.articleCount)+'건 · 최종 확인 '+date(s.lastCheckedAt),'source-meta'));
const recent=s.recentAttempt;const bad=Boolean(s.lastError)||recent?.status==='error';
row.append(left,elt('div',bad?'오류':recent?.status==='not_modified'?'변경 없음':recent?.status==='ok'?'수집 성공':'확인 필요','status '+(bad?'bad':recent?'':'neutral')));
if(bad)row.title=s.lastError||'최근 수집 오류';box.append(row);}
}
async function load(){
$('refresh').disabled=true;const e=$('error');e.hidden=true;
try{
const source=$('sourceFilter').value;
const r=await fetch('/api/v1/dashboard?limit=300'+(source?'&source='+encodeURIComponent(source):''),{cache:'no-store'});
if(!r.ok)throw Error('서버 응답 '+r.status+' — 뉴스 DB 연결을 확인하세요.');
current=await r.json();
$('articlesCount').textContent=n(current.totals.articles);$('eventsCount').textContent=n(current.totals.events);
$('impactsCount').textContent=n(current.totals.impacts);
const auto=current.automaticAnalysisCounts||{};
$('runsCount').textContent=n(auto.COMPLETED);
const errors=current.automaticFailureCodes||{};
const byCode=Object.entries(errors).map(([code,count])=>code+': '+count).join(', ');
$('runsFoot').textContent='완료 '+n(auto.COMPLETED)+' · 실패 '+n(auto.FAILED)+(byCode?' · '+byCode:'');
$('lastObserved').textContent='최근 기사 최초 관측: '+date(current.lastArticleDetectedAt);
$('connection').replaceChildren(elt('span',null,'dot ok'),document.createTextNode(' 로컬 DB 연결됨'));
const select=$('sourceFilter');const selectedSource=select.value;select.replaceChildren(new Option('모든 출처',''));
for(const s of current.sources)select.add(new Option(s.source,s.source));select.value=selectedSource;
renderArticles();renderSources();
const chosen=current.articles.find(x=>x.id===selected);if(chosen)renderDetail(chosen);
else if(selected!==null){selected=null;$('detail').textContent='현재 조건에서 선택한 기사가 없습니다.';}
}catch(err){e.textContent=err.message;e.hidden=false;$('connection').replaceChildren(elt('span',null,'dot bad'),document.createTextNode(' 연결 오류'));}
finally{$('refresh').disabled=false;}
}
$('refresh').onclick=load;$('sourceFilter').onchange=load;$('stageFilter').onchange=renderArticles;$('search').oninput=renderArticles;
load();setInterval(()=>{if(!document.hidden)load();},30000);
</script></body></html>'''
