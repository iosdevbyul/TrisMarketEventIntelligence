"""Read-only, article-level audit of sealed Gemma predictions and price outcomes.

Never reruns an LLM, fetches quotes, or rewrites the journal. The source title
and content are read from the checksum-verified archived news cohort when present.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sqlite3
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from tris_market_event_intelligence.historical_news_cohort import historical_news_cohort


def _timestamp(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError("Expected timezone-aware timestamp")
    return dt.astimezone(UTC)


def _percent(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):+.3f}%"


def _text(value: Any) -> str:
    return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")


def _article_metadata(source_db: Path | None, cutoff: datetime | None,
                      ids: set[int]) -> dict[int, dict[str, Any]]:
    if source_db is None:
        return {}
    if not source_db.is_file() or source_db.is_symlink():
        raise ValueError("Archived source database is unavailable or symlinked")
    if cutoff is None:
        raise ValueError("Source archive cutoff is required")
    metadata: dict[int, dict[str, Any]] = {}
    cursor = 0
    while len(metadata) < len(ids):
        cohort = historical_news_cohort(
            source_db=source_db, as_of=cutoff, after_id=cursor, limit=5000,
        )
        page = cohort.get("items", [])
        if not page:
            break
        for article in page:
            number = int(article["articleId"])
            if number in ids:
                metadata[number] = article
        next_id = max(int(item["articleId"]) for item in page)
        if next_id <= cursor:
            break
        cursor = next_id
        if len(page) < 5000:
            break
    return metadata


def _tables(db: sqlite3.Connection) -> set[str]:
    return {
        row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'",
        )
    }


def inspect(journal: Path, *, limit: int = 1000,
            source_db: Path | None = None) -> dict[str, Any]:
    """Return an auditable, literal reconstruction of persisted study records."""
    if journal.is_symlink() or not journal.is_file():
        raise ValueError("Existing non-symlink journal required")
    if limit < 1 or limit > 10000:
        raise ValueError("limit must be between 1 and 10000")
    with sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        tables = _tables(db)
        if not {"runs", "steps"}.issubset(tables):
            raise ValueError("No sequential learning journal in file")
        settings_row = db.execute(
            "SELECT settings_json FROM runs WHERE id=1",
        ).fetchone()
        settings = json.loads(settings_row[0]) if settings_row else {}
        cutoff = _timestamp(settings["cutoff"]) if settings.get("cutoff") else None
        if source_db is None and settings.get("sourceDb"):
            source_db = Path(settings["sourceDb"])
        if source_db is not None and not source_db.is_file():
            source_db = None
        entries = db.execute(
            """SELECT seq, article_id, observed_at, prediction_json,
                      evaluation_json, feedback_json, phase
               FROM steps ORDER BY seq LIMIT ?""",
            (limit,),
        ).fetchall()
        metadata = _article_metadata(
            source_db, cutoff, {int(e[1]) for e in entries},
        )
        has_applied = "memory_rule_applications" in tables
        has_rules = "memory_rules" in tables
        has_shadow = "memory_rule_baselines" in tables
        has_trials = "memory_rule_trials" in tables
        records = []
        count = Counter()
        outcome_times = Counter()
        neutral_wrong = 0
        relative_verdicts: Counter[str] = Counter()
        attribution_verdicts: Counter[str] = Counter()
        for seq, article_id, observed_at, raw_prediction, raw_evaluation, raw_feedback, phase in entries:
            prediction = json.loads(raw_prediction)
            evaluation = json.loads(raw_evaluation) if raw_evaluation else {}
            feedback = json.loads(raw_feedback) if raw_feedback else {}
            info = metadata.get(article_id, {})
            predicted = prediction.get("impacts", [])
            assessed = evaluation.get("impacts", [])
            by_target = {
                (x.get("targetType"), x.get("target")): x for x in predicted
            }
            outcomes = []
            assessed_keys = {(x.get("targetType"), x.get("target")) for x in assessed}
            for item in predicted:
                if (item.get("targetType"), item.get("target")) not in assessed_keys:
                    outcomes.append({
                        "targetType": item.get("targetType"),
                        "target": item.get("target"),
                        "priceProxy": item.get("priceProxy"),
                        "predictedDirection": item.get("direction"),
                        "predictionReason": item.get("reason"),
                        "observedDirection": None, "verdict": "UNVERIFIED",
                        "absoluteReturn": None, "excessReturn": None,
                        "benchmarkReturn": None, "baselineAt": None,
                        "outcomeAt": None, "baselinePrice": None,
                        "outcomePrice": None,
                        "unverifiedReason": "NO_STORED_EVALUATION",
                    })
            for result in assessed:
                status = result.get("status", "UNVERIFIED")
                count[status] += 1
                if result.get("relativeDirectionVerdict"):
                    relative_verdicts[result["relativeDirectionVerdict"]] += 1
                if result.get("eventAttributionVerdict"):
                    attribution_verdicts[result["eventAttributionVerdict"]] += 1
                if status == "WRONG" and result.get("direction") == "NEUTRAL":
                    neutral_wrong += 1
                if result.get("outcomeAt"):
                    outcome_times[str(result["outcomeAt"])] += 1
                matched = by_target.get((result.get("targetType"), result.get("target")), {})
                outcomes.append({
                    "targetType": result.get("targetType"),
                    "target": result.get("target"),
                    "priceProxy": result.get("priceProxy"),
                    "predictedDirection": result.get("direction"),
                    "predictionReason": matched.get("reason", result.get("reason")),
                    "observedDirection": result.get("actualDirection"),
                    "verdict": status,
                    "priceDirectionVerdict": result.get("priceDirectionVerdict", status),
                    "relativeDirection": result.get("relativeDirection"),
                    "relativeDirectionVerdict": result.get("relativeDirectionVerdict"),
                    "eventAttributionVerdict": result.get("eventAttributionVerdict"),
                    "absoluteReturn": result.get("absoluteReturn"),
                    "benchmarkReturn": result.get("benchmarkReturn"),
                    "excessReturn": result.get("excessReturn"),
                    "baselineAt": result.get("baselineAt"),
                    "outcomeAt": result.get("outcomeAt"),
                    "baselinePrice": result.get("baselinePrice"),
                    "outcomePrice": result.get("outcomePrice"),
                    "unverifiedReason": result.get("reasonCode"),
                })
            rules = []
            if has_applied and has_rules:
                rules = [
                    {"id": r[0], "version": r[1], "roleAtApplication": r[2],
                     "statusNow": r[3], "rule": r[4]}
                    for r in db.execute(
                        """SELECT a.rule_id,a.rule_version,a.role,
                                  r.status,r.lesson
                           FROM memory_rule_applications a
                           JOIN memory_rules r ON r.id=a.rule_id
                           WHERE a.seq=? ORDER BY a.rule_id""",
                        (seq,),
                    )
                ]
            shadow = None
            if has_shadow:
                row = db.execute(
                    "SELECT rule_id,prediction_json FROM memory_rule_baselines WHERE seq=?",
                    (seq,),
                ).fetchone()
                if row:
                    shadow = {"excludedRuleId": row[0], "prediction": json.loads(row[1])}
                    if has_trials:
                        trial = db.execute(
                            """SELECT eligible,comparable,wins,losses,ties
                               FROM memory_rule_trials WHERE seq=? AND rule_id=?""",
                            (seq, row[0]),
                        ).fetchone()
                        if trial:
                            shadow["comparison"] = dict(zip(
                                ("eligible", "comparable", "wins", "losses", "ties"),
                                trial, strict=True,
                            ))
            records.append({
                "seq": seq,
                "articleId": article_id,
                "title": info.get("title"),
                "articleSummary": info.get("summary"),
                "source": info.get("source"),
                "sourceUrl": info.get("sourceUrl"),
                "publishedAtReported": info.get("publishedAtReported"),
                "firstObservedAt": observed_at,
                "originalPredictionSummary": prediction.get("summary"),
                "eventFacts": prediction.get("eventFacts"),
                "eventGroundingStatus": prediction.get("eventGroundingStatus"),
                "eventGroundingReasons": prediction.get("eventGroundingReasons"),
                "surpriseAssessment": prediction.get("surpriseAssessment"),
                "marketContext": prediction.get("marketContext"),
                "eventMarket": prediction.get("eventMarket"),
                "sourceContentQuality": prediction.get("sourceContentQuality"),
                "contentEvidence": prediction.get("contentEvidence"),
                "targetMappingRejections": prediction.get("targetMappingRejections", []),
                "recoveryStatus": prediction.get("recoveryStatus"),
                "predictedImpacts": predicted,
                "outcomes": outcomes,
                "diagnosis": feedback.get("diagnosis"),
                "correctiveRule": feedback.get("lesson"),
                "unverifiedHypothesis": feedback.get("unverifiedHypothesis"),
                "hypothesisStatus": feedback.get("hypothesisStatus"),
                "rulesApplied": rules,
                "pairedShadow": shadow,
                "phase": phase,
                "evaluationSource": evaluation.get("source"),
                "interval": evaluation.get("interval"),
                "horizonHours": evaluation.get("horizonHours"),
            })
        correct = count["CORRECT"]
        wrong = count["WRONG"]
        completed = sum(r["phase"] == "COMPLETE" for r in records)
        warnings = [
            "CORRECT/WRONG are original stored labels, NOT independently verified accuracy.",
            "Daily candle timestamps may not equal the exchange's actual closing time.",
            "Earlier stored results may lack actual baseline/outcome price levels; missing levels are shown as unavailable.",
            "Legacy journals may include later-horizon hindsight in rule feedback; new runs filter by eligibleAfter.",
            "New study: incomplete article bodies are excluded, not scored as neutral.",
            "Korean market and sector ETF proxies require exchange-specific price validation.",
            "Market movement is not evidence that the news event caused the price change.",
            "Price direction labels and benchmark-relative labels are descriptive, not causal validation.",
            "Unverified critic hypotheses are not approved memory rules.",
            "No causal effect or price timestamp validity is established by this report.",
        ]
        if source_db is None:
            warnings.append("Source news database unavailable; some article titles are missing.")
        if neutral_wrong:
            warnings.append(
                f"{neutral_wrong} WRONG outcomes predicted NEUTRAL; inspect the +/-0.1% threshold."
            )
        repeated = [(t, n) for t, n in outcome_times.most_common(5) if n >= 3]
        if repeated:
            warnings.append(
                "Multiple verdicts share identical outcome timestamps. "
                "Check event/price alignment and market calendar before learning."
            )
        return {
            "schemaVersion": "sequential_article_audit_v1",
            "journal": str(journal),
            "sourceArchive": str(source_db) if source_db is not None else None,
            "settings": settings,
            "summary": {
                "articlesShown": len(records),
                "articlesCompleted": completed,
                "predictionsCorrect": correct,
                "predictionsWrong": wrong,
                "predictionsUnverified": count["UNVERIFIED"],
                "storedVerdictAccuracy": (
                    correct / (correct + wrong) if correct + wrong else None
                ),
                "neutralPredictionsMarkedWrong": neutral_wrong,
                "relativeDirectionVerdicts": dict(relative_verdicts),
                "eventAttributionVerdicts": dict(attribution_verdicts),
                "priceDirectionsAreNotEventAttribution": True,
                "commonOutcomeTimestamps": [
                    {"outcomeAt": t, "count": n} for t, n in repeated
                ],
                "independentlyVerified": False,
            },
            "warnings": warnings,
            "articles": records,
        }


def markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# 기사별 시장 영향 분석 감사 보고서",
        "",
        (f"- 처리 기사: {summary['articlesShown']}건 "
         f"(COMPLETE {summary['articlesCompleted']}건)"),
        (f"- 저장된 판정: 정답 {summary['predictionsCorrect']} / "
         f"오답 {summary['predictionsWrong']} / "
         f"검증 불가 {summary['predictionsUnverified']}"),
        "- 주의: 정답·오답은 기존 시스템 판정이며 검증된 정답이 아닙니다.",
        "",
        "## 검증 경고",
    ]
    lines.extend("- " + warning for warning in report["warnings"])
    lines.extend(["", "## 기사별 예측 대조표", "",
                  "| 번호 | 기사 | 대상 | AI 예측 | 실제 방향 | 수익률 | 시장 대비 | 판정 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    for record in report["articles"]:
        rows = record["outcomes"] or [{
            "target": "—", "predictedDirection": "—",
            "observedDirection": None, "absoluteReturn": None,
            "excessReturn": None, "verdict": "NO_TARGETS",
        }]
        for impact in rows:
            lines.append(
                f"| {record['seq']} | {_text(record['title'] or '[원문 없음]')} "
                f"| {_text(impact.get('target'))} | "
                f"{_text(impact.get('predictedDirection'))} | "
                f"{_text(impact.get('observedDirection'))} | "
                f"{_percent(impact.get('absoluteReturn'))} | "
                f"{_percent(impact.get('excessReturn'))} | "
                f"{_text(impact.get('verdict'))} |"
            )
    lines.extend(["", "## 기사별 상세 근거"])
    for record in report["articles"]:
        lines.extend([
            "", f"### #{record['seq']} · {_text(record['title'] or '원본 기사 정보 없음')}",
            f"- 기사 ID: {record['articleId']}",
            f"- 최초 관측 시각(UTC): {_text(record['firstObservedAt'])}",
            f"- 기사 출처: {_text(record['source'])}",
            f"- 시장 관할: {_text(record.get('eventMarket'))}",
            f"- 본문 완전성: {_text(record.get('sourceContentQuality'))}",
            f"- 확보된 원문 증거: {_text(record.get('contentEvidence'))}",
            f"- 기사 URL: {_text(record['sourceUrl'])}",
            f"- AI 분석 요약: {_text(record['originalPredictionSummary'])}",
            f"- 발표 전 시장 근거: {_text(record.get('marketContext'))}",
            f"- 최초 발표 시각: {_text(record.get('publishedAtReported'))}",
            f"- 최초 수집 시각: {_text(record.get('firstObservedAt'))}",
            f"- 기대 대비 분석: {_text(record.get('surpriseAssessment'))}",
            f"- 대상 매핑 거부 기록: {_text(record.get('targetMappingRejections'))}",
            f"- 기사 내용: {_text(record['articleSummary'])}",
            "",
        ])
        if not record["outcomes"]:
            lines.append("예측된 대상이 없거나 가격 평가가 기록되지 않았습니다.")
        for x in record["outcomes"]:
            lines.extend([
                (f"**{_text(x['target'])} ({_text(x['priceProxy'])}) — "
                 f"{_text(x['verdict'])}**"),
                (f"- AI 예측: {_text(x['predictedDirection'])} "
                 f"(근거: {_text(x['predictionReason'])})"),
                (f"- 실제 방향: {_text(x['observedDirection'])}; "
                 f"절대 수익률 {_percent(x['absoluteReturn'])}; "
                 f"벤치마크 수익률 {_percent(x['benchmarkReturn'])}; "
                 f"초과 수익률 {_percent(x['excessReturn'])}"),
                (f"- 기준 가격: {_text(x['baselinePrice'])}; "
                 f"결과 가격: {_text(x['outcomePrice'])}"),
                (f"- 기준 가격 시각: {_text(x['baselineAt'])}; "
                 f"결과 가격 시각: {_text(x['outcomeAt'])}"),
                f"- 미검증 사유: {_text(x['unverifiedReason'])}",
                "",
            ])
        lines.extend([
            f"- 기사 의미 확인: {_text(record.get('eventFacts'))}",
            f"- 근거 필터: {_text(record.get('eventGroundingStatus'))}",
            f"- 검증 전 학습 가설: {_text(record.get('unverifiedHypothesis'))}",
            f"- Gemma 오답 진단: {_text(record['diagnosis'])}",
            f"- 생성된 개선 규칙: {_text(record['correctiveRule'])}",
            "- 이번 분석에 전달한 기억 규칙:",
        ])
        for rule in record["rulesApplied"]:
            lines.append(
                f"  - R{rule['id']} v{rule['version']} "
                f"({rule['roleAtApplication']}): {_text(rule['rule'])}"
            )
        if not record["rulesApplied"]:
            lines.append("  - 기록 없음 (구버전 실행 또는 적용 규칙 없음)")
        if record["pairedShadow"]:
            lines.append(
                f"- 규칙 제외 비교: R{record['pairedShadow']['excludedRuleId']}; "
                f"비교 결과 {_text(record['pairedShadow'].get('comparison'))}"
            )
    return "\n".join(lines) + "\n"



def html_report(report: dict[str, Any]) -> str:
    """Generate a standalone, searchable report for opening in any browser."""
    esc = lambda value: html.escape(str(value if value is not None else "—"), quote=True)
    summary = report["summary"]
    def article_link(url: Any) -> str:
        if not isinstance(url, str):
            return "원문 링크 없음"
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            return "원문 링크 없음"
        return (f'<a href="{esc(url)}" rel="noopener noreferrer" '
                f'target="_blank">원문 기사 열기</a>')

    labels = {"SECTOR": "연결된 섹터", "STOCK": "연결된 종목",
              "MARKET": "시장·지수"}
    directions = {"POSITIVE": "상승", "NEGATIVE": "하락", "NEUTRAL": "중립"}
    sections = []
    for record in report["articles"]:
        grouped: dict[str, list[str]] = {kind: [] for kind in labels}
        for impact in record["outcomes"]:
            status = impact["verdict"]
            cls = "good" if status == "CORRECT" else (
                "bad" if status == "WRONG" else "unknown"
            )
            category = impact.get("targetType")
            if category not in grouped:
                category = "MARKET"
            grouped[category].append(
                "<tr>"
                f"<td>{esc(impact['target'])} ({esc(impact['priceProxy'])})</td>"
                f"<td>{esc(directions.get(impact['predictedDirection'], '미분석'))}</td>"
                f"<td>{esc(directions.get(impact['observedDirection'], '검증 불가'))}</td>"
                f"<td>{esc(_percent(impact['absoluteReturn']))}</td>"
                f"<td>{esc(_percent(impact['excessReturn']))}</td>"
                f'<td><span class="{cls}">{esc(status)}</span></td>'
                "</tr>"
                "<tr><td colspan='6' class='details'>"
                f"<b>AI 근거</b>: {esc(impact['predictionReason'])}"
                "<br>"
                f"<b>기준 가격</b>: {esc(impact['baselinePrice'])} "
                f"({esc(impact['baselineAt'])})"
                "<br>"
                f"<b>결과 가격</b>: {esc(impact['outcomePrice'])} "
                f"({esc(impact['outcomeAt'])})"
                "<br>"
                f"<b>시장 수익률</b>: {esc(_percent(impact['benchmarkReturn']))}"
                "<br>"
                f"<b>검증 불가 원인</b>: {esc(impact['unverifiedReason'])}"
                "</td></tr>"
            )
        tables = []
        for kind, label in labels.items():
            rows = "".join(grouped[kind]) or (
                "<tr><td colspan='6'>연결된 대상 없음</td></tr>"
            )
            tables.append(
                f"<h3>{esc(label)} ({len(grouped[kind])})</h3>"
                "<table><thead><tr><th>대상 (가격 프록시)</th><th>AI 예측</th>"
                "<th>실제 방향</th><th>수익률</th><th>시장 대비</th>"
                "<th>저장된 판정</th></tr></thead><tbody>"
                + rows + "</tbody></table>"
            )
        rejected_targets = record.get("targetMappingRejections") or []
        rejected_list = "".join(
            f"<li>{esc(item.get('targetType'))}: {esc(item.get('target'))} "
            f"— {esc(item.get('reasonCode'))}</li>"
            for item in rejected_targets
        ) or "<li>없음</li>"
        mapping_notice = (
            f"<p><b>대상 매핑 상태:</b> {esc(record.get('recoveryStatus'))}</p>"
            f"<p><b>거부된 미지원 대상 (예측·평가 제외):</b></p>"
            f"<ul>{rejected_list}</ul>"
        )
        context = record.get("marketContext") or {}
        surprise = record.get("surpriseAssessment") or {}
        earlier = "".join(
            f"<li>#{esc(item.get('articleId'))}: {esc(item.get('title'))} "
            f"({esc(item.get('firstObservedAt'))})</li>"
            for item in context.get("earlierRelatedArticles", [])
        ) or "<li>관련 과거 기사 없음</li>"
        market = context.get("marketPreTrend") or {}
        assessment_text = (
            f"<p><b>경제적 영향:</b> {esc(surprise.get('economicImpact'))} · "
            f"<b>시장 예상 대비:</b> {esc(surprise.get('expectationStatus'))} · "
            f"<b>정보 신규성:</b> {esc(surprise.get('novelty'))} · "
            f"<b>선반영 가능성:</b> {esc(surprise.get('pricedInStatus'))}</p>"
            f"<p><b>판단 근거:</b> {esc(surprise.get('rationale'))}</p>"
            f"<p><b>발표 전 {esc(market.get('symbol'))} 흐름:</b> {esc(_percent(market.get('return')))} "
            f"({esc(market.get('status'))}) · "
            f"<b>컨센서스:</b> {esc((context.get('consensus') or {}).get('status'))}</p>"
            f"<p><b>발표 전 관련 기사:</b></p><ul>{earlier}</ul>"
        )
        applied = "".join(
            f"<li>R{esc(r['id'])} v{esc(r['version'])} "
            f"({esc(r['roleAtApplication'])}): {esc(r['rule'])}</li>"
            for r in record["rulesApplied"]
        ) or "<li>기록 없음</li>"
        evidence = record.get("contentEvidence") or {}
        source_notice = (
            f"<p><b>시장 관할:</b> {esc(record.get('eventMarket'))} · "
            f"<b>본문 완전성:</b> {esc((record.get('sourceContentQuality') or {}).get('status'))}</p>"
            f"<p><b>확보 원문:</b> {article_link(evidence.get('attachmentUrl'))} · "
            f"<b>파일 SHA256:</b> {esc(evidence.get('contentSha256'))} · "
            f"<b>당시 원문 검증:</b> {esc(evidence.get('pointInTimeCertified'))}</p>"
        )
        sections.append(
            "<details>"
            f"<summary>#{esc(record['seq'])} "
            f"{esc(record['title'] or '제목을 원본 DB에서 찾을 수 없음')}"
            f" <small>기사 ID {esc(record['articleId'])}</small></summary>"
            "<div class='body'>"
            f"<p><b>원래 발표 시각:</b> {esc(record.get('publishedAtReported'))} · "
            f"<b>최초 수집 시각:</b> {esc(record['firstObservedAt'])} · "
            f"<b>출처:</b> {esc(record['source'])}</p>"
            f"<p><b>기사 원문 링크:</b> {article_link(record['sourceUrl'])}</p>"
            + source_notice
            + f"<p><b>저장된 기사 요약:</b> {esc(record['articleSummary'])}</p>"
            f"<p><b>Gemma 분석:</b> {esc(record['originalPredictionSummary'])}</p>"
            + "<h3>발표 전 시장 기대·새로운 정보 분석</h3>"
            + assessment_text
            + "".join(tables)
            + mapping_notice
            + f"<p><b>실패 원인 진단:</b> {esc(record['diagnosis'])}</p>"
            f"<p><b>다음 기사 수정 규칙:</b> {esc(record['correctiveRule'])}</p>"
            f"<p><b>해당 분석에 적용한 규칙:</b></p><ul>{applied}</ul>"
            "</div></details>"
        )
    warnings = "".join(f"<li>{esc(w)}</li>" for w in report["warnings"])
    return (
        "<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>TrisMarketEventIntelligence 기사별 분석 감사</title>"
        "<style>"
        ":root{color-scheme:light dark;font-family:-apple-system,BlinkMacSystemFont,"
        "'Segoe UI',sans-serif}body{max-width:1200px;margin:35px auto;padding:0 18px;"
        "line-height:1.6}h1{font-size:1.7rem}.stats{display:flex;flex-wrap:wrap;"
        "gap:14px}.stats>div{border:1px solid #8886;border-radius:10px;padding:12px 20px}"
        ".stats strong{font-size:1.5rem;display:block}.warn{border-left:3px solid #ce9638;"
        "padding:10px 20px;background:rgba(208,150,60,.1);margin:20px 0}"
        "details{border:1px solid #8886;border-radius:9px;margin:12px 0}"
        "summary{cursor:pointer;padding:14px;font-weight:650}"
        "summary small{font-weight:400;opacity:.7;margin-left:10px}"
        ".body{padding:8px 15px 15px;overflow-wrap:anywhere}table{border-collapse:collapse;"
        "width:100%;font-size:.9rem}th,td{border-bottom:1px solid #8884;"
        "text-align:left;padding:9px}.details{opacity:.85;font-size:.88rem}"
        ".good{color:#168057;font-weight:700}.bad{color:#b73838;font-weight:700}"
        ".unknown{color:#9a782b;font-weight:700}"
        "input{padding:12px;width:100%;box-sizing:border-box;font:inherit;"
        "border:1px solid #8888;border-radius:8px;background:transparent}"
        "</style></head><body>"
        "<h1>TrisMarketEventIntelligence · 기사별 AI 예측 감사</h1>"
        "<p>실제 DB에 저장된 Gemma 예측과 주가 판정을 대조합니다. "
        "<strong>기존 CORRECT/WRONG 표시는 아직 독립 검증되지 않았습니다.</strong></p>"
        "<div class='stats'>"
        f"<div>분석 기사<strong>{summary['articlesShown']}</strong></div>"
        f"<div>저장된 정답<strong>{summary['predictionsCorrect']}</strong></div>"
        f"<div>저장된 오답<strong>{summary['predictionsWrong']}</strong></div>"
        f"<div>가격 검증 불가<strong>{summary['predictionsUnverified']}</strong></div>"
        "</div>"
        f"<section class='warn'><b>검증 경고</b><ul>{warnings}</ul></section>"
        "<label for='search'>기사 번호·제목으로 검색</label>"
        "<input id='search' placeholder='예: 20, central bank, inflation' "
        "oninput=\"document.querySelectorAll('details').forEach"
        "(e=>e.hidden=!e.textContent.toLowerCase().includes(this.value.toLowerCase()))\">"
        + "".join(sections) +
        "</body></html>\n"
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only article prediction vs historical price audit",
    )
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--source-db", type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--format", choices=("markdown", "json", "html"), default="html")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = inspect(args.journal, limit=args.limit, source_db=args.source_db)
        if args.format == "markdown":
            result = markdown(report)
        elif args.format == "html":
            result = html_report(report)
        else:
            result = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            if args.output.is_symlink() or not args.output.parent.is_dir():
                raise ValueError("Output parent must exist; symlinks prohibited")
            temp_name = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=args.output.parent,
                    prefix=".sequential-audit-", suffix=".tmp", delete=False,
                ) as f:
                    temp_name = f.name
                    f.write(result)
                    if not result.endswith("\n"):
                        f.write("\n")
                    f.flush()
                    os.fsync(f.fileno())
                if args.output.is_symlink():
                    raise ValueError("Symlink output prohibited")
                os.replace(temp_name, args.output)
            finally:
                if temp_name is not None and os.path.exists(temp_name):
                    os.unlink(temp_name)
        else:
            print(result)
    except (ValueError, OSError, sqlite3.Error, KeyError, TypeError) as exc:
        parser.exit(2, f"sequential-audit: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
