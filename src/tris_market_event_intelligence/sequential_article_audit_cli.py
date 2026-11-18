"""Read-only, article-level audit of sealed Gemma predictions and price outcomes.

Never reruns an LLM, fetches quotes, or rewrites the journal. The source title
and content are read from the checksum-verified archived news cohort when present.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
            for result in assessed:
                status = result.get("status", "UNVERIFIED")
                count[status] += 1
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
                    "absoluteReturn": result.get("absoluteReturn"),
                    "benchmarkReturn": result.get("benchmarkReturn"),
                    "excessReturn": result.get("excessReturn"),
                    "baselineAt": result.get("baselineAt"),
                    "outcomeAt": result.get("outcomeAt"),
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
                "predictedImpacts": predicted,
                "outcomes": outcomes,
                "diagnosis": feedback.get("diagnosis"),
                "correctiveRule": feedback.get("lesson"),
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
            "Retrospective rule feedback can include outcomes learned after a later article's historical timestamp.",
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
            f"- 기사 URL: {_text(record['sourceUrl'])}",
            f"- AI 분석 요약: {_text(record['originalPredictionSummary'])}",
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
                (f"- 기준 가격 시각: {_text(x['baselineAt'])}; "
                 f"결과 가격 시각: {_text(x['outcomeAt'])}"),
                f"- 미검증 사유: {_text(x['unverifiedReason'])}",
                "",
            ])
        lines.extend([
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only article prediction vs historical price audit",
    )
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--source-db", type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = inspect(args.journal, limit=args.limit, source_db=args.source_db)
        result = (markdown(report) if args.format == "markdown" else
                  json.dumps(report, ensure_ascii=False, indent=2))
        if args.output:
            if args.output.is_symlink() or not args.output.parent.is_dir():
                raise ValueError("Output parent must exist; symlinks prohibited")
            with args.output.open("x", encoding="utf-8") as f:
                f.write(result)
                if not result.endswith("\n"):
                    f.write("\n")
        else:
            print(result)
    except (ValueError, OSError, sqlite3.Error, KeyError, TypeError) as exc:
        parser.exit(2, f"sequential-audit: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
