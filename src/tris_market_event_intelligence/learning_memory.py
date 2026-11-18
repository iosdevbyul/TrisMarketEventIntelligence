"""Durable, evidence-accountable rule memory for sequential news research.

Rules are hypotheses until *later* paired, point-in-time-eligible cases support them.
Outcomes from overlapping articles may be used for exploratory prompt feedback,
but never count toward rule promotion.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

FAMILIES = {
    "EARNINGS": ("earnings", "profit", "revenue", "guidance", "실적", "매출"),
    "RATES": ("interest rate", "federal reserve", "fed ", "inflation", "금리", "물가"),
    "REGULATION": ("regulation", "sanction", "export ban", "tariff", "규제", "관세"),
    "GEOPOLITICS": ("war", "attack", "missile", "conflict", "전쟁", "공격"),
    "SUPPLY": ("supply", "factory", "production", "capacity", "공급", "생산"),
    "DEMAND": ("demand", "orders", "sales", "consumer", "수요", "소비"),
}
STOP = {"the", "for", "and", "with", "from", "that", "this", "when",
        "then", "into", "before", "after", "would", "could", "should"}


def family(text: str) -> str:
    lower = text.lower()
    for name, words in FAMILIES.items():
        if any(word in lower for word in words):
            return name
    return "GENERAL"


def tokens(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z0-9가-힣]{3,}", text.lower())
        if word not in STOP
    }


def overlap(first: str, second: str) -> float:
    left, right = tokens(first), tokens(second)
    return len(left & right) / max(1, min(len(left), len(right)))


def initialize(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS memory_failure_cases (
            seq INTEGER PRIMARY KEY, article_id INTEGER NOT NULL,
            family TEXT NOT NULL, sector TEXT NOT NULL,
            wrong_count INTEGER NOT NULL, diagnosis TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memory_rules (
            id INTEGER PRIMARY KEY,
            version INTEGER NOT NULL DEFAULT 1,
            family TEXT NOT NULL, sector TEXT NOT NULL,
            lesson TEXT NOT NULL, diagnosis TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN
              ('CANDIDATE','VALIDATING','ACTIVE','RETIRED')),
            created_seq INTEGER NOT NULL, eligible_after TEXT NOT NULL,
            failure_count INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS memory_rule_applications (
            seq INTEGER NOT NULL, rule_id INTEGER NOT NULL,
            rule_version INTEGER NOT NULL, role TEXT NOT NULL,
            PRIMARY KEY(seq, rule_id),
            FOREIGN KEY(rule_id) REFERENCES memory_rules(id)
        );
        CREATE TABLE IF NOT EXISTS memory_rule_baselines (
            seq INTEGER PRIMARY KEY, rule_id INTEGER NOT NULL,
            prediction_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memory_rule_trials (
            seq INTEGER NOT NULL, rule_id INTEGER NOT NULL,
            eligible INTEGER NOT NULL, comparable INTEGER NOT NULL,
            wins INTEGER NOT NULL, losses INTEGER NOT NULL,
            ties INTEGER NOT NULL,
            baseline_json TEXT NOT NULL, candidate_json TEXT NOT NULL,
            PRIMARY KEY(seq, rule_id)
        );
        CREATE TABLE IF NOT EXISTS memory_rule_history (
            id INTEGER PRIMARY KEY,
            rule_id INTEGER NOT NULL, seq INTEGER NOT NULL,
            old_status TEXT NOT NULL, new_status TEXT NOT NULL,
            reason TEXT NOT NULL
        );
    """)


def _date(value: str) -> datetime:
    date = datetime.fromisoformat(value)
    if date.tzinfo is None:
        raise ValueError("Learning-memory dates must be timezone-aware")
    return date.astimezone(UTC)


def _sector(evaluation: dict[str, Any]) -> str:
    wrong = [row for row in evaluation.get("impacts", [])
             if row.get("status") == "WRONG"]
    if not wrong:
        return "GENERAL"
    first = wrong[0]
    if first.get("targetType") == "SECTOR":
        return str(first["target"])
    if first.get("targetType") == "MARKET":
        return "BROAD_MARKET"
    return "STOCK"


def _eligible_after(evaluation: dict[str, Any], observed: str) -> str:
    dates = [_date(observed)]
    horizon = float(evaluation.get("horizonHours") or 24)
    dates.append(_date(observed) + timedelta(hours=horizon))
    for impact in evaluation.get("impacts", []):
        if impact.get("status") != "UNVERIFIED" and impact.get("outcomeAt"):
            dates.append(_date(impact["outcomeAt"]))
    return max(dates).isoformat()


def learn(db: sqlite3.Connection, *, seq: int, article_id: int,
          article: dict[str, Any], evaluation: dict[str, Any],
          feedback: dict[str, Any]) -> int | None:
    """Idempotently record observed failures and create/merge a hypothesis."""
    lesson = feedback.get("lesson") or ""
    diagnosis = feedback.get("diagnosis") or ""
    wrong_count = sum(row.get("status") == "WRONG"
                      for row in evaluation.get("impacts", []))
    if wrong_count == 0 or not lesson.strip():
        return None
    scope = family(str(article.get("title") or "") + " " +
                   str(article.get("summary") or ""))
    sector = _sector(evaluation)
    existing_case = db.execute(
        "SELECT seq FROM memory_failure_cases WHERE seq=?", (seq,),
    ).fetchone()
    if existing_case:
        return None
    db.execute(
        "INSERT INTO memory_failure_cases VALUES (?,?,?,?,?,?)",
        (seq, article_id, scope, sector, wrong_count, diagnosis),
    )
    # A single originating article must not create the same rule on resume.
    existing = db.execute(
        "SELECT id FROM memory_rules WHERE created_seq=?", (seq,),
    ).fetchone()
    if existing:
        return int(existing[0])

    candidates = db.execute(
        """SELECT id, lesson, status FROM memory_rules
           WHERE family=? AND sector=? AND status!='RETIRED'
           ORDER BY id""",
        (scope, sector),
    ).fetchall()
    for candidate in candidates:
        if overlap(str(candidate[1]), lesson) >= 0.65:
            db.execute(
                "UPDATE memory_rules SET failure_count=failure_count+1 WHERE id=?",
                (candidate[0],),
            )
            return int(candidate[0])
    result = db.execute(
        """INSERT INTO memory_rules
           (family, sector, lesson, diagnosis, status, created_seq,
            eligible_after) VALUES (?,?,?,?,?,?,?)""",
        (scope, sector, lesson[:600], diagnosis[:1200], "CANDIDATE",
         seq, _eligible_after(evaluation, str(article["firstObservedAt"]))),
    )
    return int(result.lastrowid)


def import_legacy(db: sqlite3.Connection, articles: list[dict[str, Any]]) -> None:
    """Migrate previous journal feedback without changing immutable predictions."""
    index = {int(article["articleId"]): article for article in articles}
    for seq, article_id, evaluation, feedback in db.execute(
        """SELECT seq, article_id, evaluation_json, feedback_json FROM steps
           WHERE feedback_json IS NOT NULL ORDER BY seq"""
    ).fetchall():
        article = index.get(article_id)
        if article is None or not evaluation:
            continue
        learn(db, seq=seq, article_id=article_id, article=article,
              evaluation=json.loads(evaluation), feedback=json.loads(feedback))


def select(db: sqlite3.Connection, *, seq: int,
           article: dict[str, Any], limit: int = 7) -> list[dict[str, Any]]:
    """Retrieve relevant active rules and one exploratory candidate, not last N."""
    scope = family(str(article.get("title") or "") + " " +
                   str(article.get("summary") or ""))
    words = tokens(str(article.get("title") or "") + " " +
                   str(article.get("summary") or ""))
    rows = db.execute(
        """SELECT id, version, family, sector, lesson, status, created_seq,
                  eligible_after, failure_count
           FROM memory_rules
           WHERE created_seq < ? AND status != 'RETIRED'""",
        (seq,),
    ).fetchall()
    ranked = []
    for row in rows:
        rule = dict(zip(
            ("id", "version", "family", "sector", "lesson", "status",
             "createdSeq", "eligibleAfter", "failureCount"), row,
            strict=True,
        ))
        similarity = len(words & tokens(str(rule["lesson"])))
        score = (
            (6 if rule["family"] == scope else 0)
            + (5 if rule["status"] == "ACTIVE" else 0)
            + min(similarity, 3)
            + min(rule["failureCount"], 3)
        )
        if scope != "GENERAL" and rule["family"] not in {scope, "GENERAL"}:
            continue
        ranked.append((score, rule))
    ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
    active = [rule for _, rule in ranked if rule["status"] == "ACTIVE"][:limit - 1]
    candidate = next(
        (rule for _, rule in ranked if rule["status"] in
         {"CANDIDATE", "VALIDATING"}), None,
    )
    return active + ([candidate] if candidate is not None else [])


def apply(db: sqlite3.Connection, seq: int, rules: list[dict[str, Any]]) -> None:
    for rule in rules:
        db.execute(
            """INSERT OR IGNORE INTO memory_rule_applications
               VALUES (?,?,?,?)""",
            (seq, rule["id"], rule["version"], rule["status"]),
        )


def rule_prompt(rules: list[dict[str, Any]]) -> list[str]:
    return [
        f"[{rule['status']} R{rule['id']} v{rule['version']}] {rule['lesson']}"
        for rule in rules
    ]


def compare_trial(db: sqlite3.Connection, *, seq: int,
                  article: dict[str, Any], candidate: dict[str, Any],
                  primary: dict[str, Any], baseline: dict[str, Any],
                  evaluation: dict[str, Any]) -> dict[str, Any]:
    """Paired directional comparison, only on matching supported targets."""
    observed = _date(str(article["firstObservedAt"]))
    eligible = observed > _date(str(candidate["eligibleAfter"]))
    primary_map = {(row["targetType"], row["target"]): row["direction"]
                   for row in primary["impacts"]}
    baseline_map = {(row["targetType"], row["target"]): row["direction"]
                    for row in baseline["impacts"]}
    wins = losses = ties = 0
    for row in evaluation["impacts"]:
        key = (row["targetType"], row["target"])
        if row.get("status") == "UNVERIFIED" or key not in baseline_map:
            continue
        actual = row.get("actualDirection")
        candidate_hit = primary_map.get(key) == actual
        baseline_hit = baseline_map[key] == actual
        wins += int(candidate_hit and not baseline_hit)
        losses += int(baseline_hit and not candidate_hit)
        ties += int(baseline_hit == candidate_hit)
    comparable = wins + losses + ties > 0
    db.execute(
        """INSERT OR IGNORE INTO memory_rule_trials
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (seq, candidate["id"], int(eligible), int(comparable),
         wins, losses, ties,
         json.dumps(baseline, ensure_ascii=False),
         json.dumps(primary, ensure_ascii=False)),
    )
    return {"eligible": eligible, "comparable": comparable,
            "wins": wins, "losses": losses, "ties": ties}


def reconcile(db: sqlite3.Connection, *, seq: int) -> None:
    """Promote only on independent, chronologically eligible paired evidence."""
    rules = db.execute(
        """SELECT id, status FROM memory_rules
           WHERE status IN ('CANDIDATE','VALIDATING')"""
    ).fetchall()
    for rule_id, old_status in rules:
        events, wins, losses = db.execute(
            """SELECT COUNT(*), COALESCE(SUM(wins),0), COALESCE(SUM(losses),0)
               FROM memory_rule_trials
               WHERE rule_id=? AND eligible=1 AND comparable=1""",
            (rule_id,),
        ).fetchone()
        new_status = old_status
        reason = ""
        if events >= 5 and wins >= 2 and wins > losses:
            new_status, reason = "ACTIVE", "PAIRED_PIT_ELIGIBLE_IMPROVEMENT"
        elif events >= 8 and losses >= 3 and losses > wins:
            new_status, reason = "RETIRED", "PAIRED_PIT_ELIGIBLE_REGRESSION"
        elif events >= 1 and old_status == "CANDIDATE":
            new_status, reason = "VALIDATING", "PAIRED_PIT_ELIGIBLE_TRIAL_STARTED"
        if new_status != old_status:
            db.execute(
                "UPDATE memory_rules SET status=? WHERE id=?",
                (new_status, rule_id),
            )
            db.execute(
                """INSERT INTO memory_rule_history
                   (rule_id, seq, old_status, new_status, reason)
                   VALUES (?,?,?,?,?)""",
                (rule_id, seq, old_status, new_status, reason),
            )


def report(db: sqlite3.Connection) -> dict[str, Any]:
    counts = {status: count for status, count in db.execute(
        "SELECT status, COUNT(*) FROM memory_rules GROUP BY status",
    ).fetchall()}
    total, right, wrong, unverified = 0, 0, 0, 0
    has_steps = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='steps'"
    ).fetchone()
    if has_steps:
        for (raw,) in db.execute(
            "SELECT evaluation_json FROM steps WHERE evaluation_json IS NOT NULL"
        ):
            for impact in json.loads(raw)["impacts"]:
                total += 1
                right += impact["status"] == "CORRECT"
                wrong += impact["status"] == "WRONG"
                unverified += impact["status"] == "UNVERIFIED"
    eligible_trials = db.execute(
        "SELECT COUNT(*) FROM memory_rule_trials WHERE eligible=1 AND comparable=1",
    ).fetchone()[0]
    return {
        "predictions": total, "correct": right, "wrong": wrong,
        "unverified": unverified,
        "evaluatedAccuracy": right / (right + wrong) if right + wrong else None,
        "rules": {name: counts.get(name, 0) for name in
                  ("CANDIDATE", "VALIDATING", "ACTIVE", "RETIRED")},
        "pairedPITEligibleTrials": eligible_trials,
        "failedCases": db.execute(
            "SELECT COUNT(*) FROM memory_failure_cases",
        ).fetchone()[0],
        "applications": db.execute(
            "SELECT COUNT(*) FROM memory_rule_applications",
        ).fetchone()[0],
        "verifiedImprovement": counts.get("ACTIVE", 0) > 0,
        "independentHoldoutEvaluated": False,
    }
