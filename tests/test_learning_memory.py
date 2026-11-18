"""Rule memory lifecycle, persistence, retrieval and leakage-aware promotion tests."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from tris_market_event_intelligence import learning_memory as lm


def article(seq: int, day: int = 0):
    return {
        "articleId": seq,
        "title": "Semiconductor export regulation affects chip demand",
        "summary": "New export restrictions on semiconductor suppliers",
        "firstObservedAt": (
            datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=day)
        ).isoformat(),
    }


def evaluation(observed, actual="POSITIVE"):
    date = datetime.fromisoformat(observed)
    return {
        "horizonHours": 24,
        "impacts": [{
            "targetType": "STOCK", "target": "NVDA",
            "direction": "NEGATIVE", "actualDirection": actual,
            "status": "WRONG", "outcomeAt": (date + timedelta(days=2)).isoformat(),
        }],
    }


def feedback():
    return {
        "diagnosis": "A revenue downside overlooked market pricing power",
        "lesson": "Evaluate demand strength and pricing power before assuming supply "
                  "restrictions will reduce semiconductor stock prices.",
    }


def test_persistent_memory_select_and_idempotent_merge(tmp_path):
    path = tmp_path / "journal.sqlite3"
    with sqlite3.connect(path) as db:
        lm.initialize(db)
        first = article(1)
        result = lm.learn(
            db, seq=1, article_id=1, article=first,
            evaluation=evaluation(first["firstObservedAt"]), feedback=feedback(),
        )
        assert result == 1
        assert lm.select(db, seq=1, article=first) == []
        selected = lm.select(db, seq=2, article=article(2, 7))
        assert len(selected) == 1
        assert "pricing power" in lm.rule_prompt(selected)[0]
        lm.apply(db, 2, selected)
        assert lm.learn(
            db, seq=1, article_id=1, article=first,
            evaluation=evaluation(first["firstObservedAt"]), feedback=feedback(),
        ) is None
        second = article(3, 8)
        lm.learn(
            db, seq=3, article_id=3, article=second,
            evaluation=evaluation(second["firstObservedAt"]), feedback=feedback(),
        )
        assert db.execute("SELECT COUNT(*) FROM memory_rules").fetchone()[0] == 1
        assert db.execute(
            "SELECT failure_count FROM memory_rules"
        ).fetchone()[0] == 2
        db.commit()
    with sqlite3.connect(path) as db:
        lm.initialize(db)
        assert lm.report(db)["rules"]["CANDIDATE"] == 1
        assert lm.report(db)["applications"] == 1
        assert lm.report(db)["failedCases"] == 2


def test_no_promotion_from_overlapping_outcomes():
    with sqlite3.connect(":memory:") as db:
        lm.initialize(db)
        origin = article(1)
        lm.learn(
            db, seq=1, article_id=1, article=origin,
            evaluation=evaluation(origin["firstObservedAt"]), feedback=feedback(),
        )
        rule = lm.select(db, seq=2, article=article(2))[0]
        primary = {"impacts": [
            {"targetType": "STOCK", "target": "NVDA", "direction": "POSITIVE"},
        ]}
        baseline = {"impacts": [
            {"targetType": "STOCK", "target": "NVDA", "direction": "NEGATIVE"},
        ]}
        for seq in range(2, 12):
            test = lm.compare_trial(
                db, seq=seq, article=article(seq),
                candidate=rule, primary=primary, baseline=baseline,
                evaluation=evaluation(article(seq)["firstObservedAt"]),
            )
            assert test["wins"] == 1
            assert not test["eligible"]
            lm.reconcile(db, seq=seq)
        assert db.execute(
            "SELECT status FROM memory_rules"
        ).fetchone()[0] == "CANDIDATE"
        assert lm.report(db)["pairedPITEligibleTrials"] == 0


def test_paired_later_independent_cases_promote_and_retire():
    with sqlite3.connect(":memory:") as db:
        lm.initialize(db)
        origin = article(1)
        lm.learn(
            db, seq=1, article_id=1, article=origin,
            evaluation=evaluation(origin["firstObservedAt"]), feedback=feedback(),
        )
        rule = lm.select(db, seq=2, article=article(2, 10))[0]
        winner = {"impacts": [
            {"targetType": "STOCK", "target": "NVDA", "direction": "POSITIVE"},
        ]}
        loser = {"impacts": [
            {"targetType": "STOCK", "target": "NVDA", "direction": "NEGATIVE"},
        ]}
        for seq in range(2, 7):
            lm.compare_trial(
                db, seq=seq, article=article(seq, 10 + seq),
                candidate=rule, primary=winner, baseline=loser,
                evaluation=evaluation(article(seq, 10 + seq)["firstObservedAt"]),
            )
            lm.reconcile(db, seq=seq)
        assert db.execute(
            "SELECT status FROM memory_rules WHERE id=1"
        ).fetchone()[0] == "ACTIVE"
        assert lm.report(db)["verifiedImprovement"]
        assert lm.report(db)["independentHoldoutEvaluated"] is False
        assert db.execute(
            "SELECT COUNT(*) FROM memory_rule_history WHERE new_status='ACTIVE'"
        ).fetchone()[0] == 1


def test_unverified_and_disjoint_targets_cannot_count_as_evidence():
    with sqlite3.connect(":memory:") as db:
        lm.initialize(db)
        origin = article(1)
        lm.learn(
            db, seq=1, article_id=1, article=origin,
            evaluation=evaluation(origin["firstObservedAt"]), feedback=feedback(),
        )
        rule = lm.select(db, seq=2, article=article(2, 10))[0]
        row = lm.compare_trial(
            db, seq=2, article=article(2, 10), candidate=rule,
            primary={"impacts": [{
                "targetType": "STOCK", "target": "NVDA", "direction": "POSITIVE",
            }]},
            baseline={"impacts": [{
                "targetType": "STOCK", "target": "AMD", "direction": "NEGATIVE",
            }]},
            evaluation=evaluation(article(2, 10)["firstObservedAt"]),
        )
        assert row["eligible"]
        assert not row["comparable"]
        lm.reconcile(db, seq=2)
        assert lm.report(db)["pairedPITEligibleTrials"] == 0
