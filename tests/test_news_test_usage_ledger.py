"""Final NEWS-only TEST must not be repeatedly reserved for model tuning."""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_news_holdout_plan import cohort

from tris_market_event_intelligence.news_holdout_plan import plan_news_holdout
from tris_market_event_intelligence.news_test_usage_ledger import (
    audit_news_test_usage,
    initialize_usage_ledger,
    reserve_news_test_use,
)


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    source = cohort(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan = plan_news_holdout(
        source, train_end=datetime(2026, 11, 2, tzinfo=UTC),
        validation_end=datetime(2026, 11, 6, tzinfo=UTC),
    )
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    journal = tmp_path / "research-usage.sqlite3"
    assert initialize_usage_ledger(journal)["status"] == "INITIALIZED"
    return plan_path, journal


def _reserve(
    plan: Path, journal: Path, purpose: str, experiment: str,
) -> dict[str, object]:
    return reserve_news_test_use(
        ledger_path=journal, plan_path=plan, experiment_id=experiment,
        candidate_sha256="a" * 64, purpose=purpose,
        justification="Explicit human-authored research purpose for locked split.",
    )


def test_validation_can_repeat_but_final_test_only_once(tmp_path: Path) -> None:
    plan, ledger = _setup(tmp_path)
    original_plan = plan.read_bytes()
    assert _reserve(plan, ledger, "TRAIN", "training-1")["purpose"] == "TRAIN"
    assert _reserve(plan, ledger, "VALIDATION", "tune-1")["purpose"] == "VALIDATION"
    assert _reserve(plan, ledger, "VALIDATION", "tune-2")["purpose"] == "VALIDATION"
    one = _reserve(plan, ledger, "TEST_FINAL", "final-1")
    assert one["finalTestReuseBlockedByLedger"] is True
    assert one["status"] == "RESERVATION_RECORDED_NOT_EVALUATED"
    with pytest.raises(ValueError, match="already reserved"):
        _reserve(plan, ledger, "TEST_FINAL", "final-2")
    result = audit_news_test_usage(ledger)
    assert result["totalReservations"] == 4
    assert sum(
        row["purpose"] == "TEST_FINAL" for row in result["reservations"]
    ) == 1
    assert result["testSetAccessOutsideLedgerVerified"] is False
    assert plan.read_bytes() == original_plan


def test_one_final_test_across_modified_plan_for_same_cohort(tmp_path: Path) -> None:
    plan, ledger = _setup(tmp_path)
    _reserve(plan, ledger, "TEST_FINAL", "final-1")
    doc = json.loads(plan.read_text())
    doc["embargoHours"] = 12
    plan.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="already reserved"):
        _reserve(plan, ledger, "TEST_FINAL", "final-2")


def test_same_experiment_id_is_rejected_without_overwrite(tmp_path: Path) -> None:
    plan, ledger = _setup(tmp_path)
    _reserve(plan, ledger, "VALIDATION", "trial")
    with pytest.raises(ValueError, match="Duplicate experiment"):
        _reserve(plan, ledger, "VALIDATION", "trial")
    assert audit_news_test_usage(ledger)["totalReservations"] == 1


@pytest.mark.parametrize("bad", [
    {"candidate_sha256": "xyz"},
    {"candidate_sha256": "B" * 64},
    {"experiment_id": ""},
    {"purpose": "TEST"},
    {"purpose": "MARKET_RETURN"},
    {"justification": "short"},
])
def test_invalid_experiment_is_rejected(tmp_path: Path, bad: dict) -> None:
    plan, ledger = _setup(tmp_path)
    arguments = {
        "ledger_path": ledger, "plan_path": plan,
        "experiment_id": "candidate-1", "candidate_sha256": "b" * 64,
        "purpose": "VALIDATION",
        "justification": "Independent news model evaluation intent.",
    }
    arguments.update(bad)
    with pytest.raises(ValueError):
        reserve_news_test_use(**arguments)
    assert audit_news_test_usage(ledger)["totalReservations"] == 0


def test_missing_partition_or_tampered_counts_fail_closed(tmp_path: Path) -> None:
    plan, ledger = _setup(tmp_path)
    doc = json.loads(plan.read_text())
    doc["counts"]["TEST"] = 0
    plan.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="counts"):
        _reserve(plan, ledger, "TEST_FINAL", "final-1")


def test_uninitialized_or_unsafe_ledger_rejected(tmp_path: Path) -> None:
    plan, ledger = _setup(tmp_path)
    with pytest.raises(ValueError, match="Initialize only"):
        initialize_usage_ledger(ledger)
    fake = tmp_path / "uninitialized.sqlite3"
    with sqlite3.connect(fake):
        pass
    with pytest.raises(ValueError, match="not initialized"):
        _reserve(plan, fake, "VALIDATION", "trial")
    symlink = tmp_path / "alias.sqlite3"
    symlink.symlink_to(ledger)
    with pytest.raises(ValueError, match="symlink"):
        _reserve(plan, symlink, "VALIDATION", "trial")
    assert audit_news_test_usage(ledger)["totalReservations"] == 0
