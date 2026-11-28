"""Focused failure-analysis diagnostics tests, without Gemma inference."""
from pathlib import Path

import pytest

from tris_market_event_intelligence.prediction_failure_analysis_cli import (
    _matrix,
    _safe_dir,
)


def test_confusion_matrix_and_balanced_accuracy():
    rows = [
        {"prediction": "UP", "actualDirection": "UP"},
        {"prediction": "UP", "actualDirection": "DOWN"},
        {"prediction": "DOWN", "actualDirection": "DOWN"},
        {"prediction": "NEUTRAL", "actualDirection": "NEUTRAL"},
        {"prediction": "UP", "actualDirection": "NEUTRAL"},
    ]
    result = _matrix(rows, actual_key="actualDirection")
    assert result["count"] == 5
    assert result["correct"] == 3
    assert result["accuracy"] == pytest.approx(0.6)
    assert result["matrixActualRowsPredictedColumns"]["DOWN"]["UP"] == 1
    assert result["perClass"]["UP"]["precision"] == pytest.approx(1 / 3)
    assert result["perClass"]["DOWN"]["recall"] == pytest.approx(0.5)
    assert result["perClass"]["NEUTRAL"]["recall"] == pytest.approx(0.5)
    assert result["balancedAccuracyPresentClasses"] == pytest.approx(2 / 3)
    assert result["alwaysBaselines"]["DOWN"] == pytest.approx(0.4)


def test_absent_actual_class_not_silently_treated_as_measured_recall():
    result = _matrix(
        [{"prediction": "DOWN", "actualDirection": "UP"}],
        actual_key="actualDirection",
    )
    assert result["perClass"]["DOWN"]["recall"] is None
    assert result["balancedAccuracyPresentClasses"] == 0
    assert result["balancedAccuracyAllThreeClasses"] == 0


def test_empty_cohort_has_undefined_accuracy():
    result = _matrix([], actual_key="actualDirection")
    assert result["accuracy"] is None
    assert result["balancedAccuracyPresentClasses"] is None
    assert all(value is None for value in result["alwaysBaselines"].values())


def test_output_rejects_overwriting_inputs_and_previous_diagnostics(tmp_path: Path):
    research = tmp_path / "research"
    research.mkdir()
    with pytest.raises(ValueError, match="outside protected"):
        _safe_dir(research / "audit", exclude=[research])
    output = tmp_path / "audit"
    _safe_dir(output, exclude=[research])
    (output / "failure-analysis.json").write_text("original", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        _safe_dir(output, exclude=[research])
    assert (output / "failure-analysis.json").read_text(encoding="utf-8") == "original"
