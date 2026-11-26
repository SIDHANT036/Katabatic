"""Numeric columns returned as text must not break saving or consistency."""

import sys
from pathlib import Path

import pandas as pd

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from runner import align_numeric_dtypes, save_synthetic  # noqa: E402

from katabatic.evaluate.consistency.evaluation import ConsistencyEvaluation  # noqa: E402


def _train():
    return pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "colour": ["r", "g", "b", "r"],
            "label": ["a", "b", "a", "b"],
        }
    )


def test_text_numbers_become_numeric():
    synthetic = pd.DataFrame(
        {"age": ["21", "35", "44"], "colour": ["r", "g", "b"], "label": list("aba")}
    )
    out = align_numeric_dtypes(synthetic, _train(), categorical_cols=["colour"])
    assert pd.api.types.is_numeric_dtype(out["age"])
    assert out["age"].tolist() == [21, 35, 44]
    assert out["colour"].tolist() == ["r", "g", "b"]  # categoricals untouched


def test_unparseable_values_become_nan_and_are_reported(capsys):
    synthetic = pd.DataFrame(
        {"age": ["21", "oops"], "colour": ["r", "g"], "label": ["a", "b"]}
    )
    out = align_numeric_dtypes(synthetic, _train(), categorical_cols=["colour"])
    assert out["age"].isna().tolist() == [False, True]
    assert "1 non-numeric values in 'age'" in capsys.readouterr().out


def test_numeric_columns_pass_through_unchanged():
    synthetic = pd.DataFrame(
        {"age": [21.5, 30.0], "colour": ["r", "g"], "label": ["a", "b"]}
    )
    out = align_numeric_dtypes(synthetic, _train(), categorical_cols=["colour"])
    pd.testing.assert_frame_equal(out, synthetic)


def test_save_synthetic_returns_numeric_age(tmp_path):
    synthetic = pd.DataFrame(
        {"age": ["21", "35"], "colour": ["r", "g"], "label": ["a", "b"]}
    )
    paths = {"synthetic_dir": str(tmp_path)}
    out = save_synthetic(synthetic, _train(), paths, categorical_cols=["colour"])
    assert pd.api.types.is_numeric_dtype(out["age"])


def test_consistency_constraint_check_survives_text_column():
    real = pd.DataFrame({"age": range(20, 60), "label": ["a", "b"] * 20})
    synthetic = pd.DataFrame(
        {"age": ["25", "30", "999", "x"] * 10, "label": ["a", "b"] * 20}
    )
    ev = ConsistencyEvaluation(
        real, synthetic, target_col="label", constraints={"age": (17, 90)}, n_folds=2
    )
    violations = ev._compute_constraint_violations()
    # 999 is out of range and "x" is not a number: half the rows violate.
    assert violations["overall_violation_rate"] == 0.5
