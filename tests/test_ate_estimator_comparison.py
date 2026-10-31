from __future__ import annotations

import pandas as pd
import pytest

from orgcate_sim.ate_comparison import summarize_estimator_comparison


def test_comparison_summary_reports_bias_and_relative_deviation() -> None:
    long = pd.DataFrame(
        {
            "situation": ["inverted-u", "inverted-u"],
            "true_ate": [0.20, 0.20],
            "ols_estimate": [0.25, 0.15],
            "ols_ci_covers_true_ate": [True, True],
            "dml_estimate": [0.21, 0.19],
            "dml_ci_covers_true_ate": [True, True],
            "ols_dml_relative_deviation": [0.04 / 0.21, 0.04 / 0.19],
        }
    )
    row = summarize_estimator_comparison(long).iloc[0]
    assert row["ols_bias"] == pytest.approx(0.0)
    assert row["dml_bias"] == pytest.approx(0.0)
    assert row["ols_rmse"] == pytest.approx(0.05)
    assert row["dml_rmse"] == pytest.approx(0.01)
    assert row["ols_95ci_coverage"] == 1.0
    assert row["dml_95ci_coverage"] == 1.0
