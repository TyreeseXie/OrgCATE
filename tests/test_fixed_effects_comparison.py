from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from orgcate_sim.fixed_effects_comparison import (
    build_auxiliary_panel,
    fit_two_way_fixed_effects,
    summarize_fixed_effects_comparison,
)


def test_two_way_fixed_effects_recovers_constant_effect_without_noise() -> None:
    treated = np.array([0, 0, 1, 1], dtype=int)
    post = np.array([False, False, True, True])
    firm = np.array([1.0, -0.5, 0.2, 1.4])[:, None]
    period = np.array([-0.2, 0.0, 0.1, 0.4])[None, :]
    outcome = firm + period + 0.75 * treated[:, None] * post[None, :]
    result = fit_two_way_fixed_effects(outcome, treated, post)
    assert result["estimate"] == pytest.approx(0.75)
    assert result["standard_error"] == pytest.approx(0.0, abs=1e-12)


def test_auxiliary_panel_preserves_declared_dimensions_and_effect() -> None:
    config = {
        "relative_periods": [-2, -1, 0, 1],
        "post_period_starts_at": 0,
        "period_effects": [-0.1, 0.0, 0.1, 0.2],
        "panel_noise_standard_deviation": 0.0,
        "panel_noise_ar1": 0.0,
    }
    outcome, post = build_auxiliary_panel(
        np.array([1.0, 2.0]),
        np.array([0, 1]),
        np.array([0.5, 0.8]),
        config,
        seed=17,
    )
    assert outcome.shape == (2, 4)
    assert post.tolist() == [False, False, True, True]
    treated_difference = outcome[1] - outcome[0] - 1.0
    assert treated_difference[:2] == pytest.approx([0.0, 0.0])
    assert treated_difference[2:] == pytest.approx([0.8, 0.8])


def test_summary_contains_fixed_effects_dml_and_cate() -> None:
    long = pd.DataFrame(
        {
            "situation": ["inverted-u", "inverted-u"],
            "true_ate": [0.2, 0.2],
            "true_ate_train": [0.2, 0.2],
            "true_att_train": [0.25, 0.25],
            "true_ate_test": [0.21, 0.19],
            "fixed_effects_estimate": [0.24, 0.26],
            "fixed_effects_ci_covers_ate": [True, False],
            "fixed_effects_ci_covers_att": [True, True],
            "dml_ate": [0.21, 0.19],
            "cate_test_mean": [0.22, 0.18],
        }
    )
    row = summarize_fixed_effects_comparison(long).iloc[0]
    assert row["mean_fixed_effects_estimate"] == pytest.approx(0.25)
    assert row["mean_dml_ate"] == pytest.approx(0.20)
    assert row["mean_cate_test"] == pytest.approx(0.20)
    assert row["fixed_effects_bias_vs_att"] == pytest.approx(0.0)


def test_fixed_effects_freeze_protects_existing_results() -> None:
    root = Path(__file__).resolve().parents[1]
    report_dir = root / "reports" / "fixed_effects_comparison_v1"
    freeze = json.loads(
        (report_dir / "FIXED_EFFECTS_COMPARISON_V1_METHOD_FREEZE.json").read_text(
            encoding="utf-8"
        )
    )
    for record in freeze["frozen_inputs"].values():
        observed = hashlib.sha256((root / record["path"]).read_bytes()).hexdigest()
        assert observed == record["sha256"]

    manifest = json.loads((report_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["new_causal_forests_fitted"] == 0
    assert manifest["frozen_dml_and_cate_rows_reused"] == 90
