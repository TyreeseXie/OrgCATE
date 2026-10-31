from __future__ import annotations

import numpy as np
import pytest

from orgcate_sim.public_contract import (
    conventional_ate_diagnostics,
    evaluate_ate_cate_coherence,
    evaluate_design_eligibility,
    property_support,
    standardize_curve_to_target,
    summarize_shape_quantities_en,
    theory_audit_template,
    three_layer_taxonomy,
    validate_theory_audit,
)
from orgcate_sim.release import validate_practical_result

CONFIG = {
    "eligibility": {
        "treatment_rate_min": 0.10,
        "treatment_rate_max": 0.90,
        "propensity_q01_min": 0.01,
        "propensity_q99_max": 0.99,
        "propensity_clip_fraction_max": 0.10,
    },
    "ate_cate": {
        "max_standardized_gap": 0.10,
        "sensitivity_thresholds": [0.05, 0.15],
    },
}


def test_eligibility_gate_routes_weak_overlap_before_morphology() -> None:
    result = evaluate_design_eligibility(
        [0, 1] * 50,
        {"propensity_q01_oof": 0.005, "propensity_q99_oof": 0.80},
        0.02,
        CONFIG,
    )
    assert not result["eligible"]
    assert any("lower propensity" in reason for reason in result["reasons"])


def test_ate_cate_gate_uses_train_outcome_scale_and_fixed_threshold() -> None:
    y = np.array([-1.0, 0.0, 1.0])
    passed = evaluate_ate_cate_coherence(0.20, 0.25, y, CONFIG)
    failed = evaluate_ate_cate_coherence(0.20, 0.31, y, CONFIG)
    assert passed["passed"]
    assert not failed["passed"]
    assert failed["failure_state"] == "INCONCLUSIVE"


def test_ate_cate_gate_standardizes_curve_to_dml_target_sample() -> None:
    curve_u = np.array([0.10, 0.50, 0.90])
    curve_cate = np.array([0.0, 1.0, 3.0])
    train_target = np.array([0.10, 0.10, 0.50, 0.90])
    test_target = np.array([0.90, 0.90])

    train_mean = standardize_curve_to_target(curve_u, curve_cate, train_target)
    test_mean = standardize_curve_to_target(curve_u, curve_cate, test_target)

    assert train_mean == pytest.approx(1.0)
    assert test_mean == pytest.approx(3.0)
    result = evaluate_ate_cate_coherence(
        1.0,
        train_mean,
        np.array([-1.0, 0.0, 1.0]),
        CONFIG,
        descriptive_cate_test_mean=test_mean,
    )
    assert result["passed"]
    assert result["final_cate_train_target_mean"] == pytest.approx(1.0)
    assert result["descriptive_cate_test_mean"] == pytest.approx(3.0)


def test_conventional_ate_diagnostics_report_ols_benchmark() -> None:
    x = np.arange(20, dtype=float).reshape(10, 2)
    t = np.array([0, 1] * 5)
    y = 0.5 * t + x[:, 0] * 0.01
    result = conventional_ate_diagnostics(x, y, t, dml_ate=0.49)
    assert result["dml_ate"] == 0.49
    assert result["adjustment_count"] == 2
    assert "DML/AIPW" in result["method_note"]
    assert result["conventional_estimator"].endswith("OLS")
    assert result["adjusted_ols_dml_relative_deviation"] >= 0
    assert result["adjusted_ols_standard_error_hc3"] >= 0
    assert result["adjusted_ols_ci_low"] <= result["adjusted_ols_ci_high"]


def test_taxonomy_and_support_preserve_shape_property_gate() -> None:
    taxonomy = three_layer_taxonomy(
        "sign-reversal", [-0.2, 0.0, 0.3], 0.1, selected_family="tanh"
    )
    assert taxonomy["taxonomy_cell"] == 4
    assert taxonomy["requires_abduction"]
    supported = property_support(
        "sign-reversal", "SUPPORTED_CANDIDATE", {"passed": True}
    )
    assert supported["claim_allowed"]
    withheld = property_support("sign-reversal", "INCONCLUSIVE", {"passed": True})
    assert not withheld["claim_allowed"]


def test_result_validator_enforces_state_precedence() -> None:
    record = {
        "schema_version": "orgcate-result-v1",
        "run_kind": "unit-test",
        "scenario": "sign-reversal",
        "practical_state": "SUPPORTED_CANDIDATE",
        "eligibility": {
            "eligible": True,
            "treatment_rate": 0.5,
            "propensity_q01_oof": 0.2,
            "propensity_q99_oof": 0.8,
            "propensity_clip_fraction": 0.0,
            "thresholds": {},
            "reasons": [],
        },
        "ate": {
            "method": "test",
            "estimate": 0.2,
            "standard_error": 0.1,
            "ci_low": 0.0,
            "ci_high": 0.4,
            "conventional_diagnostics": {},
        },
        "ate_cate_coherence": {
            "passed": False,
            "dml_ate_train": 0.2,
            "final_cate_train_target_mean": 0.5,
            "descriptive_cate_test_mean": 0.5,
            "signed_gap": 0.3,
            "outcome_sd_train": 1.0,
            "standardized_absolute_gap": 0.3,
            "threshold": 0.1,
            "sensitivity_thresholds": [0.05, 0.15],
            "rule": "test",
            "failure_state": "INCONCLUSIVE",
        },
        "forest": {},
        "moderator": {},
        "property_support": {
            "ordering": True,
            "magnitude": True,
            "curvature": False,
            "boundaries": False,
            "shape_requirements": ["ordering", "magnitude"],
            "claim_allowed": True,
            "reason": "test",
        },
        "morphology": {
            "label": "sign-reversal",
            "subtype": "",
            "selected_family": "tanh",
            "shape_quantities": [],
            "key_quantities": {},
            "interpretation_en": "test",
            "continuity": "continuous",
            "monotonicity": "nonlinear",
            "sign_relation": "opposite",
            "taxonomy_cell": 4,
            "requires_abduction": True,
            "classification_basis": "test",
        },
        "confirmation": {},
        "theory_audit": {
            "status": "PENDING_RESEARCHER_REVIEW",
            "artifact": "audit.json",
            "automated_theory_generation": False,
        },
        "audit": {},
    }
    with pytest.raises(ValueError, match="coherence"):
        validate_practical_result(record)


def test_researcher_audit_is_created_blank_and_validated() -> None:
    practical = {
        "moderator": {"focal_characteristic": "cash_at"},
        "morphology": {"label": "inverted-u"},
    }
    audit = theory_audit_template(practical)
    validate_theory_audit(audit)
    assert audit["status"] == "PENDING_RESEARCHER_REVIEW"
    assert audit["candidate_mechanisms"] == []
    broken = dict(audit)
    broken.pop("supporting_sources")
    with pytest.raises(ValueError, match="missing fields"):
        validate_theory_audit(broken)


@pytest.mark.parametrize(
    ("label", "subtype", "details", "expected_keys"),
    [
        (
            "homogeneous",
            "",
            [{"level": 0.2}, {"level": 0.22}],
            {
                "constant_effect_level_median_across_refits",
                "global_slope_over_support",
            },
        ),
        (
            "linear-monotone",
            "",
            [{"slope": 1.2}, {"slope": 1.0}],
            {"slope_median_across_refits", "effect_direction"},
        ),
        (
            "nonlinear-monotone",
            "",
            [{"center": 0.5, "k": 6.0, "amplitude": 1.0}],
            {
                "transition_center_median_across_refits",
                "steepness_k_median_across_refits",
                "central_transition_width_25_75",
                "effective_center_gradient",
                "amplitude_median_across_refits",
                "slope_at_lower_support",
                "slope_at_transition_center",
                "slope_at_upper_support",
            },
        ),
        (
            "steep-continuous",
            "",
            [{"center": 0.5, "k": 10.0, "amplitude": 1.0}],
            {
                "transition_center_median_across_refits",
                "steepness_k_median_across_refits",
                "central_transition_width_25_75",
                "effective_center_gradient",
                "amplitude_median_across_refits",
                "slope_at_lower_support",
                "slope_at_transition_center",
                "slope_at_upper_support",
            },
        ),
        (
            "inverted-u",
            "",
            [{"turning_point": 0.5, "curvature": -8.0}],
            {
                "turning_point_median_across_refits",
                "curvature_median_across_refits",
                "effect_at_turning_point",
                "slope_below_turning_point",
                "slope_above_turning_point",
            },
        ),
        (
            "u-shaped",
            "",
            [{"turning_point": 0.5, "curvature": 8.0}],
            {
                "turning_point_median_across_refits",
                "curvature_median_across_refits",
                "effect_at_turning_point",
                "slope_below_turning_point",
                "slope_above_turning_point",
            },
        ),
        (
            "kinked",
            "",
            [{"kink": 0.5, "left_slope": 0.5, "right_slope": 1.5}],
            {
                "kink_location_median_across_refits",
                "left_slope_median_across_refits",
                "right_slope_median_across_refits",
                "slope_change_median_across_refits",
                "effect_at_kink",
            },
        ),
        (
            "threshold-step",
            "single-threshold-step",
            [{"boundaries": [0.5], "levels": [0.2, 1.2], "jumps": [1.0]}],
            {
                "boundary_count",
                "boundary_locations_median_across_refits",
                "band_effects_median_across_refits",
                "jumps_median_across_refits",
            },
        ),
        (
            "sign-reversal",
            "",
            [{"crossing": 0.6}],
            {
                "zero_crossing_median_across_refits",
                "sign_below_crossing",
                "sign_above_crossing",
                "global_slope_over_support",
                "slope_below_crossing",
                "slope_above_crossing",
            },
        ),
    ],
)
def test_all_nine_shapes_have_explicit_english_key_outputs(
    label: str,
    subtype: str,
    details: list[dict[str, object]],
    expected_keys: set[str],
) -> None:
    u = np.linspace(0.1, 0.9, 9)
    curves = {
        "homogeneous": np.full(9, 0.2),
        "linear-monotone": u,
        "nonlinear-monotone": np.tanh(4 * (u - 0.5)),
        "steep-continuous": np.tanh(10 * (u - 0.5)),
        "inverted-u": -((u - 0.5) ** 2) + 1.0,
        "u-shaped": (u - 0.5) ** 2,
        "kinked": np.where(u < 0.5, 0.5 * u, 0.25 + 1.5 * (u - 0.5)),
        "threshold-step": np.where(u < 0.5, 0.2, 1.2),
        "sign-reversal": u - 0.6,
    }
    result = summarize_shape_quantities_en(label, subtype, details, u, curves[label])
    assert expected_keys.issubset(result["key_quantities"])
    assert {
        "effect_at_lower_support",
        "effect_at_upper_support",
        "cate_mean_on_reported_grid",
        "cate_range_on_reported_grid",
    }.issubset(result["key_quantities"])
    assert result["interpretation_en"].strip()
    if label != "threshold-step":
        assert "slope" in result["interpretation_en"].lower()
