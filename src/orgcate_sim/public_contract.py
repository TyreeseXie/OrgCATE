"""Small, config-driven gates for the public OrgCATE workflow.

These functions deliberately use analyst-side estimates only.  They do not
fit a second forest and they never accept evaluator truth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .theory_contract import validate_no_oracle_fields


def evaluate_design_eligibility(
    treatment: Sequence[int],
    propensity: Mapping[str, float],
    clip_fraction: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the pre-morphology treatment and overlap gate."""

    t = np.asarray(treatment, dtype=int)
    spec = config["eligibility"]
    reasons: list[str] = []
    if t.size == 0 or not np.isin(t, [0, 1]).all() or np.unique(t).size != 2:
        reasons.append("treatment must be binary with both arms present")
    rate = float(np.mean(t)) if t.size else float("nan")
    if np.isfinite(rate) and not (
        float(spec["treatment_rate_min"]) <= rate <= float(spec["treatment_rate_max"])
    ):
        reasons.append("treatment prevalence is outside the configured support")
    q01 = float(propensity.get("propensity_q01_oof", float("nan")))
    q99 = float(propensity.get("propensity_q99_oof", float("nan")))
    if not np.isfinite(q01) or not np.isfinite(q99):
        reasons.append("out-of-fold propensity support is unavailable")
    else:
        if q01 < float(spec["propensity_q01_min"]):
            reasons.append("lower propensity support is below the configured minimum")
        if q99 > float(spec["propensity_q99_max"]):
            reasons.append("upper propensity support is above the configured maximum")
    clipped = float(clip_fraction)
    if not np.isfinite(clipped) or clipped > float(
        spec["propensity_clip_fraction_max"]
    ):
        reasons.append("propensity clipping exceeds the configured maximum")
    record = {
        "eligible": not reasons,
        "treatment_rate": rate,
        "propensity_q01_oof": q01,
        "propensity_q99_oof": q99,
        "propensity_clip_fraction": clipped,
        "thresholds": {
            "treatment_rate_min": float(spec["treatment_rate_min"]),
            "treatment_rate_max": float(spec["treatment_rate_max"]),
            "propensity_q01_min": float(spec["propensity_q01_min"]),
            "propensity_q99_max": float(spec["propensity_q99_max"]),
            "propensity_clip_fraction_max": float(spec["propensity_clip_fraction_max"]),
        },
        "reasons": reasons,
    }
    validate_no_oracle_fields(record)
    return record


def conventional_ate_diagnostics(
    x_train: np.ndarray,
    y_train: Sequence[float],
    t_train: Sequence[int],
    dml_ate: float,
) -> dict[str, Any]:
    """Place naive and covariate-adjusted OLS estimates beside cross-fitted DML."""

    x = np.asarray(x_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    t = np.asarray(t_train, dtype=float)
    if x.ndim != 2 or len(y) != len(t) or len(y) != len(x):
        raise ValueError("ATE diagnostic arrays have incompatible shapes")
    naive = float(y[t == 1].mean() - y[t == 0].mean())
    design = np.column_stack([np.ones(len(t)), t, x])
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    adjusted = float(coefficient[1])
    residual = y - design @ coefficient
    xtx_inverse = np.linalg.pinv(design.T @ design)
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inverse, design)
    hc3_residual = residual / np.maximum(1.0 - leverage, 1e-8)
    score_matrix = design * hc3_residual[:, None]
    hc3_covariance = xtx_inverse @ (score_matrix.T @ score_matrix) @ xtx_inverse
    adjusted_se = float(np.sqrt(max(hc3_covariance[1, 1], 0.0)))
    normal_critical_value = 1.959963984540054
    dml = float(dml_ate)
    gap = adjusted - dml
    denominator = abs(dml)
    relative_deviation = abs(gap) / denominator if denominator > 1e-12 else None
    record = {
        "dml_ate": dml,
        "naive_difference": naive,
        "adjusted_ols": adjusted,
        "adjusted_ols_standard_error_hc3": adjusted_se,
        "adjusted_ols_ci_low": adjusted - normal_critical_value * adjusted_se,
        "adjusted_ols_ci_high": adjusted + normal_critical_value * adjusted_se,
        "conventional_estimator": "covariate-adjusted constant-treatment-coefficient OLS",
        "naive_minus_dml": naive - dml,
        "adjusted_ols_minus_dml": gap,
        "absolute_adjusted_ols_dml_gap": abs(gap),
        "adjusted_ols_dml_relative_deviation": relative_deviation,
        "adjusted_ols_dml_relative_deviation_percent": (
            100.0 * relative_deviation if relative_deviation is not None else None
        ),
        "adjustment_count": int(x.shape[1]),
        "method_note": "DML/AIPW is the primary ATE estimate; OLS is the conventional benchmark",
    }
    validate_no_oracle_fields(record)
    return record


def evaluate_ate_cate_coherence(
    dml_ate: float,
    cate_train_target_mean: float,
    y_train: Sequence[float],
    config: Mapping[str, Any],
    *,
    descriptive_cate_test_mean: float | None = None,
) -> dict[str, Any]:
    """Apply the frozen standardized ATE-CATE failure test."""

    y = np.asarray(y_train, dtype=float)
    scale = float(np.std(y, ddof=1))
    gap = float(cate_train_target_mean) - float(dml_ate)
    ratio = abs(gap) / scale if scale > 0 else float("inf")
    limit = float(config["ate_cate"]["max_standardized_gap"])
    record = {
        "passed": bool(np.isfinite(ratio) and ratio <= limit),
        "dml_ate_train": float(dml_ate),
        "final_cate_train_target_mean": float(cate_train_target_mean),
        "descriptive_cate_test_mean": float(
            cate_train_target_mean
            if descriptive_cate_test_mean is None
            else descriptive_cate_test_mean
        ),
        "signed_gap": gap,
        "outcome_sd_train": scale,
        "standardized_absolute_gap": ratio,
        "threshold": limit,
        "sensitivity_thresholds": [
            float(value) for value in config["ate_cate"]["sensitivity_thresholds"]
        ],
        "rule": "abs(CATE_TRAIN_TARGET - DML_ATE_TRAIN) / SD(Y_TRAIN)",
        "failure_state": "INCONCLUSIVE",
    }
    validate_no_oracle_fields(record)
    return record


def standardize_curve_to_target(
    curve_u: Sequence[float],
    curve_cate: Sequence[float],
    target_u: Sequence[float],
) -> float:
    """Average the reported CATE curve over a declared target distribution."""

    support = np.asarray(curve_u, dtype=float)
    effects = np.asarray(curve_cate, dtype=float)
    target = np.asarray(target_u, dtype=float)
    if (
        support.ndim != 1
        or effects.ndim != 1
        or target.ndim != 1
        or len(support) != len(effects)
        or len(support) < 2
        or len(target) == 0
        or not np.isfinite(support).all()
        or not np.isfinite(effects).all()
        or not np.isfinite(target).all()
        or not np.all(np.diff(support) > 0)
    ):
        raise ValueError("invalid CATE curve or target distribution")
    standardized = np.interp(target, support, effects)
    return float(np.mean(standardized))


_CELL = {
    ("continuous", "linear", "same"): 1,
    ("continuous", "linear", "opposite"): 2,
    ("continuous", "nonlinear", "same"): 3,
    ("continuous", "nonlinear", "opposite"): 4,
    ("continuous", "non-monotone", "same"): 5,
    ("continuous", "non-monotone", "opposite"): 6,
    ("discontinuous", "linear", "same"): 7,
    ("discontinuous", "linear", "opposite"): 8,
    ("discontinuous", "nonlinear", "same"): 9,
    ("discontinuous", "nonlinear", "opposite"): 10,
    ("discontinuous", "non-monotone", "same"): 11,
    ("discontinuous", "non-monotone", "opposite"): 12,
}


def three_layer_taxonomy(
    label: str,
    curve_values: Sequence[float],
    ate: float,
    *,
    selected_family: str = "",
) -> dict[str, Any]:
    """Map the readable shape label to the prototype's three-layer cell."""

    if label == "undetermined":
        return {
            "continuity": "undetermined",
            "monotonicity": "undetermined",
            "sign_relation": "undetermined",
            "taxonomy_cell": 0,
            "requires_abduction": False,
            "classification_basis": "upstream gate withheld classification",
        }
    continuity = "discontinuous" if label == "threshold-step" else "continuous"
    if label == "homogeneous":
        monotonicity = "linear"
    elif label in {"inverted-u", "u-shaped"}:
        monotonicity = "non-monotone"
    elif label in {"linear-monotone"} or selected_family in {
        "constant",
        "linear",
        "step",
        "step1",
        "step2",
        "step3",
    }:
        monotonicity = "linear"
    else:
        monotonicity = "nonlinear"
    values = np.asarray(curve_values, dtype=float)
    scale = 1.0 if abs(float(ate)) < 1e-12 else np.sign(float(ate))
    opposite = bool(np.any(values * scale < -1e-10)) if values.size else False
    if label in {"homogeneous", "inverted-u", "u-shaped"}:
        sign_relation = "same"
    else:
        sign_relation = "opposite" if opposite or label == "sign-reversal" else "same"
    cell = _CELL.get((continuity, monotonicity, sign_relation), 0)
    record = {
        "continuity": continuity,
        "monotonicity": monotonicity,
        "sign_relation": sign_relation,
        "taxonomy_cell": int(cell),
        "requires_abduction": bool(sign_relation == "opposite"),
        "classification_basis": "readable shape label plus practical CATE curve sign check",
    }
    validate_no_oracle_fields(record)
    return record


def property_support(
    label: str,
    practical_state: str,
    coherence: Mapping[str, Any],
) -> dict[str, Any]:
    """Declare which claims are licensed by the validated practical output."""

    supported_state = practical_state == "SUPPORTED_CANDIDATE"
    coherent = bool(coherence.get("passed", False))
    support = {
        "ordering": supported_state,
        "magnitude": supported_state and coherent,
        "curvature": supported_state
        and label
        in {
            "nonlinear-monotone",
            "steep-continuous",
            "inverted-u",
            "u-shaped",
            "kinked",
        },
        "boundaries": supported_state and label == "threshold-step",
    }
    requirements = {
        "homogeneous": [],
        "linear-monotone": ["ordering"],
        "nonlinear-monotone": ["ordering", "curvature"],
        "steep-continuous": ["ordering", "magnitude", "curvature"],
        "inverted-u": ["ordering", "curvature"],
        "u-shaped": ["ordering", "curvature"],
        "kinked": ["ordering", "curvature"],
        "threshold-step": ["ordering", "magnitude", "boundaries"],
        "sign-reversal": ["ordering", "magnitude"],
    }
    record = {
        "ordering": bool(support["ordering"]),
        "magnitude": bool(support["magnitude"]),
        "curvature": bool(support["curvature"]),
        "boundaries": bool(support["boundaries"]),
        "shape_requirements": requirements.get(label, []),
        "claim_allowed": bool(
            all(support[item] for item in requirements.get(label, []))
        ),
        "reason": "requires SUPPORTED_CANDIDATE and the relevant validated property",
    }
    validate_no_oracle_fields(record)
    return record


def _median_scalar(details: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [
        float(item[key]) for item in details if key in item and np.isfinite(item[key])
    ]
    return float(np.median(values)) if values else None


def _median_vectors(details: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    vectors = [np.asarray(item[key], dtype=float) for item in details if key in item]
    if not vectors or len({len(value) for value in vectors}) != 1:
        return []
    return np.median(np.vstack(vectors), axis=0).astype(float).tolist()


def _direction(value: float | None) -> str:
    if value is None or abs(value) <= 1e-12:
        return "approximately flat"
    return "increasing" if value > 0 else "decreasing"


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or float(np.ptp(x)) <= 1e-12:
        return None
    centered = x - float(np.mean(x))
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-12:
        return None
    return float(np.dot(centered, y - float(np.mean(y))) / denominator)


def _slope_at(x: np.ndarray, y: np.ndarray, location: float | None) -> float | None:
    if location is None or len(x) < 3:
        return None
    derivative = np.gradient(y, x)
    return float(np.interp(location, x, derivative))


def _format_number(value: float | None) -> str:
    return "not available" if value is None else f"{value:.3f}"


def summarize_shape_quantities_en(
    label: str,
    subtype: str,
    details: Sequence[Mapping[str, Any]],
    curve_u: Sequence[float],
    curve_values: Sequence[float],
) -> dict[str, Any]:
    """Create one English, researcher-facing summary for the selected shape."""

    u = np.asarray(curve_u, dtype=float)
    curve = np.asarray(curve_values, dtype=float)
    if len(u) != len(curve) or not len(u) or not np.isfinite(np.c_[u, curve]).all():
        raise ValueError("a finite non-empty morphology curve is required")
    order = np.argsort(u)
    u = u[order]
    curve = curve[order]
    records = [dict(item) for item in details]
    quantities: dict[str, Any] = {
        "cate_mean_on_reported_grid": float(np.mean(curve)),
        "cate_min_on_reported_grid": float(np.min(curve)),
        "cate_max_on_reported_grid": float(np.max(curve)),
        "cate_range_on_reported_grid": float(np.ptp(curve)),
        "effect_at_lower_support": float(curve[0]),
        "effect_at_upper_support": float(curve[-1]),
    }

    if label == "homogeneous":
        level = _median_scalar(records, "level")
        slope = _linear_slope(u, curve)
        quantities.update(
            constant_effect_level_median_across_refits=level,
            global_slope_over_support=slope,
        )
        interpretation = (
            "The validated CATE surface is approximately constant: "
            f"level={_format_number(level)}, global slope={_format_number(slope)}, "
            f"and reported CATE range={float(np.ptp(curve)):.3f}."
        )
    elif label == "linear-monotone":
        slope = _median_scalar(records, "slope")
        quantities.update(
            slope_median_across_refits=slope,
            effect_direction=_direction(slope),
        )
        interpretation = (
            f"CATE is {_direction(slope)} across the moderator: linear "
            f"slope={_format_number(slope)}, lower-support effect={curve[0]:.3f}, "
            f"and upper-support effect={curve[-1]:.3f}."
        )
    elif label in {"nonlinear-monotone", "steep-continuous"}:
        center = _median_scalar(records, "center")
        steepness = _median_scalar(records, "k")
        amplitude = _median_scalar(records, "amplitude")
        transition_width = (
            float(2.0 * np.arctanh(0.5) / steepness)
            if steepness is not None and steepness > 0
            else None
        )
        effective_gradient = (
            float(abs(amplitude) * steepness)
            if amplitude is not None and steepness is not None
            else None
        )
        center_slope = _slope_at(u, curve, center)
        lower_slope = _slope_at(u, curve, float(u[0]))
        upper_slope = _slope_at(u, curve, float(u[-1]))
        quantities.update(
            transition_center_median_across_refits=center,
            steepness_k_median_across_refits=steepness,
            amplitude_median_across_refits=amplitude,
            central_transition_width_25_75=transition_width,
            effective_center_gradient=effective_gradient,
            effect_direction=_direction(amplitude),
            slope_at_lower_support=lower_slope,
            slope_at_transition_center=center_slope,
            slope_at_upper_support=upper_slope,
        )
        kind = (
            "sharp but continuous"
            if label == "steep-continuous"
            else "nonlinear monotone"
        )
        interpretation = (
            f"CATE follows a {kind} {_direction(amplitude)} pattern: transition "
            f"center={_format_number(center)}, amplitude={_format_number(amplitude)}, "
            f"25%-75% transition width={_format_number(transition_width)}, effective "
            f"center gradient={_format_number(effective_gradient)}, and slopes at the lower support, "
            f"center, and upper support are {_format_number(lower_slope)}, "
            f"{_format_number(center_slope)}, and {_format_number(upper_slope)}. "
            f"Raw k={_format_number(steepness)} is retained as a diagnostic; endpoint "
            f"effects are {curve[0]:.3f} and {curve[-1]:.3f}."
        )
    elif label in {"inverted-u", "u-shaped"}:
        turning = _median_scalar(records, "turning_point")
        curvature = _median_scalar(records, "curvature")
        extremum = float(np.interp(turning, u, curve)) if turning is not None else None
        left_effect = float(curve[0])
        right_effect = float(curve[-1])
        below = u <= turning if turning is not None else np.zeros(len(u), dtype=bool)
        above = u >= turning if turning is not None else np.zeros(len(u), dtype=bool)
        left_slope = _linear_slope(u[below], curve[below])
        right_slope = _linear_slope(u[above], curve[above])
        quantities.update(
            turning_point_median_across_refits=turning,
            curvature_median_across_refits=curvature,
            effect_at_turning_point=extremum,
            effect_at_lower_support=left_effect,
            effect_at_upper_support=right_effect,
            turning_point_minus_lower_support=(
                extremum - left_effect if extremum is not None else None
            ),
            turning_point_minus_upper_support=(
                extremum - right_effect if extremum is not None else None
            ),
            slope_below_turning_point=left_slope,
            slope_above_turning_point=right_slope,
        )
        name = "peak" if label == "inverted-u" else "trough"
        lower_contrast = extremum - left_effect if extremum is not None else None
        upper_contrast = extremum - right_effect if extremum is not None else None
        interpretation = (
            f"CATE has an interior {name}: turning point={_format_number(turning)}, "
            f"curvature={_format_number(curvature)}, effect at the turning point="
            f"{_format_number(extremum)}, and lower/upper-support effects="
            f"{left_effect:.3f}/{right_effect:.3f}. Turning-point-minus-endpoint "
            f"contrasts are {_format_number(lower_contrast)} and "
            f"{_format_number(upper_contrast)}; fitted slopes below and above the "
            f"turning point are {_format_number(left_slope)} and "
            f"{_format_number(right_slope)}."
        )
    elif label == "kinked":
        kink = _median_scalar(records, "kink")
        if kink is None:
            kink = _median_scalar(records, "center")
        left_slope = _median_scalar(records, "left_slope")
        right_slope = _median_scalar(records, "right_slope")
        slope_change = _median_scalar(records, "slope_change")
        if slope_change is None and left_slope is not None and right_slope is not None:
            slope_change = right_slope - left_slope
        effect_at_kink = float(np.interp(kink, u, curve)) if kink is not None else None
        quantities.update(
            kink_location_median_across_refits=kink,
            left_slope_median_across_refits=left_slope,
            right_slope_median_across_refits=right_slope,
            slope_change_median_across_refits=slope_change,
            left_segment_direction=_direction(left_slope),
            right_segment_direction=_direction(right_slope),
            effect_at_kink=effect_at_kink,
        )
        interpretation = (
            "CATE remains continuous but changes slope: kink location="
            f"{_format_number(kink)}, effect at the kink={_format_number(effect_at_kink)}, "
            f"left slope={_format_number(left_slope)}, right slope="
            f"{_format_number(right_slope)}, and slope change="
            f"{_format_number(slope_change)}. The left/right segments are "
            f"{_direction(left_slope)}/{_direction(right_slope)}; endpoint effects "
            f"are {curve[0]:.3f} and {curve[-1]:.3f}."
        )
    elif label == "threshold-step":
        boundaries = _median_vectors(records, "boundaries")
        levels = _median_vectors(records, "levels")
        jumps = _median_vectors(records, "jumps")
        if not boundaries:
            boundary = _median_scalar(records, "boundary")
            boundaries = [boundary] if boundary is not None else []
        if not jumps:
            jump = _median_scalar(records, "jump")
            jumps = [jump] if jump is not None else []
        if not levels:
            left = _median_scalar(records, "left_effect")
            right = _median_scalar(records, "right_effect")
            levels = [left, right] if left is not None and right is not None else []
        quantities.update(
            boundary_count=len(boundaries),
            boundary_locations_median_across_refits=boundaries,
            band_effects_median_across_refits=levels,
            jumps_median_across_refits=jumps,
            boundary_claim_scope="operational candidate boundary with localized two-side contrast",
        )
        boundary_text = ", ".join(f"{value:.3f}" for value in boundaries)
        level_text = ", ".join(f"{value:.3f}" for value in levels)
        jump_text = ", ".join(f"{value:.3f}" for value in jumps)
        interpretation = (
            f"CATE forms {len(levels)} effect bands separated by {len(boundaries)} "
            f"candidate boundary/boundaries at {boundary_text}. Band effects are "
            f"[{level_text}], and adjacent-band jumps are [{jump_text}]. These are "
            "operational candidate boundaries with localized two-side effect contrasts."
            if boundaries
            else "CATE contains a threshold-like candidate boundary."
        )
        if subtype:
            quantities["threshold_subtype"] = subtype
    elif label == "sign-reversal":
        crossing = _median_scalar(records, "crossing")
        lower = float(curve[0])
        upper = float(curve[-1])
        global_slope = _linear_slope(u, curve)
        below = u <= crossing if crossing is not None else np.ones(len(u), dtype=bool)
        above = u >= crossing if crossing is not None else np.ones(len(u), dtype=bool)
        lower_slope = _linear_slope(u[below], curve[below])
        upper_slope = _linear_slope(u[above], curve[above])
        quantities.update(
            zero_crossing_median_across_refits=crossing,
            effect_at_lower_support=lower,
            effect_at_upper_support=upper,
            sign_below_crossing="positive"
            if lower > 0
            else "negative"
            if lower < 0
            else "zero",
            sign_above_crossing="positive"
            if upper > 0
            else "negative"
            if upper < 0
            else "zero",
            global_slope_over_support=global_slope,
            slope_below_crossing=lower_slope,
            slope_above_crossing=upper_slope,
        )
        interpretation = (
            f"CATE crosses zero near {_format_number(crossing)}. The lower-support "
            f"effect is {lower:.3f} ({quantities['sign_below_crossing']}) and the "
            f"upper-support effect is {upper:.3f} "
            f"({quantities['sign_above_crossing']}). The global slope is "
            f"{_format_number(global_slope)}; fitted slopes below and above the "
            f"crossing are {_format_number(lower_slope)} and "
            f"{_format_number(upper_slope)}."
        )
    else:
        quantities = {}
        interpretation = (
            "No shape-specific quantities are reported because morphology is withheld."
        )

    record = {"key_quantities": quantities, "interpretation_en": interpretation}
    validate_no_oracle_fields(record)
    return record


def theory_audit_template(practical: Mapping[str, Any]) -> dict[str, Any]:
    """Create a blank interpretation record for the researcher."""

    morphology = practical.get("morphology", {})
    template = {
        "schema_version": "orgcate-theory-audit-v1",
        "status": "PENDING_RESEARCHER_REVIEW",
        "candidate_moderator": morphology.get(
            "candidate_moderator",
            practical.get("moderator", {}).get("focal_characteristic"),
        ),
        "morphology_label": morphology.get("label"),
        "empirical_pattern": "",
        "candidate_mechanisms": [],
        "supporting_sources": [],
        "rejected_mechanisms": [],
        "researcher_cues": [],
        "unresolved_readings": [],
        "adoption_decision": "PENDING",
        "decision_rationale": "",
        "researcher_name": "",
        "review_date": "",
    }
    validate_no_oracle_fields(template)
    return template


def validate_theory_audit(record: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "status",
        "candidate_moderator",
        "morphology_label",
        "empirical_pattern",
        "candidate_mechanisms",
        "supporting_sources",
        "rejected_mechanisms",
        "researcher_cues",
        "unresolved_readings",
        "adoption_decision",
        "decision_rationale",
        "researcher_name",
        "review_date",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"theory audit missing fields: {missing}")
    if record["schema_version"] != "orgcate-theory-audit-v1":
        raise ValueError("wrong theory audit schema")
    if record["status"] not in {"PENDING_RESEARCHER_REVIEW", "REVIEWED"}:
        raise ValueError("invalid theory audit status")
    if record["adoption_decision"] not in {"PENDING", "ADOPT", "ABSTAIN", "REVISE"}:
        raise ValueError("invalid theory adoption decision")
    if not isinstance(record["candidate_mechanisms"], list):
        raise TypeError("candidate_mechanisms must be a list")
    validate_no_oracle_fields(record)
