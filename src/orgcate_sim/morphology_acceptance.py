"""Common curve and key-quantity checks for morphology acceptance."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr


def median_scalar(details: list[dict[str, Any]], key: str) -> float:
    values = [
        float(item[key])
        for item in details
        if key in item and np.isfinite(float(item[key]))
    ]
    return float(np.median(values)) if values else float("nan")


def median_vector(details: list[dict[str, Any]], key: str) -> list[float]:
    values = [np.asarray(item[key], dtype=float) for item in details if key in item]
    if not values or len({len(value) for value in values}) != 1:
        return []
    return np.median(np.vstack(values), axis=0).astype(float).tolist()


def linear_slope(u: np.ndarray, curve: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(u)), u])
    return float(np.linalg.lstsq(design, curve, rcond=None)[0][1])


def curve_metrics(
    truth: np.ndarray,
    estimate: np.ndarray,
    homogeneous: bool,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    rmse = float(np.sqrt(np.mean((estimate - truth) ** 2)))
    scale = max(float(np.ptp(truth)), abs(float(np.mean(truth))), 1e-12)
    normalized_rmse = rmse / scale
    correlation = (
        float("nan")
        if homogeneous
        else float(spearmanr(truth, estimate).statistic)
    )
    rank_pass = bool(
        homogeneous or correlation >= float(thresholds["curve_spearman"])
    )
    magnitude_pass = bool(
        normalized_rmse <= float(thresholds["normalized_curve_rmse"])
    )
    return {
        "curve_spearman": correlation,
        "curve_rmse": rmse,
        "normalized_curve_rmse": normalized_rmse,
        "curve_rank_pass": rank_pass,
        "curve_magnitude_pass": magnitude_pass,
        "curve_fidelity_pass": bool(rank_pass and magnitude_pass),
    }


def quantity_check(
    estimate: float,
    truth: float,
    kind: str,
    thresholds: dict[str, float],
    truth_cate_range: float,
) -> dict[str, Any]:
    error = (
        abs(float(estimate) - float(truth))
        if np.isfinite(estimate)
        else float("inf")
    )
    if kind == "location":
        scaled_error = error
        limit = float(thresholds["location_abs_error_ecdf"])
    elif kind == "relative":
        scaled_error = error / max(abs(float(truth)), 1e-12)
        limit = float(thresholds["dimensionless_parameter_relative_error"])
    else:
        scale = max(float(truth_cate_range), 1e-12)
        multipliers = {
            "effect": 1.0,
            "slope": 0.80,
            "side_slope": 0.40,
            "curvature": 0.08,
        }
        if kind not in multipliers:
            raise ValueError(f"unsupported quantity kind: {kind}")
        scaled_error = error * multipliers[kind] / scale
        limit = float(
            thresholds["effect_quantity_error_fraction_truth_cate_range"]
        )
    return {
        "estimate": float(estimate),
        "truth": float(truth),
        "error": error,
        "scaled_error": scaled_error,
        "kind": kind,
        "limit": limit,
        "passed": bool(np.isfinite(scaled_error) and scaled_error <= limit),
    }


def single_quantities(
    shape: str,
    details: list[dict[str, Any]],
    u: np.ndarray,
    estimate: np.ndarray,
    truth_curve: np.ndarray,
    truth_spec: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    truth_range = float(np.ptp(truth_curve))

    def add(name: str, value: float, kind: str = "effect") -> None:
        checks[name] = quantity_check(
            value,
            float(truth_spec[name]),
            kind,
            thresholds,
            truth_range,
        )

    if shape == "homogeneous":
        add("level", median_scalar(details, "level"), "relative")
    elif shape == "linear-monotone":
        add("slope", median_scalar(details, "slope"), "slope")
    elif shape in {"nonlinear-monotone", "steep-continuous"}:
        add("center", median_scalar(details, "center"), "location")
        add("k", median_scalar(details, "k"), "relative")
        add("amplitude", median_scalar(details, "amplitude"))
    elif shape in {"inverted-u", "u-shaped"}:
        add("turning_point", median_scalar(details, "turning_point"), "location")
        add("curvature", median_scalar(details, "curvature"), "curvature")
    elif shape == "kinked":
        add("center", median_scalar(details, "center"), "location")
        add("left_slope", median_scalar(details, "left_slope"), "side_slope")
        add("right_slope", median_scalar(details, "right_slope"), "side_slope")
    elif shape in {"threshold-step", "multi-threshold-step"}:
        boundaries = median_vector(details, "boundaries")
        levels = median_vector(details, "levels")
        jumps = median_vector(details, "jumps")
        for index, value in enumerate(truth_spec["boundaries"]):
            estimated = boundaries[index] if index < len(boundaries) else float("nan")
            checks[f"boundary_{index + 1}"] = quantity_check(
                estimated,
                float(value),
                "location",
                thresholds,
                truth_range,
            )
        for name, estimated_values, true_values in (
            ("level", levels, truth_spec["levels"]),
            ("jump", jumps, truth_spec["jumps"]),
        ):
            for index, value in enumerate(true_values):
                estimated = (
                    estimated_values[index]
                    if index < len(estimated_values)
                    else float("nan")
                )
                checks[f"{name}_{index + 1}"] = quantity_check(
                    estimated,
                    float(value),
                    "effect",
                    thresholds,
                    truth_range,
                )
    elif shape == "sign-reversal":
        add("crossing", median_scalar(details, "crossing"), "location")
        add("global_slope", linear_slope(u, estimate), "slope")
        add("lower_effect", float(estimate[0]))
        add("upper_effect", float(estimate[-1]))
    else:
        raise ValueError(shape)
    return checks
