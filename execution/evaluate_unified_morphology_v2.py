"""Evaluate fresh unified morphology v2 output under the frozen common standard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "unified_morphology_v2_acceptance.json"
sys.path.insert(0, str(ROOT / "src"))
from orgcate_sim import morphology_acceptance as acceptance_helpers


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(config_path: Path, value: str) -> Path:
    return (config_path.parent / value).resolve()


def validate_config(config: dict[str, Any]) -> None:
    if config["schema_version"] != "unified-morphology-v2-acceptance":
        raise ValueError("wrong acceptance schema")
    if config["thresholds"] != {
        "scenario_success_rate": 0.90,
        "curve_spearman": 0.95,
        "normalized_curve_rmse": 0.20,
        "location_abs_error_ecdf": 0.10,
        "effect_quantity_error_fraction_truth_cate_range": 0.20,
        "dimensionless_parameter_relative_error": 0.20,
        "blind_curve_stability_spearman": 0.80,
    }:
        raise ValueError("common recovery thresholds changed")


def morphology_quantities(
    shape: str,
    details: list[dict[str, Any]],
    u: np.ndarray,
    estimate: np.ndarray,
    truth_curve: np.ndarray,
    truth_specification: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, dict[str, Any]]:
    if shape not in {"nonlinear-monotone", "steep-continuous", "kinked"}:
        return acceptance_helpers.single_quantities(
            shape,
            details,
            u,
            estimate,
            truth_curve,
            truth_specification,
            thresholds,
        )
    truth_range = float(np.ptp(truth_curve))
    checks: dict[str, dict[str, Any]] = {}
    if shape in {"nonlinear-monotone", "steep-continuous"}:
        specifications = (
            ("center", "location"),
            ("amplitude", "effect"),
            ("transition_width_25_75", "location"),
        )
    else:
        specifications = (
            ("kink", "location"),
            ("left_slope", "side_slope"),
            ("right_slope", "side_slope"),
        )
    for key, kind in specifications:
        checks[key] = acceptance_helpers.quantity_check(
            acceptance_helpers.median_scalar(details, key),
            float(truth_specification[key]),
            kind,
            thresholds,
            truth_range,
        )
    return checks


def evaluate(config_path: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_config(config)
    results = pd.read_csv(resolve(config_path, config["results"]))
    curves = pd.read_csv(resolve(config_path, config["curves"]))
    rows: list[dict[str, Any]] = []
    for result in results.loc[results["status"] == "completed"].itertuples(index=False):
        if result.dgp == "multi-threshold-step":
            continue
        frame = curves.loc[
            (curves["dgp"] == result.dgp)
            & (curves["replication"] == int(result.replication))
        ].sort_values("u")
        u = frame["u"].to_numpy(float)
        truth = frame["truth_curve"].to_numpy(float)
        estimate = frame["estimate_curve"].to_numpy(float)
        curve = acceptance_helpers.curve_metrics(
            truth,
            estimate,
            result.dgp == "homogeneous",
            config["thresholds"],
        )
        quantities = morphology_quantities(
            result.dgp,
            json.loads(result.parameter_details_json),
            u,
            estimate,
            truth,
            config["truth"][result.dgp],
            config["thresholds"],
        )
        blind = bool(
            result.practical_state == "NO_THEORY_CLAIM"
            and result.predicted_label == "homogeneous"
            if result.dgp == "homogeneous"
            else result.practical_state == "SUPPORTED_CANDIDATE"
        )
        exact = bool(result.predicted_label == result.dgp)
        quantity_pass = bool(
            quantities and all(item["passed"] for item in quantities.values())
        )
        success = bool(blind and exact and curve["curve_fidelity_pass"] and quantity_pass)
        rows.append(
            {
                "scenario": result.dgp,
                "replication": int(result.replication),
                "blind_evidence_pass": blind,
                "exact_morphology_pass": exact,
                "curve_fidelity_pass": curve["curve_fidelity_pass"],
                "key_quantities_pass": quantity_pass,
                "recovery_success": success,
                "predicted_label": result.predicted_label,
                "practical_state": result.practical_state,
                "curve_stability_spearman": result.curve_stability_spearman,
                "key_quantity_checks_json": json.dumps(quantities, sort_keys=True),
                **curve,
            }
        )
    long = pd.DataFrame(rows)
    summary = (
        long.groupby("scenario", as_index=False, sort=False)
        .agg(
            replications=("replication", "size"),
            blind_evidence_rate=("blind_evidence_pass", "mean"),
            exact_morphology_rate=("exact_morphology_pass", "mean"),
            curve_fidelity_rate=("curve_fidelity_pass", "mean"),
            key_quantities_rate=("key_quantities_pass", "mean"),
            unified_recovery_rate=("recovery_success", "mean"),
            median_curve_spearman=("curve_spearman", "median"),
            median_normalized_rmse=("normalized_curve_rmse", "median"),
        )
    )
    summary["replications_complete"] = summary["replications"].eq(
        int(config["required_replications_per_scenario"])
    )
    summary["scenario_pass"] = summary["replications_complete"] & (
        summary["unified_recovery_rate"]
        >= float(config["thresholds"]["scenario_success_rate"])
    )
    return long, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    config = read_json(config_path)
    long, summary = evaluate(config_path, config)
    output = resolve(config_path, config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    long.to_csv(output / "UNIFIED_MORPHOLOGY_V2_ACCEPTANCE_LONG.csv", index=False)
    summary.to_csv(output / "UNIFIED_MORPHOLOGY_V2_ACCEPTANCE_SUMMARY.csv", index=False)
    confusion = pd.crosstab(long["scenario"], long["predicted_label"], dropna=False)
    confusion.to_csv(output / "UNIFIED_MORPHOLOGY_V2_CONFUSION_MATRIX.csv")
    decision = {
        "all_nine_pass": bool(len(summary) == 9 and summary["scenario_pass"].all()),
        "all_scenarios_complete": bool(summary["replications_complete"].all()),
        "required_replications_per_scenario": int(
            config["required_replications_per_scenario"]
        ),
        "scenario_replications": dict(
            zip(summary["scenario"], summary["replications"].astype(int), strict=True)
        ),
        "scenario_rates": dict(
            zip(summary["scenario"], summary["unified_recovery_rate"], strict=True)
        ),
        "thresholds": config["thresholds"],
        "scenario_specific_dispatch": False,
    }
    (output / "UNIFIED_MORPHOLOGY_V2_ACCEPTANCE.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    return 0 if decision["all_nine_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
