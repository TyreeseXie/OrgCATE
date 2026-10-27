"""Run the property-first two/three-component coexistence refinement."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import traceback
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "additive_coexistence" / "shared" / "property_config.json"
BASE_RUNNER = ROOT / "additive_coexistence" / "src" / "additive_base.py"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_RUNNER, "coexistence_property_v2_base")
ORIGINAL_COMPONENT_RAW = BASE.component_raw


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(config_path: Path, value: str) -> Path:
    return (config_path.parent / value).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_grid(master_seed: int, replications: int) -> dict[int, int]:
    children = np.random.SeedSequence(int(master_seed)).spawn(int(replications))
    return {
        index: int(child.generate_state(1, dtype=np.uint32)[0])
        for index, child in enumerate(children)
    }


def validate_config(
    config_path: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if config["schema_version"] != "multi-hte-property-first-coexistence-v2":
        raise ValueError("wrong schema version")
    fixed = {
        "n": 3000,
        "backfit_iterations": 6,
        "same_data_nuisance_refits": 3,
        "nuisance_crossfit_folds": 5,
        "shape_cv_folds": 5,
    }
    for key, value in fixed.items():
        if int(config[key]) != value:
            raise ValueError(f"{key} must remain {value}")
    if not np.isclose(float(config["test_size"]), 0.20):
        raise ValueError("the external split must remain 80/20")
    if int(config["development"]["replications"]) != 5:
        raise ValueError("development requires five replications")
    if int(config["confirmation"]["primary_replications"]) != 20:
        raise ValueError("primary confirmation requires twenty replications")
    if int(config["confirmation"]["diagnostic_replications"]) != 10:
        raise ValueError("diagnostic confirmation requires ten replications")
    development_seeds = set(
        seed_grid(
            config["development"]["master_seed"],
            config["development"]["replications"],
        ).values()
    )
    confirmation_seeds = set(
        seed_grid(
            config["confirmation"]["master_seed"],
            config["confirmation"]["primary_replications"],
        ).values()
    )
    if development_seeds.intersection(confirmation_seeds):
        raise ValueError("development and confirmation seeds overlap")
    expected_policy = {
        "one_joint_multivariable_forest": True,
        "active_features_prespecified_for_reconstruction": True,
        "test_used_for_shape_selection": False,
        "oracle_used_for_shape_selection": False,
        "oracle_used_only_for_evaluation": True,
        "common_random_numbers_across_scenarios": True,
        "interactions_included": False,
    }
    if config["selection_policy"] != expected_policy:
        raise ValueError("selection policy changed")
    if list(config["scenarios"]) != [
        "primary_count_2",
        "primary_count_3",
        "stress_count_3_threshold",
        "control_count_3_steep",
    ]:
        raise ValueError("scenario set changed")

    source = read_json(resolve(config_path, config["source_config"]))
    method = read_json(resolve(config_path, config["method_config"]))
    expected_forest = {
        "class": "econml.dml.CausalForestDML",
        "n_estimators": 1000,
        "min_samples_leaf": 5,
        "max_samples": 0.45,
        "cv": 2,
        "honest": True,
        "inference": False,
        "n_jobs": 2,
    }
    if method["primary_forest"] != expected_forest:
        raise ValueError("primary forest contract changed")
    for name, scenario in config["scenarios"].items():
        components = scenario["components"]
        if len(components) != int(scenario["component_count"]):
            raise ValueError(f"{name} component count changed")
        features = [item["feature"] for item in components]
        if len(features) != len(set(features)) or not set(features).issubset(source["features"]):
            raise ValueError(f"{name} has invalid active features")
    if config["scenarios"]["primary_count_2"]["components"] != config["scenarios"]["primary_count_3"]["components"][:2]:
        raise ValueError("primary count stages are not nested")
    return source, method


def component_raw(u: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    value = np.asarray(u, dtype=float)
    shape = spec["shape"]
    if shape in {"linear", "inverted_u", "sign_reversal", "threshold"}:
        return ORIGINAL_COMPONENT_RAW(value, spec)
    if shape == "tanh":
        return float(spec["half_range"]) * np.tanh(
            float(spec["steepness"]) * (value - float(spec["center"]))
        )
    if shape == "kink":
        location = float(spec["location"])
        left = float(spec["left_slope"])
        right = float(spec["right_slope"])
        return left * (value - location) + (right - left) * np.maximum(
            value - location, 0.0
        )
    raise ValueError(f"unsupported property-v2 shape: {shape}")


def _aggregate_table(libraries: list[dict[str, Any]]) -> pd.DataFrame:
    return BASE.aggregate_family_losses(libraries)


def _family_models(
    libraries: list[dict[str, Any]], family: str
) -> list[dict[str, Any]]:
    return [item["models"][family] for item in libraries]


def _centered_curves(
    models: list[dict[str, Any]],
    u_train: list[np.ndarray],
    grid: np.ndarray,
) -> tuple[list[np.ndarray], list[float], np.ndarray]:
    curves = []
    centers = []
    for model, values in zip(models, u_train, strict=True):
        center = float(np.mean(BASE.HELPER.predict_model(model, values)))
        centers.append(center)
        curves.append(BASE.HELPER.predict_model(model, grid) - center)
    return curves, centers, np.median(curves, axis=0)


def _selection_from_family(
    family: str,
    libraries: list[dict[str, Any]],
    u_train: list[np.ndarray],
    grid: np.ndarray,
    ate_value: float,
    diagnostics: dict[str, Any],
    table: pd.DataFrame,
) -> dict[str, Any]:
    models = _family_models(libraries, family)
    centered_curves, centers, consensus = _centered_curves(models, u_train, grid)
    marginal_curve = float(ate_value) + consensus
    parameter = float("nan")
    details: dict[str, Any] = {"property_selector": diagnostics}
    if family == "linear":
        slopes = [float(np.asarray(model["coefficient"])[1]) for model in models]
        parameter = BASE.first_crossing(grid, marginal_curve)
        label = "sign-reversal" if np.isfinite(parameter) else "linear-monotone"
        details.update(slope=float(np.median(slopes)), crossing=parameter)
    elif family == "quadratic":
        curvatures = [float(2.0 * np.asarray(model["coefficient"])[1]) for model in models]
        locations = [float(model["params"]["c"]) for model in models]
        parameter = float(np.median(locations))
        label = "u-shaped" if float(np.median(curvatures)) > 0 else "inverted-u"
        details.update(turning_point=parameter, curvature=float(np.median(curvatures)))
    elif family == "step":
        locations = [float(model["params"]["c"]) for model in models]
        contrasts = [float(np.asarray(model["coefficient"])[1]) for model in models]
        parameter = float(np.median(locations))
        label = "threshold-step"
        details.update(boundary=parameter, contrast=float(np.median(contrasts)))
    elif family == "kink":
        locations = [float(model["params"]["c"]) for model in models]
        left = [float(np.asarray(model["coefficient"])[1]) for model in models]
        right = [
            float(np.asarray(model["coefficient"])[1] + np.asarray(model["coefficient"])[2])
            for model in models
        ]
        parameter = float(np.median(locations))
        label = "kinked"
        details.update(
            kink=parameter,
            left_slope=float(np.median(left)),
            right_slope=float(np.median(right)),
        )
    elif family == "tanh":
        ks = [float(model["params"]["k"]) for model in models]
        locations = [float(model["params"]["c"]) for model in models]
        crossing = BASE.first_crossing(grid, marginal_curve)
        if np.isfinite(crossing):
            label = "sign-reversal"
            parameter = crossing
            details.update(crossing=parameter, k=float(np.median(ks)))
        else:
            parameter = float(np.median(locations))
            label = "steep-continuous" if float(np.median(ks)) >= 8.0 else "nonlinear-monotone"
            details.update(center=parameter, k=float(np.median(ks)))
    else:
        raise ValueError(family)
    stability = []
    for index, left in enumerate(centered_curves):
        for right in centered_curves[index + 1 :]:
            if np.std(left) > 1e-12 and np.std(right) > 1e-12:
                stability.append(float(spearmanr(left, right).statistic))
    return {
        "family": family,
        "label": label,
        "parameter": parameter,
        "details": details,
        "models": models,
        "centers": centers,
        "consensus_component": consensus,
        "curve_stability": float(np.nanmedian(stability)) if stability else float("nan"),
        "loss_table": table.drop(columns="fold_losses").to_dict(orient="records"),
    }


def property_select_component(
    libraries: list[dict[str, Any]],
    u_train: list[np.ndarray],
    grid: np.ndarray,
    ate_value: float,
    selector: dict[str, Any],
) -> dict[str, Any]:
    table = _aggregate_table(libraries)
    lookup = table.set_index("family")
    continuous = ["linear", "quadratic", "tanh", "kink"]
    best_continuous = min(
        continuous,
        key=lambda family: (float(lookup.loc[family, "mean_loss"]), family),
    )
    step_losses = np.asarray(lookup.loc["step", "fold_losses"], dtype=float)
    continuous_losses = np.asarray(
        lookup.loc[best_continuous, "fold_losses"], dtype=float
    )
    differences = step_losses - continuous_losses
    folds = int(len(differences) / len(libraries))
    refit_wins = [
        float(np.mean(differences[index * folds : (index + 1) * folds])) < 0.0
        for index in range(len(libraries))
    ]
    step_models = _family_models(libraries, "step")
    step_locations = [float(model["params"]["c"]) for model in step_models]
    step_diagnostics = {
        "step_vs_best_continuous_mean_loss_difference": float(np.mean(differences)),
        "step_refit_win_rate": float(np.mean(refit_wins)),
        "step_location_range": float(np.ptp(step_locations)),
        "best_continuous_family": best_continuous,
    }
    step_selected = bool(
        float(np.mean(differences)) < 0.0
        and float(np.mean(refit_wins))
        >= float(selector["step_minimum_refit_win_rate"])
        and float(np.ptp(step_locations)) <= float(selector["maximum_location_range"])
    )
    if step_selected:
        return _selection_from_family(
            "step", libraries, u_train, grid, ate_value, step_diagnostics, table
        )

    quadratic_models = _family_models(libraries, "quadratic")
    quadratic_locations = [float(model["params"]["c"]) for model in quadratic_models]
    quadratic_curves, _, quadratic_consensus = _centered_curves(
        quadratic_models, u_train, grid
    )
    del quadratic_curves
    location = float(np.median(quadratic_locations))
    location_index = int(np.argmin(np.abs(grid - location)))
    peak_tail = float(
        quadratic_consensus[location_index]
        - np.mean([quadratic_consensus[0], quadratic_consensus[-1]])
    )
    best_nonstep = min(
        continuous,
        key=lambda family: (float(lookup.loc[family, "mean_loss"]), family),
    )
    quadratic_eligible = float(lookup.loc["quadratic", "mean_loss"]) <= float(
        lookup.loc[best_nonstep, "mean_loss"] + lookup.loc[best_nonstep, "se_loss"]
    )
    support_low, support_high = (float(value) for value in selector["turning_support"])
    turning_diagnostics = {
        **step_diagnostics,
        "quadratic_within_one_se": bool(quadratic_eligible),
        "quadratic_location": location,
        "quadratic_location_range": float(np.ptp(quadratic_locations)),
        "quadratic_peak_tail_contrast": peak_tail,
    }
    turning_selected = bool(
        quadratic_eligible
        and support_low <= location <= support_high
        and float(np.ptp(quadratic_locations)) <= float(selector["maximum_location_range"])
        and abs(peak_tail) >= float(selector["minimum_peak_tail_contrast"])
    )
    if turning_selected:
        return _selection_from_family(
            "quadratic",
            libraries,
            u_train,
            grid,
            ate_value,
            turning_diagnostics,
            table,
        )

    fallback_families = ["linear", "tanh", "kink"]
    fallback_best = min(
        fallback_families,
        key=lambda family: (float(lookup.loc[family, "mean_loss"]), family),
    )
    fallback_eligible = [
        family
        for family in fallback_families
        if float(lookup.loc[family, "mean_loss"])
        <= float(
            lookup.loc[fallback_best, "mean_loss"]
            + lookup.loc[fallback_best, "se_loss"]
        )
    ]
    complexity = {"linear": 1, "tanh": 2, "kink": 2}
    fallback = min(
        fallback_eligible,
        key=lambda family: (
            complexity[family],
            float(lookup.loc[family, "mean_loss"]),
            family,
        ),
    )
    diagnostics = {**turning_diagnostics, "fallback_family": fallback}
    return _selection_from_family(
        fallback, libraries, u_train, grid, ate_value, diagnostics, table
    )


def paired_difference_confirmation(
    difference: np.ndarray, alpha: float
) -> dict[str, Any]:
    values = np.asarray(difference, dtype=float)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    upper = mean + float(norm.ppf(1.0 - alpha)) * se
    return {"difference": mean, "se": se, "upper": upper, "confirmed": bool(upper < 0.0)}


def run_cell(
    backbone: pd.DataFrame,
    source: dict[str, Any],
    method: dict[str, Any],
    config_path: Path,
    config: dict[str, Any],
    phase: str,
    scenario_name: str,
    replication: int,
    master_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    lineage = BASE.BASE.lineage(master_seed, 0)
    BASE.component_raw = component_raw
    base_parameters = read_json(
        (
            resolve(config_path, config["method_config"]).parent
            / method["base_config"]
        ).resolve()
    )["shape_parameters"]
    data = BASE.build_data(
        backbone, source, method, config, scenario_name, lineage, base_parameters
    )
    train, test = BASE.V2.v2.split_once(
        data["T"], data["unit_id"], float(config["test_size"]), lineage["split_seed"]
    )
    specs = config["scenarios"][scenario_name]["components"]
    features = [item["feature"] for item in specs]
    u_train = {feature: data["u_values"][feature][train] for feature in features}
    u_test = {feature: data["u_values"][feature][test] for feature in features}

    ate = BASE.PRODUCT.estimate_ate(
        data["X"][train], data["Y"][train], data["T"][train], lineage["nuisance_seed"], method
    )
    ate_value = float(ate["ate_dml"])
    common = dict(source["common"])
    common["n_jobs"] = int(method["primary_forest"]["n_jobs"])
    forest = BASE.V2.v2._new_m1(common, lineage["forest_seed"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forest.fit(data["Y"][train], data["T"][train], X=data["X"][train], W=None)
    forest_test = np.asarray(forest.effect(data["X"][test]), dtype=float).reshape(-1)
    importance = np.asarray(forest.feature_importances_, dtype=float)
    ranks = {
        feature: int(np.sum(importance > importance[source["features"].index(feature)]) + 1)
        for feature in features
    }

    library_config = BASE.library_config(method)
    theory_libraries = {feature: [] for feature in features}
    nuisances = []
    for refit in range(int(config["same_data_nuisance_refits"])):
        seeds = BASE.HELPER.refit_seeds(master_seed, refit)
        nuisance = BASE.V2.crossfit_nuisance(
            data["X"][train],
            data["Y"][train],
            data["T"][train],
            data["X"][test],
            seeds["nuisance_refit_seed"],
            int(config["nuisance_crossfit_folds"]),
        )
        flexible = BASE.backfit_components(
            features,
            u_train,
            nuisance["y_residual_train"],
            nuisance["t_residual_train"],
            int(config["backfit_iterations"]),
        )
        for index, feature in enumerate(features):
            other = sum(
                (flexible[name] for name in features if name != feature),
                start=np.zeros(len(train), dtype=float),
            )
            adjusted = nuisance["y_residual_train"] - nuisance["t_residual_train"] * other
            theory_libraries[feature].append(
                BASE.HELPER.fit_library(
                    u_train[feature],
                    adjusted,
                    nuisance["t_residual_train"],
                    library_config,
                    seeds["shape_cv_refit_seed"] + 1009 * index,
                )
            )
        nuisances.append(nuisance)

    grid = np.linspace(0.10, 0.90, 161)
    selections = {
        feature: property_select_component(
            theory_libraries[feature],
            [u_train[feature]] * len(theory_libraries[feature]),
            grid,
            ate_value,
            config["property_selector"],
        )
        for feature in features
    }

    alpha = float(config["test_confirmation_alpha"])
    full_refit_confirmations = []
    component_refit_confirmations = {feature: [] for feature in features}
    full_differences = []
    component_differences = {feature: [] for feature in features}
    reconstructed_test = []
    for refit, nuisance in enumerate(nuisances):
        components_test = {}
        for feature in features:
            selection = selections[feature]
            raw = BASE.HELPER.predict_model(selection["models"][refit], u_test[feature])
            components_test[feature] = raw - float(selection["centers"][refit])
        full = ate_value + sum(components_test.values())
        reconstructed_test.append(full)
        y_test = data["Y"][test] - nuisance["m_test"]
        t_test = data["T"][test] - nuisance["e_test"]
        constant = np.full(len(test), ate_value, dtype=float)
        full_loss_difference = BASE.HELPER.observation_loss(
            y_test, t_test, full
        ) - BASE.HELPER.observation_loss(y_test, t_test, constant)
        full_differences.append(full_loss_difference)
        full_refit_confirmations.append(
            paired_difference_confirmation(full_loss_difference, alpha)
        )
        for feature in features:
            ablated = full - components_test[feature]
            difference = BASE.HELPER.observation_loss(
                y_test, t_test, full
            ) - BASE.HELPER.observation_loss(y_test, t_test, ablated)
            component_differences[feature].append(difference)
            component_refit_confirmations[feature].append(
                paired_difference_confirmation(difference, alpha)
            )

    full_consensus = paired_difference_confirmation(
        np.mean(full_differences, axis=0), alpha
    )
    component_consensus = {
        feature: paired_difference_confirmation(
            np.mean(component_differences[feature], axis=0), alpha
        )
        for feature in features
    }
    reconstructed = np.median(reconstructed_test, axis=0)
    component_records = []
    curve_rows = []
    for spec in specs:
        feature = spec["feature"]
        selection = selections[feature]
        estimated_curve = ate_value + selection["consensus_component"]
        truth_curve = float(config["target_ate"]) + component_raw(grid, spec) - float(
            data["component_centers"][feature]
        )
        nominal = spec["truth_parameter"]
        if spec["truth_label"] == "sign-reversal":
            nominal = BASE.first_crossing(grid, truth_curve)
        parameter_error = (
            float(abs(selection["parameter"] - nominal))
            if nominal is not None and np.isfinite(selection["parameter"])
            else float("nan")
        )
        exact = bool(selection["label"] == spec["truth_label"])
        curve_spearman = float(spearmanr(estimated_curve, truth_curve).statistic)
        parameter_pass = bool(
            np.isfinite(parameter_error)
            and parameter_error <= float(config["property_selector"]["parameter_tolerance_ecdf"])
        )
        strict_parameter_pass = bool(
            np.isfinite(parameter_error)
            and parameter_error
            <= float(config["property_selector"]["strict_parameter_tolerance_ecdf"])
        )
        curve_pass = bool(
            curve_spearman
            >= float(config["property_selector"]["curve_fidelity_spearman"])
        )
        confirmed = bool(component_consensus[feature]["confirmed"])
        component_records.append(
            {
                "feature": feature,
                "truth_label": spec["truth_label"],
                "predicted_label": selection["label"],
                "exact_label_correct": exact,
                "selected_family": selection["family"],
                "test_refit_confirmation_rate": float(
                    np.mean(
                        [item["confirmed"] for item in component_refit_confirmations[feature]]
                    )
                ),
                "test_component_confirmed": confirmed,
                "test_consensus_difference": component_consensus[feature]["difference"],
                "test_consensus_upper": component_consensus[feature]["upper"],
                "curve_rmse": float(np.sqrt(np.mean((estimated_curve - truth_curve) ** 2))),
                "curve_spearman": curve_spearman,
                "curve_fidelity_pass": curve_pass,
                "curve_stability": selection["curve_stability"],
                "estimated_parameter": selection["parameter"],
                "truth_parameter": nominal,
                "parameter_abs_error": parameter_error,
                "parameter_tolerance_pass": parameter_pass,
                "strict_parameter_tolerance_pass": strict_parameter_pass,
                "feature_rank": ranks[feature],
                "functional_recovered": bool(exact and curve_pass and parameter_pass),
                "theory_recovered": bool(exact and confirmed and curve_pass and parameter_pass),
                "details": selection["details"],
                "loss_table": selection["loss_table"],
            }
        )
        for value, truth, estimate in zip(grid, truth_curve, estimated_curve, strict=True):
            curve_rows.append(
                {
                    "phase": phase,
                    "scenario": scenario_name,
                    "replication": replication,
                    "feature": feature,
                    "truth_label": spec["truth_label"],
                    "predicted_label": selection["label"],
                    "u": float(value),
                    "truth_curve": float(truth),
                    "estimate_curve": float(estimate),
                    "ate_dml": ate_value,
                }
            )

    all_exact = bool(all(item["exact_label_correct"] for item in component_records))
    all_confirmed = bool(all(item["test_component_confirmed"] for item in component_records))
    all_functional = bool(all(item["functional_recovered"] for item in component_records))
    all_theory = bool(all(item["theory_recovered"] for item in component_records))
    practical_state = (
        "SUPPORTED_MULTIPLE_COMPONENTS"
        if all_confirmed and full_consensus["confirmed"]
        else "INCONCLUSIVE"
    )
    result = {
        "status": "completed",
        "phase": phase,
        "scenario": scenario_name,
        "scenario_role": config["scenarios"][scenario_name]["role"],
        "component_count": len(features),
        "replication": replication,
        "master_seed": master_seed,
        "n": int(config["n"]),
        "practical_state": practical_state,
        "all_labels_exact": all_exact,
        "all_components_test_confirmed": all_confirmed,
        "all_components_functionally_recovered": all_functional,
        "all_components_theory_recovered": all_theory,
        "full_test_consensus_confirmed": bool(full_consensus["confirmed"]),
        "full_test_consensus_difference": full_consensus["difference"],
        "full_test_consensus_upper": full_consensus["upper"],
        "full_test_refit_confirmation_rate": float(
            np.mean([item["confirmed"] for item in full_refit_confirmations])
        ),
        "component_records_json": json.dumps(component_records, sort_keys=True),
        "forest_test_cate_rmse": float(np.sqrt(np.mean((forest_test - data["tau_true"][test]) ** 2))),
        "forest_test_cate_spearman": float(spearmanr(forest_test, data["tau_true"][test]).statistic),
        "reconstructed_test_cate_rmse": float(np.sqrt(np.mean((reconstructed - data["tau_true"][test]) ** 2))),
        "reconstructed_test_cate_spearman": float(spearmanr(reconstructed, data["tau_true"][test]).statistic),
        "true_ate": float(np.mean(data["tau_true"])),
        "dml_ate": ate_value,
        "dml_ate_error": float(ate_value - np.mean(data["tau_true"])),
        "mean_reconstructed_cate_test": float(np.mean(reconstructed)),
        "ate_cate_gap_sd_y": float(abs(np.mean(reconstructed) - ate_value) / np.std(data["Y"][train], ddof=1)),
        "active_feature_ranks_json": json.dumps(ranks, sort_keys=True),
        "warning_count": len(caught),
        "runtime_seconds": float(time.perf_counter() - started),
        "one_joint_multivariable_forest": True,
        "test_used_for_shape_selection": False,
        "oracle_used_for_shape_selection": False,
    }
    return result, curve_rows


def output_paths(config_path: Path, config: dict[str, Any], phase: str) -> dict[str, Path]:
    output = resolve(config_path, config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    prefix = phase.upper().replace("-", "_")
    return {
        "output": output,
        "results": output / f"PROPERTY_V2_{prefix}_LONG.csv",
        "curves": output / f"PROPERTY_V2_{prefix}_CURVES.csv",
    }


def append(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def load_inputs(
    config_path: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    source, method = validate_config(config_path, config)
    source_path = resolve(config_path, config["source_config"])
    backbone = BASE.V2.add_rank_columns(
        BASE.V2.v2.load_frozen_backbone(ROOT, source, source_path), source["features"]
    )
    return source, method, backbone


def run_phase(config_path: Path, config: dict[str, Any], phase: str) -> None:
    if phase == "confirmation":
        verify_freeze(config_path, config)
    source, method, backbone = load_inputs(config_path, config)
    paths = output_paths(config_path, config, phase)
    phase_config = config[phase]
    maximum_replications = (
        int(phase_config["replications"])
        if phase == "development"
        else int(phase_config["primary_replications"])
    )
    seeds = seed_grid(phase_config["master_seed"], maximum_replications)
    completed: set[tuple[str, int]] = set()
    if paths["results"].is_file():
        existing = pd.read_csv(paths["results"])
        completed = set(zip(existing["scenario"], existing["replication"].astype(int), strict=False))
    scenario_replications = {
        scenario: (
            int(phase_config["replications"])
            if phase == "development"
            else (
                int(phase_config["primary_replications"])
                if config["scenarios"][scenario]["role"] == "primary"
                else int(phase_config["diagnostic_replications"])
            )
        )
        for scenario in config["scenarios"]
    }
    total = sum(scenario_replications.values())
    done = len(completed)
    for replication in range(maximum_replications):
        for scenario in config["scenarios"]:
            if replication >= scenario_replications[scenario]:
                continue
            if (scenario, replication) in completed:
                continue
            print(f"{phase.upper()} {done + 1}/{total} scenario={scenario} replication={replication}", flush=True)
            try:
                result, curves = run_cell(
                    backbone,
                    source,
                    method,
                    config_path,
                    config,
                    phase,
                    scenario,
                    replication,
                    seeds[replication],
                )
            except Exception as error:  # noqa: BLE001 - failed cells remain visible
                result = {
                    "status": "failed",
                    "phase": phase,
                    "scenario": scenario,
                    "replication": replication,
                    "master_seed": seeds[replication],
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
                curves = []
            append(paths["results"], [result])
            if curves:
                append(paths["curves"], curves)
            done += 1


def component_frame(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results.itertuples(index=False):
        for component in json.loads(result.component_records_json):
            rows.append(
                {
                    "scenario": result.scenario,
                    "scenario_role": result.scenario_role,
                    "component_count": int(result.component_count),
                    "replication": int(result.replication),
                    **component,
                }
            )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    components = component_frame(results)
    scenario = results.groupby(
        ["scenario", "scenario_role", "component_count"], as_index=False
    ).agg(
        replications=("replication", "size"),
        all_labels_exact_rate=("all_labels_exact", "mean"),
        all_components_confirmed_rate=("all_components_test_confirmed", "mean"),
        all_component_functional_recovery_rate=(
            "all_components_functionally_recovered",
            "mean",
        ),
        all_component_theory_recovery_rate=("all_components_theory_recovered", "mean"),
        reconstructed_cate_spearman_median=("reconstructed_test_cate_spearman", "median"),
        reconstructed_cate_spearman_p10=("reconstructed_test_cate_spearman", lambda value: float(value.quantile(0.10))),
        reconstructed_cate_rmse_median=("reconstructed_test_cate_rmse", "median"),
        dml_ate_abs_error_median=("dml_ate_error", lambda value: float(value.abs().median())),
    )
    component = components.groupby(
        ["scenario", "scenario_role", "component_count", "feature", "truth_label"],
        as_index=False,
    ).agg(
        replications=("replication", "size"),
        exact_label_rate=("exact_label_correct", "mean"),
        test_confirmation_rate=("test_component_confirmed", "mean"),
        curve_fidelity_rate=("curve_fidelity_pass", "mean"),
        parameter_tolerance_rate=("parameter_tolerance_pass", "mean"),
        strict_parameter_tolerance_rate=("strict_parameter_tolerance_pass", "mean"),
        functional_recovery_rate=("functional_recovered", "mean"),
        theory_recovery_rate=("theory_recovered", "mean"),
        curve_spearman_median=("curve_spearman", "median"),
        curve_rmse_median=("curve_rmse", "median"),
        parameter_abs_error_median=("parameter_abs_error", "median"),
        feature_rank_median=("feature_rank", "median"),
    )
    confusion = (
        components.groupby(
            ["scenario", "feature", "truth_label", "predicted_label"], as_index=False
        )
        .size()
        .rename(columns={"size": "replications"})
    )
    return scenario, component, confusion


def write_development_report(config_path: Path, config: dict[str, Any]) -> None:
    paths = output_paths(config_path, config, "development")
    results = pd.read_csv(paths["results"])
    expected = len(config["scenarios"]) * int(config["development"]["replications"])
    if len(results) != expected or (results["status"] != "completed").any():
        raise RuntimeError("development is incomplete")
    scenario, component, confusion = summarize(results)
    scenario.to_csv(paths["output"] / "PROPERTY_V2_DEVELOPMENT_SUMMARY.csv", index=False)
    component.to_csv(paths["output"] / "PROPERTY_V2_DEVELOPMENT_COMPONENTS.csv", index=False)
    confusion.to_csv(paths["output"] / "PROPERTY_V2_DEVELOPMENT_CONFUSION.csv", index=False)
    report = f"""# Property-First Coexistence v2 Development

Development evidence only. These seeds may tune the property hierarchy but
cannot support the final claim.

## Scenario summary

{scenario.to_markdown(index=False, floatfmt='.3f')}

## Component summary

{component.to_markdown(index=False, floatfmt='.3f')}

## Confusion

{confusion.to_markdown(index=False)}
"""
    (paths["output"] / "PROPERTY_V2_DEVELOPMENT_REPORT.md").write_text(report, encoding="utf-8")


def freeze_payload(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    source, _ = validate_config(config_path, config)
    development = output_paths(config_path, config, "development")
    expected = len(config["scenarios"]) * int(config["development"]["replications"])
    if not development["results"].is_file() or not development["curves"].is_file():
        raise RuntimeError("development artifacts are missing")
    results = pd.read_csv(development["results"])
    if len(results) != expected or (results["status"] != "completed").any():
        raise RuntimeError("development artifacts are incomplete")
    source_path = resolve(config_path, config["source_config"])
    backbone = (source_path.parent / source["backbone_path"]).resolve()
    return {
        "status": "FROZEN_AFTER_DEVELOPMENT_BEFORE_CONFIRMATION",
        "frozen_at": "2026-10-25",
        "confirmation_seed_grid": seed_grid(
            config["confirmation"]["master_seed"],
            config["confirmation"]["primary_replications"],
        ),
        "acceptance": config["acceptance"],
        "hashes": {
            "config_sha256": sha256_file(config_path),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "base_runner_sha256": sha256_file(BASE_RUNNER),
            "source_config_sha256": sha256_file(source_path),
            "method_config_sha256": sha256_file(resolve(config_path, config["method_config"])),
            "backbone_sha256": sha256_file(backbone),
            "development_results_sha256": sha256_file(development["results"]),
            "development_curves_sha256": sha256_file(development["curves"]),
        },
    }


def create_freeze(config_path: Path, config: dict[str, Any]) -> Path:
    path = resolve(config_path, config["freeze_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(freeze_payload(config_path, config), indent=2) + "\n", encoding="utf-8")
    return path


def verify_freeze(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = resolve(config_path, config["freeze_path"])
    if not path.is_file():
        raise RuntimeError("property-v2 method freeze is missing")
    frozen = read_json(path)
    current = freeze_payload(config_path, config)
    seeds = {str(key): value for key, value in current["confirmation_seed_grid"].items()}
    if frozen["hashes"] != current["hashes"] or frozen["confirmation_seed_grid"] != seeds:
        raise RuntimeError("property-v2 method freeze mismatch")
    return frozen


def render_figures(curves: pd.DataFrame, output: Path) -> list[Path]:
    figures = []
    for scenario, group in curves.groupby("scenario", sort=False):
        path = output / f"PROPERTY_V2_{scenario.upper()}.svg"
        BASE.render_svg(group, path)
        title = f"Property-first additive recovery: {scenario}"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "Two coexisting CATE morphologies recovered from one joint forest", title
            ),
            encoding="utf-8",
        )
        figures.append(path)
    return figures


def write_confirmation_report(config_path: Path, config: dict[str, Any]) -> None:
    freeze = verify_freeze(config_path, config)
    paths = output_paths(config_path, config, "confirmation")
    results = pd.read_csv(paths["results"])
    expected = sum(
        int(config["confirmation"]["primary_replications"])
        if scenario["role"] == "primary"
        else int(config["confirmation"]["diagnostic_replications"])
        for scenario in config["scenarios"].values()
    )
    if len(results) != expected or (results["status"] != "completed").any():
        raise RuntimeError("confirmation is incomplete")
    curves = pd.read_csv(paths["curves"])
    scenario, component, confusion = summarize(results)
    scenario_path = paths["output"] / "PROPERTY_V2_CONFIRMATION_SUMMARY.csv"
    component_path = paths["output"] / "PROPERTY_V2_CONFIRMATION_COMPONENTS.csv"
    confusion_path = paths["output"] / "PROPERTY_V2_CONFIRMATION_CONFUSION.csv"
    scenario.to_csv(scenario_path, index=False)
    component.to_csv(component_path, index=False)
    confusion.to_csv(confusion_path, index=False)

    primary = scenario.loc[scenario["scenario_role"] == "primary"].set_index("scenario")
    primary_components = component.loc[component["scenario_role"] == "primary"]
    acceptance = config["acceptance"]
    checks = {
        "primary_count_2_functional_recovery": bool(
            primary.loc["primary_count_2", "all_component_functional_recovery_rate"]
            >= float(acceptance["primary_all_component_functional_recovery_rate"])
        ),
        "primary_count_3_functional_recovery": bool(
            primary.loc["primary_count_3", "all_component_functional_recovery_rate"]
            >= float(acceptance["primary_all_component_functional_recovery_rate"])
        ),
        "each_primary_component_functional_recovery": bool(
            (
                primary_components["functional_recovery_rate"]
                >= float(acceptance["minimum_primary_component_functional_recovery_rate"])
            ).all()
        ),
        "primary_cate_median": bool(
            (primary["reconstructed_cate_spearman_median"] >= float(acceptance["minimum_reconstructed_cate_spearman_median"])).all()
        ),
        "primary_cate_p10": bool(
            (primary["reconstructed_cate_spearman_p10"] >= float(acceptance["minimum_reconstructed_cate_spearman_p10"])).all()
        ),
    }
    component_records = component_frame(results)
    steep_control = component_records.loc[
        (component_records["scenario"] == "control_count_3_steep")
        & (component_records["feature"] == "ppe_at")
    ]
    threshold_stress = component_records.loc[
        (component_records["scenario"] == "stress_count_3_threshold")
        & (component_records["feature"] == "ppe_at")
    ]
    false_step_rate = float(np.mean(steep_control["predicted_label"] == "threshold-step"))
    threshold_stress_metrics = {
        "exact_label_rate": float(threshold_stress["exact_label_correct"].mean()),
        "functional_recovery_rate": float(threshold_stress["functional_recovered"].mean()),
        "test_confirmation_rate": float(threshold_stress["test_component_confirmed"].mean()),
        "parameter_tolerance_rate": float(threshold_stress["parameter_tolerance_pass"].mean()),
    }
    decision = "PASS_PRIMARY_CAPABILITY" if all(checks.values()) else "PRIMARY_CAPABILITY_NOT_ESTABLISHED"
    figures = render_figures(curves, paths["output"])
    report_path = paths["output"] / "PROPERTY_FIRST_COEXISTENCE_V2_REPORT_ZH.md"
    report = f"""# Property-First 两/三组件共生恢复 v2

## 决策

**{decision}**

本报告只使用冻结后全新 confirmation seeds。开发结果不进入成功率。

主验收回答的是：在已知两个或三个加性异质性成分存在时，能否同时恢复每个成分的形态、曲线和关键位置。untouched TEST 显著性确认单独报告，因为它衡量单次样本中的推断把握，而不是形态恢复本身。

## 情境结果

{scenario.to_markdown(index=False, floatfmt='.3f')}

## 逐组件结果

{component.to_markdown(index=False, floatfmt='.3f')}

## 标签混淆

{confusion.to_markdown(index=False)}

## TEST 确认与压力诊断

- 主情境逐组件 TEST confirmation 率见“逐组件结果”；它不参与形态恢复主门槛。
- threshold-step 压力情境：准确命名 {threshold_stress_metrics['exact_label_rate']:.1%}，完整功能恢复 {threshold_stress_metrics['functional_recovery_rate']:.1%}，TEST confirmation {threshold_stress_metrics['test_confirmation_rate']:.1%}，边界误差在 0.10 ECDF 内 {threshold_stress_metrics['parameter_tolerance_rate']:.1%}。
- steep-continuous 邻近对照被误报为 threshold-step：{false_step_rate:.1%}。该项用于公开能力边界，不参与两/三组件主结论的 PASS 判定。

## 预冻结验收项

{pd.DataFrame([{"check": key, "passed": value} for key, value in checks.items()]).to_markdown(index=False)}

## 图

"""
    for figure in figures:
        report += f"![{figure.stem}]({figure.name})\n\n"
    report_path.write_text(report, encoding="utf-8")
    artifacts = [
        paths["results"],
        paths["curves"],
        scenario_path,
        component_path,
        confusion_path,
        report_path,
        *figures,
    ]
    manifest = {
        "schema_version": "property-first-coexistence-result-v2",
        "decision": decision,
        "checks": checks,
        "false_step_rate": false_step_rate,
        "threshold_stress_metrics": threshold_stress_metrics,
        "cells": len(results),
        "failed_cells": int((results["status"] == "failed").sum()),
        "one_joint_forest_per_cell": True,
        "test_used_for_shape_selection": False,
        "oracle_used_for_shape_selection": False,
        "freeze": freeze,
        "artifact_hashes": {path.name: sha256_file(path) for path in artifacts},
    }
    (paths["output"] / "PROPERTY_V2_CONFIRMATION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase",
        choices=("development", "development-report", "freeze", "confirmation", "report"),
        required=True,
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    if args.phase == "development":
        run_phase(config_path, config, "development")
    elif args.phase == "development-report":
        write_development_report(config_path, config)
    elif args.phase == "freeze":
        print(create_freeze(config_path, config))
    elif args.phase == "confirmation":
        run_phase(config_path, config, "confirmation")
    else:
        write_confirmation_report(config_path, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
