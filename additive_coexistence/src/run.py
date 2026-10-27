"""Develop a joint orthogonal-score confirmation for coexisting HTE components."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, spearmanr

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "additive_coexistence" / "two_component" / "config.json"
PROPERTY_RUNNER = ROOT / "additive_coexistence" / "src" / "component_reconstruction.py"


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROPERTY = load_module(PROPERTY_RUNNER, "property_first_for_joint_confirmation")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(config_path: Path, value: str) -> Path:
    return (config_path.parent / value).resolve()


def seed_grid(master_seed: int, replications: int) -> dict[int, int]:
    children = np.random.SeedSequence(int(master_seed)).spawn(int(replications))
    return {
        index: int(child.generate_state(1, dtype=np.uint32)[0])
        for index, child in enumerate(children)
    }


def holm_adjusted_pvalues(p_values: dict[str, float]) -> dict[str, float]:
    """Return Holm-adjusted p-values for one family of component tests."""

    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running_maximum = 0.0
    count = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        running_maximum = max(running_maximum, (count - index) * float(p_value))
        adjusted[name] = float(min(1.0, running_maximum))
    return adjusted


def validate_config(
    config_path: Path, config: dict[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Any], pd.DataFrame]:
    if int(config["n"]) != 3000 or float(config["test_size"]) != 0.2:
        raise ValueError("pilot requires n=3000 and one 80/20 external split")
    property_path = resolve(config_path, config["property_config"])
    property_config = read_json(property_path)
    source, method, backbone = PROPERTY.load_inputs(property_path, property_config)
    required = {
        "one_joint_multivariable_forest": True,
        "test_used_for_shape_or_parameter_selection": False,
        "oracle_used_for_shape_or_parameter_selection": False,
        "interactions_included": False,
    }
    for key, expected in required.items():
        if config["selection_policy"].get(key) is not expected:
            raise ValueError(f"selection policy changed: {key}")
    if len(source["features"]) != 12:
        raise ValueError("all 12 candidate characteristics must enter one forest")
    forest = method["primary_forest"]
    if not (
        int(forest["n_estimators"]) == 1000
        and int(forest["cv"]) == 2
        and forest["honest"] is True
    ):
        raise ValueError("primary forest changed")
    familywise_alpha = config.get("familywise_confirmation_alpha")
    if familywise_alpha is not None and not 0.0 < float(familywise_alpha) < 1.0:
        raise ValueError("familywise confirmation alpha must be between zero and one")
    if config.get("acceptance_mode") == "componentwise-v1":
        if config.get("joint_train_shape_scaling") is not True:
            raise ValueError("componentwise acceptance requires joint TRAIN scaling")
        if config.get("use_joint_train_effect_level") is not True:
            raise ValueError("componentwise acceptance requires the joint TRAIN effect level")
        if familywise_alpha is None:
            raise ValueError("componentwise acceptance requires family-wise correction")
        required_acceptance = {
            "development_minimum_component_success_rate",
            "development_minimum_total_cate_fidelity_rate",
            "maximum_null_familywise_false_confirmation_rate",
            "curve_spearman",
            "normalized_curve_rmse",
            "location_tolerance_ecdf",
            "effect_tolerance_share_true_cate_range",
            "minimum_curve_stability_spearman",
            "maximum_location_range_ecdf",
        }
        missing = required_acceptance.difference(config["acceptance"])
        if missing:
            raise ValueError(f"componentwise acceptance keys missing: {sorted(missing)}")
    for name, scenario in config["scenarios"].items():
        base = property_config["scenarios"][scenario["base_scenario"]]
        components = scenario.get("components", base["components"])
        if len(components) != int(scenario["component_count"]):
            raise ValueError(f"component count mismatch: {name}")
        features = [item["feature"] for item in components]
        if len(features) != len(set(features)):
            raise ValueError(f"duplicate component feature: {name}")
        if not set(features).issubset(source["features"]):
            raise ValueError(f"component feature missing from 12-variable forest: {name}")
    return property_path, property_config, method, backbone


def _zero_component(spec: dict[str, Any]) -> None:
    if spec["shape"] == "linear":
        spec["half_range"] = 0.0
    elif spec["shape"] == "inverted_u":
        spec["amplitude"] = 0.0
    elif spec["shape"] == "tanh":
        spec["half_range"] = 0.0
    elif spec["shape"] == "threshold":
        spec["contrast"] = 0.0
    else:
        raise ValueError(f"unsupported null component: {spec['shape']}")


def build_cell_data(
    backbone: pd.DataFrame,
    source: dict[str, Any],
    method: dict[str, Any],
    property_path: Path,
    property_config: dict[str, Any],
    base_scenario: str,
    role: str,
    lineage: dict[str, int],
    component_override: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    data_config = copy.deepcopy(property_config)
    if component_override is not None:
        data_config["scenarios"][base_scenario]["components"] = copy.deepcopy(
            component_override
        )
        data_config["scenarios"][base_scenario]["component_count"] = len(
            component_override
        )
    if role == "all-component-null":
        for spec in data_config["scenarios"][base_scenario]["components"]:
            _zero_component(spec)
    base_parameters = read_json(
        (
            resolve(property_path, property_config["method_config"]).parent
            / method["base_config"]
        ).resolve()
    )["shape_parameters"]
    PROPERTY.BASE.component_raw = PROPERTY.component_raw
    return PROPERTY.BASE.build_data(
        backbone,
        source,
        method,
        data_config,
        base_scenario,
        lineage,
        base_parameters,
    )


def joint_cluster_score(
    y_residuals: list[np.ndarray],
    t_residuals: list[np.ndarray],
    component_predictions: list[dict[str, np.ndarray]],
    features: list[str],
    alpha: float,
    familywise_alpha: float | None = None,
) -> dict[str, Any]:
    """Joint fixed-basis regression with TEST-firm clustered covariance."""

    if not (len(y_residuals) == len(t_residuals) == len(component_predictions)):
        raise ValueError("refit collections differ in length")
    n_firms = len(y_residuals[0])
    designs: list[np.ndarray] = []
    responses: list[np.ndarray] = []
    for y_res, t_res, predictions in zip(
        y_residuals, t_residuals, component_predictions, strict=True
    ):
        basis = np.column_stack(
            [np.ones(n_firms, dtype=float)]
            + [np.asarray(predictions[feature], dtype=float) for feature in features]
        )
        designs.append(np.asarray(t_res, dtype=float)[:, None] * basis)
        responses.append(np.asarray(y_res, dtype=float))
    design = np.vstack(designs)
    response = np.concatenate(responses)
    bread = np.linalg.pinv(design.T @ design)
    beta = bread @ design.T @ response
    residual = response - design @ beta
    cluster_scores = np.zeros((n_firms, design.shape[1]), dtype=float)
    for refit, current in enumerate(designs):
        start = refit * n_firms
        stop = start + n_firms
        cluster_scores += current * residual[start:stop, None]
    meat = cluster_scores.T @ cluster_scores
    observations = len(response)
    parameters = design.shape[1]
    correction = (n_firms / (n_firms - 1.0)) * (
        (observations - 1.0) / (observations - parameters)
    )
    covariance = correction * bread @ meat @ bread
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    components: dict[str, dict[str, Any]] = {}
    for index, feature in enumerate(features, start=1):
        se = float(standard_error[index])
        estimate = float(beta[index])
        z_value = estimate / se if se > 0 else float("nan")
        p_value = float(1.0 - norm.cdf(z_value)) if np.isfinite(z_value) else 1.0
        components[feature] = {
            "coefficient": estimate,
            "standard_error": se,
            "z_value": z_value,
            "one_sided_p_value": p_value,
            "confirmed": bool(estimate > 0.0 and p_value < alpha),
        }
    if familywise_alpha is not None:
        adjusted = holm_adjusted_pvalues(
            {
                feature: float(record["one_sided_p_value"])
                for feature, record in components.items()
            }
        )
        for feature, record in components.items():
            record["holm_adjusted_p_value"] = adjusted[feature]
            record["confirmed"] = bool(
                float(record["coefficient"]) > 0.0
                and adjusted[feature] < float(familywise_alpha)
            )
            record["confirmation_method"] = "holm-family-wise"
    else:
        for record in components.values():
            record["holm_adjusted_p_value"] = float(record["one_sided_p_value"])
            record["confirmation_method"] = "unadjusted-component"
    component_beta = beta[1:]
    component_covariance = covariance[1:, 1:]
    omnibus = float(
        component_beta @ np.linalg.pinv(component_covariance) @ component_beta
    )
    omnibus_p = float(chi2.sf(omnibus, len(features)))
    return {
        "ate_adjustment": float(beta[0]),
        "ate_adjustment_standard_error": float(standard_error[0]),
        "components": components,
        "omnibus_wald": omnibus,
        "omnibus_p_value": omnibus_p,
        "omnibus_confirmed": bool(omnibus_p < alpha),
        "clusters": n_firms,
        "stacked_rows": observations,
    }


def _location_values(
    selection: dict[str, Any], grid: np.ndarray, ate_value: float
) -> list[float]:
    if selection["family"] == "linear":
        return [
            PROPERTY.BASE.first_crossing(
                grid,
                ate_value
                + PROPERTY.BASE.HELPER.predict_model(model, grid)
                - float(center),
            )
            for model, center in zip(
                selection["models"], selection["centers"], strict=True
            )
        ]
    if selection["family"] == "quadratic":
        return [float(model["params"]["c"]) for model in selection["models"]]
    return [float(selection["parameter"])]


def _pooled_parameters(selection: dict[str, Any]) -> dict[str, float]:
    family = selection["family"]
    if family == "linear":
        return {}
    keys = {
        "quadratic": ("c",),
        "step": ("c",),
        "kink": ("c",),
        "tanh": ("c", "k"),
    }
    if family not in keys:
        raise ValueError(f"unsupported pooled family: {family}")
    return {
        key: float(np.median([model["params"][key] for model in selection["models"]]))
        for key in keys[family]
    }


def pool_train_coefficients(
    selection: dict[str, Any],
    u_train: np.ndarray,
    adjusted_residuals: list[np.ndarray],
    t_residuals: list[np.ndarray],
    grid: np.ndarray,
    ate_value: float,
) -> dict[str, Any]:
    """Refit one fixed TRAIN-selected family on stacked nuisance residuals."""

    family = selection["family"]
    params = _pooled_parameters(selection)
    model = PROPERTY.BASE.HELPER.fit_model(
        family,
        np.tile(np.asarray(u_train, dtype=float), len(adjusted_residuals)),
        np.concatenate(adjusted_residuals),
        np.concatenate(t_residuals),
        params,
    )
    center = float(np.mean(PROPERTY.BASE.HELPER.predict_model(model, u_train)))
    consensus = PROPERTY.BASE.HELPER.predict_model(model, grid) - center
    marginal = ate_value + consensus
    label = selection["label"]
    parameter = selection["parameter"]
    if family == "linear":
        parameter = PROPERTY.BASE.first_crossing(grid, marginal)
        label = "sign-reversal" if np.isfinite(parameter) else "linear-monotone"
    elif family == "quadratic":
        parameter = float(params["c"])
        label = (
            "u-shaped"
            if float(np.asarray(model["coefficient"])[1]) > 0.0
            else "inverted-u"
        )
    pooled = dict(selection)
    pooled.update(
        label=label,
        parameter=float(parameter),
        consensus_component=np.asarray(consensus, dtype=float),
        pooled_model=model,
        pooled_center=center,
        train_coefficient_pooling=True,
    )
    return pooled


def joint_train_scale_components(
    selections: dict[str, dict[str, Any]],
    u_train: dict[str, np.ndarray],
    nuisances: list[dict[str, Any]],
    features: list[str],
    grid: np.ndarray,
    ate_value: float,
    use_joint_effect_level: bool = False,
) -> dict[str, dict[str, Any]]:
    """Jointly rescale fixed TRAIN-selected component shapes on TRAIN only."""

    basis_train: dict[str, np.ndarray] = {}
    for feature in features:
        selection = selections[feature]
        predictions = [
            PROPERTY.BASE.HELPER.predict_model(model, u_train[feature])
            - float(center)
            for model, center in zip(
                selection["models"], selection["centers"], strict=True
            )
        ]
        basis_train[feature] = np.median(predictions, axis=0)
    designs = []
    responses = []
    for nuisance in nuisances:
        t_residual = np.asarray(nuisance["t_residual_train"], dtype=float)
        basis = np.column_stack(
            [np.ones(len(t_residual), dtype=float)]
            + [basis_train[feature] for feature in features]
        )
        designs.append(t_residual[:, None] * basis)
        responses.append(np.asarray(nuisance["y_residual_train"], dtype=float))
    design = np.vstack(designs)
    response = np.concatenate(responses)
    coefficient, *_ = np.linalg.lstsq(design, response, rcond=None)
    effect_level = float(coefficient[0]) if use_joint_effect_level else float(ate_value)
    result: dict[str, dict[str, Any]] = {}
    for index, feature in enumerate(features, start=1):
        selection = dict(selections[feature])
        scale = float(coefficient[index])
        consensus = scale * np.asarray(selection["consensus_component"], dtype=float)
        label = selection["label"]
        parameter = selection["parameter"]
        if selection["family"] == "linear":
            parameter = PROPERTY.BASE.first_crossing(grid, effect_level + consensus)
            label = (
                "sign-reversal" if np.isfinite(parameter) else "linear-monotone"
            )
        elif selection["family"] == "quadratic" and scale < 0.0:
            label = (
                "u-shaped"
                if selection["label"] == "inverted-u"
                else "inverted-u"
            )
        selection.update(
            label=label,
            parameter=float(parameter),
            consensus_component=consensus,
            joint_train_scale=scale,
            joint_train_shape_scaling=True,
            joint_train_ate_adjustment=float(coefficient[0]),
            joint_train_effect_level=effect_level,
        )
        result[feature] = selection
    return result


def _key_quantity_pass(
    truth_label: str,
    grid: np.ndarray,
    estimated: np.ndarray,
    truth: np.ndarray,
    parameter_error: float,
    acceptance: dict[str, Any],
) -> tuple[bool, dict[str, float]]:
    true_range = float(np.ptp(truth))
    tolerance = float(acceptance["effect_tolerance_share_true_cate_range"]) * true_range
    details: dict[str, float] = {
        "true_cate_range": true_range,
        "effect_scale_tolerance": tolerance,
    }
    location_pass = bool(
        np.isfinite(parameter_error)
        and parameter_error <= float(acceptance["location_tolerance_ecdf"])
    )
    if truth_label == "sign-reversal":
        estimate_slope = float(np.polyfit(grid, estimated, 1)[0])
        truth_slope = float(np.polyfit(grid, truth, 1)[0])
        slope_effect_error = abs(estimate_slope - truth_slope) * float(np.ptp(grid))
        endpoint_error = float(
            max(abs(estimated[0] - truth[0]), abs(estimated[-1] - truth[-1]))
        )
        details.update(
            slope_effect_error=slope_effect_error,
            endpoint_effect_error=endpoint_error,
        )
        return bool(
            location_pass
            and slope_effect_error <= tolerance
            and endpoint_error <= tolerance
        ), details
    if truth_label in {"inverted-u", "u-shaped"}:
        estimate_fit = np.polyfit(grid, estimated, 2)
        truth_fit = np.polyfit(grid, truth, 2)
        half_support = float(np.ptp(grid) / 2.0)
        curvature_effect_error = abs(estimate_fit[0] - truth_fit[0]) * half_support**2
        estimate_contrast = float(
            abs(np.polyval(estimate_fit, -estimate_fit[1] / (2.0 * estimate_fit[0])) - np.mean([estimated[0], estimated[-1]]))
        )
        truth_contrast = float(
            abs(np.polyval(truth_fit, -truth_fit[1] / (2.0 * truth_fit[0])) - np.mean([truth[0], truth[-1]]))
        )
        contrast_error = abs(estimate_contrast - truth_contrast)
        details.update(
            curvature_effect_error=curvature_effect_error,
            contrast_error=contrast_error,
        )
        return bool(
            location_pass
            and curvature_effect_error <= tolerance
            and contrast_error <= tolerance
        ), details
    raise ValueError(f"unsupported primary truth label: {truth_label}")


def run_cell(
    config_path: Path,
    config: dict[str, Any],
    property_path: Path,
    property_config: dict[str, Any],
    method: dict[str, Any],
    backbone: pd.DataFrame,
    source: dict[str, Any],
    scenario_name: str,
    replication: int,
    master_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    del config_path
    started = time.perf_counter()
    scenario = config["scenarios"][scenario_name]
    base_scenario = scenario["base_scenario"]
    role = scenario["role"]
    lineage = PROPERTY.BASE.BASE.lineage(master_seed, 0)
    data = build_cell_data(
        backbone,
        source,
        method,
        property_path,
        property_config,
        base_scenario,
        role,
        lineage,
        scenario.get("components"),
    )
    train, test = PROPERTY.BASE.V2.v2.split_once(
        data["T"], data["unit_id"], float(config["test_size"]), lineage["split_seed"]
    )
    specs = scenario.get(
        "components", property_config["scenarios"][base_scenario]["components"]
    )
    features = [item["feature"] for item in specs]
    u_train = {feature: data["u_values"][feature][train] for feature in features}
    u_test = {feature: data["u_values"][feature][test] for feature in features}
    ate = PROPERTY.BASE.PRODUCT.estimate_ate(
        data["X"][train],
        data["Y"][train],
        data["T"][train],
        lineage["nuisance_seed"],
        method,
    )
    ate_value = float(ate["ate_dml"])
    common = dict(source["common"])
    common["n_jobs"] = int(method["primary_forest"]["n_jobs"])
    forest = PROPERTY.BASE.V2.v2._new_m1(common, lineage["forest_seed"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forest.fit(data["Y"][train], data["T"][train], X=data["X"][train], W=None)
    importance = np.asarray(forest.feature_importances_, dtype=float)
    ranks = {
        feature: int(
            np.sum(importance > importance[source["features"].index(feature)]) + 1
        )
        for feature in features
    }
    library_config = PROPERTY.BASE.library_config(method)
    theory_libraries = {feature: [] for feature in features}
    adjusted_residuals = {feature: [] for feature in features}
    train_t_residuals: list[np.ndarray] = []
    nuisances = []
    for refit in range(int(config["nuisance_refits"])):
        seeds = PROPERTY.BASE.HELPER.refit_seeds(master_seed, refit)
        nuisance = PROPERTY.BASE.V2.crossfit_nuisance(
            data["X"][train],
            data["Y"][train],
            data["T"][train],
            data["X"][test],
            seeds["nuisance_refit_seed"],
            int(config["nuisance_crossfit_folds"]),
        )
        flexible = PROPERTY.BASE.backfit_components(
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
            adjusted = (
                nuisance["y_residual_train"]
                - nuisance["t_residual_train"] * other
            )
            adjusted_residuals[feature].append(adjusted)
            theory_libraries[feature].append(
                PROPERTY.BASE.HELPER.fit_library(
                    u_train[feature],
                    adjusted,
                    nuisance["t_residual_train"],
                    library_config,
                    seeds["shape_cv_refit_seed"] + 1009 * index,
                )
            )
        nuisances.append(nuisance)
        train_t_residuals.append(nuisance["t_residual_train"])
    grid = np.linspace(0.10, 0.90, 161)
    selections = {
        feature: PROPERTY.property_select_component(
            theory_libraries[feature],
            [u_train[feature]] * len(theory_libraries[feature]),
            grid,
            ate_value,
            property_config["property_selector"],
        )
        for feature in features
    }
    if config.get("joint_train_shape_scaling") is True:
        selections = joint_train_scale_components(
            selections,
            u_train,
            nuisances,
            features,
            grid,
            ate_value,
            bool(config.get("use_joint_train_effect_level")),
        )
    elif config.get("pool_train_coefficients") is True:
        selections = {
            feature: pool_train_coefficients(
                selections[feature],
                u_train[feature],
                adjusted_residuals[feature],
                train_t_residuals,
                grid,
                ate_value,
            )
            for feature in features
        }
    effect_level = ate_value
    if config.get("use_joint_train_effect_level") is True:
        effect_levels = {
            float(selection["joint_train_effect_level"])
            for selection in selections.values()
        }
        if len(effect_levels) != 1:
            raise AssertionError("joint TRAIN effect level differs across components")
        effect_level = effect_levels.pop()
    component_predictions: list[dict[str, np.ndarray]] = []
    y_residuals: list[np.ndarray] = []
    t_residuals: list[np.ndarray] = []
    old_differences = {feature: [] for feature in features}
    reconstructed = []
    for refit, nuisance in enumerate(nuisances):
        predictions: dict[str, np.ndarray] = {}
        for feature in features:
            selection = selections[feature]
            if selection.get("joint_train_shape_scaling") is True:
                raw = PROPERTY.BASE.HELPER.predict_model(
                    selection["models"][refit], u_test[feature]
                )
                predictions[feature] = (
                    raw - float(selection["centers"][refit])
                ) * float(selection["joint_train_scale"])
            elif selection.get("train_coefficient_pooling") is True:
                raw = PROPERTY.BASE.HELPER.predict_model(
                    selection["pooled_model"], u_test[feature]
                )
                predictions[feature] = raw - float(selection["pooled_center"])
            else:
                raw = PROPERTY.BASE.HELPER.predict_model(
                    selection["models"][refit], u_test[feature]
                )
                predictions[feature] = raw - float(selection["centers"][refit])
        y_res = data["Y"][test] - nuisance["m_test"]
        t_res = data["T"][test] - nuisance["e_test"]
        full = effect_level + sum(predictions.values())
        reconstructed.append(full)
        for feature in features:
            ablated = full - predictions[feature]
            old_differences[feature].append(
                PROPERTY.BASE.HELPER.observation_loss(y_res, t_res, full)
                - PROPERTY.BASE.HELPER.observation_loss(y_res, t_res, ablated)
            )
        component_predictions.append(predictions)
        y_residuals.append(y_res)
        t_residuals.append(t_res)
    score = joint_cluster_score(
        y_residuals,
        t_residuals,
        component_predictions,
        features,
        float(config["test_alpha"]),
        (
            float(config["familywise_confirmation_alpha"])
            if config.get("familywise_confirmation_alpha") is not None
            else None
        ),
    )
    old_confirmation = {
        feature: PROPERTY.paired_difference_confirmation(
            np.mean(old_differences[feature], axis=0), float(config["test_alpha"])
        )
        for feature in features
    }
    records: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    acceptance = config["acceptance"]
    for spec in specs:
        feature = spec["feature"]
        selection = selections[feature]
        locations = [
            value
            for value in _location_values(selection, grid, effect_level)
            if np.isfinite(value)
        ]
        location_range = float(np.ptp(locations)) if locations else float("inf")
        stability_pass = bool(
            selection["curve_stability"]
            >= float(acceptance["minimum_curve_stability_spearman"])
            and location_range <= float(acceptance["maximum_location_range_ecdf"])
        )
        joint_result = score["components"][feature]
        if role == "all-component-null":
            records.append(
                {
                    "feature": feature,
                    "truth_label": "null",
                    "predicted_label": selection["label"],
                    "joint_score_confirmed": joint_result["confirmed"],
                    "joint_score_coefficient": joint_result["coefficient"],
                    "joint_score_standard_error": joint_result["standard_error"],
                    "joint_score_p_value": joint_result["one_sided_p_value"],
                    "joint_score_adjusted_p_value": joint_result[
                        "holm_adjusted_p_value"
                    ],
                    "joint_score_confirmation_method": joint_result[
                        "confirmation_method"
                    ],
                    "old_ablation_confirmed": old_confirmation[feature]["confirmed"],
                    "curve_stability_pass": stability_pass,
                    "feature_rank": ranks[feature],
                }
            )
            continue
        estimated_curve = effect_level + selection["consensus_component"]
        truth_curve = (
            float(property_config["target_ate"])
            + PROPERTY.component_raw(grid, spec)
            - float(data["component_centers"][feature])
        )
        nominal = spec["truth_parameter"]
        if spec["truth_label"] == "sign-reversal":
            nominal = PROPERTY.BASE.first_crossing(grid, truth_curve)
        parameter_error = float(abs(selection["parameter"] - nominal))
        rmse = float(np.sqrt(np.mean((estimated_curve - truth_curve) ** 2)))
        normalized_rmse = rmse / float(np.ptp(truth_curve))
        curve_spearman = float(spearmanr(estimated_curve, truth_curve).statistic)
        curve_pass = bool(
            curve_spearman >= float(acceptance["curve_spearman"])
            and normalized_rmse <= float(acceptance["normalized_curve_rmse"])
        )
        key_pass, key_details = _key_quantity_pass(
            spec["truth_label"],
            grid,
            estimated_curve,
            truth_curve,
            parameter_error,
            acceptance,
        )
        exact = bool(selection["label"] == spec["truth_label"])
        blind = bool(joint_result["confirmed"] and stability_pass)
        recovery = bool(blind and exact and curve_pass and key_pass)
        records.append(
            {
                "feature": feature,
                "truth_label": spec["truth_label"],
                "predicted_label": selection["label"],
                "joint_score_confirmed": joint_result["confirmed"],
                "joint_score_coefficient": joint_result["coefficient"],
                "joint_score_standard_error": joint_result["standard_error"],
                "joint_score_p_value": joint_result["one_sided_p_value"],
                "joint_score_adjusted_p_value": joint_result[
                    "holm_adjusted_p_value"
                ],
                "joint_score_confirmation_method": joint_result[
                    "confirmation_method"
                ],
                "old_ablation_confirmed": old_confirmation[feature]["confirmed"],
                "curve_stability_pass": stability_pass,
                "location_range_ecdf": location_range,
                "exact_morphology": exact,
                "curve_spearman": curve_spearman,
                "normalized_curve_rmse": normalized_rmse,
                "curve_fidelity_pass": curve_pass,
                "parameter_error_ecdf": parameter_error,
                "key_quantities_pass": key_pass,
                "key_quantity_details": key_details,
                "component_recovery_success": recovery,
                "feature_rank": ranks[feature],
            }
        )
        for u_value, truth_value, estimate_value in zip(
            grid, truth_curve, estimated_curve, strict=True
        ):
            curve_rows.append(
                {
                    "scenario": scenario_name,
                    "replication": replication,
                    "feature": feature,
                    "truth_label": spec["truth_label"],
                    "u": float(u_value),
                    "truth_curve": float(truth_value),
                    "estimate_curve": float(estimate_value),
                }
            )
    active = role == "active"
    joint_success = bool(
        active and all(record["component_recovery_success"] for record in records)
    )
    if active:
        reconstructed_curve = np.median(reconstructed, axis=0)
        truth_test = np.asarray(data["tau_true"][test], dtype=float)
        reconstructed_cate_spearman = float(
            spearmanr(reconstructed_curve, truth_test).statistic
        )
        reconstructed_cate_normalized_rmse = float(
            np.sqrt(np.mean((reconstructed_curve - truth_test) ** 2))
            / np.ptp(truth_test)
        )
        total_cate_fidelity_pass = bool(
            reconstructed_cate_spearman >= float(acceptance["curve_spearman"])
            and reconstructed_cate_normalized_rmse
            <= float(acceptance["normalized_curve_rmse"])
        )
    else:
        reconstructed_cate_spearman = float("nan")
        reconstructed_cate_normalized_rmse = float("nan")
        total_cate_fidelity_pass = False
    result = {
        "status": "completed",
        "scenario": scenario_name,
        "scenario_role": role,
        "component_count": len(features),
        "replication": replication,
        "master_seed": master_seed,
        "joint_recovery_success": joint_success,
        "all_joint_scores_confirmed": bool(
            all(record["joint_score_confirmed"] for record in records)
        ),
        "all_old_ablations_confirmed": bool(
            all(record["old_ablation_confirmed"] for record in records)
        ),
        "any_false_confirmation": bool(
            not active and any(record["joint_score_confirmed"] for record in records)
        ),
        "omnibus_confirmed": score["omnibus_confirmed"],
        "omnibus_p_value": score["omnibus_p_value"],
        "component_records_json": json.dumps(records, sort_keys=True),
        "dml_ate": ate_value,
        "morphology_effect_level": effect_level,
        "true_ate": float(np.mean(data["tau_true"])),
        "reconstructed_cate_spearman": reconstructed_cate_spearman,
        "reconstructed_cate_normalized_rmse": reconstructed_cate_normalized_rmse,
        "total_cate_fidelity_pass": total_cate_fidelity_pass,
        "warning_count": len(caught),
        "runtime_seconds": float(time.perf_counter() - started),
        "one_joint_multivariable_forest": True,
        "test_used_for_shape_or_parameter_selection": False,
        "oracle_used_for_shape_or_parameter_selection": False,
        "train_coefficient_pooling": bool(config.get("pool_train_coefficients")),
        "joint_train_shape_scaling": bool(
            config.get("joint_train_shape_scaling")
        ),
    }
    return result, curve_rows


def component_frame(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
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


def wilson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    z_value = float(norm.ppf(0.5 + confidence / 2.0))
    rate = successes / trials
    denominator = 1.0 + z_value**2 / trials
    center = (rate + z_value**2 / (2.0 * trials)) / denominator
    radius = (
        z_value
        * np.sqrt(rate * (1.0 - rate) / trials + z_value**2 / (4.0 * trials**2))
        / denominator
    )
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def write_componentwise_report(
    config: dict[str, Any],
    results: pd.DataFrame,
    components: pd.DataFrame,
    output: Path,
) -> None:
    active = results.loc[results["scenario_role"] == "active"].copy()
    active_components = components.loc[components["scenario_role"] == "active"].copy()
    null = results.loc[results["scenario_role"] == "all-component-null"].copy()
    scenario = active.groupby(["scenario", "component_count"], as_index=False).agg(
        replications=("replication", "nunique"),
        total_cate_fidelity_rate=("total_cate_fidelity_pass", "mean"),
        reconstructed_cate_spearman_median=("reconstructed_cate_spearman", "median"),
        reconstructed_cate_normalized_rmse_median=(
            "reconstructed_cate_normalized_rmse",
            "median",
        ),
    )
    component_rows = []
    for keys, group in active_components.groupby(
        ["scenario", "component_count", "feature", "truth_label"], sort=True
    ):
        successes = int(group["component_recovery_success"].sum())
        trials = int(group["replication"].nunique())
        lower, upper = wilson_interval(successes, trials)
        component_rows.append(
            {
                "scenario": keys[0],
                "component_count": int(keys[1]),
                "feature": keys[2],
                "truth_label": keys[3],
                "replications": trials,
                "successes": successes,
                "component_recovery_rate": successes / trials,
                "wilson_ci_low": lower,
                "wilson_ci_high": upper,
                "blind_evidence_rate": float(
                    (group["joint_score_confirmed"] & group["curve_stability_pass"]).mean()
                ),
                "exact_morphology_rate": float(group["exact_morphology"].mean()),
                "curve_fidelity_rate": float(group["curve_fidelity_pass"].mean()),
                "key_quantities_rate": float(group["key_quantities_pass"].mean()),
            }
        )
    component = pd.DataFrame(component_rows)
    null_rows = []
    for keys, group in null.groupby(["scenario", "component_count"], sort=True):
        false_count = int(group["any_false_confirmation"].sum())
        trials = int(group["replication"].nunique())
        lower, upper = wilson_interval(false_count, trials)
        null_rows.append(
            {
                "scenario": keys[0],
                "component_count": int(keys[1]),
                "replications": trials,
                "any_false_confirmation_count": false_count,
                "familywise_false_confirmation_rate": false_count / trials,
                "wilson_ci_low": lower,
                "wilson_ci_high": upper,
            }
        )
    null_summary = pd.DataFrame(null_rows)
    scenario.to_csv(output / "MULTI_COMPONENT_ACCEPTANCE_SCENARIOS.csv", index=False)
    component.to_csv(output / "MULTI_COMPONENT_ACCEPTANCE_COMPONENTS.csv", index=False)
    null_summary.to_csv(output / "MULTI_COMPONENT_ACCEPTANCE_NULL.csv", index=False)
    acceptance = config["acceptance"]
    checks: dict[str, bool] = {}
    counts = sorted(int(value) for value in scenario["component_count"].unique())
    for count in counts:
        active_name = f"active_count_{count}"
        null_name = f"null_count_{count}"
        component_minimum = float(
            component.loc[
                component["scenario"] == active_name, "component_recovery_rate"
            ].min()
        )
        total_rate = float(
            scenario.set_index("scenario").loc[active_name, "total_cate_fidelity_rate"]
        )
        familywise_rate = float(
            null_summary.set_index("scenario").loc[
                null_name, "familywise_false_confirmation_rate"
            ]
        )
        checks[f"count_{count}_componentwise_recovery"] = bool(
            component_minimum
            >= float(acceptance["development_minimum_component_success_rate"])
        )
        checks[f"count_{count}_total_cate_fidelity"] = bool(
            total_rate
            >= float(acceptance["development_minimum_total_cate_fidelity_rate"])
        )
        checks[f"count_{count}_familywise_null_control"] = bool(
            familywise_rate
            <= float(acceptance["maximum_null_familywise_false_confirmation_rate"])
        )
    phase = config.get("report_phase", "development")
    if phase == "confirmation":
        decision = "PASS" if all(checks.values()) else "FAIL"
        count_label = "/".join(str(value) for value in counts)
        title = f"{count_label} 组件共生恢复正式确认报告 v1"
        opening = (
            "本轮使用冻结后的 fresh seeds，构成正式 confirmation。每个组件继续执行"
            "完整单情境验收；场景按每个组件各自成功率、总 CATE fidelity 和匹配全 "
            "null family-wise 假确认率评价。"
        )
    else:
        decision = "PROCEED_TO_METHOD_FREEZE" if all(checks.values()) else "REVISE_METHOD"
        count_label = "/".join(str(value) for value in counts)
        title = f"{count_label} 组件验收方法开发报告 v1"
        opening = (
            "本轮仅使用既有开发 seeds。每个组件继续执行完整单情境验收；场景按"
            "每个组件各自成功率、总 CATE fidelity 和匹配全 null family-wise "
            "假确认率评价。"
        )
    if decision == "PASS":
        interpretation = (
            "本次结果表明，在冻结的加性设计中，每个共生成分都保持了单情境的完整"
            "形态和关键量标准；总体 CATE fidelity 与全 null family-wise 控制也同时"
            "通过。这里验证的是逐成分剥离能力，而不只是合并效应的预测准确性。"
        )
    else:
        interpretation = (
            "本次结果没有同时满足全部逐成分、总体 CATE 与全 null 门槛，因此不能"
            "把该组件数量写成已经验证的共生恢复能力。"
        )
    report = f"""# {title}

## 决策

**{decision}**

{opening}所有形态与参数由 TRAIN 固定，untouched TEST 只检验固定组件，未读取
oracle 真值进行选择。Wilson 区间用于表达有限 replication 的不确定性，不替代预先
冻结的点估计门槛。

## 研究问题与贡献

本报告检验的不是总体 CATE 能否拟合，而是多个预先指定的企业特征共同产生异质性
时，能否在控制其他活动成分后分别恢复每条函数形态及其关键量。每个数据集仍只训练
一棵联合纳入 12 个候选企业特征的 `CausalForestDML`；加性成分剥离发生在后续
TRAIN 重构层，untouched TEST 只确认已经固定的成分。

## 验收口径

- 每个组件分别计算成功率；成功要求盲态证据、准确形态、曲线 fidelity 和全部关键量
  同时通过既有单情境标准。
- 总 CATE fidelity 要求排序相关不低于 `.95` 且标准化 RMSE 不高于 `.20`。
- 匹配全 null 情境中，只要任一组件被假确认，该 replication 就记为 family-wise 错误。
- 未使用“所有组件同一 replication 同时成功”的联合概率作为验收标准。

## 场景级总 CATE

{scenario.to_markdown(index=False, floatfmt='.3f')}

## 组件级恢复

{component.to_markdown(index=False, floatfmt='.3f')}

## 全 null family-wise 校准

{null_summary.to_markdown(index=False, floatfmt='.3f')}

## 冻结前检查

{json.dumps(checks, indent=2, ensure_ascii=False)}

## 结果解释

{interpretation}

## 图示

- `FIGURE_1_COMPONENT_RECOVERY.svg`：每条共生异质性形态的独立完整恢复率。
- `FIGURE_2_TOTAL_CATE_AND_NULL_CONTROL.svg`：总 CATE fidelity 与全 null family-wise 错误率。
- `FIGURE_3_COMPONENT_CURVES.svg`：各组件跨 replication 的中位估计曲线及 10%--90% 区间。
"""
    report_filename = config.get(
        "report_filename", "MULTI_COMPONENT_ACCEPTANCE_DEVELOPMENT_REPORT_ZH.md"
    )
    (output / report_filename).write_text(report, encoding="utf-8")
    manifest = {
        "schema_version": "multi-component-acceptance-development-v1",
        "decision": decision,
        "checks": checks,
        "cells": len(results),
        "failed_cells": int((results["status"] != "completed").sum()),
        "acceptance_mode": config["acceptance_mode"],
    }
    manifest_filename = config.get(
        "manifest_filename", "MULTI_COMPONENT_ACCEPTANCE_DEVELOPMENT_MANIFEST.json"
    )
    (output / manifest_filename).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def render_componentwise_figures(
    config: dict[str, Any],
    results: pd.DataFrame,
    components: pd.DataFrame,
    curves: pd.DataFrame,
    output: Path,
) -> None:
    """Render fixed descriptive figures after all metrics have been computed."""

    def text(
        x: float,
        y: float,
        value: str,
        *,
        size: int = 13,
        anchor: str = "start",
        weight: str = "normal",
        fill: str = "#252525",
    ) -> str:
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
            f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
            f'fill="{fill}">{escape(value)}</text>'
        )

    def polyline(
        x_values: np.ndarray,
        y_values: np.ndarray,
        left: float,
        top: float,
        width: float,
        height: float,
        low: float,
        high: float,
    ) -> str:
        span = max(high - low, 1e-12)
        points = " ".join(
            f"{left + width * (x - 0.10) / 0.80:.1f},{top + height * (high - y) / span:.1f}"
            for x, y in zip(x_values, y_values, strict=True)
        )
        return points

    active_components = components.loc[components["scenario_role"] == "active"].copy()
    recovery = active_components.groupby(
        ["scenario", "component_count", "feature", "truth_label"], as_index=False
    ).agg(rate=("component_recovery_success", "mean"))
    feature_names = {
        "cash_at_slope_L": "Cash-holding change",
        "intangibles_at": "Intangible-asset intensity",
        "log_at_slope_L": "Firm-size growth",
        "ppe_at": "Fixed-asset intensity",
        "debt_at_slope_L": "Leverage change",
    }
    width, height = 1260, 660
    left, top, plot_width, plot_height = 90.0, 90.0, 1110.0, 410.0
    figure_one = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 38, "Recovery of each coexisting heterogeneity component", size=24, anchor="middle", weight="bold"),
        text(width / 2, 62, "Complete single-component standard applied separately within each joint model", size=13, anchor="middle", fill="#626c75"),
    ]
    for tick in np.linspace(0.0, 1.0, 6):
        y_value = top + plot_height * (1.0 - tick)
        figure_one.extend(
            [
                f'<line x1="{left}" y1="{y_value:.1f}" x2="{left + plot_width}" y2="{y_value:.1f}" stroke="#d9dee2"/>',
                text(left - 12, y_value + 4, f"{tick:.0%}", size=12, anchor="end", fill="#5f6972"),
            ]
        )
    threshold = float(config["acceptance"]["development_minimum_component_success_rate"])
    threshold_y = top + plot_height * (1.0 - threshold)
    figure_one.append(
        f'<line x1="{left}" y1="{threshold_y:.1f}" x2="{left + plot_width}" y2="{threshold_y:.1f}" stroke="#252525" stroke-width="1.5" stroke-dasharray="7 5"/>'
    )
    figure_one.append(text(left + plot_width, threshold_y - 8, f"Threshold {threshold:.0%}", size=12, anchor="end"))
    slot = plot_width / len(recovery)
    bar_width = slot * 0.56
    for index, row in enumerate(recovery.itertuples(index=False)):
        center = left + slot * (index + 0.5)
        bar_height = plot_height * float(row.rate)
        y_value = top + plot_height - bar_height
        color = "#1F6F78" if int(row.component_count) == 2 else "#C76D3A"
        figure_one.append(
            f'<rect x="{center - bar_width / 2:.1f}" y="{y_value:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}"/>'
        )
        figure_one.append(text(center, y_value - 9, f"{float(row.rate):.0%}", size=13, anchor="middle", weight="bold"))
        figure_one.append(text(center, 530, f"{row.component_count} components", size=12, anchor="middle", weight="bold"))
        figure_one.append(text(center, 552, feature_names.get(row.feature, row.feature), size=11, anchor="middle"))
        figure_one.append(text(center, 572, row.truth_label, size=11, anchor="middle", fill="#626c75"))
    figure_one.extend([text(30, 300, "Recovery rate", size=13, anchor="middle"), "</svg>"])
    (output / "FIGURE_1_COMPONENT_RECOVERY.svg").write_text(
        "\n".join(figure_one), encoding="utf-8"
    )

    active = results.loc[results["scenario_role"] == "active"]
    null = results.loc[results["scenario_role"] == "all-component-null"]
    fidelity = active.groupby("component_count")["total_cate_fidelity_pass"].mean()
    familywise = null.groupby("component_count")["any_false_confirmation"].mean()
    figure_two = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="570" viewBox="0 0 1160 570">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(580, 38, "Total-effect recovery and family-wise null calibration", size=24, anchor="middle", weight="bold"),
    ]
    panel_specs = [
        (60.0, "Total CATE fidelity", fidelity, float(config["acceptance"]["development_minimum_total_cate_fidelity_rate"]), "#1F6F78"),
        (620.0, "Any false confirmation under all-null", familywise, float(config["acceptance"]["maximum_null_familywise_false_confirmation_rate"]), "#C76D3A"),
    ]
    for panel_left, title, values, threshold_value, color in panel_specs:
        panel_top, panel_width, panel_height = 100.0, 470.0, 330.0
        figure_two.append(text(panel_left + panel_width / 2, 76, title, size=17, anchor="middle", weight="bold"))
        for tick in np.linspace(0.0, 1.0, 6):
            y_value = panel_top + panel_height * (1.0 - tick)
            figure_two.append(f'<line x1="{panel_left}" y1="{y_value:.1f}" x2="{panel_left + panel_width}" y2="{y_value:.1f}" stroke="#d9dee2"/>')
            figure_two.append(text(panel_left - 8, y_value + 4, f"{tick:.0%}", size=11, anchor="end", fill="#5f6972"))
        threshold_y = panel_top + panel_height * (1.0 - threshold_value)
        figure_two.append(f'<line x1="{panel_left}" y1="{threshold_y:.1f}" x2="{panel_left + panel_width}" y2="{threshold_y:.1f}" stroke="#252525" stroke-width="1.4" stroke-dasharray="7 5"/>')
        counts = list(values.index.astype(int))
        for index, count in enumerate(counts):
            center = panel_left + panel_width * (index + 1) / (len(counts) + 1)
            value = float(values.loc[count])
            bar_height = panel_height * value
            y_value = panel_top + panel_height - bar_height
            figure_two.append(f'<rect x="{center - 55:.1f}" y="{y_value:.1f}" width="110" height="{bar_height:.1f}" fill="{color}"/>')
            figure_two.append(text(center, y_value - 8, f"{value:.0%}", size=13, anchor="middle", weight="bold"))
            figure_two.append(text(center, 458, f"{count} components", size=12, anchor="middle"))
        figure_two.append(text(panel_left + panel_width, threshold_y - 7, f"Threshold {threshold_value:.0%}", size=11, anchor="end"))
    figure_two.append("</svg>")
    (output / "FIGURE_2_TOTAL_CATE_AND_NULL_CONTROL.svg").write_text(
        "\n".join(figure_two), encoding="utf-8"
    )

    active_curves = curves.loc[curves["scenario"].str.startswith("active_")].copy()
    curve_summary = active_curves.groupby(
        ["scenario", "feature", "truth_label", "u"], as_index=False
    ).agg(
        truth=("truth_curve", "mean"),
        estimate=("estimate_curve", "median"),
        estimate_q10=("estimate_curve", lambda values: values.quantile(0.10)),
        estimate_q90=("estimate_curve", lambda values: values.quantile(0.90)),
    )
    panels = list(
        curve_summary[["scenario", "feature", "truth_label"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    figure_three = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="820" viewBox="0 0 1320 820">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(660, 34, "Recovered component curves across fresh replications", size=24, anchor="middle", weight="bold"),
        '<line x1="455" y1="58" x2="495" y2="58" stroke="#202020" stroke-width="2.5"/>',
        text(502, 63, "Truth", size=12),
        '<line x1="585" y1="58" x2="625" y2="58" stroke="#1F6F78" stroke-width="2.5"/>',
        text(632, 63, "Median estimate", size=12),
        '<rect x="790" y="49" width="40" height="16" fill="#A7CED1" fill-opacity="0.55"/>',
        text(837, 63, "10%--90% across replications", size=12),
    ]
    panel_width, panel_height = 360.0, 260.0
    positions = [(70.0, 110.0), (480.0, 110.0), (890.0, 110.0), (275.0, 475.0), (685.0, 475.0)]
    for (scenario_name, feature, truth_label), (panel_left, panel_top) in zip(
        panels, positions[: len(panels)], strict=True
    ):
        current = curve_summary.loc[
            (curve_summary["scenario"] == scenario_name)
            & (curve_summary["feature"] == feature)
        ]
        values = current[["truth", "estimate", "estimate_q10", "estimate_q90"]].to_numpy(float)
        low = float(np.min(values))
        high = float(np.max(values))
        padding = max((high - low) * 0.10, 0.02)
        low -= padding
        high += padding
        x_values = current["u"].to_numpy(float)
        upper = polyline(x_values, current["estimate_q90"].to_numpy(float), panel_left, panel_top, panel_width, panel_height, low, high)
        lower = polyline(x_values[::-1], current["estimate_q10"].to_numpy(float)[::-1], panel_left, panel_top, panel_width, panel_height, low, high)
        truth_points = polyline(x_values, current["truth"].to_numpy(float), panel_left, panel_top, panel_width, panel_height, low, high)
        estimate_points = polyline(x_values, current["estimate"].to_numpy(float), panel_left, panel_top, panel_width, panel_height, low, high)
        figure_three.extend(
            [
                f'<rect x="{panel_left}" y="{panel_top}" width="{panel_width}" height="{panel_height}" fill="#ffffff" stroke="#cfd6dc"/>',
                f'<polygon points="{upper} {lower}" fill="#A7CED1" fill-opacity="0.55"/>',
                f'<polyline points="{truth_points}" fill="none" stroke="#202020" stroke-width="2.5"/>',
                f'<polyline points="{estimate_points}" fill="none" stroke="#1F6F78" stroke-width="2.5"/>',
            ]
        )
        count = 2 if scenario_name.endswith("_2") else 3
        figure_three.append(text(panel_left + panel_width / 2, panel_top - 36, f"{count} components | {feature_names.get(feature, feature)}", size=14, anchor="middle", weight="bold"))
        figure_three.append(text(panel_left + panel_width / 2, panel_top - 17, truth_label, size=12, anchor="middle", fill="#626c75"))
        figure_three.append(text(panel_left, panel_top + panel_height + 20, "0.10", size=10, anchor="middle"))
        figure_three.append(text(panel_left + panel_width, panel_top + panel_height + 20, "0.90", size=10, anchor="middle"))
        figure_three.append(text(panel_left + panel_width / 2, panel_top + panel_height + 39, "Reference ECDF", size=11, anchor="middle"))
        figure_three.append(text(panel_left - 8, panel_top + 4, f"{high:.2f}", size=10, anchor="end"))
        figure_three.append(text(panel_left - 8, panel_top + panel_height, f"{low:.2f}", size=10, anchor="end"))
    figure_three.append("</svg>")
    (output / "FIGURE_3_COMPONENT_CURVES.svg").write_text(
        "\n".join(figure_three), encoding="utf-8"
    )


def write_report(
    config: dict[str, Any], results: pd.DataFrame, components: pd.DataFrame, output: Path
) -> None:
    if config.get("acceptance_mode") == "componentwise-v1":
        write_componentwise_report(config, results, components, output)
        return
    active = results.loc[results["scenario_role"] == "active"].copy()
    null = components.loc[components["scenario_role"] == "all-component-null"].copy()
    scenario = active.groupby(
        ["scenario", "component_count"], as_index=False
    ).agg(
        replications=("replication", "nunique"),
        joint_score_all_confirmed_rate=("all_joint_scores_confirmed", "mean"),
        old_ablation_all_confirmed_rate=("all_old_ablations_confirmed", "mean"),
        joint_recovery_success_rate=("joint_recovery_success", "mean"),
        reconstructed_cate_spearman_median=("reconstructed_cate_spearman", "median"),
    )
    active_components = components.loc[components["scenario_role"] == "active"]
    component = active_components.groupby(
        ["scenario", "component_count", "feature", "truth_label"], as_index=False
    ).agg(
        replications=("replication", "nunique"),
        joint_score_confirmation_rate=("joint_score_confirmed", "mean"),
        old_ablation_confirmation_rate=("old_ablation_confirmed", "mean"),
        exact_morphology_rate=("exact_morphology", "mean"),
        curve_fidelity_rate=("curve_fidelity_pass", "mean"),
        key_quantities_rate=("key_quantities_pass", "mean"),
        component_recovery_rate=("component_recovery_success", "mean"),
    )
    pooled_null_rate = float(null["joint_score_confirmed"].mean())
    old_null_rate = float(null["old_ablation_confirmed"].mean())
    scenario.to_csv(output / "JOINT_CONFIRMATION_PILOT_SUMMARY.csv", index=False)
    component.to_csv(output / "JOINT_CONFIRMATION_PILOT_COMPONENTS.csv", index=False)
    checks = {
        "active_count_2_development_signal": bool(
            scenario.set_index("scenario").loc[
                "active_count_2", "joint_recovery_success_rate"
            ]
            >= float(config["acceptance"]["development_minimum_joint_success_rate"])
        ),
        "active_count_3_development_signal": bool(
            scenario.set_index("scenario").loc[
                "active_count_3", "joint_recovery_success_rate"
            ]
            >= float(config["acceptance"]["development_minimum_joint_success_rate"])
        ),
        "pooled_null_false_confirmation_controlled": bool(
            pooled_null_rate
            <= float(
                config["acceptance"]["maximum_pooled_null_false_confirmation_rate"]
            )
        ),
    }
    decision = "PROCEED_TO_METHOD_FREEZE" if all(checks.values()) else "REVISE_METHOD"
    report = f"""# 多组件联合确认开发 Pilot v1

## 决策

**{decision}**

本轮只使用开发种子，不构成正式 confirmation。两/三组件的形态和参数仍由 TRAIN
固定；untouched TEST 只估计同时控制其他组件后的联合正交确认系数。三次 nuisance
refit 以企业为 cluster 合并，未把同一 TEST 企业重复当作独立样本。

## 主情境

{scenario.to_markdown(index=False, floatfmt='.3f')}

## 逐组件

{component.to_markdown(index=False, floatfmt='.3f')}

## Null 校准

- 联合 score 的 pooled false-confirmation rate：{pooled_null_rate:.1%}
- 旧 ablation-loss 的 pooled false-confirmation rate：{old_null_rate:.1%}
- Null 组件检验数：{len(null)}

## 检查

{json.dumps(checks, indent=2, ensure_ascii=False)}

如果本轮通过，下一步是冻结 runner、配置、阈值和 fresh seeds，然后对两组件与三组件
各运行 20 次独立 confirmation。正式成功标准保持不变：一个 replication 只有所有
组件同时通过盲态证据、形态、曲线和关键量才成功；场景门槛为 90%。
"""
    (output / "JOINT_CONFIRMATION_PILOT_REPORT_ZH.md").write_text(
        report, encoding="utf-8"
    )
    manifest = {
        "schema_version": "multi-component-joint-confirmation-pilot-result-v1",
        "decision": decision,
        "checks": checks,
        "pooled_null_false_confirmation_rate": pooled_null_rate,
        "cells": len(results),
        "failed_cells": int((results["status"] != "completed").sum()),
    }
    (output / "JOINT_CONFIRMATION_PILOT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    property_path, property_config, method, backbone = validate_config(
        config_path, config
    )
    source = read_json(resolve(property_path, property_config["source_config"]))
    output = resolve(config_path, config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "JOINT_CONFIRMATION_PILOT_LONG.csv"
    work_path = output / "JOINT_CONFIRMATION_PILOT_WORK.jsonl"
    curve_path = output / "JOINT_CONFIRMATION_PILOT_CURVES.csv"
    seeds = seed_grid(config["master_seed"], config["replications"])
    completed: set[tuple[str, int]] = set()
    if work_path.is_file():
        existing = pd.DataFrame(
            [
                json.loads(line)
                for line in work_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )
        completed = set(
            zip(existing["scenario"], existing["replication"].astype(int), strict=False)
        )
    total = len(config["scenarios"]) * int(config["replications"])
    done = len(completed)
    for replication in range(int(config["replications"])):
        for scenario in config["scenarios"]:
            if (scenario, replication) in completed:
                continue
            print(
                f"PILOT {done + 1}/{total} scenario={scenario} replication={replication}",
                flush=True,
            )
            try:
                result, curves = run_cell(
                    config_path,
                    config,
                    property_path,
                    property_config,
                    method,
                    backbone,
                    source,
                    scenario,
                    replication,
                    seeds[replication],
                )
            except Exception as error:  # noqa: BLE001 - failures must remain visible
                result = {
                    "status": "failed",
                    "scenario": scenario,
                    "scenario_role": config["scenarios"][scenario]["role"],
                    "component_count": config["scenarios"][scenario]["component_count"],
                    "replication": replication,
                    "master_seed": seeds[replication],
                    "error": f"{type(error).__name__}: {error}",
                    "component_records_json": "[]",
                }
                curves = []
            with work_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, sort_keys=True) + "\n")
            if curves:
                pd.DataFrame(curves).to_csv(
                    curve_path,
                    mode="a",
                    header=not curve_path.exists(),
                    index=False,
                )
            done += 1
    results = pd.DataFrame(
        [
            json.loads(line)
            for line in work_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    )
    if (results["status"] != "completed").any():
        raise RuntimeError("pilot contains failed cells")
    results.to_csv(result_path, index=False)
    components = component_frame(results)
    components.to_csv(output / "JOINT_CONFIRMATION_PILOT_COMPONENT_LONG.csv", index=False)
    if config.get("acceptance_mode") == "componentwise-v1":
        curves = pd.read_csv(curve_path)
        render_componentwise_figures(config, results, components, curves, output)
    write_report(config, results, components, output)
    print(f"Completed {len(results)} cells; outputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
