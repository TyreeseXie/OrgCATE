"""Exploratory pilot for two additive CATE morphologies in one DGP."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
import traceback
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "multi_morphology_pilot_v1" / "config" / "pilot.json"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import execution.run_morphology_product_v1 as PRODUCT  # noqa: I001


HELPER = PRODUCT.HELPER
BASE = PRODUCT.BASE
V2 = PRODUCT.V2
FAMILIES = ("constant", "linear", "quadratic", "tanh", "kink", "step")
FAMILY_COMPLEXITY = {
    "linear": 1,
    "quadratic": 2,
    "tanh": 2,
    "kink": 2,
    "step": 2,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_config(
    config_path: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if config["schema_version"] != "multi-morphology-additive-pilot-v1":
        raise ValueError("wrong schema version")
    if int(config["n"]) != 3000 or not np.isclose(config["test_size"], 0.20):
        raise ValueError("pilot must retain n=3000 and the frozen 80/20 split")
    if int(config["backfit_iterations"]) != 3:
        raise ValueError("three additive backfitting iterations are frozen")
    if int(config["same_data_nuisance_refits"]) != 3:
        raise ValueError("three nuisance refits are required")
    if config["selection_policy"] != {
        "one_joint_multivariable_forest": True,
        "active_features_prespecified_for_reconstruction": True,
        "test_used_for_shape_selection": False,
        "oracle_used_for_shape_selection": False,
        "rank_is_information_only": True,
    }:
        raise ValueError("selection policy changed")
    source = read_json(resolve(config_path, config["source_config"]))
    method = read_json(resolve(config_path, config["method_config"]))
    if method["primary_forest"] != {
        "class": "econml.dml.CausalForestDML",
        "n_estimators": 1000,
        "min_samples_leaf": 5,
        "max_samples": 0.45,
        "cv": 2,
        "honest": True,
        "inference": False,
        "n_jobs": 2,
    }:
        raise ValueError("primary forest contract changed")
    active = [
        component["feature"]
        for scenario in config["scenarios"].values()
        for component in scenario["components"]
    ]
    missing = sorted(set(active) - set(source["features"]))
    if missing:
        raise ValueError(f"active features are not in the 12-feature forest: {missing}")
    if any(len(item["components"]) != 2 for item in config["scenarios"].values()):
        raise ValueError("this pilot requires exactly two additive components")
    return source, method


def load_inputs(
    config_path: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    source, method = validate_config(config_path, config)
    source_path = resolve(config_path, config["source_config"])
    backbone = V2.add_rank_columns(
        V2.v2.load_frozen_backbone(ROOT, source, source_path), source["features"]
    )
    return source, method, backbone


def component_raw(u: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    value = np.asarray(u, dtype=float)
    shape = spec["shape"]
    if shape == "linear":
        return 2.0 * float(spec["half_range"]) * (value - 0.5)
    if shape == "inverted_u":
        center = float(spec["center"])
        return float(spec["amplitude"]) * (1.0 - 4.0 * (value - center) ** 2)
    if shape == "sign_reversal":
        return float(spec["slope"]) * (value - float(spec["center"]))
    if shape == "threshold":
        return float(spec["contrast"]) * (value >= float(spec["boundary"]))
    raise ValueError(f"unsupported component shape: {shape}")


def build_data(
    backbone: pd.DataFrame,
    source: dict[str, Any],
    method: dict[str, Any],
    config: dict[str, Any],
    scenario_name: str,
    lineage: dict[str, int],
    base_parameters: dict[str, Any],
) -> dict[str, Any]:
    base = BASE.build_data(
        backbone,
        source,
        int(config["n"]),
        "homogeneous",
        lineage,
        base_parameters,
    )
    target = float(config["target_ate"])
    component_truth: dict[str, np.ndarray] = {}
    component_centers: dict[str, float] = {}
    u_values: dict[str, np.ndarray] = {}
    for spec in config["scenarios"][scenario_name]["components"]:
        feature = spec["feature"]
        u = BASE.reference_u(backbone, base["unit_id"], feature)
        raw = component_raw(u, spec)
        center = float(np.mean(raw))
        u_values[feature] = u
        component_centers[feature] = center
        component_truth[feature] = raw - center
    tau = target + sum(component_truth.values())
    tau = tau + (target - float(np.mean(tau)))
    y0 = base["Y"] - base["T"] * base["tau_true"]
    y1 = y0 + tau
    return {
        **base,
        "Y": y0 + base["T"] * tau,
        "tau_true": tau,
        "y0_true": y0,
        "y1_true": y1,
        "component_truth": component_truth,
        "component_centers": component_centers,
        "u_values": u_values,
        "method_n": int(method["n"]),
    }


def library_config(method: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(method)
    result["selection"] = {
        "structured_families": list(FAMILIES),
        "flexible_family": "spline",
        "crossing_support": [0.10, 0.90],
    }
    return result


def backfit_components(
    features: list[str],
    u_train: dict[str, np.ndarray],
    y_residual: np.ndarray,
    t_residual: np.ndarray,
    iterations: int,
) -> dict[str, np.ndarray]:
    components = {
        feature: np.zeros(len(y_residual), dtype=float) for feature in features
    }
    for _ in range(int(iterations)):
        for feature in features:
            other = sum(
                (components[name] for name in features if name != feature),
                start=np.zeros(len(y_residual), dtype=float),
            )
            adjusted = y_residual - t_residual * other
            model = HELPER.fit_model(
                "spline",
                u_train[feature],
                adjusted,
                t_residual,
                {"knots": 6, "alpha": 0.01},
            )
            fitted = HELPER.predict_model(model, u_train[feature])
            components[feature] = fitted - float(np.mean(fitted))
    return components


def aggregate_family_losses(libraries: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        losses = np.concatenate(
            [
                np.asarray(item["models"][family]["fold_losses"], dtype=float)
                for item in libraries
            ]
        )
        rows.append(
            {
                "family": family,
                "mean_loss": float(np.mean(losses)),
                "se_loss": float(np.std(losses, ddof=1) / math.sqrt(len(losses))),
                "fold_losses": losses,
            }
        )
    return pd.DataFrame(rows)


def first_crossing(grid: np.ndarray, curve: np.ndarray) -> float:
    indices = np.flatnonzero(curve[:-1] * curve[1:] <= 0.0)
    if not len(indices):
        return float("nan")
    index = int(indices[0])
    left, right = float(curve[index]), float(curve[index + 1])
    if np.isclose(left, right):
        return float((grid[index] + grid[index + 1]) / 2.0)
    return float(grid[index] - left * (grid[index + 1] - grid[index]) / (right - left))


def select_component(
    libraries: list[dict[str, Any]],
    u_train: list[np.ndarray],
    grid: np.ndarray,
    ate_value: float,
) -> dict[str, Any]:
    table = aggregate_family_losses(libraries)
    nonconstant = table[table["family"] != "constant"]
    best = nonconstant.sort_values(["mean_loss", "family"]).iloc[0]
    eligible = nonconstant[
        nonconstant["mean_loss"] <= float(best["mean_loss"] + best["se_loss"])
    ].copy()
    eligible["complexity"] = eligible["family"].map(FAMILY_COMPLEXITY)
    selected_family = str(
        eligible.sort_values(["complexity", "mean_loss", "family"]).iloc[0]["family"]
    )
    models = [item["models"][selected_family] for item in libraries]
    centered_curves = []
    centers = []
    for model, values in zip(models, u_train, strict=True):
        center = float(np.mean(HELPER.predict_model(model, values)))
        centers.append(center)
        centered_curves.append(HELPER.predict_model(model, grid) - center)
    consensus = np.median(centered_curves, axis=0)
    marginal_curve = float(ate_value) + consensus
    family = selected_family
    parameter = float("nan")
    details: dict[str, float] = {}
    if family == "constant":
        label = "homogeneous"
    elif family == "linear":
        slopes = [float(np.asarray(model["coefficient"])[1]) for model in models]
        parameter = first_crossing(grid, marginal_curve)
        label = "sign-reversal" if np.isfinite(parameter) else "linear-monotone"
        details = {"slope": float(np.median(slopes)), "crossing": parameter}
    elif family == "quadratic":
        curvatures = [
            float(2.0 * np.asarray(model["coefficient"])[1]) for model in models
        ]
        centers_param = [float(model["params"]["c"]) for model in models]
        parameter = float(np.median(centers_param))
        label = "u-shaped" if float(np.median(curvatures)) > 0 else "inverted-u"
        details = {
            "turning_point": parameter,
            "curvature": float(np.median(curvatures)),
        }
    elif family == "step":
        boundaries = [float(model["params"]["c"]) for model in models]
        jumps = [float(np.asarray(model["coefficient"])[1]) for model in models]
        parameter = float(np.median(boundaries))
        label = "threshold-step"
        details = {"boundary": parameter, "contrast": float(np.median(jumps))}
    elif family == "kink":
        locations = [float(model["params"]["c"]) for model in models]
        parameter = float(np.median(locations))
        label = "kinked"
        details = {"kink": parameter}
    else:
        ks = [float(model["params"]["k"]) for model in models]
        centers_param = [float(model["params"]["c"]) for model in models]
        parameter = first_crossing(grid, marginal_curve)
        if np.isfinite(parameter):
            label = "sign-reversal"
            details = {"crossing": parameter, "k": float(np.median(ks))}
        else:
            parameter = float(np.median(centers_param))
            label = (
                "steep-continuous"
                if float(np.median(ks)) >= 8.0
                else "nonlinear-monotone"
            )
            details = {"center": parameter, "k": float(np.median(ks))}
    stability_values = []
    for left, right in combinations(centered_curves, 2):
        if np.std(left) > 1e-12 and np.std(right) > 1e-12:
            stability_values.append(float(spearmanr(left, right).statistic))
    return {
        "family": selected_family,
        "label": label,
        "parameter": parameter,
        "details": details,
        "models": models,
        "centers": centers,
        "curves": centered_curves,
        "consensus_component": consensus,
        "curve_stability": (
            float(np.nanmedian(stability_values)) if stability_values else float("nan")
        ),
        "loss_table": table.drop(columns="fold_losses").to_dict(orient="records"),
    }


def paired_loss_confirmation(
    y_residual: np.ndarray,
    t_residual: np.ndarray,
    candidate: np.ndarray,
    comparator: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    difference = HELPER.observation_loss(
        y_residual, t_residual, candidate
    ) - HELPER.observation_loss(y_residual, t_residual, comparator)
    mean = float(np.mean(difference))
    se = float(np.std(difference, ddof=1) / math.sqrt(len(difference)))
    upper = mean + float(norm.ppf(1.0 - alpha)) * se
    return {
        "difference": mean,
        "se": se,
        "upper": upper,
        "confirmed": bool(upper < 0.0),
    }


def truth_component_curve(
    grid: np.ndarray,
    spec: dict[str, Any],
    center: float,
    target_ate: float,
) -> np.ndarray:
    return float(target_ate) + component_raw(grid, spec) - float(center)


def run_cell(
    backbone: pd.DataFrame,
    source: dict[str, Any],
    method: dict[str, Any],
    config: dict[str, Any],
    phase: str,
    scenario_name: str,
    replication: int,
    master_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    lineage = BASE.lineage(master_seed, 0)
    data = build_data(
        backbone,
        source,
        method,
        config,
        scenario_name,
        lineage,
        read_json(
            (
                resolve(DEFAULT_CONFIG, config["method_config"]).parent
                / method["base_config"]
            ).resolve()
        )["shape_parameters"],
    )
    train, test = V2.v2.split_once(
        data["T"], data["unit_id"], float(config["test_size"]), lineage["split_seed"]
    )
    specs = config["scenarios"][scenario_name]["components"]
    features = [item["feature"] for item in specs]
    u_train = {feature: data["u_values"][feature][train] for feature in features}
    u_test = {feature: data["u_values"][feature][test] for feature in features}

    ate = PRODUCT.estimate_ate(
        data["X"][train],
        data["Y"][train],
        data["T"][train],
        lineage["nuisance_seed"],
        method,
    )
    ate_value = float(ate["ate_dml"])
    common = dict(source["common"])
    common["n_jobs"] = int(method["primary_forest"]["n_jobs"])
    forest = V2.v2._new_m1(common, lineage["forest_seed"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forest.fit(data["Y"][train], data["T"][train], X=data["X"][train], W=None)
    forest_test = np.asarray(forest.effect(data["X"][test]), dtype=float).reshape(-1)
    importance = np.asarray(forest.feature_importances_, dtype=float)
    ranks = {
        feature: int(
            np.sum(importance > importance[source["features"].index(feature)]) + 1
        )
        for feature in features
    }

    cfg = library_config(method)
    theory_libraries = {feature: [] for feature in features}
    nuisances = []
    for refit in range(int(config["same_data_nuisance_refits"])):
        seeds = HELPER.refit_seeds(master_seed, refit)
        nuisance = V2.crossfit_nuisance(
            data["X"][train],
            data["Y"][train],
            data["T"][train],
            data["X"][test],
            seeds["nuisance_refit_seed"],
            int(config["nuisance_crossfit_folds"]),
        )
        flexible = backfit_components(
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
                nuisance["y_residual_train"] - nuisance["t_residual_train"] * other
            )
            theory_libraries[feature].append(
                HELPER.fit_library(
                    u_train[feature],
                    adjusted,
                    nuisance["t_residual_train"],
                    cfg,
                    seeds["shape_cv_refit_seed"] + 1009 * index,
                )
            )
        nuisances.append(nuisance)

    grid = np.linspace(0.10, 0.90, 161)
    selections = {
        feature: select_component(
            theory_libraries[feature],
            [u_train[feature]] * len(theory_libraries[feature]),
            grid,
            ate_value,
        )
        for feature in features
    }

    full_confirmations = []
    ablation_confirmations = {feature: [] for feature in features}
    reconstructed_test = []
    for refit, nuisance in enumerate(nuisances):
        components_test = {}
        for feature in features:
            selection = selections[feature]
            raw = HELPER.predict_model(selection["models"][refit], u_test[feature])
            components_test[feature] = raw - float(selection["centers"][refit])
        full = ate_value + sum(components_test.values())
        reconstructed_test.append(full)
        y_test = data["Y"][test] - nuisance["m_test"]
        t_test = data["T"][test] - nuisance["e_test"]
        constant = np.full(len(test), ate_value, dtype=float)
        full_confirmations.append(
            paired_loss_confirmation(y_test, t_test, full, constant, 0.05)
        )
        for feature in features:
            ablated = full - components_test[feature]
            ablation_confirmations[feature].append(
                paired_loss_confirmation(y_test, t_test, full, ablated, 0.05)
            )

    reconstructed = np.median(reconstructed_test, axis=0)
    full_rate = float(np.mean([item["confirmed"] for item in full_confirmations]))
    component_rates = {
        feature: float(np.mean([item["confirmed"] for item in values]))
        for feature, values in ablation_confirmations.items()
    }
    component_records = []
    curve_rows = []
    for spec in specs:
        feature = spec["feature"]
        selection = selections[feature]
        estimated_curve = ate_value + selection["consensus_component"]
        truth_curve = truth_component_curve(
            grid,
            spec,
            data["component_centers"][feature],
            float(config["target_ate"]),
        )
        nominal = spec["truth_parameter"]
        if spec["truth_label"] == "sign-reversal":
            nominal = first_crossing(grid, truth_curve)
        record = {
            "feature": feature,
            "truth_label": spec["truth_label"],
            "predicted_label": selection["label"],
            "exact_label_correct": bool(selection["label"] == spec["truth_label"]),
            "selected_family": selection["family"],
            "test_ablation_confirmation_rate": component_rates[feature],
            "test_component_confirmed": bool(
                component_rates[feature]
                >= float(config["minimum_component_test_confirmation_rate"])
            ),
            "curve_rmse": float(np.sqrt(np.mean((estimated_curve - truth_curve) ** 2))),
            "curve_spearman": float(spearmanr(estimated_curve, truth_curve).statistic),
            "curve_stability": selection["curve_stability"],
            "estimated_parameter": selection["parameter"],
            "truth_parameter": nominal,
            "parameter_abs_error": (
                float(abs(selection["parameter"] - nominal))
                if nominal is not None and np.isfinite(selection["parameter"])
                else float("nan")
            ),
            "feature_rank": ranks[feature],
            "details": selection["details"],
        }
        component_records.append(record)
        for value, truth, estimate in zip(
            grid, truth_curve, estimated_curve, strict=True
        ):
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

    both_exact = bool(all(item["exact_label_correct"] for item in component_records))
    both_confirmed = bool(
        all(item["test_component_confirmed"] for item in component_records)
    )
    state = (
        "SUPPORTED_MULTIPLE_COMPONENTS"
        if both_confirmed
        and full_rate >= float(config["minimum_full_test_confirmation_rate"])
        else "INCONCLUSIVE"
    )
    result = {
        "status": "completed",
        "phase": phase,
        "scenario": scenario_name,
        "replication": replication,
        "master_seed": master_seed,
        "n": int(config["n"]),
        "practical_state": state,
        "both_labels_exact": both_exact,
        "both_components_test_confirmed": both_confirmed,
        "full_test_confirmation_rate": full_rate,
        "component_records_json": json.dumps(component_records, sort_keys=True),
        "forest_test_cate_rmse": float(
            np.sqrt(np.mean((forest_test - data["tau_true"][test]) ** 2))
        ),
        "forest_test_cate_spearman": float(
            spearmanr(forest_test, data["tau_true"][test]).statistic
        ),
        "reconstructed_test_cate_rmse": float(
            np.sqrt(np.mean((reconstructed - data["tau_true"][test]) ** 2))
        ),
        "reconstructed_test_cate_spearman": float(
            spearmanr(reconstructed, data["tau_true"][test]).statistic
        ),
        "true_ate": float(np.mean(data["tau_true"])),
        "dml_ate": ate_value,
        "dml_ate_error": float(ate_value - np.mean(data["tau_true"])),
        "mean_reconstructed_cate_test": float(np.mean(reconstructed)),
        "ate_cate_gap_sd_y": float(
            abs(np.mean(reconstructed) - ate_value) / np.std(data["Y"][train], ddof=1)
        ),
        "active_feature_ranks_json": json.dumps(ranks, sort_keys=True),
        "warning_count": len(caught),
        "runtime_seconds": float(time.perf_counter() - started),
        "one_joint_multivariable_forest": True,
        "test_used_for_shape_selection": False,
        "oracle_used_for_shape_selection": False,
    }
    return result, curve_rows


def paths(config_path: Path, config: dict[str, Any], phase: str) -> dict[str, Path]:
    output = resolve(config_path, config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    prefix = phase.upper()
    return {
        "output": output,
        "results": output / f"MULTI_MORPHOLOGY_{prefix}_LONG.csv",
        "curves": output / f"MULTI_MORPHOLOGY_{prefix}_CURVES.csv",
    }


def append(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def create_freeze(config_path: Path, config: dict[str, Any]) -> Path:
    source, _ = validate_config(config_path, config)
    source_path = resolve(config_path, config["source_config"])
    backbone_path = (source_path.parent / source["backbone_path"]).resolve()
    freeze = {
        "status": "REV3_FINAL_FROZEN_BEFORE_CONFIRMATION",
        "frozen_at": "2026-10-23",
        "development_disclosure": (
            "Revision 0 collapsed genuine components to constant under an over-conservative "
            "one-SE shortcut. Revision 1 separated TRAIN nonconstant-family discovery from "
            "TEST ablation but exposed an unintended 0.3-0.7 component range versus the "
            "existing 2-5 unit single-shape capability range. Revision 2 freezes components "
            "at approximately 50% of the established single-shape ranges. Revision 3 corrects "
            "the strengthened linear component's implied label to sign reversal and applies "
            "one-SE simplicity only among nonconstant families. Confirmation seeds remain "
            "unseen and unchanged."
        ),
        "confirmation_seed_grid": seed_grid(
            config["confirmation"]["master_seed"],
            config["confirmation"]["replications"],
        ),
        "hashes": {
            "config_sha256": sha256_file(config_path),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "source_config_sha256": sha256_file(source_path),
            "method_config_sha256": sha256_file(
                resolve(config_path, config["method_config"])
            ),
            "backbone_sha256": sha256_file(backbone_path),
        },
    }
    freeze_path = resolve(config_path, config["freeze_path"])
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    return freeze_path


def verify_freeze(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    freeze_path = resolve(config_path, config["freeze_path"])
    if not freeze_path.is_file():
        raise RuntimeError("multi-morphology method freeze is missing")
    freeze = read_json(freeze_path)
    source = read_json(resolve(config_path, config["source_config"]))
    source_path = resolve(config_path, config["source_config"])
    backbone_path = (source_path.parent / source["backbone_path"]).resolve()
    current = {
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "source_config_sha256": sha256_file(source_path),
        "method_config_sha256": sha256_file(
            resolve(config_path, config["method_config"])
        ),
        "backbone_sha256": sha256_file(backbone_path),
    }
    if freeze["hashes"] != current:
        raise RuntimeError("multi-morphology method freeze mismatch")
    return freeze


def run_phase(config_path: Path, config: dict[str, Any], phase: str) -> None:
    verify_freeze(config_path, config)
    source, method, backbone = load_inputs(config_path, config)
    output = paths(config_path, config, phase)
    phase_config = config[phase]
    seeds = seed_grid(phase_config["master_seed"], phase_config["replications"])
    completed: set[tuple[str, int]] = set()
    if output["results"].is_file():
        existing = pd.read_csv(output["results"])
        completed = set(
            zip(existing["scenario"], existing["replication"].astype(int), strict=False)
        )
    total = len(config["scenarios"]) * int(phase_config["replications"])
    done = len(completed)
    for replication in range(int(phase_config["replications"])):
        for scenario in config["scenarios"]:
            if (scenario, replication) in completed:
                continue
            print(
                f"{phase.upper()} {done + 1}/{total} scenario={scenario} replication={replication}",
                flush=True,
            )
            try:
                result, curves = run_cell(
                    backbone,
                    source,
                    method,
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
            append(output["results"], [result])
            if curves:
                append(output["curves"], curves)
            done += 1


def render_svg(curves: pd.DataFrame, output: Path) -> None:
    panels = list(
        curves[["scenario", "feature", "truth_label"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    width, height = 1280, 760
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#20252b}.title{font-size:24px;font-weight:700}.sub{font-size:13px;fill:#5c6670}.panel{fill:#fff;stroke:#d6dce1}.name{font-size:14px;font-weight:700}.small{font-size:11px}.truth{fill:none;stroke:#20252b;stroke-width:2.2;stroke-dasharray:6 4}.estimate{fill:none;stroke:#00796b;stroke-width:2.8}.ate{stroke:#c96b2c;stroke-width:1.5;stroke-dasharray:3 4}</style>",
        '<rect width="100%" height="100%" fill="#f7f8f9"/>',
        '<text x="30" y="38" class="title">Two coexisting CATE morphologies recovered from one joint forest</text>',
        '<text x="30" y="62" class="sub">black dashed = oracle component curve; green = median TRAIN reconstruction; orange = DML ATE</text>',
    ]
    for index, (scenario, feature, truth_label) in enumerate(panels):
        row, column = divmod(index, 2)
        x0, y0 = 30 + column * 620, 90 + row * 320
        frame = curves[
            (curves["scenario"] == scenario) & (curves["feature"] == feature)
        ]
        grouped = frame.groupby("u", as_index=False).agg(
            truth=("truth_curve", "mean"),
            estimate=("estimate_curve", "median"),
            ate=("ate_dml", "median"),
        )
        low = float(grouped[["truth", "estimate", "ate"]].min().min()) - 0.05
        high = float(grouped[["truth", "estimate", "ate"]].max().max()) + 0.05

        def point(
            u: float,
            value: float,
            panel_x: float = x0,
            panel_y: float = y0,
            minimum: float = low,
            maximum: float = high,
        ) -> tuple[float, float]:
            return (
                panel_x + 45 + (u - 0.10) / 0.80 * 510,
                panel_y + 255 - (value - minimum) / (maximum - minimum) * 190,
            )

        def poly(column_name: str, panel: pd.DataFrame = grouped) -> str:
            return " ".join(
                f"{point(float(u), float(value))[0]:.1f},{point(float(u), float(value))[1]:.1f}"
                for u, value in zip(panel["u"], panel[column_name], strict=True)
            )

        ate_y = point(0.10, float(grouped["ate"].iloc[0]))[1]
        lines.extend(
            [
                f'<rect x="{x0}" y="{y0}" width="590" height="290" class="panel"/>',
                f'<text x="{x0 + 18}" y="{y0 + 25}" class="name">{feature}: {truth_label}</text>',
                f'<text x="{x0 + 18}" y="{y0 + 44}" class="small">{scenario}</text>',
                f'<line x1="{x0 + 45}" x2="{x0 + 555}" y1="{ate_y:.1f}" y2="{ate_y:.1f}" class="ate"/>',
                f'<polyline points="{poly("truth")}" class="truth"/>',
                f'<polyline points="{poly("estimate")}" class="estimate"/>',
                f'<text x="{x0 + 45}" y="{y0 + 275}" class="small">10th percentile support</text>',
                f'<text x="{x0 + 455}" y="{y0 + 275}" class="small">90th percentile</text>',
            ]
        )
    lines.append("</svg>")
    output.write_text("\n".join(lines), encoding="utf-8")


def write_report(config_path: Path, config: dict[str, Any]) -> None:
    verify_freeze(config_path, config)
    output = paths(config_path, config, "confirmation")
    results = pd.read_csv(output["results"])
    curves = pd.read_csv(output["curves"])
    if len(results) != len(config["scenarios"]) * int(
        config["confirmation"]["replications"]
    ):
        raise RuntimeError("confirmation batch is incomplete")
    if (results["status"] != "completed").any():
        raise RuntimeError("confirmation contains failed cells")
    component_rows = []
    for row in results.itertuples(index=False):
        for component in json.loads(row.component_records_json):
            component_rows.append(
                {"scenario": row.scenario, "replication": row.replication, **component}
            )
    components = pd.DataFrame(component_rows)
    summary = results.groupby("scenario", as_index=False).agg(
        replications=("replication", "size"),
        joint_exact_rate=("both_labels_exact", "mean"),
        joint_component_confirmation_rate=("both_components_test_confirmed", "mean"),
        supported_multiple_rate=(
            "practical_state",
            lambda value: float(np.mean(value == "SUPPORTED_MULTIPLE_COMPONENTS")),
        ),
        forest_cate_spearman_median=("forest_test_cate_spearman", "median"),
        reconstructed_cate_spearman_median=(
            "reconstructed_test_cate_spearman",
            "median",
        ),
        forest_cate_rmse_median=("forest_test_cate_rmse", "median"),
        reconstructed_cate_rmse_median=("reconstructed_test_cate_rmse", "median"),
        ate_cate_gap_sd_y_max=("ate_cate_gap_sd_y", "max"),
        runtime_seconds_median=("runtime_seconds", "median"),
    )
    component_summary = components.groupby(
        ["scenario", "feature", "truth_label"], as_index=False
    ).agg(
        exact_label_rate=("exact_label_correct", "mean"),
        test_component_confirmation_rate=("test_component_confirmed", "mean"),
        curve_spearman_median=("curve_spearman", "median"),
        curve_rmse_median=("curve_rmse", "median"),
        parameter_abs_error_median=("parameter_abs_error", "median"),
        feature_rank_median=("feature_rank", "median"),
    )
    summary.to_csv(
        output["output"] / "MULTI_MORPHOLOGY_PILOT_V1_SUMMARY.csv", index=False
    )
    component_summary.to_csv(
        output["output"] / "MULTI_MORPHOLOGY_PILOT_V1_COMPONENT_SUMMARY.csv",
        index=False,
    )
    render_svg(curves, output["output"] / "MULTI_MORPHOLOGY_PILOT_V1_FIGURE.svg")
    summary_md = summary.to_markdown(index=False, floatfmt=".3f")
    component_md = component_summary.to_markdown(index=False, floatfmt=".3f")
    report = f"""# Multi-Morphology Additive Recovery Pilot v1

## Question

Can one joint multivariable causal forest recover two different additive CATE
morphologies that coexist in the same DGP? This is an exploratory capability
pilot. The two focal characteristics were prespecified; moderator discovery and
correlated-proxy stress are outside its claim.

## Design

- Real L=3 firm-covariate backbone; simulated treatment, outcomes, and truth.
- One 12-characteristic CausalForestDML per replication.
- n=3000; one external 80/20 TRAIN/untouched-TEST split.
- Three cross-fitted nuisance refits and three-iteration additive spline
  backfitting before theory-family competition.
- TRAIN chooses each component shape. TEST compares the full additive candidate
  with a constant effect and with each component ablated.
- Development: 3 replications per scenario. Fresh confirmation: 5 per scenario.

## Scenario-level results

{summary_md}

## Component-level results

{component_md}

## Interpretation rule

Success requires more than an accurate full CATE ranking. Both component labels
must be recovered and both must improve untouched-TEST orthogonal loss relative
to their own ablations. Oracle truth is used only for the accuracy columns.

The pilot does not establish that arbitrary correlated or interacting effects
can be uniquely decomposed. It determines whether the current architecture can
be extended to a clean two-component additive setting without fitting one
forest per characteristic.
"""
    (output["output"] / "MULTI_MORPHOLOGY_ADDITIVE_RECOVERY_PILOT_V1.md").write_text(
        report, encoding="utf-8"
    )
    manifest = {
        "schema_version": "multi-morphology-additive-pilot-result-v1",
        "cells": len(results),
        "failed_cells": int((results["status"] == "failed").sum()),
        "selection_uses_test": False,
        "selection_uses_oracle": False,
        "one_joint_forest_per_cell": True,
        "freeze": verify_freeze(config_path, config),
        "hashes": {
            "confirmation_results": sha256_file(output["results"]),
            "confirmation_curves": sha256_file(output["curves"]),
            "summary": sha256_file(
                output["output"] / "MULTI_MORPHOLOGY_PILOT_V1_SUMMARY.csv"
            ),
            "component_summary": sha256_file(
                output["output"] / "MULTI_MORPHOLOGY_PILOT_V1_COMPONENT_SUMMARY.csv"
            ),
        },
    }
    (output["output"] / "MULTI_MORPHOLOGY_PILOT_V1_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase",
        choices=["freeze", "development", "confirmation", "report"],
        required=True,
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    if args.phase == "freeze":
        print(create_freeze(config_path, config))
    elif args.phase in {"development", "confirmation"}:
        run_phase(config_path, config, args.phase)
    else:
        write_report(config_path, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
