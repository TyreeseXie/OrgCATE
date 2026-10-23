"""Canonical morphology product v1 runner.

This runner keeps the frozen real-backbone, DGP, 80/20 split and one joint
CausalForestDML contract.  It adds an independent DML/AIPW ATE anchor,
multiscale kink diagnostics, and a one/two/three-boundary threshold subtype.
TRAIN selects morphology; untouched TEST only confirms the selected candidate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
V4_PATH = ROOT / "execution" / "run_nine_shape_capability_demo_v4.py"
DEFAULT_CONFIG = ROOT / "configs" / "morphology_product_v1.json"
sys.path.insert(0, str(ROOT / "src"))
from orgcate_sim.ate import ATEAnchorConfig, fit_cross_fitted_aipw_ate


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V4 = load_module(V4_PATH, "morphology_product_v1_v4")
HELPER = V4.HELPER
BASE = V4.BASE
V2 = V4.V2

HIGH_LEVEL_SHAPES = (
    "homogeneous",
    "linear-monotone",
    "nonlinear-monotone",
    "steep-continuous",
    "inverted-u",
    "u-shaped",
    "kinked",
    "threshold-step",
    "sign-reversal",
)
CONFIRMATION_DGPS = (*HIGH_LEVEL_SHAPES, "multi-threshold-step")
DEVELOPMENT_DGPS = (
    "kinked",
    "linear-monotone",
    "nonlinear-monotone",
    "steep-continuous",
)
PRACTICAL_STATES = {
    "SUPPORTED_CANDIDATE",
    "INCONCLUSIVE",
    "NO_THEORY_CLAIM",
    "DESIGN_NOT_ELIGIBLE",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_grid(master_seed: int, replications: int) -> dict[int, int]:
    children = np.random.SeedSequence(int(master_seed)).spawn(int(replications))
    return {
        replication: int(child.generate_state(1, dtype=np.uint32)[0])
        for replication, child in enumerate(children)
    }


def validate_config(config: dict[str, Any]) -> None:
    if config["schema_version"] != "morphology-product-v1":
        raise ValueError("wrong schema_version")
    if int(config["n"]) != 3000 or not np.isclose(float(config["test_size"]), 0.20):
        raise ValueError("n=3000 and one 80/20 split are frozen")
    expected = {
        "class": "econml.dml.CausalForestDML",
        "n_estimators": 1000,
        "min_samples_leaf": 5,
        "max_samples": 0.45,
        "cv": 2,
        "honest": True,
        "inference": False,
        "n_jobs": 2,
    }
    if config["primary_forest"] != expected:
        raise ValueError("primary forest contract changed")
    if int(config["nuisance_crossfit_folds"]) != 5:
        raise ValueError("TRAIN nuisance cross-fitting must use five folds")
    if int(config["same_data_nuisance_refits"]) != 3:
        raise ValueError("three nuisance refits are frozen")
    if int(config["development"]["replications"]) != 5:
        raise ValueError("development must contain 20 cells")
    if int(config["confirmation"]["replications"]) != 10:
        raise ValueError("confirmation must contain 100 cells")
    if config["development"]["dgps"] != list(DEVELOPMENT_DGPS):
        raise ValueError("development controls changed")
    if config["confirmation"]["dgps"] != list(CONFIRMATION_DGPS):
        raise ValueError("confirmation DGPs changed")
    if config["selection_policy"] != {
        "test_used_for_shape_selection": False,
        "oracle_used_for_shape_selection": False,
        "focal_characteristic_prespecified": True,
        "rank_is_information_only": True,
    }:
        raise ValueError("selection policy changed")
    if sorted(config["practical_states"]) != sorted(PRACTICAL_STATES):
        raise ValueError("practical-state vocabulary changed")
    if config["multi_step_truth"] != {
        "boundaries": [0.30, 0.70],
        "levels": [0.20, 1.20, 2.20],
        "high_level_label": "threshold-step",
        "subtype": "multi-threshold-step",
    }:
        raise ValueError("multi-step truth changed")


def multi_step_tau(u: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    truth = config["multi_step_truth"]
    value = np.asarray(u, dtype=float)
    left, right = (float(item) for item in truth["boundaries"])
    low, middle, high = (float(item) for item in truth["levels"])
    return np.where(value < left, low, np.where(value < right, middle, high))


def truth_curve(dgp: str, u: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    if dgp == "multi-threshold-step":
        return multi_step_tau(u, config)
    return V4.prototype_tau(dgp, u)


def build_data(
    backbone: pd.DataFrame,
    source: dict[str, Any],
    method: dict[str, Any],
    dgp: str,
    lineage: dict[str, int],
    base_parameters: dict[str, Any],
) -> dict[str, Any]:
    if dgp != "multi-threshold-step":
        return V4.build_capability_data(
            backbone, source, method, dgp, lineage, base_parameters
        )
    base = BASE.build_data(
        backbone,
        source,
        int(method["n"]),
        "homogeneous",
        lineage,
        base_parameters,
    )
    tau = multi_step_tau(base["u_m1_true"], method)
    y0 = base["Y"] - base["T"] * base["tau_true"]
    y = y0 + base["T"] * tau
    return {**base, "Y": y, "tau_true": tau, "y0_true": y0}


def analyst_frame(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    columns = [f"X_{index:02d}" for index in range(x.shape[1])]
    frame = pd.DataFrame(np.asarray(x, dtype=float), columns=columns)
    frame.insert(0, "T", np.asarray(t, dtype=int))
    frame.insert(0, "Y", np.asarray(y, dtype=float))
    return frame, columns


def estimate_ate(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    frame, columns = analyst_frame(x, y, t)
    spec = config["dml_ate"]
    estimate = fit_cross_fitted_aipw_ate(
        frame,
        adjustment_columns=columns,
        config=ATEAnchorConfig(
            n_splits=int(spec["n_splits"]),
            propensity_clip_low=float(spec["propensity_clip_low"]),
            propensity_clip_high=float(spec["propensity_clip_high"]),
            confidence_level=float(spec["confidence_level"]),
            random_state=int(seed),
        ),
    )
    return estimate.to_record()


def piecewise_basis(u: np.ndarray, boundaries: tuple[float, ...]) -> np.ndarray:
    value = np.asarray(u, dtype=float)
    return np.column_stack(
        [np.ones(len(value)), *[(value >= boundary).astype(float) for boundary in boundaries]]
    )


def fit_piecewise_model(
    u: np.ndarray,
    y_residual: np.ndarray,
    t_residual: np.ndarray,
    boundaries: tuple[float, ...],
) -> dict[str, Any]:
    matrix = piecewise_basis(u, boundaries)
    design = np.asarray(t_residual, dtype=float)[:, None] * matrix
    coefficient, *_ = np.linalg.lstsq(design, y_residual, rcond=None)
    return {
        "family": f"step{len(boundaries)}",
        "params": {"boundaries": list(boundaries)},
        "coefficient": np.asarray(coefficient, dtype=float),
    }


def predict_model(model: dict[str, Any], u: np.ndarray) -> np.ndarray:
    if model["family"].startswith("step") and model["family"] != "step":
        boundaries = tuple(float(item) for item in model["params"]["boundaries"])
        return piecewise_basis(u, boundaries) @ np.asarray(model["coefficient"], dtype=float)
    return HELPER.predict_model(model, u)


def piecewise_candidates(config: dict[str, Any], count: int) -> list[tuple[float, ...]]:
    if count == 0:
        return [()]
    spec = config["multi_step_search"]
    values = HELPER.inclusive_grid(spec["boundary_grid"])
    separation = float(spec["minimum_separation"])
    return [
        tuple(float(value) for value in candidate)
        for candidate in itertools.combinations(values, count)
        if all(np.diff(candidate) >= separation - 1e-12)
    ]


def fit_piecewise_counts(
    u: np.ndarray,
    y_residual: np.ndarray,
    t_residual: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> dict[int, dict[str, Any]]:
    splitter = KFold(
        n_splits=int(config["shape_cv_folds"]), shuffle=True, random_state=int(seed)
    )
    folds = list(splitter.split(u))
    result: dict[int, dict[str, Any]] = {}
    for count in range(4):
        records = []
        for boundaries in piecewise_candidates(config, count):
            losses = []
            for fit_index, validation_index in folds:
                model = fit_piecewise_model(
                    u[fit_index],
                    y_residual[fit_index],
                    t_residual[fit_index],
                    boundaries,
                )
                loss = HELPER.observation_loss(
                    y_residual[validation_index],
                    t_residual[validation_index],
                    predict_model(model, u[validation_index]),
                )
                losses.append(float(np.mean(loss)))
            records.append((float(np.mean(losses)), float(np.std(losses, ddof=1) / math.sqrt(len(losses))), losses, boundaries))
        mean, se, losses, boundaries = min(records, key=lambda item: (item[0], item[3]))
        model = fit_piecewise_model(u, y_residual, t_residual, boundaries)
        model.update(cv_mean_r_loss=mean, cv_se_r_loss=se, fold_losses=losses)
        result[count] = model
    return result


def fit_extended_library(
    u: np.ndarray,
    y_residual: np.ndarray,
    t_residual: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    library = HELPER.fit_library(u, y_residual, t_residual, config, seed)
    piecewise = fit_piecewise_counts(u, y_residual, t_residual, config, seed + 7919)
    library["models"]["step0"] = piecewise[0]
    library["models"]["step1"] = piecewise[1]
    library["models"]["step2"] = piecewise[2]
    library["models"]["step3"] = piecewise[3]
    library["u_train"] = np.asarray(u, dtype=float)
    library["y_residual"] = np.asarray(y_residual, dtype=float)
    library["t_residual"] = np.asarray(t_residual, dtype=float)
    return library


def aggregate_losses(libraries: list[dict[str, Any]], families: list[str]) -> pd.DataFrame:
    rows = []
    for family in families:
        losses = np.concatenate(
            [np.asarray(item["models"][family]["fold_losses"], dtype=float) for item in libraries]
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


def paired_loss(table: pd.DataFrame, left: str, right: str) -> tuple[float, float]:
    left_loss = table.loc[table["family"] == left, "fold_losses"].iloc[0]
    right_loss = table.loc[table["family"] == right, "fold_losses"].iloc[0]
    difference = np.asarray(left_loss) - np.asarray(right_loss)
    return float(np.mean(difference)), float(np.std(difference, ddof=1) / math.sqrt(len(difference)))


def select_step_count(libraries: list[dict[str, Any]]) -> tuple[int, pd.DataFrame]:
    table = aggregate_losses(libraries, ["step0", "step1", "step2", "step3"])
    best = table.sort_values(["mean_loss", "family"]).iloc[0]
    threshold = float(best["mean_loss"] + best["se_loss"])
    eligible = table[table["mean_loss"] <= threshold].copy()
    eligible["count"] = eligible["family"].str.replace("step", "", regex=False).astype(int)
    return int(eligible.sort_values("count").iloc[0]["count"]), table


def local_side_slope(
    u: np.ndarray,
    y_residual: np.ndarray,
    t_residual: np.ndarray,
    center: float,
    width: float,
    side: str,
) -> tuple[float, int]:
    if side == "left":
        mask = (u >= center - width) & (u < center)
    else:
        mask = (u >= center) & (u <= center + width)
    if int(mask.sum()) < 25:
        return float("nan"), int(mask.sum())
    centered = u[mask] - center
    design = t_residual[mask, None] * np.column_stack([np.ones(mask.sum()), centered])
    coefficient, *_ = np.linalg.lstsq(design, y_residual[mask], rcond=None)
    return float(coefficient[1]), int(mask.sum())


def multiscale_kink_diagnostic(
    libraries: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    windows = [float(item) for item in config["kink_diagnostic"]["windows"]]
    records = []
    for library in libraries:
        model = library["models"]["kink"]
        center = float(model["params"]["c"])
        deltas = []
        for width in windows:
            left, left_n = local_side_slope(
                library["u_train"], library["y_residual"], library["t_residual"], center, width, "left"
            )
            right, right_n = local_side_slope(
                library["u_train"], library["y_residual"], library["t_residual"], center, width, "right"
            )
            deltas.append(right - left)
            records.append(
                {"center": center, "width": width, "left_slope": left, "right_slope": right, "delta_slope": right - left, "left_n": left_n, "right_n": right_n}
            )
        large = abs(deltas[-1])
        records[-len(windows)]["persistence"] = abs(deltas[0]) / max(large, 1e-8)
    frame = pd.DataFrame(records)
    small = frame[frame["width"] == windows[0]]
    persistence = [
        row.get("persistence", float("nan"))
        for row in records[:: len(windows)]
    ]
    models = [item["models"]["kink"] for item in libraries]
    geometry = [V4.same_sign_slope_ratio(model) for model in models]
    centers = [float(model["params"]["c"]) for model in models]
    left_slopes = [float(model["coefficient"][1]) for model in models]
    right_slopes = [float(model["coefficient"][1] + model["coefficient"][2]) for model in models]
    return {
        "median_abs_delta_small": float(np.nanmedian(np.abs(small["delta_slope"]))),
        "median_persistence": float(np.nanmedian(persistence)),
        "same_sign_rate": float(np.mean([item[0] for item in geometry])),
        "median_slope_ratio": float(np.median([item[1] for item in geometry])),
        "center_range": float(np.ptp(centers)),
        "center_median": float(np.median(centers)),
        "global_left_slope_median": float(np.median(left_slopes)),
        "global_right_slope_median": float(np.median(right_slopes)),
        "global_slope_change_median": float(np.median(np.asarray(right_slopes) - np.asarray(left_slopes))),
        "records": records,
    }


def select_family(
    libraries: list[dict[str, Any]],
    config: dict[str, Any],
    grid: np.ndarray,
    ate_train: float,
) -> dict[str, Any]:
    families = ["constant", "linear", "quadratic", "tanh", "kink", "step"]
    table = aggregate_losses(libraries, families)
    lookup = table.set_index("family")["mean_loss"].to_dict()
    step_count, step_table = select_step_count(libraries)
    continuous = ["linear", "quadratic", "tanh", "kink"]
    best_continuous = min(continuous, key=lambda family: lookup[family])
    step_family = f"step{step_count}"
    step_loss = float(step_table.loc[step_table["family"] == step_family, "mean_loss"].iloc[0])
    if step_count >= 1 and step_loss < lookup[best_continuous]:
        subtype = "single-threshold-step" if step_count == 1 else "multi-threshold-step"
        return {
            "family": step_family,
            "label": "threshold-step",
            "subtype": subtype,
            "reason": f"TRAIN one-SE piecewise search selected {step_count} boundary/boundaries",
            "kink": multiscale_kink_diagnostic(libraries, config),
            "step_count": step_count,
        }
    tanh_k = float(np.median([item["models"]["tanh"]["params"]["k"] for item in libraries]))
    step_difference, step_se = paired_loss(table, "step", "tanh")
    if (
        tanh_k >= float(config["hierarchy"]["step_saturation_k"])
        and step_difference <= float(config["hierarchy"]["step_near_tie_z"]) * step_se
    ):
        return {
            "family": "step1",
            "label": "threshold-step",
            "subtype": "single-threshold-step",
            "reason": "TRAIN smooth fit saturated and was tied with one boundary",
            "kink": multiscale_kink_diagnostic(libraries, config),
            "step_count": 1,
        }
    best_monotone = min(["linear", "tanh", "kink"], key=lambda family: lookup[family])
    if lookup["quadratic"] < lookup[best_monotone]:
        models = [item["models"]["quadratic"] for item in libraries]
        coefficient = float(np.median([model["coefficient"][1] for model in models]))
        center = float(np.median([model["params"]["c"] for model in models]))
        curve = np.median([predict_model(model, grid) for model in models], axis=0)
        if V4.interior_turn_supported(
            curve,
            grid,
            center,
            config["hierarchy"]["turning_point_support"],
            float(config["hierarchy"]["minimum_two_sided_tail_contrast_share"]),
        ):
            return {
                "family": "quadratic",
                "label": "u-shaped" if coefficient > 0 else "inverted-u",
                "subtype": "",
                "reason": "TRAIN reconstruction contains a supported interior extremum",
                "kink": multiscale_kink_diagnostic(libraries, config),
                "step_count": step_count,
            }
    consensus = np.median([predict_model(item["models"][best_monotone], grid) for item in libraries], axis=0)
    monotone = V4.monotone_share(consensus, grid) >= float(config["hierarchy"]["minimum_monotone_derivative_share"])
    opposite = bool(np.any(consensus * np.sign(ate_train if abs(ate_train) > 1e-8 else 1.0) < 0.0))
    same = bool(np.any(consensus * np.sign(ate_train if abs(ate_train) > 1e-8 else 1.0) > 0.0))
    if monotone and opposite and same:
        return {
            "family": best_monotone,
            "label": "sign-reversal",
            "subtype": "",
            "reason": "TRAIN monotone curve contains regions opposite to the DML ATE sign",
            "kink": multiscale_kink_diagnostic(libraries, config),
            "step_count": step_count,
        }
    deviation = V4.normalized_deviation_from_linear(consensus, grid)
    if deviation <= float(config["hierarchy"]["maximum_normalized_deviation_from_linear"]):
        return {
            "family": "linear",
            "label": "linear-monotone",
            "subtype": "",
            "reason": "TRAIN reconstruction is within the frozen near-linear tolerance",
            "kink": multiscale_kink_diagnostic(libraries, config),
            "step_count": step_count,
        }
    best_nonlinear = min(["tanh", "kink"], key=lambda family: lookup[family])
    difference, difference_se = paired_loss(table, "linear", best_nonlinear)
    z = float(config["hierarchy"]["train_paired_loss_z"])
    if difference <= z * difference_se:
        return {
            "family": "linear",
            "label": "linear-monotone",
            "subtype": "",
            "reason": "TRAIN paired loss did not establish nonlinearity",
            "kink": multiscale_kink_diagnostic(libraries, config),
            "step_count": step_count,
        }
    diagnostic = multiscale_kink_diagnostic(libraries, config)
    kink_rule = config["kink_diagnostic"]
    kink_difference, kink_se = paired_loss(table, "kink", "tanh")
    kink_loss_upper = kink_difference + float(kink_rule["paired_loss_one_sided_z"]) * kink_se
    kink_pass = (
        diagnostic["same_sign_rate"] >= float(kink_rule["minimum_same_sign_rate"])
        and diagnostic["median_slope_ratio"] >= float(kink_rule["minimum_slope_ratio"])
        and diagnostic["center_range"] <= float(kink_rule["maximum_center_range"])
        and kink_loss_upper < 0.0
    )
    if kink_pass:
        return {
            "family": "kink",
            "label": "kinked",
            "subtype": "",
            "reason": "TRAIN hinge has stable same-direction slopes and its paired CV R-loss upper bound beats smooth tanh",
            "kink": diagnostic,
            "step_count": step_count,
        }
    label = "steep-continuous" if tanh_k >= float(config["hierarchy"]["steep_tanh_k_cutpoint"]) else "nonlinear-monotone"
    return {
        "family": "tanh",
        "label": label,
        "subtype": "",
        "reason": "TRAIN continuous nonlinear branch selected",
        "kink": diagnostic,
        "step_count": step_count,
    }


def parameter_record(label: str, models: list[dict[str, Any]], grid: np.ndarray) -> dict[str, Any]:
    if models[0]["family"] not in {"step2", "step3", "step0", "step1"}:
        return V4.parameter_record(label, models, grid)
    details = []
    primary = []
    all_boundaries = []
    for model in models:
        boundaries = [float(item) for item in model["params"]["boundaries"]]
        coefficient = np.asarray(model["coefficient"], dtype=float)
        levels = [float(coefficient[0])]
        for index in range(len(boundaries)):
            levels.append(float(coefficient[0] + coefficient[1 : index + 2].sum()))
        details.append({"boundaries": boundaries, "levels": levels, "jumps": [float(item) for item in coefficient[1:]]})
        all_boundaries.append(boundaries)
        if boundaries:
            primary.append(float(boundaries[0]))
    lengths = {len(item) for item in all_boundaries}
    boundary_range = float("nan")
    if len(lengths) == 1 and next(iter(lengths), 0) > 0:
        array = np.asarray(all_boundaries, dtype=float)
        boundary_range = float(np.max(np.ptp(array, axis=0)))
    return {
        "parameter_median": float(np.median(primary)) if primary else float("nan"),
        "parameter_range": boundary_range,
        "details_json": json.dumps(details, sort_keys=True),
    }


def confirm_candidate(
    candidate: dict[str, Any],
    constant: dict[str, Any],
    u_test: np.ndarray,
    y_residual: np.ndarray,
    t_residual: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    candidate_loss = HELPER.observation_loss(y_residual, t_residual, predict_model(candidate, u_test))
    constant_loss = HELPER.observation_loss(y_residual, t_residual, predict_model(constant, u_test))
    difference = candidate_loss - constant_loss
    mean = float(np.mean(difference))
    se = float(np.std(difference, ddof=1) / math.sqrt(len(difference)))
    upper = mean + float(norm.ppf(1 - alpha)) * se
    return {
        "test_loss_difference": mean,
        "test_loss_difference_se": se,
        "test_loss_upper_one_sided": upper,
        "test_confirmed_vs_constant": bool(upper < 0.0),
    }


def run_cell(
    backbone: pd.DataFrame,
    source: dict[str, Any],
    base_parameters: dict[str, Any],
    config: dict[str, Any],
    phase: str,
    dgp: str,
    replication: int,
    master_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    started = time.perf_counter()
    lineage = BASE.lineage(master_seed, 0)
    data = build_data(backbone, source, config, dgp, lineage, base_parameters)
    train, test = V2.v2.split_once(data["T"], data["unit_id"], float(config["test_size"]), lineage["split_seed"])
    u = BASE.reference_u(backbone, data["unit_id"], config["focal_characteristic"])
    ate_train = estimate_ate(data["X"][train], data["Y"][train], data["T"][train], lineage["nuisance_seed"], config)
    common = dict(source["common"])
    common["n_jobs"] = int(config["primary_forest"]["n_jobs"])
    estimator = V2.v2._new_m1(common, lineage["forest_seed"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(data["Y"][train], data["T"][train], X=data["X"][train], W=None)
    forest_test = np.asarray(estimator.effect(data["X"][test]), dtype=float).reshape(-1)
    importance = np.asarray(estimator.feature_importances_, dtype=float)
    focal_index = source["features"].index(config["focal_characteristic"])
    focal_rank = int(np.sum(importance > importance[focal_index]) + 1)
    libraries = []
    nuisances = []
    for refit in range(int(config["same_data_nuisance_refits"])):
        refit_seed = HELPER.refit_seeds(master_seed, refit)
        nuisance = V2.crossfit_nuisance(
            data["X"][train], data["Y"][train], data["T"][train], data["X"][test],
            refit_seed["nuisance_refit_seed"], int(config["nuisance_crossfit_folds"]),
        )
        library_config = copy.deepcopy(config)
        library_config["selection"] = {
            "structured_families": ["constant", "linear", "quadratic", "tanh", "kink", "step"],
            "flexible_family": "spline",
        }
        library = fit_extended_library(
            u[train], nuisance["y_residual_train"], nuisance["t_residual_train"],
            library_config, refit_seed["shape_cv_refit_seed"],
        )
        libraries.append(library)
        nuisances.append(nuisance)
    grid_spec = config["grid"]
    grid = np.linspace(float(grid_spec["minimum"]), float(grid_spec["maximum"]), int(grid_spec["points"]))
    selection = select_family(libraries, config, grid, float(ate_train["ate_dml"]))
    selected_family = selection["family"]
    models = [item["models"][selected_family] for item in libraries]
    estimates = [predict_model(model, grid) for model in models]
    parameter = parameter_record(selection["label"], models, grid)
    confirmations = []
    confirmation_rows = []
    for refit, (library, nuisance, model) in enumerate(zip(libraries, nuisances, models, strict=True)):
        y_test = data["Y"][test] - nuisance["m_test"]
        t_test = data["T"][test] - nuisance["e_test"]
        confirmation = confirm_candidate(
            model, library["models"]["constant"], u[test], y_test, t_test,
            float(config["test_confirmation"]["one_sided_alpha"]),
        )
        confirmations.append(bool(confirmation["test_confirmed_vs_constant"]))
        confirmation_rows.append({
            "phase": phase, "dgp": dgp, "replication": replication,
            "master_seed": master_seed, "refit": refit,
            "selected_family": selected_family, "train_selected_label": selection["label"],
            **confirmation,
        })
    confirmation_rate = float(np.mean(confirmations))
    curve_stability = V4.pairwise_curve_spearman(estimates)
    predicted_label = selection["label"]
    predicted_subtype = selection["subtype"]
    state = "SUPPORTED_CANDIDATE"
    if confirmation_rate < float(config["test_confirmation"]["minimum_confirmation_rate"]):
        predicted_label = "homogeneous"
        predicted_subtype = ""
        state = "NO_THEORY_CLAIM"
        selected_family = "constant"
        models = [item["models"]["constant"] for item in libraries]
        estimates = [predict_model(model, grid) for model in models]
        parameter = V4.parameter_record("homogeneous", models, grid)
    else:
        stable_parameter = not np.isfinite(parameter["parameter_range"]) or parameter["parameter_range"] <= float(config["supportability"]["maximum_parameter_range"])
        if curve_stability < float(config["supportability"]["minimum_pairwise_curve_spearman"]) or not stable_parameter:
            state = "INCONCLUSIVE"
    consensus = np.median(estimates, axis=0)
    consensus_test = np.median([predict_model(model, u[test]) for model in models], axis=0)
    truth_grid_values = truth_curve(dgp, grid, config)
    truth_test = data["tau_true"][test]
    ate_full = estimate_ate(data["X"], data["Y"], data["T"], lineage["artifact_seed"], config)
    true_label = "threshold-step" if dgp == "multi-threshold-step" else dgp
    true_subtype = "multi-threshold-step" if dgp == "multi-threshold-step" else ("single-threshold-step" if dgp == "threshold-step" else "")
    kink = selection["kink"]
    diagnostic_loss_table = aggregate_losses(libraries, ["linear", "tanh", "kink"])
    kink_minus_tanh, kink_minus_tanh_se = paired_loss(diagnostic_loss_table, "kink", "tanh")
    kink_minus_linear, kink_minus_linear_se = paired_loss(diagnostic_loss_table, "kink", "linear")
    tanh_centers = [float(item["models"]["tanh"]["params"]["c"]) for item in libraries]
    tanh_ks = [float(item["models"]["tanh"]["params"]["k"]) for item in libraries]
    details = json.loads(parameter["details_json"])
    boundary_count = 0
    if details and "boundaries" in details[0]:
        boundary_count = len(details[0]["boundaries"])
    elif predicted_label == "threshold-step":
        boundary_count = 1
    ate_value = float(ate_train["ate_dml"])
    ate_sign = np.sign(ate_value) if abs(ate_value) > 1e-8 else 1.0
    result = {
        "status": "completed",
        "phase": phase,
        "dgp": dgp,
        "true_label": true_label,
        "true_subtype": true_subtype,
        "replication": replication,
        "master_seed": master_seed,
        "n": int(config["n"]),
        "predicted_label": predicted_label,
        "predicted_subtype": predicted_subtype,
        "practical_state": state,
        "selected_family": selected_family,
        "train_selected_label": selection["label"],
        "selection_reason": selection["reason"],
        "exact_label_correct": bool(predicted_label == true_label),
        "exact_subtype_correct": bool(predicted_subtype == true_subtype) if true_subtype else True,
        "test_confirmation_rate": confirmation_rate,
        "curve_stability_spearman": curve_stability,
        "oracle_curve_spearman": float(spearmanr(truth_grid_values, consensus).statistic) if np.std(truth_grid_values) > 1e-12 and np.std(consensus) > 1e-12 else float("nan"),
        "curve_rmse": float(np.sqrt(np.mean((consensus - truth_grid_values) ** 2))),
        "forest_test_cate_rmse": float(np.sqrt(np.mean((forest_test - truth_test) ** 2))),
        "forest_test_cate_spearman": float(spearmanr(truth_test, forest_test).statistic) if np.std(truth_test) > 1e-12 and np.std(forest_test) > 1e-12 else float("nan"),
        "cate_test_mean": float(np.mean(consensus_test)),
        "cate_test_sd": float(np.std(consensus_test)),
        "cate_test_q10": float(np.quantile(consensus_test, 0.10)),
        "cate_test_q50": float(np.quantile(consensus_test, 0.50)),
        "cate_test_q90": float(np.quantile(consensus_test, 0.90)),
        "cate_weighted_mean_minus_ate_train": float(np.mean(consensus_test) - ate_value),
        "cate_share_above_ate": float(np.mean(consensus_test > ate_value)),
        "cate_share_below_ate": float(np.mean(consensus_test < ate_value)),
        "cate_share_opposite_ate_sign": float(np.mean(consensus_test * ate_sign < 0.0)),
        "true_ate": float(np.mean(data["tau_true"])),
        "ate_train_error": float(ate_value - np.mean(data["tau_true"])),
        "ate_full_error": float(ate_full["ate_dml"] - np.mean(data["tau_true"])),
        "parameter_median": parameter["parameter_median"],
        "parameter_range": parameter["parameter_range"],
        "parameter_details_json": parameter["details_json"],
        "estimated_boundary_count": boundary_count,
        "focal_characteristic_rank": focal_rank,
        "focal_characteristic_importance": float(importance[focal_index]),
        "importance_json": json.dumps(dict(zip(source["features"], importance.tolist())), sort_keys=True),
        "kink_median_abs_delta_small": kink["median_abs_delta_small"],
        "kink_median_persistence": kink["median_persistence"],
        "kink_same_sign_rate": kink["same_sign_rate"],
        "kink_median_slope_ratio": kink["median_slope_ratio"],
        "kink_center_range": kink["center_range"],
        "kink_center_median": kink["center_median"],
        "kink_global_left_slope_median": kink["global_left_slope_median"],
        "kink_global_right_slope_median": kink["global_right_slope_median"],
        "kink_global_slope_change_median": kink["global_slope_change_median"],
        "kink_minus_tanh_loss": kink_minus_tanh,
        "kink_minus_tanh_loss_se": kink_minus_tanh_se,
        "kink_minus_linear_loss": kink_minus_linear,
        "kink_minus_linear_loss_se": kink_minus_linear_se,
        "tanh_center_median": float(np.median(tanh_centers)),
        "tanh_k_median": float(np.median(tanh_ks)),
        "warning_count": len(caught),
        "runtime_seconds": float(time.perf_counter() - started),
        "test_used_for_shape_selection": False,
        "oracle_used_for_shape_selection": False,
        "one_joint_multivariable_forest": True,
        **{f"train_{key}": value for key, value in ate_train.items() if key != "adjustment_columns"},
        **{f"full_{key}": value for key, value in ate_full.items() if key != "adjustment_columns"},
    }
    curve_rows = [
        {
            "phase": phase,
            "dgp": dgp,
            "replication": replication,
            "master_seed": master_seed,
            "u": float(value),
            "truth_curve": float(truth),
            "estimate_curve": float(estimate),
            "ate_dml_train": ate_value,
            "ate_ci_low_train": float(ate_train["ate_ci_low"]),
            "ate_ci_high_train": float(ate_train["ate_ci_high"]),
            "true_ate": float(np.mean(data["tau_true"])),
        }
        for value, truth, estimate in zip(grid, truth_grid_values, consensus, strict=True)
    ]
    return result, confirmation_rows, curve_rows


def output_paths(config_path: Path, config: dict[str, Any], phase: str) -> dict[str, Path]:
    output = (config_path.parent / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = "DEVELOPMENT" if phase == "development" else "CONFIRMATION"
    return {
        "dir": output,
        "results": output / f"MORPHOLOGY_PRODUCT_V1_{prefix}_LONG.csv",
        "confirmations": output / f"MORPHOLOGY_PRODUCT_V1_{prefix}_TEST_CONFIRMATION_LONG.csv",
        "curves": output / f"MORPHOLOGY_PRODUCT_V1_{prefix}_CURVES_LONG.csv",
    }


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def run_phase(config_path: Path, config: dict[str, Any], phase: str) -> None:
    paths = output_paths(config_path, config, phase)
    phase_config = config[phase]
    seeds = seed_grid(int(phase_config["master_seed"]), int(phase_config["replications"]))
    completed: set[tuple[str, int]] = set()
    if paths["results"].exists():
        existing = pd.read_csv(paths["results"])
        completed = set(zip(existing["dgp"], existing["replication"].astype(int), strict=False))
    source_path = (config_path.parent / config["source_config"]).resolve()
    source = V2.load_json(source_path)
    base_config = json.loads((config_path.parent / config["base_config"]).resolve().read_text(encoding="utf-8"))
    backbone = V2.add_rank_columns(V2.v2.load_frozen_backbone(ROOT, source, source_path), source["features"])
    total = len(phase_config["dgps"]) * int(phase_config["replications"])
    done = len(completed)
    for replication in range(int(phase_config["replications"])):
        for dgp in phase_config["dgps"]:
            key = (dgp, replication)
            if key in completed:
                continue
            print(f"{phase.upper()} {done + 1}/{total} dgp={dgp} replication={replication}", flush=True)
            try:
                result, confirmations, curves = run_cell(
                    backbone, source, base_config["shape_parameters"], config,
                    phase, dgp, replication, seeds[replication],
                )
            except Exception as error:  # noqa: BLE001 - failed cells must remain in the audit table
                result = {
                    "status": "failed", "phase": phase, "dgp": dgp,
                    "replication": replication, "master_seed": seeds[replication],
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
                confirmations, curves = [], []
            append_rows(paths["results"], [result])
            if confirmations:
                append_rows(paths["confirmations"], confirmations)
            if curves:
                append_rows(paths["curves"], curves)
            done += 1


def create_freeze(config_path: Path, config: dict[str, Any]) -> Path:
    validate_config(config)
    development = output_paths(config_path, config, "development")["results"]
    if not development.exists() or len(pd.read_csv(development)) != 20:
        raise RuntimeError("complete the 20-cell development calibration before freezing")
    freeze_path = (config_path.parent / config["freeze_path"]).resolve()
    payload = {
        "status": "METHOD_FROZEN_BEFORE_100_CELL_FRESH_CONFIRMATION",
        "frozen_at": "2026-10-22",
        "development_cells": 20,
        "confirmation_seed_grid": seed_grid(
            int(config["confirmation"]["master_seed"]), int(config["confirmation"]["replications"])
        ),
        "hashes": {
            "config_sha256": sha256_file(config_path),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "source_config_sha256": sha256_file((config_path.parent / config["source_config"]).resolve()),
            "development_results_sha256": sha256_file(development),
        },
        "test_selects_shape": False,
        "oracle_selects_shape": False,
    }
    freeze_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return freeze_path


def verify_freeze(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    freeze_path = (config_path.parent / config["freeze_path"]).resolve()
    if not freeze_path.exists():
        raise RuntimeError("method freeze missing")
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    development = output_paths(config_path, config, "development")["results"]
    current = {
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "source_config_sha256": sha256_file((config_path.parent / config["source_config"]).resolve()),
        "development_results_sha256": sha256_file(development),
    }
    if current != payload["hashes"]:
        raise RuntimeError("method freeze hash mismatch")
    return payload


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    completed = results[results["status"] == "completed"].copy()
    rows = []
    for dgp, group in completed.groupby("dgp", sort=False):
        rows.append({
            "dgp": dgp,
            "replications": len(group),
            "exact_label_rate": float(group["exact_label_correct"].mean()),
            "exact_subtype_rate": float(group["exact_subtype_correct"].mean()),
            "supported_rate": float((group["practical_state"] == "SUPPORTED_CANDIDATE").mean()),
            "median_curve_spearman": float(group["oracle_curve_spearman"].median()),
            "median_curve_rmse": float(group["curve_rmse"].median()),
            "median_ate_train_abs_error": float(group["ate_train_error"].abs().median()),
            "median_ate_full_abs_error": float(group["ate_full_error"].abs().median()),
            "median_focal_rank": float(group["focal_characteristic_rank"].median()),
            "median_estimated_boundary_count": float(group["estimated_boundary_count"].median()),
            "warning_count": int(group["warning_count"].sum()),
        })
    return pd.DataFrame(rows)


def svg_polyline(x: np.ndarray, y: np.ndarray, x0: float, y0: float, width: float, height: float, low: float, high: float) -> str:
    span = max(high - low, 1e-9)
    return " ".join(
        f"{x0 + (float(xv) - .10) / .80 * width:.1f},{y0 + height - (float(yv) - low) / span * height:.1f}"
        for xv, yv in zip(x, y, strict=True)
    )


def render_figures(output: Path, results: pd.DataFrame, curves: pd.DataFrame, config: dict[str, Any]) -> None:
    order = list(CONFIRMATION_DGPS)
    width, height = 1240, 1580
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#20252b}.title{font-size:25px;font-weight:700}.sub{font-size:13px;fill:#5c6670}.panel{fill:#fff;stroke:#d6dce1}.name{font-size:14px;font-weight:700}.axis{stroke:#9aa3ab}.truth{fill:none;stroke:#222;stroke-width:1.8;stroke-dasharray:6 4}.estimate{fill:none;stroke:#238b57;stroke-width:2.6}.ate{stroke:#245b8a;stroke-width:1.6}.trueate{stroke:#777;stroke-width:1.2;stroke-dasharray:3 3}</style>',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="35" y="38" class="title">CATE morphology relative to the DML ATE anchor</text>',
        '<text x="35" y="62" class="sub">green = reconstructed CATE; black dashed = truth (evaluation only); blue = TRAIN DML ATE; gray dotted = true ATE</text>',
    ]
    for index, dgp in enumerate(order):
        row, column = divmod(index, 2)
        panel_x, panel_y = 28 + column * 605, 88 + row * 292
        plot_x, plot_y, plot_w, plot_h = panel_x + 54, panel_y + 46, 500, 190
        frame = curves[curves["dgp"] == dgp]
        grouped = frame.groupby("u", as_index=False).agg(
            truth=("truth_curve", "mean"), estimate=("estimate_curve", "median"),
            ate=("ate_dml_train", "median"), true_ate=("true_ate", "mean"),
        )
        low = float(grouped[["truth", "estimate", "ate", "true_ate"]].min().min())
        high = float(grouped[["truth", "estimate", "ate", "true_ate"]].max().max())
        margin = max(.05, .08 * (high - low))
        low, high = low - margin, high + margin
        x = grouped["u"].to_numpy(float)
        estimate = svg_polyline(x, grouped["estimate"].to_numpy(float), plot_x, plot_y, plot_w, plot_h, low, high)
        truth = svg_polyline(x, grouped["truth"].to_numpy(float), plot_x, plot_y, plot_w, plot_h, low, high)
        ate_y = plot_y + plot_h - (float(grouped["ate"].iloc[0]) - low) / (high - low) * plot_h
        true_ate_y = plot_y + plot_h - (float(grouped["true_ate"].iloc[0]) - low) / (high - low) * plot_h
        lines.extend([
            f'<rect x="{panel_x}" y="{panel_y}" width="580" height="270" class="panel"/>',
            f'<text x="{panel_x + 16}" y="{panel_y + 25}" class="name">{dgp}</text>',
            f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" class="axis"/>',
            f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" class="axis"/>',
            f'<line x1="{plot_x}" y1="{ate_y:.1f}" x2="{plot_x + plot_w}" y2="{ate_y:.1f}" class="ate"/>',
            f'<line x1="{plot_x}" y1="{true_ate_y:.1f}" x2="{plot_x + plot_w}" y2="{true_ate_y:.1f}" class="trueate"/>',
            f'<polyline points="{truth}" class="truth"/>',
            f'<polyline points="{estimate}" class="estimate"/>',
        ])
        if dgp == "multi-threshold-step":
            for boundary in config["multi_step_truth"]["boundaries"]:
                bx = plot_x + (float(boundary) - .10) / .80 * plot_w
                lines.append(f'<line x1="{bx:.1f}" y1="{plot_y}" x2="{bx:.1f}" y2="{plot_y + plot_h}" stroke="#b23a48" stroke-dasharray="3 3"/>')
    lines.append("</svg>")
    (output / "FIGURE_1_CATE_ATE_AND_MORPHOLOGY.svg").write_text("\n".join(lines), encoding="utf-8")

    summary = summarize(results).set_index("dgp").reindex(order).reset_index()
    width, height = 1220, 620
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#20252b}.title{font-size:25px;font-weight:700}.label{font-size:11px}.value{font-size:11px;font-weight:700}.axis{stroke:#9aa3ab}.exact{fill:#247b7b}.support{fill:#cf6d3d}</style>',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="35" y="38" class="title">Fresh morphology recovery and practical support</text>',
        '<line x1="70" y1="490" x2="1180" y2="490" class="axis"/>',
        '<line x1="70" y1="90" x2="70" y2="490" class="axis"/>',
        '<line x1="70" y1="130" x2="1180" y2="130" stroke="#333" stroke-dasharray="5 4"/>',
        '<text x="76" y="124" class="label">90%</text>',
    ]
    for index, row in enumerate(summary.itertuples(index=False)):
        center = 125 + index * 108
        exact_h = 400 * float(row.exact_label_rate)
        support_h = 400 * float(row.supported_rate)
        lines.extend([
            f'<rect x="{center - 28}" y="{490 - exact_h:.1f}" width="25" height="{exact_h:.1f}" class="exact"/>',
            f'<rect x="{center + 3}" y="{490 - support_h:.1f}" width="25" height="{support_h:.1f}" class="support"/>',
            f'<text x="{center - 28}" y="{480 - exact_h:.1f}" class="value">{row.exact_label_rate:.0%}</text>',
            f'<text x="{center + 3}" y="{480 - support_h:.1f}" class="value">{row.supported_rate:.0%}</text>',
            f'<text x="{center}" y="515" class="label" text-anchor="middle" transform="rotate(35 {center} 515)">{row.dgp}</text>',
        ])
    lines.extend([
        '<rect x="830" y="40" width="16" height="16" class="exact"/><text x="852" y="53" class="label">exact label</text>',
        '<rect x="960" y="40" width="16" height="16" class="support"/><text x="982" y="53" class="label">practical support</text>',
        '</svg>',
    ])
    (output / "FIGURE_2_RECOVERY_RATES.svg").write_text("\n".join(lines), encoding="utf-8")

    kink_order = ["linear-monotone", "nonlinear-monotone", "steep-continuous", "kinked"]
    threshold = float(config["kink_diagnostic"]["minimum_persistence"])
    step = results[results["dgp"].isin(["threshold-step", "multi-threshold-step"])]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1220" height="560" viewBox="0 0 1220 560">',
        '<style>text{font-family:Arial,sans-serif;fill:#20252b}.title{font-size:24px;font-weight:700}.sub{font-size:13px}.point{fill:#247b7b;fill-opacity:.7}.bar{fill:#cf6d3d}</style>',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="35" y="38" class="title">Kink persistence and discontinuity-boundary recovery</text>',
        f'<line x1="55" y1="{445 - min(threshold, 2.0) / 2.0 * 340:.1f}" x2="585" y2="{445 - min(threshold, 2.0) / 2.0 * 340:.1f}" stroke="#b23a48" stroke-dasharray="5 4"/>',
    ]
    for index, dgp in enumerate(kink_order):
        x0 = 115 + index * 125
        values = results.loc[results["dgp"] == dgp, "kink_median_persistence"].to_numpy(float)
        for offset, value in enumerate(values):
            x = x0 + (offset - 4.5) * 5
            y = 445 - min(float(value), 2.0) / 2.0 * 340
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" class="point"/>')
        lines.append(f'<text x="{x0}" y="475" class="sub" text-anchor="middle" transform="rotate(25 {x0} 475)">{dgp}</text>')
    for group_index, dgp in enumerate(["threshold-step", "multi-threshold-step"]):
        group = step[step["dgp"] == dgp]
        x0 = 760 + group_index * 230
        for count in range(4):
            value = int((group["estimated_boundary_count"] == count).sum())
            bar_h = value * 28
            lines.extend([
                f'<rect x="{x0 + count * 42}" y="{445 - bar_h}" width="30" height="{bar_h}" class="bar" opacity="{.35 + .2 * count}"/>',
                f'<text x="{x0 + count * 42 + 15}" y="{438 - bar_h}" class="sub" text-anchor="middle">{value}</text>',
                f'<text x="{x0 + count * 42 + 15}" y="465" class="sub" text-anchor="middle">{count}</text>',
            ])
        lines.append(f'<text x="{x0 + 63}" y="505" class="sub" text-anchor="middle">{dgp}</text>')
    lines.extend([
        '<text x="315" y="525" class="sub" text-anchor="middle">multiscale persistence (capped at 2 for display)</text>',
        '<text x="940" y="525" class="sub" text-anchor="middle">estimated boundary count</text>',
        '</svg>',
    ])
    (output / "FIGURE_3_KINK_AND_BOUNDARY_DIAGNOSTICS.svg").write_text("\n".join(lines), encoding="utf-8")


def acceptance(results: pd.DataFrame) -> dict[str, Any]:
    completed = results[results["status"] == "completed"].copy()
    rates = completed.groupby("true_label")["exact_label_correct"].mean().to_dict()
    adjacent = completed[completed["dgp"].isin(["linear-monotone", "nonlinear-monotone", "steep-continuous"])]
    false_kink = int((adjacent["predicted_label"] == "kinked").sum())
    homogeneous_false_support = int(
        ((completed["dgp"] == "homogeneous") & (completed["practical_state"] == "SUPPORTED_CANDIDATE")).sum()
    )
    multi = completed[completed["dgp"] == "multi-threshold-step"]
    multi_correct = int((multi["estimated_boundary_count"] == 2).sum())
    return {
        "all_high_level_at_least_9_of_10": bool(all(value >= .90 for value in rates.values())),
        "high_level_rates": rates,
        "adjacent_false_kink_count": false_kink,
        "adjacent_false_kink_at_most_1_of_30": bool(false_kink <= 1),
        "homogeneous_false_support_count": homogeneous_false_support,
        "homogeneous_false_support_zero": bool(homogeneous_false_support == 0),
        "multi_step_two_boundary_count": multi_correct,
        "multi_step_two_boundary_at_least_9_of_10": bool(multi_correct >= 9),
    }


def write_report(config_path: Path, config: dict[str, Any]) -> None:
    paths = output_paths(config_path, config, "confirmation")
    output = paths["dir"]
    results = pd.read_csv(paths["results"])
    curves = pd.read_csv(paths["curves"])
    summary = summarize(results)
    summary.to_csv(output / "MORPHOLOGY_PRODUCT_V1_SUMMARY.csv", index=False)
    ate_columns = [
        "dgp", "replication", "true_ate", "train_ate_dml", "train_ate_standard_error",
        "train_ate_ci_low", "train_ate_ci_high", "ate_train_error", "full_ate_dml",
        "full_ate_standard_error", "full_ate_ci_low", "full_ate_ci_high", "ate_full_error",
        "train_propensity_q01_oof", "train_propensity_q50_oof", "train_propensity_q99_oof",
    ]
    results[ate_columns].to_csv(output / "MORPHOLOGY_PRODUCT_V1_ATE_LONG.csv", index=False)
    render_figures(output, results, curves, config)
    decision = acceptance(results)
    table = []
    for row in summary.itertuples(index=False):
        table.append(
            f"| {row.dgp} | {row.exact_label_rate:.0%} | {row.supported_rate:.0%} | "
            f"{row.median_curve_spearman:.3f} | {row.median_ate_train_abs_error:.3f} | "
            f"{row.median_estimated_boundary_count:.1f} |"
        )
    report = f"""# Morphology Product v1：100-cell fresh confirmation

## 研究链条

本轮把现有研究收敛为一条可审计产品链：五折 cross-fitted DML/AIPW 在 TRAIN 建立总体平均效应锚点；一棵包含 12 个企业特征的 honest `CausalForestDML` 估计 CATE；预先指定的焦点企业特征进入 TRAIN 正交形态重构；untouched TEST 只确认候选形态相对常数效应是否改善。变量重要性排名保留为信息输出，不决定是否存在异质性，也不进入形态选择。

## Fresh 结果

| DGP | 形态准确率 | 获得 practical support | CATE 曲线排序相关中位数 | TRAIN DML ATE 绝对误差中位数 | 边界数中位数 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

注意：每个比例的分母只有 10，因此 90% 表示本轮观察到 9/10，不是对总体成功概率的精确证明。

## 图 1：CATE 相对 ATE

![CATE、DML ATE 与形态](FIGURE_1_CATE_ATE_AND_MORPHOLOGY.svg)

绿色线是 practical TRAIN 重构的 CATE，中间绿色带是 10 个 fresh replications 的 10%–90% 区间；蓝线与浅蓝带是 TRAIN DML ATE 及其置信区间；黑虚线和灰点线分别是真实 CATE 与真实 ATE，只用于 simulation evaluation。图直接显示平均效应会遮蔽哪些企业间差异，以及形态重构是否找回方向、转折、折点或平台。

## 图 2：形态恢复与 practical state

![Fresh recovery rates](FIGURE_2_RECOVERY_RATES.svg)

青色是高层形态标签准确率，橙色是同时通过 untouched-TEST 确认并达到稳定性要求的比例。两者不同，意味着“分类正确”与“证据足以形成理论候选”是两道独立门槛。

## 图 3：折点与多边界诊断

![Kink and boundary diagnostics](FIGURE_3_KINK_AND_BOUNDARY_DIAGNOSTICS.svg)

左图检查斜率变化在不同局部窗口缩小时是否持续存在，用来把真正连续折点与平滑弯曲区分开；右图显示单阈值和多阈值 DGP 中恢复出的边界数量。该图回答的是关键信息恢复，不把多阶梯另立为第十个高层形态。

## 预冻结 acceptance audit

```json
{json.dumps(decision, indent=2)}
```

## 边界

本轮是已知真值的 simulation confirmation，用于证明实现链能恢复预先定义的形态与关键信息；真实研究中不会看到黑色真值曲线，也不能把变量重要性或单次分类直接解释为组织理论机制。
"""
    (output / "MORPHOLOGY_PRODUCT_V1_REPORT_ZH.md").write_text(report, encoding="utf-8")
    best = results.sort_values(["dgp", "exact_label_correct", "curve_rmse"], ascending=[True, False, True]).groupby("dgp", as_index=False).first()
    (output / "MORPHOLOGY_PRODUCT_V1_EXAMPLE_OUTPUT.json").write_text(
        best.to_json(orient="records", indent=2), encoding="utf-8"
    )
    freeze = verify_freeze(config_path, config)
    manifest = {
        "status": "completed",
        "cells_expected": 100,
        "cells_completed": int((results["status"] == "completed").sum()),
        "cells_failed": int((results["status"] == "failed").sum()),
        "one_joint_multivariable_forest_per_cell": True,
        "train_test_split": "80/20 once per cell",
        "test_used_for_shape_selection": False,
        "oracle_used_for_shape_selection": False,
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "freeze_sha256": sha256_file((config_path.parent / config["freeze_path"]).resolve()),
        "freeze": freeze,
        "package_versions": V2.package_versions(),
        "acceptance": decision,
    }
    (output / "MORPHOLOGY_PRODUCT_V1_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=["development", "confirmation", "report"], required=True)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if args.freeze_only:
        print(create_freeze(config_path, config), flush=True)
        return 0
    if args.phase == "confirmation":
        verify_freeze(config_path, config)
    if args.phase in {"development", "confirmation"}:
        run_phase(config_path, config, args.phase)
    else:
        verify_freeze(config_path, config)
        write_report(config_path, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
