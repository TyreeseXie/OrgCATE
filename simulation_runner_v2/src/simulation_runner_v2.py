"""OrgCATE simulation runner v2.

Each replication has one external 80/20 train/test split. EconML owns nuisance
cross-fitting and forest honesty; this runner does not add outer OOF or a second
residualization layer.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from econml.dr import ForestDRLearner
from econml.grf import CausalForest
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split


RUNNER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RUNNER_ROOT.parent
ALLOWED_WORLDS = {"W0", "W1_weak_existing", "W1_core", "W2", "W3", "W4", "W5"}
TRUTH_ROLES = ["M1", "M2", "M3", "C1", "C2", "C3", "C4", "P1", "P2", "P3", "A1"]
MODEL_NAMES = {
    "M0_CF_RAW_HONEST",
    "M1_CF_DML_HONEST",
    "M2_CF_DML_TUNED_HONEST",
    "M3_FOREST_DR",
}
TUNED_PARAMETERS = ["min_weight_fraction_leaf", "max_depth", "min_var_fraction_leaf"]
PILOT_OUTPUTS = {
    "A_algorithm_comparison": "pilot_a_long.csv",
    "B_morphology": "pilot_b_long.csv",
    "C_n3000_confirmation": "pilot_c_n3000_long.csv",
}
ORACLE_COLUMNS = {
    "tau_true",
    "u_m1_true",
    "true_morphology",
    "true_boundary",
    "true_zero_crossing",
}

_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_CONFIG_PATH: Path | None = None
_WORKER_BACKBONE: pd.DataFrame | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "runner_version", "pilot", "backbone_path", "feature_role_mapping_path",
        "n", "test_size", "worlds", "common", "features", "dgp",
        "diagnostics", "execution",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Config missing keys: {sorted(missing)}")
    if config["runner_version"] != "v2":
        raise ValueError("Only simulation runner v2 configs are accepted")
    if not np.isclose(float(config["test_size"]), 0.20):
        raise ValueError("The external split must be exactly 80/20")
    if set(config["worlds"]).difference(ALLOWED_WORLDS):
        raise ValueError("W6-W8 and unknown worlds are prohibited")
    common = config["common"]
    if int(common["cv"]) != 2 or common["honest"] is not True:
        raise ValueError("M1/M3 require cv=2 and honest=True")
    if int(common["n_estimators"]) != 1000:
        raise ValueError("The frozen pilot requires 1000 trees")
    if len(config["features"]) != 12 or len(set(config["features"])) != 12:
        raise ValueError("Exactly 12 distinct multivariable X features are required")
    if not np.isclose(float(config["dgp"]["target_ate"]), 0.20):
        raise ValueError("The current positive-ATE family is fixed at 0.20")
    if config["pilot"] == "A_algorithm_comparison":
        if set(config["models"]) != MODEL_NAMES:
            raise ValueError("Pilot A must contain exactly M0-M3")
    elif config.get("primary_model") != "M1_CF_DML_HONEST":
        raise ValueError("Pilot B/C primary model must remain M1")


def resolve_input_path(config_path: Path, value: str) -> Path:
    path = (config_path.parent / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def pct_rank(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True).astype(float)


def load_frozen_backbone(project_root: Path, config: dict[str, Any], config_path: Path) -> pd.DataFrame:
    del project_root
    backbone_path = resolve_input_path(config_path, config["backbone_path"])
    role_path = resolve_input_path(config_path, config["feature_role_mapping_path"])
    role_registry = json.loads(role_path.read_text(encoding="utf-8"))
    roles = {
        role: value["feature"] if isinstance(value, dict) else value
        for role, value in role_registry.items()
    }
    missing_roles = set(TRUTH_ROLES).difference(roles)
    if missing_roles:
        raise ValueError(f"Role registry is missing {sorted(missing_roles)}")

    headline = pd.read_pickle(backbone_path, compression="gzip").copy()
    required = {"gvkey", "backbone_anchor_id", "candidate_t0", *config["features"]}
    required.update(roles[role] for role in TRUTH_ROLES)
    missing_columns = required.difference(headline.columns)
    if missing_columns:
        raise ValueError(f"Frozen backbone is missing {sorted(missing_columns)}")
    if "history_length_L" in headline and not (
        headline["history_length_L"] == int(config["dgp"]["history_length"])
    ).all():
        raise ValueError("Backbone includes rows outside frozen L=3")
    if not headline["gvkey"].is_unique:
        raise ValueError("Backbone must contain one anchor per firm")

    complete = [roles[role] for role in TRUTH_ROLES] + list(config["features"])
    safe = headline.loc[headline[complete].notna().all(axis=1)].copy().reset_index(drop=True)
    safe["U_M1"] = pd.to_numeric(safe[roles["M1"]], errors="raise").clip(1e-6, 1 - 1e-6)
    for role in TRUTH_ROLES[1:]:
        safe[f"U_{role}"] = pct_rank(pd.to_numeric(safe[roles[role]], errors="raise")).clip(
            1e-6, 1 - 1e-6
        )
    for role in TRUTH_ROLES:
        safe[f"Z_{role}"] = 2.0 * (safe[f"U_{role}"] - 0.5)
    x = safe[config["features"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Analyst X is not finite")
    if ORACLE_COLUMNS.intersection(x.columns):
        raise ValueError("Oracle field leaked into X")
    if len(safe) < int(config["n"]):
        raise ValueError("Truth-safe backbone is smaller than requested n")
    return safe


def stdz(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    sd = float(array.std())
    if sd < 1e-12:
        raise ValueError("Cannot standardize a constant score")
    return (array - array.mean()) / sd


def sigmoid(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    output = np.empty_like(array)
    positive = array >= 0
    output[positive] = 1 / (1 + np.exp(-array[positive]))
    exp_value = np.exp(array[~positive])
    output[~positive] = exp_value / (1 + exp_value)
    return output


def calibrate_alpha(score: np.ndarray, kappa: float, target: float) -> float:
    standardized = stdz(score)
    low, high = -20.0, 20.0
    for _ in range(200):
        midpoint = (low + high) / 2
        if sigmoid(midpoint + kappa * standardized).mean() < target:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


def treatment_propensity(data: pd.DataFrame, dgp: dict[str, Any]) -> np.ndarray:
    score = (
        0.60 * data["Z_C1"] - 0.50 * data["Z_C2"] + 0.35 * data["Z_M1"]
        + 0.30 * np.sin(np.pi * data["Z_C3"] / 2)
        + 0.25 * data["Z_C1"] * data["Z_C4"] + 0.20 * data["Z_A1"]
    ).to_numpy(dtype=float)
    kappa = float(dgp["overlap_kappa"])
    alpha = calibrate_alpha(score, kappa, float(dgp["target_treatment_prevalence"]))
    return sigmoid(alpha + kappa * stdz(score))


def untreated_outcome(data: pd.DataFrame, epsilon: np.ndarray, dgp: dict[str, Any]) -> np.ndarray:
    mean_score = (
        0.45 * data["Z_C1"] + 0.35 * data["Z_C2"]
        + 0.30 * np.sin(np.pi * data["Z_C3"] / 2) + 0.25 * data["Z_C4"]
        + 0.30 * data["Z_P1"] - 0.25 * data["Z_P2"] ** 2 + 0.20 * data["Z_P3"]
    ).to_numpy(dtype=float)
    r2 = float(dgp["outcome_r2"])
    return math.sqrt(r2) * stdz(mean_score) + math.sqrt(1 - r2) * stdz(epsilon)


def shape_tau(shape: np.ndarray, target: float, cate_sd: float) -> np.ndarray:
    tau = target + cate_sd * stdz(shape)
    return tau + target - tau.mean()


def first_zero_crossing(u: np.ndarray, values: np.ndarray) -> float:
    order = np.argsort(u)
    x = np.asarray(u, dtype=float)[order]
    y = np.asarray(values, dtype=float)[order]
    hits = np.flatnonzero(y[:-1] * y[1:] <= 0)
    if not len(hits):
        return float("nan")
    index = int(hits[0])
    if np.isclose(y[index], y[index + 1]):
        return float((x[index] + x[index + 1]) / 2)
    return float(x[index] - y[index] * (x[index + 1] - x[index]) / (y[index + 1] - y[index]))


def tau_for_world(world: str, u_m1: np.ndarray, dgp: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if world not in ALLOWED_WORLDS:
        raise ValueError(f"Unsupported or prohibited world: {world}")
    u = np.asarray(u_m1, dtype=float)
    target = float(dgp["target_ate"])
    spec = dgp[world]
    metadata: dict[str, Any] = {"world_version": spec["version"]}
    if world == "W0":
        tau = np.full(len(u), target)
        metadata.update(true_morphology="homogeneous", true_boundary=None)
    elif world == "W1_weak_existing":
        tau = shape_tau(u, target, float(spec["cate_sd"]))
        metadata.update(true_morphology="same_direction_monotone_weak", true_boundary=None)
    elif world == "W1_core":
        raw = np.tanh(float(spec["steepness"]) * (u - float(spec["center"])))
        centered = raw - raw.mean()
        scaled = centered / np.max(np.abs(centered))
        tau = target + float(spec["half_range"]) * scaled
        tau += target - tau.mean()
        metadata.update(true_morphology="same_direction_monotone_core", true_boundary=None)
    elif world in {"W2", "W4"}:
        raw = np.tanh(float(spec["steepness"]) * (u - float(spec["center"])))
        tau = shape_tau(raw, target, float(spec["cate_sd"]))
        morphology = "smooth_flip" if world == "W2" else "steep_continuous_flip"
        metadata.update(
            true_morphology=morphology,
            true_boundary=float(spec["center"]),
            true_zero_crossing=first_zero_crossing(u, tau),
        )
    elif world == "W3":
        raw = -((u - float(spec["vertex"])) ** 2)
        tau = shape_tau(raw, target, float(spec["cate_sd"]))
        metadata.update(true_morphology="inverted_u", true_boundary=float(spec["vertex"]))
    elif world == "W5":
        boundary = float(spec["boundary"])
        tau = shape_tau((u >= boundary).astype(float), target, float(spec["cate_sd"]))
        metadata.update(true_morphology="true_discontinuity", true_boundary=boundary)
    else:  # pragma: no cover
        raise ValueError(world)

    if not np.isclose(tau.mean(), target, atol=1e-12):
        raise AssertionError("Target ATE contract failed")
    delta = float(dgp["effect_sign_delta"])
    if world in {"W1_weak_existing", "W1_core"} and not (tau > delta).all():
        raise AssertionError(f"{world} violated the all-positive effect contract")
    if world == "W0" and not np.isclose(tau.std(), 0.0, atol=1e-12):
        raise AssertionError("W0 must be homogeneous")
    if world == "W2" and not ((tau < 0).any() and (tau > 0).any()):
        raise AssertionError("W2 must contain a sign flip")
    metadata.update(
        cate_true_mean=float(tau.mean()), cate_true_sd=float(tau.std()),
        cate_true_min=float(tau.min()), cate_true_max=float(tau.max()),
    )
    return tau, metadata


def make_seed_lineage(master_seed: int, replication: int) -> dict[str, int]:
    root = np.random.SeedSequence([master_seed, replication])
    children = root.spawn(5)
    names = ["backbone_sample_seed", "treatment_seed", "outcome_noise_seed", "split_seed", "model_seed"]
    result = {name: int(child.generate_state(1, dtype=np.uint32)[0]) for name, child in zip(names, children)}
    result.update(master_seed=int(master_seed), replication=int(replication))
    return result


def generate_world_dataset(
    backbone: pd.DataFrame, world: str, n: int, seeds: dict[str, int], config: dict[str, Any]
) -> dict[str, Any]:
    sample_rng = np.random.default_rng(seeds["backbone_sample_seed"])
    treatment_rng = np.random.default_rng(seeds["treatment_seed"])
    outcome_rng = np.random.default_rng(seeds["outcome_noise_seed"])
    positions = sample_rng.choice(len(backbone), size=n, replace=False)
    selected = backbone.iloc[positions].copy().reset_index(drop=True)
    if not selected["gvkey"].is_unique:
        raise AssertionError("A firm appeared more than once")
    propensity = treatment_propensity(selected, config["dgp"])
    treatment = (treatment_rng.uniform(size=n) < propensity).astype(int)
    y0 = untreated_outcome(selected, outcome_rng.normal(size=n), config["dgp"])
    tau, truth = tau_for_world(world, selected["U_M1"].to_numpy(dtype=float), config["dgp"])
    y1 = y0 + tau
    outcome = y0 + treatment * tau
    if not np.allclose(y1 - y0, tau) or not np.allclose(outcome, y0 + treatment * tau):
        raise AssertionError("Potential/observed outcome identity failed")
    x = selected[config["features"]].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise AssertionError("X contains non-finite values")
    return {
        "X": x, "T": treatment, "Y": outcome, "tau_true": tau,
        "u_m1_true": selected["U_M1"].to_numpy(dtype=float),
        "propensity_true": propensity, "unit_id": selected["gvkey"].astype(str).to_numpy(),
        "truth": truth,
    }


def _nuisance_models(seed: int) -> tuple[HistGradientBoostingRegressor, HistGradientBoostingClassifier]:
    return HistGradientBoostingRegressor(random_state=seed), HistGradientBoostingClassifier(random_state=seed)


def _m0(x_train, treatment_train, outcome_train, x_test, common, seed):
    started = time.perf_counter()
    estimator = CausalForest(
        n_estimators=int(common["n_estimators"]), min_samples_leaf=int(common["min_samples_leaf"]),
        max_samples=float(common["max_samples"]), honest=bool(common["honest"]),
        inference=False, random_state=seed, n_jobs=int(common["n_jobs"]),
    )
    estimator.fit(x_train, treatment_train.reshape(-1, 1), outcome_train)
    return np.asarray(estimator.predict(x_test), dtype=float).reshape(-1), time.perf_counter() - started, {"model": "M0_CF_RAW_HONEST"}


def _new_m1(common: dict[str, Any], seed: int) -> CausalForestDML:
    model_y, model_t = _nuisance_models(seed)
    return CausalForestDML(
        model_y=model_y, model_t=model_t, discrete_treatment=True, cv=int(common["cv"]),
        n_estimators=int(common["n_estimators"]), min_samples_leaf=int(common["min_samples_leaf"]),
        max_samples=float(common["max_samples"]), honest=bool(common["honest"]),
        inference=False, random_state=seed, n_jobs=int(common["n_jobs"]),
    )


def _m1(x_train, treatment_train, outcome_train, x_test, common, seed):
    started = time.perf_counter()
    estimator = _new_m1(common, seed)
    estimator.fit(outcome_train, treatment_train, X=x_train, W=None)
    return np.asarray(estimator.effect(x_test), dtype=float).reshape(-1), time.perf_counter() - started, {"model": "M1_CF_DML_HONEST"}


def _m2(x_train, treatment_train, outcome_train, x_test, common, seed):
    started = time.perf_counter()
    estimator = _new_m1(common, seed)
    estimator.tune(outcome_train, treatment_train, X=x_train, W=None, params="auto")
    # EconML 0.17.0 exposes tuned values as estimator attributes, not get_params().
    tuned = {name: getattr(estimator, name, None) for name in TUNED_PARAMETERS}
    estimator.fit(outcome_train, treatment_train, X=x_train, W=None)
    return np.asarray(estimator.effect(x_test), dtype=float).reshape(-1), time.perf_counter() - started, {
        "model": "M2_CF_DML_TUNED_HONEST", "tuned_params": tuned,
    }


def _m3(x_train, treatment_train, outcome_train, x_test, common, seed):
    started = time.perf_counter()
    model_y, model_t = _nuisance_models(seed)
    estimator = ForestDRLearner(
        model_regression=model_y, model_propensity=model_t, cv=int(common["cv"]),
        n_estimators=int(common["n_estimators"]), min_samples_leaf=int(common["min_samples_leaf"]),
        max_samples=float(common["max_samples"]), honest=bool(common["honest"]),
        random_state=seed, n_jobs=int(common["n_jobs"]),
    )
    estimator.fit(outcome_train, treatment_train, X=x_train, W=None)
    return np.asarray(estimator.effect(x_test), dtype=float).reshape(-1), time.perf_counter() - started, {"model": "M3_FOREST_DR"}


MODEL_FUNCS = {
    "M0_CF_RAW_HONEST": _m0, "M1_CF_DML_HONEST": _m1,
    "M2_CF_DML_TUNED_HONEST": _m2, "M3_FOREST_DR": _m3,
}


def binned_curve(u: np.ndarray, values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"u": u, "value": values})
    frame["bin"] = pd.qcut(frame["u"], q=min(bins, len(frame)), labels=False, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(u=("u", "mean"), value=("value", "mean"))
    return grouped["u"].to_numpy(dtype=float), grouped["value"].to_numpy(dtype=float)


def piecewise_diagnostic(u: np.ndarray, values: np.ndarray, diagnostic_config: dict[str, Any]) -> dict[str, Any]:
    x, y = binned_curve(u, values, int(diagnostic_config["bins"]))
    candidates = np.linspace(
        float(diagnostic_config["piecewise_candidate_min"]),
        float(diagnostic_config["piecewise_candidate_max"]),
        int(diagnostic_config["piecewise_candidate_count"]),
    )

    def fit_candidate(boundary: float, discontinuous: bool) -> tuple[float, np.ndarray]:
        columns = [np.ones(len(x)), x]
        if discontinuous:
            columns.append((x >= boundary).astype(float))
        columns.append(np.maximum(0.0, x - boundary))
        design = np.column_stack(columns)
        coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coefficient
        sse = max(float(np.sum(residual**2)), 1e-15)
        bic = len(x) * math.log(sse / len(x)) + design.shape[1] * math.log(len(x))
        return bic, coefficient

    continuous = [(fit_candidate(c, False)[0], c) for c in candidates]
    discontinuous = []
    for candidate in candidates:
        bic, coefficient = fit_candidate(candidate, True)
        discontinuous.append((bic, candidate, coefficient))
    best_continuous_bic, _ = min(continuous, key=lambda item: item[0])
    best_jump_bic, boundary, coefficient = min(discontinuous, key=lambda item: item[0])
    return {
        "morphology_pred_discontinuous": bool(best_jump_bic < best_continuous_bic),
        "morphology_bic_jump_minus_continuous": float(best_jump_bic - best_continuous_bic),
        "morphology_estimated_boundary": float(boundary),
        "morphology_estimated_jump": float(coefficient[2]),
    }


def metrics(tau_true: np.ndarray, tau_hat: np.ndarray) -> dict[str, Any]:
    error = tau_hat - tau_true
    result = {
        "cate_rmse": float(np.sqrt(np.mean(error**2))), "cate_mae": float(np.mean(np.abs(error))),
        "pred_cate_sd": float(np.std(tau_hat)),
        "pred_p95_p05": float(np.quantile(tau_hat, 0.95) - np.quantile(tau_hat, 0.05)),
        "pred_negative_share": float(np.mean(tau_hat < 0)),
        "test_ate_true": float(np.mean(tau_true)), "test_ate_hat": float(np.mean(tau_hat)),
        "test_ate_error": float(np.mean(tau_hat) - np.mean(tau_true)),
    }
    result["spearman"] = float(spearmanr(tau_true, tau_hat).statistic) if np.std(tau_true) > 1e-12 else float("nan")
    order = np.argsort(tau_hat)
    count = max(1, len(order) // 5)
    result["oracle_top_bottom_diff"] = float(np.mean(tau_true[order[-count:]]) - np.mean(tau_true[order[:count]]))
    return result


def world_metrics(world, u_m1_test, tau_true_test, tau_hat_test, config):
    result = metrics(tau_true_test, tau_hat_test)
    result.update(
        sign_region_accuracy=float(np.mean((tau_true_test >= 0) == (tau_hat_test >= 0))),
        true_negative_share=float(np.mean(tau_true_test < 0)),
    )
    bins = int(config["diagnostics"]["bins"])
    if world in {"W2", "W4"}:
        true_x, true_y = binned_curve(u_m1_test, tau_true_test, bins)
        pred_x, pred_y = binned_curve(u_m1_test, tau_hat_test, bins)
        true_crossing = first_zero_crossing(true_x, true_y)
        predicted_crossing = first_zero_crossing(pred_x, pred_y)
        result.update(
            true_zero_crossing_test=true_crossing, predicted_zero_crossing=predicted_crossing,
            zero_crossing_abs_error=(abs(predicted_crossing - true_crossing)
                if np.isfinite(predicted_crossing) and np.isfinite(true_crossing) else float("nan")),
        )
    if world == "W3":
        x, y = binned_curve(u_m1_test, tau_hat_test, bins)
        quadratic, linear, _ = np.polyfit(x, y, 2)
        turning_point = -linear / (2 * quadratic) if abs(quadratic) > 1e-12 else float("nan")
        classified = bool(quadratic < 0 and np.isfinite(turning_point) and 0.1 <= turning_point <= 0.9)
        result.update(
            morphology_inverted_u_classified=classified,
            morphology_estimated_turning_point=float(turning_point),
            turning_point_abs_error=(abs(float(turning_point) - float(config["dgp"]["W3"]["vertex"]))
                if np.isfinite(turning_point) else float("nan")),
        )
    if world in {"W4", "W5"}:
        diagnostic = piecewise_diagnostic(u_m1_test, tau_hat_test, config["diagnostics"])
        result.update(diagnostic)
        true_boundary = float(config["dgp"][world]["center" if world == "W4" else "boundary"])
        left = u_m1_test < true_boundary
        right = ~left
        result.update(
            false_discontinuity=(bool(diagnostic["morphology_pred_discontinuous"]) if world == "W4" else float("nan")),
            true_discontinuity_detected=(bool(diagnostic["morphology_pred_discontinuous"]) if world == "W5" else float("nan")),
            boundary_abs_error=(abs(float(diagnostic["morphology_estimated_boundary"]) - true_boundary) if world == "W5" else float("nan")),
            predicted_left_effect=float(np.mean(tau_hat_test[left])),
            predicted_right_effect=float(np.mean(tau_hat_test[right])),
            true_left_effect=float(np.mean(tau_true_test[left])),
            true_right_effect=float(np.mean(tau_true_test[right])),
        )
    return result


def split_once(treatment, unit_id, test_size, split_seed):
    indices = np.arange(len(treatment))
    train, test = train_test_split(indices, test_size=test_size, random_state=split_seed, stratify=treatment)
    if set(unit_id[train]).intersection(unit_id[test]):
        raise AssertionError("Firm leakage across train/test")
    if len(test) != int(round(len(treatment) * test_size)):
        raise AssertionError("Unexpected test-set size")
    return np.asarray(train), np.asarray(test)


def hash_unit(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def save_test_outputs(config, world, model_name, replication, data, test_index, prediction):
    directory = RUNNER_ROOT / "local_outputs" / config["pilot"] / world / model_name
    directory.mkdir(parents=True, exist_ok=True)
    key = [hash_unit(value) for value in data["unit_id"][test_index]]
    analyst = pd.DataFrame({"test_position": test_index, "unit_key_sha256": key, "cate_hat": prediction})
    oracle = pd.DataFrame({
        "test_position": test_index, "unit_key_sha256": key,
        "u_m1_true": data["u_m1_true"][test_index], "tau_true": data["tau_true"][test_index],
    })
    if ORACLE_COLUMNS.intersection(analyst.columns):
        raise AssertionError("Oracle truth leaked into analyst test predictions")
    analyst_path = directory / f"analyst_test_rep_{replication:03d}.parquet"
    oracle_path = directory / f"oracle_test_rep_{replication:03d}.parquet"
    analyst.to_parquet(analyst_path, index=False)
    oracle.to_parquet(oracle_path, index=False)
    return str(analyst_path), str(oracle_path)


def run_one(data, config, model_name, split_seed, model_seed, world, replication):
    train, test = split_once(data["T"], data["unit_id"], float(config["test_size"]), split_seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prediction, runtime, metadata = MODEL_FUNCS[model_name](
            data["X"][train], data["T"][train], data["Y"][train], data["X"][test],
            config["common"], model_seed,
        )
    if len(prediction) != len(test) or not np.isfinite(prediction).all():
        raise AssertionError("Model returned invalid test predictions")
    result = world_metrics(world, data["u_m1_true"][test], data["tau_true"][test], prediction, config)
    analyst_path = oracle_path = ""
    if config["execution"].get("save_test_predictions", True):
        analyst_path, oracle_path = save_test_outputs(config, world, model_name, replication, data, test, prediction)
    result.update(
        runtime_seconds=float(runtime), warning_count=len(caught),
        warning_messages="\n".join(str(item.message) for item in caught),
        train_n=len(train), test_n=len(test),
        train_treated_share=float(data["T"][train].mean()), test_treated_share=float(data["T"][test].mean()),
        external_split_count=1, train_index_sha256=sha256_arrays(train), test_index_sha256=sha256_arrays(test),
        dataset_sha256=sha256_arrays(data["X"], data["T"], data["Y"]),
        analyst_test_path=analyst_path, oracle_test_path=oracle_path,
        tuned_params_json=json.dumps(metadata.get("tuned_params", {}), sort_keys=True),
    )
    return result


def tasks_for_config(config: dict[str, Any]) -> list[tuple[str, int, str]]:
    tasks = []
    if config["pilot"] == "A_algorithm_comparison":
        for world in config["worlds"]:
            for model_name, model_config in config["models"].items():
                for replication in range(int(model_config["replications"])):
                    tasks.append((world, replication, model_name))
    else:
        for world in config["worlds"]:
            for replication in range(int(config["replications"])):
                tasks.append((world, replication, config["primary_model"]))
    return tasks


def _worker_init(config_path_value: str) -> None:
    global _WORKER_CONFIG, _WORKER_CONFIG_PATH, _WORKER_BACKBONE
    _WORKER_CONFIG_PATH = Path(config_path_value).resolve()
    _WORKER_CONFIG = load_config(_WORKER_CONFIG_PATH)
    _WORKER_BACKBONE = load_frozen_backbone(PROJECT_ROOT, _WORKER_CONFIG, _WORKER_CONFIG_PATH)


def _worker_run(task: tuple[str, int, str]) -> dict[str, Any]:
    if _WORKER_CONFIG is None or _WORKER_BACKBONE is None:
        raise RuntimeError("Worker was not initialized")
    world, replication, model_name = task
    config = _WORKER_CONFIG
    seeds = make_seed_lineage(int(config["common"]["master_seed"]), replication)
    base = {"pilot": config["pilot"], "world": world, "model": model_name,
            "replication": replication, "n": int(config["n"]), **seeds}
    try:
        data = generate_world_dataset(_WORKER_BACKBONE, world, int(config["n"]), seeds, config)
        result = run_one(data, config, model_name, seeds["split_seed"], seeds["model_seed"], world, replication)
        return {**base, "status": "completed", **data["truth"], **result,
                "error_type": "", "error_message": "", "traceback": ""}
    except Exception as error:
        return {**base, "status": "failed", "runtime_seconds": float("nan"),
                "warning_count": 0, "warning_messages": "", "error_type": type(error).__name__,
                "error_message": str(error), "traceback": traceback.format_exc()}


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def package_versions() -> dict[str, str]:
    names = ["econml", "scikit-learn", "numpy", "pandas", "scipy", "pyarrow"]
    return {name: importlib.metadata.version(name) for name in names}


def result_path_for(config: dict[str, Any]) -> Path:
    return RUNNER_ROOT / "results" / PILOT_OUTPUTS[config["pilot"]]


def write_manifest(config_path: Path, config: dict[str, Any], result_path: Path) -> Path:
    backbone_path = resolve_input_path(config_path, config["backbone_path"])
    role_path = resolve_input_path(config_path, config["feature_role_mapping_path"])
    result = pd.read_csv(result_path)
    manifest = {
        "artifact": f"OrgCATE simulation runner v2 {config['pilot']}", "created_utc": utc_now(),
        "status": "completed" if (result["status"] == "completed").all() else "completed_with_failures",
        "expected_runs": len(tasks_for_config(config)), "completed_runs": int((result["status"] == "completed").sum()),
        "failed_runs": int((result["status"] == "failed").sum()), "python": sys.version,
        "platform": platform.platform(), "packages": package_versions(), "config_path": str(config_path),
        "config_sha256": sha256_file(config_path), "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()), "backbone_path": str(backbone_path),
        "backbone_sha256": sha256_file(backbone_path), "feature_role_mapping_path": str(role_path),
        "feature_role_mapping_sha256": sha256_file(role_path), "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "scientific_contract": {
            "external_split": "one 80/20 train/test split", "outer_oof": False,
            "dml_internal_cv": 2, "forest_honesty": True,
            "one_multivariable_forest_per_dataset_model": True,
            "oracle_accuracy_scope": "untouched test only", "prohibited_worlds_run": [],
        }, "config": config,
    }
    path = RUNNER_ROOT / "results" / f"{config['pilot']}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return path


def run_config(config_path: Path) -> int:
    config_path = config_path.resolve()
    config = load_config(config_path)
    result_path = result_path_for(config)
    expected_tasks = tasks_for_config(config)
    existing = {}
    if result_path.is_file():
        previous = pd.read_csv(result_path)
        for row in previous.to_dict("records"):
            if row.get("status") == "completed":
                existing[(str(row["world"]), int(row["replication"]), str(row["model"]))] = row
    pending = [task for task in expected_tasks if task not in existing]
    rows = dict(existing)
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(
            max_workers=int(config["execution"]["dataset_workers"]),
            initializer=_worker_init, initargs=(str(config_path),),
        ) as executor:
            futures = {executor.submit(_worker_run, task): task for task in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                key = (str(row["world"]), int(row["replication"]), str(row["model"]))
                rows[key] = row
                frame = pd.DataFrame(rows.values()).sort_values(["world", "replication", "model"])
                atomic_write_csv(frame, result_path)
                if index == 1 or index % 10 == 0 or index == len(pending):
                    print(f"PROGRESS pilot={config['pilot']} this_run={index}/{len(pending)} total={len(rows)}/{len(expected_tasks)} latest={key} status={row['status']}", flush=True)
    frame = pd.DataFrame(rows.values()).sort_values(["world", "replication", "model"])
    atomic_write_csv(frame, result_path)
    failure_columns = ["pilot", "world", "model", "replication", "error_type", "error_message", "traceback"]
    failures = frame.loc[frame["status"] == "failed"].reindex(columns=failure_columns)
    atomic_write_csv(failures, RUNNER_ROOT / "results" / f"{config['pilot']}_failures.csv")
    seed_columns = ["pilot", "world", "model", "replication", "master_seed", "backbone_sample_seed",
                    "treatment_seed", "outcome_noise_seed", "split_seed", "model_seed"]
    atomic_write_csv(frame.reindex(columns=seed_columns), RUNNER_ROOT / "results" / f"{config['pilot']}_seeds.csv")
    write_manifest(config_path, config, result_path)
    print(f"DONE pilot={config['pilot']} rows={len(frame)} failures={len(failures)} wall_seconds={time.perf_counter() - started:.1f}", flush=True)
    return 0 if len(frame) == len(expected_tasks) and failures.empty else 1


def top_bottom_by_u(u: np.ndarray, tau: np.ndarray) -> float:
    order = np.argsort(u)
    count = max(1, len(order) // 5)
    return float(tau[order[-count:]].mean() - tau[order[:count]].mean())


def truth_audit(config_path: Path) -> int:
    config_path = config_path.resolve()
    config = load_config(config_path)
    backbone = load_frozen_backbone(PROJECT_ROOT, config, config_path)
    full = {}
    u = backbone["U_M1"].to_numpy(dtype=float)
    for world in ["W1_weak_existing", "W1_core"]:
        tau, metadata = tau_for_world(world, u, config["dgp"])
        full[world] = {**metadata, "top_bottom_true_effect_diff": top_bottom_by_u(u, tau)}
    sample_rows = []
    for replication in range(200):
        seeds = make_seed_lineage(int(config["common"]["master_seed"]) + 100_000, replication)
        rng = np.random.default_rng(seeds["backbone_sample_seed"])
        selected = backbone.iloc[rng.choice(len(backbone), size=int(config["n"]), replace=False)]
        sample_u = selected["U_M1"].to_numpy(dtype=float)
        for world in ["W1_weak_existing", "W1_core"]:
            tau, _ = tau_for_world(world, sample_u, config["dgp"])
            sample_rows.append({"replication": replication, "world": world, "ate": float(tau.mean()),
                                "minimum_effect": float(tau.min()),
                                "top_bottom_true_effect_diff": top_bottom_by_u(sample_u, tau)})
    samples = pd.DataFrame(sample_rows)
    summary = samples.groupby("world").agg(
        replications=("replication", "nunique"), minimum_effect_over_draws=("minimum_effect", "min"),
        mean_top_bottom_true_effect_diff=("top_bottom_true_effect_diff", "mean"),
        minimum_top_bottom_true_effect_diff=("top_bottom_true_effect_diff", "min"),
        maximum_ate_error=("ate", lambda value: float(np.max(np.abs(value - 0.20)))),
    )
    ratio = float(full["W1_core"]["top_bottom_true_effect_diff"]) / float(full["W1_weak_existing"]["top_bottom_true_effect_diff"])
    checks = {
        "target_ate_exact": bool((samples["ate"] - 0.20).abs().max() < 1e-12),
        "weak_preserved_sd_full_backbone": bool(np.isclose(full["W1_weak_existing"]["cate_true_sd"], 0.08, atol=1e-12)),
        "w1_core_all_effects_above_0_02": bool(samples.loc[samples.world == "W1_core", "minimum_effect"].min() > 0.02),
        "w1_core_full_contrast_at_least_25pct_larger": bool(ratio >= 1.25),
        "audit_completed_before_pilot_result": not result_path_for(config).exists(),
    }
    report = {
        "artifact": "W1_core pre-model truth audit", "created_utc": utc_now(),
        "config_path": str(config_path), "config_sha256": sha256_file(config_path), "draws_per_world": 200,
        "full_backbone": full, "sample_draw_summary": summary.reset_index().to_dict("records"),
        "core_to_weak_full_contrast_ratio": ratio, "checks": checks, "passed": all(checks.values()),
    }
    output = RUNNER_ROOT / "reports" / "W1_CORE_TRUTH_AUDIT.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    atomic_write_csv(samples, RUNNER_ROOT / "results" / "w1_core_truth_audit_draws.csv")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report["passed"] else 1


def summarize(config_path: Path) -> int:
    config = load_config(config_path.resolve())
    path = result_path_for(config)
    if not path.is_file():
        raise FileNotFoundError(path)
    results = pd.read_csv(path)
    completed = results.loc[results["status"] == "completed"].copy()
    metric_names = [
        "cate_rmse", "cate_mae", "spearman", "pred_cate_sd", "pred_p95_p05",
        "oracle_top_bottom_diff", "pred_negative_share", "sign_region_accuracy",
        "zero_crossing_abs_error", "morphology_inverted_u_classified", "turning_point_abs_error",
        "false_discontinuity", "true_discontinuity_detected", "boundary_abs_error", "runtime_seconds",
    ]
    aggregations = {metric: ["mean", "std"] for metric in metric_names if metric in completed.columns}
    summary = completed.groupby(["world", "model"]).agg(aggregations)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    counts = results.groupby(["world", "model"]).agg(
        result_rows=("replication", "size"),
        completed_runs=("status", lambda value: int((value == "completed").sum())),
        failed_runs=("status", lambda value: int((value == "failed").sum())),
    ).reset_index()
    summary = counts.merge(summary, on=["world", "model"], how="left")
    atomic_write_csv(summary, RUNNER_ROOT / "results" / f"{config['pilot']}_summary.csv")
    print(summary.to_string(index=False), flush=True)
    return 0


def audit_outputs(config_paths: list[Path]) -> int:
    pilot_reports = {}
    all_checks = []
    prediction_files_audited = 0
    for config_path in config_paths:
        config_path = config_path.resolve()
        config = load_config(config_path)
        result_path = result_path_for(config)
        if not result_path.is_file():
            pilot_reports[config["pilot"]] = {"present": False}
            continue
        results = pd.read_csv(result_path)
        expected = len(tasks_for_config(config))
        checks = {
            "expected_rows": len(results) == expected,
            "no_duplicate_runs": not results.duplicated(["world", "model", "replication"]).any(),
            "all_runs_completed": bool((results["status"] == "completed").all()),
            "external_split_count_one": bool((results["external_split_count"] == 1).all()),
            "train_test_sizes": bool((results["train_n"] == int(config["n"] * 0.8)).all() and (results["test_n"] == int(config["n"] * 0.2)).all()),
            "no_prohibited_world": not bool(set(results["world"]).difference(ALLOWED_WORLDS)),
            "config_hash_in_manifest": False, "result_hash_in_manifest": False,
            "matched_data_and_split_across_models": True, "analyst_oracle_separation": True,
            "test_prediction_rows": True,
        }
        for _, group in results.groupby(["world", "replication"]):
            if group["dataset_sha256"].nunique() != 1 or group["test_index_sha256"].nunique() != 1:
                checks["matched_data_and_split_across_models"] = False
        for row in results.itertuples(index=False):
            analyst = pd.read_parquet(Path(row.analyst_test_path))
            oracle = pd.read_parquet(Path(row.oracle_test_path))
            prediction_files_audited += 2
            if ORACLE_COLUMNS.intersection(analyst.columns):
                checks["analyst_oracle_separation"] = False
            if len(analyst) != int(row.test_n) or len(oracle) != int(row.test_n):
                checks["test_prediction_rows"] = False
            merged = analyst.merge(oracle, on=["test_position", "unit_key_sha256"], validate="one_to_one")
            if len(merged) != int(row.test_n):
                checks["analyst_oracle_separation"] = False
        manifest_path = RUNNER_ROOT / "results" / f"{config['pilot']}_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checks["config_hash_in_manifest"] = manifest["config_sha256"] == sha256_file(config_path)
            checks["result_hash_in_manifest"] = manifest["result_sha256"] == sha256_file(result_path)
        all_checks.extend(checks.values())
        pilot_reports[config["pilot"]] = {
            "present": True, "expected_runs": expected,
            "completed_runs": int((results["status"] == "completed").sum()),
            "failed_runs": int((results["status"] == "failed").sum()), "checks": checks,
        }
    truth_path = RUNNER_ROOT / "reports" / "W1_CORE_TRUTH_AUDIT.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8")) if truth_path.is_file() else {"passed": False}
    all_checks.append(bool(truth["passed"]))
    report = {
        "artifact": "Simulation Runner v2 execution audit", "created_utc": utc_now(),
        "pilot_reports": pilot_reports, "w1_core_truth_audit_passed": bool(truth["passed"]),
        "prediction_files_audited": prediction_files_audited,
        "guarded_gate_runner_used": False, "outer_oof_used": False,
        "all_present_checks_passed": all(all_checks),
    }
    output = RUNNER_ROOT / "reports" / "SIMULATION_RUNNER_V2_EXECUTION_AUDIT.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report["all_present_checks_passed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OrgCATE simulation runner v2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ["truth-audit", "run", "summarize"]:
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--config", type=Path, action="append", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "truth-audit":
        return truth_audit(args.config)
    if args.command == "run":
        return run_config(args.config)
    if args.command == "summarize":
        return summarize(args.config)
    return audit_outputs(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
