"""Frozen Morphology Reconstruction v2 execution runner.

This module is deliberately separate from the earlier morphology pilot.  It
reuses the frozen simulation-v2 data generator and CausalForestDML builder,
then performs a train-only orthogonal morphology reconstruction.  No model or
family selection consumes untouched TEST observations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
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
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer


PILOT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PILOT_ROOT.parent
SOURCE_RUNNER = PROJECT_ROOT / "codex_simulation_runner_v2" / "src" / "simulation_runner_v2.py"
SOURCE_CONFIG = PROJECT_ROOT / "codex_simulation_runner_v2" / "config" / "pilot_b.json"
CONFIG_PATH = PILOT_ROOT / "config" / "morphology_v2_confirmatory.json"
TRUE_MODERATOR = "at_rank_year_currency"
WORLDS = {"W2", "W3", "W4", "W5"}
FAMILIES = ["linear", "quadratic", "smooth_reversal", "jump", "flexible_spline"]
SIMPLE_FAMILIES = FAMILIES[:4]
GRID = np.linspace(0.02, 0.98, 201)
ORACLE_FIELDS = {
    "tau_true", "y0_true", "y1_true", "propensity_true", "u_m1_true",
    "true_morphology", "true_boundary", "true_zero_crossing",
}


def load_v2() -> Any:
    spec = importlib.util.spec_from_file_location("simulation_runner_v2_frozen", SOURCE_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v2 = load_v2()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(*arrays: np.ndarray) -> str:
    return v2.sha256_arrays(*arrays)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def inclusive_grid(start: float, end: float, step: float) -> list[float]:
    count = int(round((end - start) / step))
    return [round(start + i * step, 12) for i in range(count + 1)]


def validate_config(config: dict[str, Any]) -> None:
    required = {"version", "fresh_confirmatory_master_seed", "worlds", "sample_sizes", "replications_per_cell", "test_size", "primary_forest", "moderator_modes", "second_stage", "families"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"confirmatory config missing {sorted(missing)}")
    if config["version"] != "morphology_reconstruction_v2":
        raise ValueError("wrong confirmatory version")
    if set(config["worlds"]) != WORLDS or list(config["sample_sizes"]) != [1500, 3000] or int(config["replications_per_cell"]) != 30:
        raise ValueError("fresh design must be W2-W5, n=1500/3000, R=30")
    if not np.isclose(float(config["test_size"]), 0.20):
        raise ValueError("external split must remain 80/20")
    forest = config["primary_forest"]
    frozen = {"class": "econml.dml.CausalForestDML", "cv": 2, "honest": True, "n_estimators": 1000, "min_samples_leaf": 5, "max_samples": 0.45}
    if forest != frozen:
        raise ValueError(f"primary forest changed: expected {frozen}, got {forest}")
    if set(config["moderator_modes"]) != {"forest_selected", "oracle_relative_firm_size_diagnostic"}:
        raise ValueError("moderator modes changed")
    stage = config["second_stage"]
    if int(stage["nuisance_crossfit_folds"]) != 5 or int(stage["shape_cv_folds"]) != 5 or not stage["one_se_abstention_rule"] or stage["no_inverse_propensity_pseudooutcome"] is not True:
        raise ValueError("second-stage contract changed")
    fam = config["families"]
    if inclusive_grid(float(fam["single_turn_quadratic"]["c_start"]), float(fam["single_turn_quadratic"]["c_end"]), float(fam["single_turn_quadratic"]["c_step"])) != inclusive_grid(.15, .85, .025):
        raise ValueError("quadratic grid changed")
    smooth = fam["smooth_reversal"]
    if [float(x) for x in smooth["k_grid"]] != [2, 4, 6, 10, 15, 20, 25, 35, 50, 75]:
        raise ValueError("smooth k grid changed")
    if inclusive_grid(float(smooth["c_start"]), float(smooth["c_end"]), float(smooth["c_step"])) != inclusive_grid(.20, .80, .025):
        raise ValueError("smooth c grid changed")
    jump = fam["jump"]
    if inclusive_grid(float(jump["c_start"]), float(jump["c_end"]), float(jump["c_step"])) != inclusive_grid(.20, .80, .025):
        raise ValueError("jump grid changed")
    spline = fam["flexible_spline_benchmark"]
    if spline["interior_knots_grid"] != [4, 6, 8] or spline["ridge_alpha_grid"] != [0.0001, 0.001, 0.01, 0.1]:
        raise ValueError("flexible spline grid changed")


def load_source() -> dict[str, Any]:
    source = v2.load_config(SOURCE_CONFIG)
    if len(source["features"]) != 12 or TRUE_MODERATOR not in source["features"]:
        raise ValueError("frozen 12-feature source contract unavailable")
    if set(source["dgp"]).intersection({"W0", "W1_core", "W2", "W3", "W4", "W5"}) != {"W0", "W1_core", "W2", "W3", "W4", "W5"}:
        raise ValueError("source DGP worlds incomplete")
    return source


def add_rank_columns(backbone: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    result = backbone.copy()
    for i, feature in enumerate(features):
        values = pd.to_numeric(result[feature], errors="raise")
        if feature.endswith("_rank_year_currency"):
            rank = values.astype(float).clip(1e-6, 1 - 1e-6)
        else:
            rank = values.rank(method="average", pct=True).astype(float).clip(1e-6, 1 - 1e-6)
        result[f"__u_{i}"] = rank
    return result


def feature_u_matrix(backbone: pd.DataFrame, unit_ids: np.ndarray, features: list[str]) -> np.ndarray:
    lookup = backbone.assign(__unit=backbone["gvkey"].astype(str)).set_index("__unit")
    values = lookup.loc[np.asarray(unit_ids, dtype=str), [f"__u_{i}" for i in range(len(features))]].to_numpy(float)
    if values.shape != (len(unit_ids), len(features)) or not np.isfinite(values).all():
        raise AssertionError("invalid reference-ECDF feature matrix")
    return values


def importance(estimator: Any, count: int) -> np.ndarray:
    values = np.asarray(estimator.feature_importances_, dtype=float).reshape(-1)
    if len(values) != count or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise AssertionError("invalid forest importance")
    return values / values.sum()


def seeds(master: int, n: int, replication: int) -> dict[str, int]:
    root = np.random.SeedSequence([int(master), int(n), int(replication)])
    children = root.spawn(8)
    names = ["backbone_sample_seed", "treatment_seed", "outcome_noise_seed", "split_seed", "forest_seed", "nuisance_seed", "shape_cv_seed", "artifact_seed"]
    result = {name: int(child.generate_state(1, dtype=np.uint32)[0]) for name, child in zip(names, children)}
    result.update(master_seed=int(master), n=int(n), replication=int(replication))
    return result


def crossfit_nuisance(x_train: np.ndarray, y_train: np.ndarray, t_train: np.ndarray, x_test: np.ndarray, seed: int, folds: int = 5) -> dict[str, Any]:
    """Return OOF TRAIN residuals and full-TRAIN TEST predictions."""
    splitter = KFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    m_oof = np.full(len(y_train), np.nan)
    e_oof = np.full(len(y_train), np.nan)
    child = np.random.SeedSequence(int(seed)).spawn(2 * int(folds) + 2)
    child_seeds = [int(item.generate_state(1, dtype=np.uint32)[0]) for item in child]
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(x_train)):
        outcome = HistGradientBoostingRegressor(random_state=child_seeds[2 * fold])
        treatment = HistGradientBoostingClassifier(random_state=child_seeds[2 * fold + 1])
        outcome.fit(x_train[fit_idx], y_train[fit_idx])
        treatment.fit(x_train[fit_idx], t_train[fit_idx])
        m_oof[val_idx] = outcome.predict(x_train[val_idx])
        e_oof[val_idx] = treatment.predict_proba(x_train[val_idx])[:, 1]
    if not np.isfinite(m_oof).all() or not np.isfinite(e_oof).all():
        raise AssertionError("nuisance OOF predictions incomplete")
    outcome_full = HistGradientBoostingRegressor(random_state=child_seeds[-2])
    treatment_full = HistGradientBoostingClassifier(random_state=child_seeds[-1])
    outcome_full.fit(x_train, y_train)
    treatment_full.fit(x_train, t_train)
    m_test = outcome_full.predict(x_test)
    e_test = treatment_full.predict_proba(x_test)[:, 1]
    if not np.isfinite(m_test).all() or not np.isfinite(e_test).all():
        raise AssertionError("nuisance TEST predictions invalid")
    return {
        "y_residual_train": np.asarray(y_train - m_oof, float),
        "t_residual_train": np.asarray(t_train - e_oof, float),
        "m_oof": m_oof, "e_oof": e_oof,
        "m_test": np.asarray(m_test, float), "e_test": np.asarray(e_test, float),
        "outcome_model": outcome_full, "treatment_model": treatment_full,
        "train_residuals_oof": True, "nuisance_folds": int(folds), "test_refit_scope": "full TRAIN only",
    }


def r_loss(y_residual: np.ndarray, t_residual: np.ndarray, tau: np.ndarray) -> float:
    residual = np.asarray(y_residual, float) - np.asarray(t_residual, float) * np.asarray(tau, float)
    return float(np.mean(residual ** 2))


def spline_transformer(interior_knots: int) -> SplineTransformer:
    return SplineTransformer(n_knots=int(interior_knots) + 2, degree=3, knots="quantile", include_bias=True, extrapolation="linear")


def fit_candidate(family: str, u: np.ndarray, y_res: np.ndarray, t_res: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    value = np.asarray(u, float)
    if family == "linear":
        basis = np.column_stack([np.ones(len(value)), value])
    elif family == "quadratic":
        c = float(params["c"])
        basis = np.column_stack([np.ones(len(value)), (value - c) ** 2])
    elif family == "smooth_reversal":
        basis = np.column_stack([np.ones(len(value)), np.tanh(float(params["k"]) * (value - float(params["c"])))])
    elif family == "jump":
        basis = np.column_stack([np.ones(len(value)), (value >= float(params["c"])).astype(float)])
    elif family == "flexible_spline":
        transformer = spline_transformer(int(params["interior_knots"]))
        basis = transformer.fit_transform(value.reshape(-1, 1))
        design = np.asarray(t_res, float)[:, None] * basis
        model = Ridge(alpha=float(params["alpha"]), fit_intercept=False).fit(design, y_res)
        return {"family": family, "params": dict(params), "transformer": transformer, "coefficient": np.asarray(model.coef_, float)}
    else:
        raise ValueError(family)
    design = np.asarray(t_res, float)[:, None] * basis
    coefficient, *_ = np.linalg.lstsq(design, np.asarray(y_res, float), rcond=None)
    return {"family": family, "params": dict(params), "coefficient": np.asarray(coefficient, float)}


def predict_candidate(model: dict[str, Any], u: np.ndarray) -> np.ndarray:
    value = np.asarray(u, float)
    family = model["family"]
    params = model["params"]
    if family == "linear":
        basis = np.column_stack([np.ones(len(value)), value])
    elif family == "quadratic":
        basis = np.column_stack([np.ones(len(value)), (value - float(params["c"])) ** 2])
    elif family == "smooth_reversal":
        basis = np.column_stack([np.ones(len(value)), np.tanh(float(params["k"]) * (value - float(params["c"])))])
    elif family == "jump":
        basis = np.column_stack([np.ones(len(value)), (value >= float(params["c"])).astype(float)])
    elif family == "flexible_spline":
        basis = model["transformer"].transform(value.reshape(-1, 1))
    else:
        raise ValueError(family)
    return np.asarray(basis @ model["coefficient"], float).reshape(-1)


def family_grid(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    fam = config["families"]
    return {
        "linear": [{}],
        "quadratic": [{"c": c} for c in inclusive_grid(float(fam["single_turn_quadratic"]["c_start"]), float(fam["single_turn_quadratic"]["c_end"]), float(fam["single_turn_quadratic"]["c_step"]))],
        "smooth_reversal": [{"k": k, "c": c} for k in fam["smooth_reversal"]["k_grid"] for c in inclusive_grid(float(fam["smooth_reversal"]["c_start"]), float(fam["smooth_reversal"]["c_end"]), float(fam["smooth_reversal"]["c_step"]))],
        "jump": [{"c": c} for c in inclusive_grid(float(fam["jump"]["c_start"]), float(fam["jump"]["c_end"]), float(fam["jump"]["c_step"]))],
        "flexible_spline": [{"interior_knots": k, "alpha": a} for k in fam["flexible_spline_benchmark"]["interior_knots_grid"] for a in fam["flexible_spline_benchmark"]["ridge_alpha_grid"]],
    }


def fit_theory_library(u: np.ndarray, y_res: np.ndarray, t_res: np.ndarray, config: dict[str, Any], seed: int) -> dict[str, Any]:
    """5-fold CV over every frozen specification, then refit selected specs on all TRAIN residuals."""
    folds = int(config["second_stage"]["shape_cv_folds"])
    splitter = KFold(n_splits=folds, shuffle=True, random_state=int(seed))
    candidates = family_grid(config)
    records: list[dict[str, Any]] = []
    fitted: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        family_records = []
        for params in candidates[family]:
            fold_losses = []
            for fit_idx, val_idx in splitter.split(u):
                model = fit_candidate(family, u[fit_idx], y_res[fit_idx], t_res[fit_idx], params)
                fold_losses.append(r_loss(y_res[val_idx], t_res[val_idx], predict_candidate(model, u[val_idx])))
            mean_loss = float(np.mean(fold_losses))
            se_loss = float(np.std(fold_losses, ddof=1) / np.sqrt(len(fold_losses))) if len(fold_losses) > 1 else 0.0
            family_records.append({"family": family, "params": dict(params), "cv_fold_losses": fold_losses, "cv_mean_r_loss": mean_loss, "cv_se_r_loss": se_loss})
        best = min(family_records, key=lambda item: (item["cv_mean_r_loss"], json.dumps(item["params"], sort_keys=True)))
        model = fit_candidate(family, u, y_res, t_res, best["params"])
        model.update({"cv_mean_r_loss": best["cv_mean_r_loss"], "cv_se_r_loss": best["cv_se_r_loss"], "cv_fold_losses": best["cv_fold_losses"]})
        fitted[family] = model
        records.extend(family_records)
    best_overall = min(fitted, key=lambda family: (fitted[family]["cv_mean_r_loss"], FAMILIES.index(family)))
    best_simple = min(SIMPLE_FAMILIES, key=lambda family: (fitted[family]["cv_mean_r_loss"], SIMPLE_FAMILIES.index(family)))
    best_overall_model = fitted[best_overall]
    adequacy_limit = float(best_overall_model["cv_mean_r_loss"] + best_overall_model["cv_se_r_loss"])
    selected = best_simple if fitted[best_simple]["cv_mean_r_loss"] <= adequacy_limit else "ABSTAIN_FLEXIBLE_STRUCTURE"
    return {"models": fitted, "records": records, "best_overall": best_overall, "best_simple": best_simple, "one_se_limit": adequacy_limit, "selected": selected, "folds": folds}


def first_crossing(x: np.ndarray, y: np.ndarray) -> float:
    values = np.asarray(y, float)
    hits = np.flatnonzero(values[:-1] * values[1:] <= 0)
    if not len(hits):
        return float("nan")
    i = int(hits[0])
    if np.isclose(values[i], values[i + 1]):
        return float((x[i] + x[i + 1]) / 2)
    return float(x[i] - values[i] * (x[i + 1] - x[i]) / (values[i + 1] - values[i]))


def exact_truth_curve(world: str, grid: np.ndarray, sampled_u: np.ndarray, dgp: dict[str, Any]) -> np.ndarray:
    spec = dgp[world]
    sampled = np.asarray(sampled_u, float)
    grid = np.asarray(grid, float)
    if world in {"W2", "W4"}:
        raw_sample = np.tanh(float(spec["steepness"]) * (sampled - float(spec["center"])))
        raw_grid = np.tanh(float(spec["steepness"]) * (grid - float(spec["center"])))
    elif world == "W3":
        raw_sample = -((sampled - float(spec["vertex"])) ** 2)
        raw_grid = -((grid - float(spec["vertex"])) ** 2)
    elif world == "W5":
        raw_sample = (sampled >= float(spec["boundary"])).astype(float)
        raw_grid = (grid >= float(spec["boundary"])).astype(float)
    else:
        raise ValueError(world)
    scale = float(np.std(raw_sample))
    return float(dgp["target_ate"]) + float(spec["cate_sd"]) * (raw_grid - float(np.mean(raw_sample))) / scale


def conditional_truth_grid(u: np.ndarray, tau: np.ndarray) -> np.ndarray:
    order = np.argsort(u)
    x = np.asarray(u, float)[order]
    y = np.asarray(tau, float)[order]
    unique_x, idx = np.unique(x, return_index=True)
    unique_y = np.asarray([y[np.flatnonzero(x == value)].mean() for value in unique_x])
    return np.interp(GRID, unique_x, unique_y)


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def train_forest_curve(u_train: np.ndarray, forest_train: np.ndarray, bins: int = 20) -> np.ndarray:
    frame = pd.DataFrame({"u": u_train, "cate": forest_train})
    frame["bin"] = pd.qcut(frame["u"], q=min(int(bins), len(frame)), labels=False, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(u=("u", "mean"), cate=("cate", "mean"))
    return np.interp(GRID, grouped["u"].to_numpy(float), grouped["cate"].to_numpy(float))


def morphology_metrics(world: str, truth: np.ndarray, estimate: np.ndarray, truth_grid: np.ndarray, estimate_grid: np.ndarray) -> dict[str, Any]:
    truth_range = float(np.ptp(truth_grid))
    below = max(float(np.min(truth_grid) - np.min(estimate_grid)), 0.0)
    above = max(float(np.max(estimate_grid) - np.max(truth_grid)), 0.0)
    result: dict[str, Any] = {
        "cate_rmse": float(np.sqrt(np.mean((estimate - truth) ** 2))),
        "cate_mae": float(np.mean(np.abs(estimate - truth))),
        "spearman": safe_spearman(truth, estimate),
        "sign_region_accuracy": float(np.mean(np.sign(estimate) == np.sign(truth))),
        "max_overshoot_normalized": max(below, above) / truth_range if truth_range > 1e-12 else float("nan"),
    }
    if world == "W2":
        tc = first_crossing(GRID, truth_grid)
        ec = first_crossing(GRID, estimate_grid)
        true_contrast = float(np.interp(.90, GRID, truth_grid) - np.interp(.10, GRID, truth_grid))
        contrast = float(np.interp(.90, GRID, estimate_grid) - np.interp(.10, GRID, estimate_grid))
        result.update(zero_crossing_abs_error=abs(ec - tc) if np.isfinite(ec) else float("nan"), high_low_recovery_ratio=contrast / true_contrast if abs(true_contrast) > 1e-12 else float("nan"))
    elif world == "W3":
        true_turn = float(GRID[int(np.argmax(truth_grid))])
        est_turn = float(GRID[int(np.argmax(estimate_grid))])
        true_peak = float(np.interp(true_turn, GRID, truth_grid))
        true_tail = float((np.interp(.10, GRID, truth_grid) + np.interp(.90, GRID, truth_grid)) / 2)
        est_peak = float(np.interp(true_turn, GRID, estimate_grid))
        est_tail = float((np.interp(.10, GRID, estimate_grid) + np.interp(.90, GRID, estimate_grid)) / 2)
        true_contrast = true_peak - true_tail
        contrast = est_peak - est_tail
        curvature = np.polyfit(GRID, estimate_grid, 2)[0]
        true_curvature = np.polyfit(GRID, truth_grid, 2)[0]
        result.update(turning_point_abs_error=abs(est_turn - true_turn), peak_to_tail_recovery_ratio=contrast / true_contrast if abs(true_contrast) > 1e-12 else float("nan"), curvature_recovery_ratio=curvature / true_curvature if abs(true_curvature) > 1e-12 else float("nan"), inverted_u_classified=float(curvature < 0 and .20 <= est_turn <= .80 and contrast > 0))
    elif world in {"W4", "W5"}:
        derivative = np.gradient(estimate_grid, GRID)
        location = float(GRID[int(np.argmax(np.abs(derivative)))])
        true_location = .40
        result.update(boundary_abs_error=abs(location - true_location))
    return result


def fit_forest(x_train: np.ndarray, y_train: np.ndarray, t_train: np.ndarray, x_test: np.ndarray, seed: int, common: dict[str, Any]) -> tuple[Any, np.ndarray, np.ndarray]:
    estimator = v2._new_m1(common, int(seed))
    estimator.fit(y_train, t_train, X=x_train, W=None)
    return estimator, np.asarray(estimator.effect(x_train), float).reshape(-1), np.asarray(estimator.effect(x_test), float).reshape(-1)


def save_prediction_artifacts(data: dict[str, Any], train: np.ndarray, test: np.ndarray, forest_train: np.ndarray, forest_test: np.ndarray, selected_feature: str, u_all: np.ndarray, n: int, world: str, replication: int, seed: int) -> tuple[str, str]:
    key = [hashlib.sha256(str(value).encode("utf-8")).hexdigest() for value in data["unit_id"]]
    analyst = pd.DataFrame({"position": np.arange(len(key)), "unit_key_sha256": key, "sample": np.where(np.isin(np.arange(len(key)), test), "test", "train"), "treatment": data["T"], "outcome": data["Y"], "forest_cate_train": np.nan, "forest_cate_test": np.nan, "forest_selected_moderator": selected_feature, "selected_moderator_u": u_all})
    analyst.loc[train, "forest_cate_train"] = forest_train
    analyst.loc[test, "forest_cate_test"] = forest_test
    if ORACLE_FIELDS.intersection(analyst.columns):
        raise AssertionError("oracle field leaked into analyst artifact")
    oracle = pd.DataFrame({"position": np.arange(len(key)), "unit_key_sha256": key, "tau_true": data["tau_true"], "u_m1_true": data["u_m1_true"], "propensity_true": data["propensity_true"]})
    directory = PILOT_ROOT / "local_outputs" / "confirmatory" / f"n_{n}" / world
    analyst_path = directory / f"analyst_rep_{replication:03d}.parquet"
    oracle_path = directory / f"oracle_rep_{replication:03d}.parquet"
    atomic_write_parquet(analyst, analyst_path)
    atomic_write_parquet(oracle, oracle_path)
    return str(analyst_path), str(oracle_path)


def run_one(n: int, world: str, replication: int, config: dict[str, Any], source: dict[str, Any], backbone: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    started = time.perf_counter()
    lineage = seeds(int(config["fresh_confirmatory_master_seed"]), n, replication)
    data = v2.generate_world_dataset(backbone, world, n, lineage, source)
    train, test = v2.split_once(data["T"], data["unit_id"], float(config["test_size"]), lineage["split_seed"])
    u_matrix = feature_u_matrix(backbone, data["unit_id"], source["features"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator, forest_train, forest_test = fit_forest(data["X"][train], data["Y"][train], data["T"][train], data["X"][test], lineage["forest_seed"], source["common"])
    imp = importance(estimator, len(source["features"]))
    selected_index = int(np.argmax(imp))
    selected_feature = source["features"][selected_index]
    nuisance = crossfit_nuisance(data["X"][train], data["Y"][train], data["T"][train], data["X"][test], lineage["nuisance_seed"], int(config["second_stage"]["nuisance_crossfit_folds"]))
    nuisance["y_residual_test"] = data["Y"][test] - nuisance["m_test"]
    nuisance["t_residual_test"] = data["T"][test] - nuisance["e_test"]
    analyst_path, oracle_path = save_prediction_artifacts(data, train, test, forest_train, forest_test, selected_feature, u_matrix[:, selected_index], n, world, replication, lineage["artifact_seed"])
    rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    for mode, feature_index in [("forest_selected", selected_index), ("oracle_relative_firm_size_diagnostic", source["features"].index(TRUE_MODERATOR))]:
        feature = source["features"][feature_index]
        u_all = u_matrix[:, feature_index]
        u_train, u_test = u_all[train], u_all[test]
        theory = fit_theory_library(u_train, nuisance["y_residual_train"], nuisance["t_residual_train"], config, lineage["shape_cv_seed"] + feature_index)
        if theory["selected"] == "ABSTAIN_FLEXIBLE_STRUCTURE":
            theory_test = np.full(len(test), np.nan)
            theory_grid = np.full(len(GRID), np.nan)
        else:
            model = theory["models"][theory["selected"]]
            theory_test = predict_candidate(model, u_test)
            theory_grid = predict_candidate(model, GRID)
        raw_grid = train_forest_curve(u_train, forest_train)
        truth_grid = exact_truth_curve(world, GRID, data["u_m1_true"], source["dgp"]) if feature == TRUE_MODERATOR else conditional_truth_grid(u_test, data["tau_true"][test])
        raw_metrics = morphology_metrics(world, data["tau_true"][test], forest_test, truth_grid, raw_grid)
        theory_metrics = morphology_metrics(world, data["tau_true"][test], theory_test, truth_grid, theory_grid) if np.isfinite(theory_test).all() else {"cate_rmse": float("nan"), "cate_mae": float("nan"), "spearman": float("nan"), "sign_region_accuracy": float("nan")}
        losses_test = {family: r_loss(nuisance["y_residual_test"], nuisance["t_residual_test"], predict_candidate(model, u_test)) for family, model in theory["models"].items()}
        delta = losses_test["jump"] - losses_test["smooth_reversal"]
        best_smooth_loss = min(losses_test[family] for family in SIMPLE_FAMILIES if family != "jump")
        smooth_model = theory["models"]["smooth_reversal"]
        jump_model = theory["models"]["jump"]
        jump_coefficient = float(jump_model["coefficient"][1])
        true_jump = float(np.max(truth_grid) - np.min(truth_grid)) if world == "W5" else float("nan")
        base = {"status": "completed", "n": n, "world": world, "replication": replication, "moderator_mode": mode, "forest_selected_moderator": selected_feature, "route": "raw_forest", "route_moderator": feature, "selected_theory_family": theory["selected"], "selected_theory_params_json": json.dumps(theory["models"][theory["selected"]]["params"], sort_keys=True) if theory["selected"] != "ABSTAIN_FLEXIBLE_STRUCTURE" else "{}", "best_overall_family": theory["best_overall"], "best_simple_family": theory["best_simple"], "one_se_limit": theory["one_se_limit"], "forest_selected_true_moderator": selected_feature == TRUE_MODERATOR, "route_is_true_moderator": feature == TRUE_MODERATOR, "selected_moderator_importance": float(imp[selected_index]), "true_moderator_importance": float(imp[source["features"].index(TRUE_MODERATOR)]), "true_moderator_rank": int(np.sum(imp > imp[source["features"].index(TRUE_MODERATOR)]) + 1), "raw_forest_selected": selected_feature, "best_smooth_k": float(smooth_model["params"]["k"]), "best_smooth_c": float(smooth_model["params"]["c"]), "jump_comparator_c": float(jump_model["params"]["c"]), "jump_comparator_boundary_error": abs(float(jump_model["params"]["c"]) - .40), "jump_comparator_magnitude": jump_coefficient, "jump_comparator_recovery_ratio": jump_coefficient / true_jump if world == "W5" and abs(true_jump) > 1e-12 else float("nan"), "test_loss_jump": losses_test["jump"], "test_loss_smooth_reversal": losses_test["smooth_reversal"], "test_loss_jump_minus_smooth_reversal": delta, "test_loss_jump_minus_best_smooth": losses_test["jump"] - best_smooth_loss, "nuisance_train_residuals_oof": True, "nuisance_crossfit_folds": nuisance["nuisance_folds"], "test_nuisance_refit_scope": nuisance["test_refit_scope"], "test_used_for_selection": False, "external_split_count": 1, "train_n": len(train), "test_n": len(test), "train_index_sha256": sha256_arrays(train), "test_index_sha256": sha256_arrays(test), "dataset_sha256": sha256_arrays(data["X"], data["T"], data["Y"]), "analyst_artifact_path": analyst_path, "oracle_artifact_path": oracle_path, "warning_count": len(caught), "warning_messages": "\n".join(str(w.message) for w in caught), "runtime_seconds": 0.0, **lineage}
        rows.append({**base, **raw_metrics})
        rows.append({**base, "route": "theory_reconstruction", **theory_metrics, "selected_theory_family": theory["selected"]})
        for record in theory["records"]:
            selected_within = record["params"] == theory["models"][record["family"]]["params"]
            family_rows.append({"status": "completed", "n": n, "world": world, "replication": replication, "moderator_mode": mode, "family": record["family"], "params_json": json.dumps(record["params"], sort_keys=True), "cv_fold_losses_json": json.dumps(record["cv_fold_losses"]), "cv_mean_r_loss": record["cv_mean_r_loss"], "cv_se_r_loss": record["cv_se_r_loss"], "selected_within_family": selected_within, "selected_final": record["family"] == theory["selected"] and selected_within, "best_overall": record["family"] == theory["best_overall"] and selected_within, "abstained": theory["selected"] == "ABSTAIN_FLEXIBLE_STRUCTURE", **lineage})
        curve_path = PILOT_ROOT / "local_outputs" / "confirmatory" / f"n_{n}" / world / f"curves_{mode}_rep_{replication:03d}.parquet"
        curve_frame = pd.DataFrame({"u": GRID, "truth_curve": truth_grid, "raw_forest_curve": raw_grid, "theory_curve": theory_grid, "selected_family": theory["selected"], "moderator_mode": mode, "world": world, "n": n, "replication": replication})
        atomic_write_parquet(curve_frame, curve_path)
        for row in rows[-2:]:
            row["curve_artifact_path"] = str(curve_path)
    for row in rows:
        row["runtime_seconds"] = float(time.perf_counter() - started)
    return rows, family_rows


_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_SOURCE: dict[str, Any] | None = None
_WORKER_BACKBONE: pd.DataFrame | None = None


def worker_init(config_path: str) -> None:
    global _WORKER_CONFIG, _WORKER_SOURCE, _WORKER_BACKBONE
    _WORKER_CONFIG = load_json(Path(config_path))
    validate_config(_WORKER_CONFIG)
    _WORKER_SOURCE = load_source()
    _WORKER_BACKBONE = add_rank_columns(v2.load_frozen_backbone(PROJECT_ROOT, _WORKER_SOURCE, SOURCE_CONFIG), _WORKER_SOURCE["features"])


def worker(task: tuple[int, str, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    n, world, replication = task
    try:
        if _WORKER_CONFIG is None or _WORKER_SOURCE is None or _WORKER_BACKBONE is None:
            raise RuntimeError("worker not initialized")
        rows, families = run_one(n, world, replication, _WORKER_CONFIG, _WORKER_SOURCE, _WORKER_BACKBONE)
        return rows, families, None
    except Exception as error:
        return [], [], {"status": "failed", "n": n, "world": world, "replication": replication, "error_type": type(error).__name__, "error_message": str(error), "traceback": traceback.format_exc(), **seeds(int(_WORKER_CONFIG["fresh_confirmatory_master_seed"]) if _WORKER_CONFIG else 0, n, replication)}


def tasks(config: dict[str, Any]) -> list[tuple[int, str, int]]:
    return [(int(n), world, r) for n in config["sample_sizes"] for world in config["worlds"] for r in range(int(config["replications_per_cell"]))]


def run_confirmatory(config_path: Path) -> int:
    config = load_json(config_path.resolve())
    validate_config(config)
    source = load_source()
    backbone = add_rank_columns(v2.load_frozen_backbone(PROJECT_ROOT, source, SOURCE_CONFIG), source["features"])
    result_path = PILOT_ROOT / "results" / "MORPHOLOGY_V2_LONG.csv"
    family_path = PILOT_ROOT / "results" / "MORPHOLOGY_V2_FAMILY_SELECTION.csv"
    failure_path = PILOT_ROOT / "results" / "MORPHOLOGY_V2_FAILURES.csv"
    existing = pd.read_csv(result_path).to_dict("records") if result_path.is_file() else []
    existing_frame = pd.DataFrame(existing)
    if existing_frame.empty:
        existing_keys: set[tuple[int, str, int]] = set()
    else:
        counts = existing_frame.loc[existing_frame["status"] == "completed"].groupby(["n", "world", "replication"]).size()
        existing_keys = {(int(key[0]), str(key[1]), int(key[2])) for key, count in counts.items() if int(count) == 4}
    pending = [task for task in tasks(config) if task not in existing_keys]
    rows = existing
    family_rows = pd.read_csv(family_path).to_dict("records") if family_path.is_file() else []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1), initializer=worker_init, initargs=(str(config_path.resolve()),)) as executor:
            futures = {executor.submit(worker, task): task for task in pending}
            for i, future in enumerate(as_completed(futures), 1):
                task_rows, task_family, failure = future.result()
                rows.extend(task_rows)
                family_rows.extend(task_family)
                if failure is not None:
                    failures.append(failure)
                atomic_write_csv(pd.DataFrame(rows), result_path)
                atomic_write_csv(pd.DataFrame(family_rows), family_path)
                if i == 1 or i % 5 == 0 or i == len(pending):
                    print(f"PROGRESS {i}/{len(pending)} task={futures[future]} failed={failure is not None}", flush=True)
    if failures:
        atomic_write_csv(pd.DataFrame(failures), failure_path)
    elif not failure_path.is_file():
        atomic_write_csv(pd.DataFrame(columns=["status", "n", "world", "replication", "error_type", "error_message", "traceback"]), failure_path)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["n", "world", "replication", "moderator_mode", "route"])
        atomic_write_csv(frame, result_path)
    print(f"DONE confirmatory datasets={len(tasks(config))} result_rows={len(frame)} failures={len(failures)} wall={time.perf_counter()-started:.1f}s", flush=True)
    return 0 if len(failures) == 0 and len({(int(r["n"]), str(r["world"]), int(r["replication"])) for r in rows if r.get("status") == "completed"}) == len(tasks(config)) else 1


def summarize_results(config_path: Path) -> int:
    config = load_json(config_path.resolve())
    validate_config(config)
    frame = pd.read_csv(PILOT_ROOT / "results" / "MORPHOLOGY_V2_LONG.csv")
    completed = frame[frame["status"] == "completed"].copy()
    numeric = ["cate_rmse", "cate_mae", "spearman", "sign_region_accuracy", "max_overshoot_normalized", "zero_crossing_abs_error", "high_low_recovery_ratio", "turning_point_abs_error", "peak_to_tail_recovery_ratio", "curvature_recovery_ratio", "boundary_abs_error", "best_smooth_k", "best_smooth_c", "jump_comparator_c", "jump_comparator_boundary_error", "jump_comparator_magnitude", "jump_comparator_recovery_ratio", "test_loss_jump_minus_smooth_reversal", "test_loss_jump_minus_best_smooth", "runtime_seconds"]
    rows = []
    for key, group in completed.groupby(["n", "world", "moderator_mode", "route"], dropna=False):
        row = dict(zip(["n", "world", "moderator_mode", "route"], key))
        row["replications"] = int(group["replication"].nunique())
        for metric in numeric:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean()) if values.notna().any() else float("nan")
            row[f"{metric}_median"] = float(values.median()) if values.notna().any() else float("nan")
            row[f"{metric}_sd"] = float(values.std()) if values.notna().sum() > 1 else float("nan")
        if group["route"].iloc[0] == "theory_reconstruction":
            row["abstention_rate"] = float((group["selected_theory_family"] == "ABSTAIN_FLEXIBLE_STRUCTURE").mean())
            for family in FAMILIES:
                row[f"selected_{family}_rate"] = float((group["selected_theory_family"] == family).mean())
            row["moderator_top1_rate"] = float(group["forest_selected_true_moderator"].mean())
        rows.append(row)
    atomic_write_csv(pd.DataFrame(rows), PILOT_ROOT / "results" / "MORPHOLOGY_V2_SUMMARY.csv")
    return 0


def write_execution_audit(config_path: Path) -> int:
    config = load_json(config_path.resolve())
    validate_config(config)
    result_path = PILOT_ROOT / "results" / "MORPHOLOGY_V2_LONG.csv"
    family_path = PILOT_ROOT / "results" / "MORPHOLOGY_V2_FAMILY_SELECTION.csv"
    failure_path = PILOT_ROOT / "results" / "MORPHOLOGY_V2_FAILURES.csv"
    frame = pd.read_csv(result_path) if result_path.is_file() else pd.DataFrame()
    checks = {
        "expected_dataset_count": len({tuple(x) for x in frame[["n", "world", "replication"]].to_numpy()}) == 240 if not frame.empty else False,
        "expected_long_rows": len(frame) == 960,
        "no_duplicate_rows": not frame.duplicated(["n", "world", "replication", "moderator_mode", "route"]).any() if not frame.empty else False,
        "all_worlds_allowed": set(frame["world"]).issubset(WORLDS) if not frame.empty else False,
        "one_external_split": bool((frame["external_split_count"] == 1).all()) if not frame.empty else False,
        "train_test_sizes": bool(((frame["train_n"] == frame["n"] * .8) & (frame["test_n"] == frame["n"] * .2)).all()) if not frame.empty else False,
        "train_nuisance_oof": bool(frame["nuisance_train_residuals_oof"].all()) if not frame.empty else False,
        "no_test_selection": bool((~frame["test_used_for_selection"]).all()) if not frame.empty else False,
        "no_oracle_in_analyst": True,
        "families_present": family_path.is_file() and len(pd.read_csv(family_path)) > 0,
        "failures_explicit": failure_path.is_file(),
    }
    if frame.empty:
        checks["no_oracle_in_analyst"] = False
    else:
        for path in frame["analyst_artifact_path"].drop_duplicates():
            if ORACLE_FIELDS.intersection(pd.read_parquet(path).columns):
                checks["no_oracle_in_analyst"] = False
    report = {"artifact": "Morphology Reconstruction v2 execution audit", "created_utc": utc_now(), "checks": checks, "passed": all(checks.values()), "expected_datasets": 240, "result_rows": len(frame), "runner_sha256": sha256_file(Path(__file__).resolve()), "config_sha256": sha256_file(config_path.resolve()), "source_runner_sha256": sha256_file(SOURCE_RUNNER), "source_config_sha256": sha256_file(SOURCE_CONFIG), "backbone_sha256": sha256_file(v2.resolve_input_path(SOURCE_CONFIG, load_source()["backbone_path"]))}
    out = PILOT_ROOT / "results" / "MORPHOLOGY_V2_EXECUTION_AUDIT.json"
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report["passed"] else 1


def package_versions() -> dict[str, str]:
    names = ["econml", "scikit-learn", "numpy", "pandas", "scipy", "pyarrow"]
    return {name: importlib.metadata.version(name) for name in names}


def preflight_audit(config_path: Path) -> int:
    config_path = config_path.resolve()
    config = load_json(config_path)
    validate_config(config)
    source = load_source()
    smooth_grid = family_grid(config)["smooth_reversal"]
    jump_grid = family_grid(config)["jump"]
    result_path = PILOT_ROOT / "results" / "MORPHOLOGY_V2_LONG.csv"
    test_command = [sys.executable, "-m", "pytest", str(PILOT_ROOT / "tests"), "-q"]
    test_run = subprocess.run(test_command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    checks = {
        "fresh_results_absent_before_freeze": not result_path.exists(),
        "only_w2_w5": set(config["worlds"]) == WORLDS,
        "n_and_replication_contract": config["sample_sizes"] == [1500, 3000] and int(config["replications_per_cell"]) == 30,
        "primary_forest_frozen": config["primary_forest"] == {"class": "econml.dml.CausalForestDML", "cv": 2, "honest": True, "n_estimators": 1000, "min_samples_leaf": 5, "max_samples": 0.45},
        "twelve_joint_features": len(source["features"]) == len(set(source["features"])) == 12,
        "w4_true_k_supported": float(source["dgp"]["W4"]["steepness"]) == 25.0 and any(float(item["k"]) == 25.0 for item in smooth_grid),
        "w4_true_center_supported": any(np.isclose(float(item["c"]), float(source["dgp"]["W4"]["center"])) for item in smooth_grid),
        "w5_true_boundary_supported": any(np.isclose(float(item["c"]), float(source["dgp"]["W5"]["boundary"])) for item in jump_grid),
        "automated_acceptance_tests_passed": test_run.returncode == 0,
        "analyst_x_has_no_oracle_fields": not bool(ORACLE_FIELDS.intersection(source["features"])),
        "one_se_rule_enabled": config["second_stage"]["one_se_abstention_rule"] is True,
    }
    report = {"artifact": "Morphology Reconstruction v2 pre-confirmation acceptance audit", "created_utc": utc_now(), "checks": checks, "passed": all(checks.values()), "test_command": " ".join(test_command), "test_exit_code": test_run.returncode, "test_stdout": test_run.stdout, "test_stderr": test_run.stderr, "runner_sha256": sha256_file(Path(__file__).resolve()), "config_sha256": sha256_file(config_path), "acceptance_test_sha256": sha256_file(PILOT_ROOT / "tests" / "test_morphology_v2_contract.py"), "dgp_support_audit_sha256": sha256_file(PILOT_ROOT / "reports" / "MORPHOLOGY_V2_DGP_AND_FAMILY_SUPPORT_AUDIT.md")}
    out = PILOT_ROOT / "results" / "MORPHOLOGY_V2_PRECONFIRMATION_AUDIT.json"
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report["passed"] else 1


def create_method_freeze(config_path: Path) -> int:
    config = load_json(config_path.resolve())
    validate_config(config)
    source = load_source()
    freeze = {
        "artifact": "MORPHOLOGY_V2_METHOD_FREEZE",
        "runner_path": str(Path(__file__).resolve()), "runner_sha256": sha256_file(Path(__file__).resolve()),
        "config_path": str(config_path.resolve()), "config_sha256": sha256_file(config_path.resolve()),
        "source_runner_path": str(SOURCE_RUNNER), "source_runner_sha256": sha256_file(SOURCE_RUNNER),
        "source_config_path": str(SOURCE_CONFIG), "source_config_sha256": sha256_file(SOURCE_CONFIG),
        "acceptance_tests_path": str(PILOT_ROOT / "tests" / "test_morphology_v2_contract.py"), "acceptance_tests_sha256": sha256_file(PILOT_ROOT / "tests" / "test_morphology_v2_contract.py"),
        "preconfirmation_audit_path": str(PILOT_ROOT / "results" / "MORPHOLOGY_V2_PRECONFIRMATION_AUDIT.json"), "preconfirmation_audit_sha256": sha256_file(PILOT_ROOT / "results" / "MORPHOLOGY_V2_PRECONFIRMATION_AUDIT.json"),
        "dgp_support_audit_path": str(PILOT_ROOT / "reports" / "MORPHOLOGY_V2_DGP_AND_FAMILY_SUPPORT_AUDIT.md"), "dgp_support_audit_sha256": sha256_file(PILOT_ROOT / "reports" / "MORPHOLOGY_V2_DGP_AND_FAMILY_SUPPORT_AUDIT.md"),
        "backbone_path": str(v2.resolve_input_path(SOURCE_CONFIG, source["backbone_path"])), "backbone_sha256": sha256_file(v2.resolve_input_path(SOURCE_CONFIG, source["backbone_path"])),
        "feature_role_mapping_path": str(v2.resolve_input_path(SOURCE_CONFIG, source["feature_role_mapping_path"])), "feature_role_mapping_sha256": sha256_file(v2.resolve_input_path(SOURCE_CONFIG, source["feature_role_mapping_path"])),
        "python": sys.version, "platform": platform.platform(), "packages": package_versions(),
        "primary_forest": config["primary_forest"], "candidate_features": source["features"], "true_moderator_diagnostic": TRUE_MODERATOR,
        "nuisance_definitions": {"m": "HistGradientBoostingRegressor", "e": "HistGradientBoostingClassifier", "train_crossfit_folds": 5, "test_predictions": "models refit on full TRAIN only", "pseudo_outcome": "none"},
        "family_grids": config["families"], "shape_cv_folds": config["second_stage"]["shape_cv_folds"], "one_se_rule": "best simple family CV loss <= best overall CV loss + best overall fold-loss SE; otherwise ABSTAIN_FLEXIBLE_STRUCTURE",
        "metric_definitions": {"r_loss": "mean((Y_residual - T_residual * tau_hat)^2)", "oracle_accuracy": "untouched TEST only", "moderator_selection": "TRAIN forest feature importance Top-1"},
        "seed_schedule": {"master_seed": config["fresh_confirmatory_master_seed"], "lineage": "SeedSequence(master, n, replication) -> backbone, treatment, outcome, split, forest, nuisance, shape_cv, artifact"},
        "prohibited": ["W0", "W1", "W6", "W7", "W8", "outer OOF", "BLP/GATE", "discontinuity p-value gate", "tuned forest", "leaf-size sensitivity"],
    }
    canonical = json.dumps(freeze, sort_keys=True, separators=(",", ":")).encode("utf-8")
    freeze["freeze_sha256"] = hashlib.sha256(canonical).hexdigest()
    freeze["created_utc"] = utc_now()
    out = PILOT_ROOT / "results" / "MORPHOLOGY_V2_METHOD_FREEZE.json"
    if out.exists():
        old = load_json(out)
        if old.get("freeze_sha256") != freeze["freeze_sha256"]:
            raise RuntimeError("existing method freeze differs; do not overwrite")
    else:
        out.write_text(json.dumps(freeze, indent=2, allow_nan=False), encoding="utf-8")
    print(f"METHOD_FREEZE_SHA256 {freeze['freeze_sha256']}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["preflight", "freeze", "run", "summarize", "audit"]:
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "preflight":
        return preflight_audit(args.config)
    if args.command == "freeze":
        return create_method_freeze(args.config)
    if args.command == "run":
        return run_confirmatory(args.config)
    if args.command == "summarize":
        return summarize_results(args.config)
    return write_execution_audit(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
