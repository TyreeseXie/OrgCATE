"""Focused kink-resolution v2 experiment.

Product v1 is immutable. This runner changes only the final kink-versus-smooth
readout: nested TRAIN cross-fitting localizes a stable interior hinge and tests
its out-of-fold improvement over a single line. Untouched TEST confirms the
TRAIN-selected nonconstant hinge against a constant effect. Oracle truth is
evaluation-only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
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
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
PRODUCT_PATH = ROOT / "execution" / "run_morphology_product_v1.py"
DEFAULT_CONFIG = ROOT / "configs" / "kink_resolution_v2.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PRODUCT = load_module(PRODUCT_PATH, "kink_resolution_product_v1")
HELPER = PRODUCT.HELPER
V2 = PRODUCT.V2
BASE = PRODUCT.BASE
TARGET_DGPS = ("kinked", "nonlinear-monotone", "steep-continuous")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def inclusive_grid(spec: dict[str, float]) -> list[float]:
    start, stop, step = (float(spec[key]) for key in ("minimum", "maximum", "step"))
    count = round((stop - start) / step)
    return [round(start + index * step, 12) for index in range(count + 1)]


def seed_grid(master_seed: int, replications: int) -> dict[int, int]:
    children = np.random.SeedSequence(int(master_seed)).spawn(int(replications))
    return {
        replication: int(child.generate_state(1, dtype=np.uint32)[0])
        for replication, child in enumerate(children)
    }


def validate_config(config: dict[str, Any]) -> None:
    if config["schema_version"] != "kink-resolution-v2":
        raise ValueError("wrong schema")
    if int(config["n"]) != 3000 or not np.isclose(float(config["test_size"]), .20):
        raise ValueError("n and external split changed")
    if config["development"]["dgps"] != list(TARGET_DGPS):
        raise ValueError("development controls changed")
    if config["confirmation"]["dgps"] != list(TARGET_DGPS):
        raise ValueError("confirmation controls changed")
    if int(config["development"]["replications"]) != 5:
        raise ValueError("development must contain 15 cells")
    if int(config["confirmation"]["replications"]) != 10:
        raise ValueError("confirmation must contain 30 cells")
    if int(config["same_data_nuisance_refits"]) != 3:
        raise ValueError("nuisance refit contract changed")
    if int(config["nested_kink_evidence"]["outer_folds"]) != 5:
        raise ValueError("outer kink folds changed")
    if int(config["nested_kink_evidence"]["inner_folds"]) != 3:
        raise ValueError("inner kink folds changed")
    if config["selection_policy"] != {
        "test_used_for_kink_selection": False,
        "oracle_used_for_kink_selection": False,
        "product_v1_results_used_only_as_historical_diagnosis": True,
    }:
        raise ValueError("selection policy changed")


def fit_family(
    family: str,
    u: np.ndarray,
    y_residual: np.ndarray,
    t_residual: np.ndarray,
    center: float | None = None,
) -> dict[str, Any]:
    params = {} if center is None else {"c": float(center)}
    return HELPER.fit_model(family, u, y_residual, t_residual, params)


def select_center_inner_cv(
    u: np.ndarray,
    y_residual: np.ndarray,
    t_residual: np.ndarray,
    centers: list[float],
    folds: int,
    seed: int,
) -> float:
    splitter = KFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    candidates = []
    for center in centers:
        losses = []
        for fit_index, validation_index in splitter.split(u):
            model = fit_family(
                "kink", u[fit_index], y_residual[fit_index],
                t_residual[fit_index], center,
            )
            prediction = HELPER.predict_model(model, u[validation_index])
            losses.append(float(np.mean(HELPER.observation_loss(
                y_residual[validation_index], t_residual[validation_index], prediction
            ))))
        candidates.append((float(np.mean(losses)), float(center)))
    return min(candidates, key=lambda item: (item[0], item[1]))[1]


def nested_kink_evidence(
    u: np.ndarray,
    y_residual: np.ndarray,
    t_residual: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    specification = config["nested_kink_evidence"]
    centers = inclusive_grid(specification["candidate_center_grid"])
    outer = KFold(
        n_splits=int(specification["outer_folds"]),
        shuffle=True,
        random_state=int(seed),
    )
    loss_differences = np.full(len(u), np.nan)
    records = []
    for fold, (fit_index, validation_index) in enumerate(outer.split(u)):
        center = select_center_inner_cv(
            u[fit_index], y_residual[fit_index], t_residual[fit_index], centers,
            int(specification["inner_folds"]), int(seed) + 1009 * (fold + 1),
        )
        kink = fit_family(
            "kink", u[fit_index], y_residual[fit_index], t_residual[fit_index], center
        )
        linear = fit_family(
            "linear", u[fit_index], y_residual[fit_index], t_residual[fit_index]
        )
        kink_prediction = HELPER.predict_model(kink, u[validation_index])
        linear_prediction = HELPER.predict_model(linear, u[validation_index])
        kink_loss = HELPER.observation_loss(
            y_residual[validation_index], t_residual[validation_index], kink_prediction
        )
        linear_loss = HELPER.observation_loss(
            y_residual[validation_index], t_residual[validation_index], linear_prediction
        )
        loss_differences[validation_index] = kink_loss - linear_loss
        coefficient = np.asarray(kink["coefficient"], dtype=float)
        left_slope = float(coefficient[1])
        right_slope = float(coefficient[1] + coefficient[2])
        minimum = min(abs(left_slope), abs(right_slope))
        ratio = max(abs(left_slope), abs(right_slope)) / max(minimum, 1e-10)
        records.append({
            "fold": fold,
            "center": center,
            "left_slope": left_slope,
            "right_slope": right_slope,
            "slope_change": right_slope - left_slope,
            "same_sign": bool(left_slope * right_slope > 0),
            "slope_ratio": float(ratio),
            "validation_loss_difference": float(np.mean(kink_loss - linear_loss)),
        })
    if not np.isfinite(loss_differences).all():
        raise RuntimeError("nested kink OOF losses incomplete")
    mean = float(np.mean(loss_differences))
    standard_error = float(np.std(loss_differences, ddof=1) / math.sqrt(len(loss_differences)))
    z = float(specification["one_sided_z"])
    center_values = np.asarray([item["center"] for item in records], dtype=float)
    lower, upper = (float(item) for item in specification["interior_support"])
    return {
        "loss_difference": mean,
        "loss_difference_se": standard_error,
        "loss_difference_upper": mean + z * standard_error,
        "center_median": float(np.median(center_values)),
        "center_iqr": float(np.quantile(center_values, .75) - np.quantile(center_values, .25)),
        "center_range": float(np.ptp(center_values)),
        "interior_center_share": float(np.mean((center_values >= lower) & (center_values <= upper))),
        "same_sign_rate": float(np.mean([item["same_sign"] for item in records])),
        "median_slope_ratio": float(np.median([item["slope_ratio"] for item in records])),
        "median_slope_change": float(np.median([item["slope_change"] for item in records])),
        "records": records,
    }


def combine_evidence(items: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    rule = config["kink_rule"]
    centers = np.asarray([item["center_median"] for item in items], dtype=float)
    loss_pass_rate = float(np.mean([item["loss_difference_upper"] < 0 for item in items]))
    interior_share = float(np.mean([item["interior_center_share"] for item in items]))
    same_sign_rate = float(np.mean([item["same_sign_rate"] for item in items]))
    slope_ratio = float(np.median([item["median_slope_ratio"] for item in items]))
    slope_change = float(np.median([item["median_slope_change"] for item in items]))
    center = float(np.median(centers))
    center_iqr = float(np.quantile(centers, .75) - np.quantile(centers, .25))
    center_range = float(np.ptp(centers))
    lower, upper = (float(value) for value in rule["interior_support"])
    checks = {
        "loss_pass": loss_pass_rate >= float(rule["minimum_refit_loss_pass_rate"]),
        "interior_location": lower <= center <= upper,
        "interior_stability": interior_share >= float(rule["minimum_interior_center_share"]),
        "center_iqr": center_iqr <= float(rule["maximum_center_iqr"]),
        "same_sign": same_sign_rate >= float(rule["minimum_same_sign_rate"]),
        "slope_ratio": slope_ratio >= float(rule["minimum_slope_ratio"]),
        "slope_change": abs(slope_change) >= float(rule["minimum_absolute_slope_change"]),
    }
    return {
        "kink_supported_train": bool(all(checks.values())),
        "checks": checks,
        "loss_pass_rate": loss_pass_rate,
        "center_median": center,
        "center_iqr": center_iqr,
        "center_range": center_range,
        "interior_center_share": interior_share,
        "same_sign_rate": same_sign_rate,
        "median_slope_ratio": slope_ratio,
        "median_slope_change": slope_change,
        "refit_loss_upper_median": float(np.median([item["loss_difference_upper"] for item in items])),
        "refits": items,
    }


def run_cell(
    backbone: pd.DataFrame,
    source: dict[str, Any],
    base_parameters: dict[str, Any],
    product_config: dict[str, Any],
    config: dict[str, Any],
    phase: str,
    dgp: str,
    replication: int,
    master_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    lineage = BASE.lineage(master_seed, 0)
    data = PRODUCT.build_data(
        backbone, source, product_config, dgp, lineage, base_parameters
    )
    train, test = V2.v2.split_once(
        data["T"], data["unit_id"], float(config["test_size"]), lineage["split_seed"]
    )
    u = BASE.reference_u(backbone, data["unit_id"], config["focal_characteristic"])
    common = dict(source["common"])
    common["n_jobs"] = int(product_config["primary_forest"]["n_jobs"])
    estimator = V2.v2._new_m1(common, lineage["forest_seed"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(data["Y"][train], data["T"][train], X=data["X"][train], W=None)
    importance = np.asarray(estimator.feature_importances_, dtype=float)
    focal_index = source["features"].index(config["focal_characteristic"])
    focal_rank = int(np.sum(importance > importance[focal_index]) + 1)
    libraries = []
    nuisances = []
    evidence = []
    for refit in range(int(config["same_data_nuisance_refits"])):
        refit_seeds = HELPER.refit_seeds(master_seed, refit)
        nuisance = V2.crossfit_nuisance(
            data["X"][train], data["Y"][train], data["T"][train], data["X"][test],
            refit_seeds["nuisance_refit_seed"], int(config["nuisance_crossfit_folds"]),
        )
        library_config = copy.deepcopy(product_config)
        library_config["selection"] = {
            "structured_families": ["constant", "linear", "quadratic", "tanh", "kink", "step"],
            "flexible_family": "spline",
        }
        library = HELPER.fit_library(
            u[train], nuisance["y_residual_train"], nuisance["t_residual_train"],
            library_config, refit_seeds["shape_cv_refit_seed"],
        )
        evidence.append(nested_kink_evidence(
            u[train], nuisance["y_residual_train"], nuisance["t_residual_train"],
            config, refit_seeds["shape_cv_refit_seed"] + 7001,
        ))
        libraries.append(library)
        nuisances.append(nuisance)
    combined = combine_evidence(evidence, config)
    confirmations = []
    for library, nuisance in zip(libraries, nuisances, strict=True):
        y_test = data["Y"][test] - nuisance["m_test"]
        t_test = data["T"][test] - nuisance["e_test"]
        confirmation = PRODUCT.confirm_candidate(
            library["models"]["kink"], library["models"]["constant"],
            u[test], y_test, t_test, float(config["test_confirmation"]["one_sided_alpha"]),
        )
        confirmations.append(bool(confirmation["test_confirmed_vs_constant"]))
    confirmation_rate = float(np.mean(confirmations))
    kink_supported = (
        combined["kink_supported_train"]
        and confirmation_rate >= float(config["test_confirmation"]["minimum_confirmation_rate"])
    )
    predicted = "kinked" if kink_supported else "kink-not-supported"
    state = "SUPPORTED_CANDIDATE" if kink_supported else "INCONCLUSIVE"
    models = [library["models"]["kink"] for library in libraries]
    grid = np.linspace(.10, .90, 161)
    estimates = [HELPER.predict_model(model, grid) for model in models]
    consensus = np.median(estimates, axis=0)
    truth = PRODUCT.truth_curve(dgp, grid, product_config)
    details = []
    for model in models:
        coefficient = np.asarray(model["coefficient"], dtype=float)
        details.append({
            "center": float(model["params"]["c"]),
            "left_slope": float(coefficient[1]),
            "right_slope": float(coefficient[1] + coefficient[2]),
            "slope_change": float(coefficient[2]),
        })
    true_is_kink = dgp == "kinked"
    result = {
        "status": "completed",
        "phase": phase,
        "dgp": dgp,
        "replication": replication,
        "master_seed": master_seed,
        "n": int(config["n"]),
        "predicted_kink_state": predicted,
        "practical_state": state,
        "true_is_kink_evaluator": true_is_kink,
        "classification_correct_evaluator": bool(kink_supported == true_is_kink),
        "train_kink_supported": combined["kink_supported_train"],
        "test_confirmation_rate": confirmation_rate,
        "candidate_center": combined["center_median"],
        "candidate_center_iqr": combined["center_iqr"],
        "candidate_center_range": combined["center_range"],
        "center_abs_error_evaluator": abs(combined["center_median"] - .50) if true_is_kink else float("nan"),
        "interior_center_share": combined["interior_center_share"],
        "same_sign_rate": combined["same_sign_rate"],
        "median_slope_ratio": combined["median_slope_ratio"],
        "median_slope_change": combined["median_slope_change"],
        "nested_loss_pass_rate": combined["loss_pass_rate"],
        "nested_loss_upper_median": combined["refit_loss_upper_median"],
        "check_json": json.dumps(combined["checks"], sort_keys=True),
        "parameter_details_json": json.dumps(details, sort_keys=True),
        "focal_characteristic_rank": focal_rank,
        "focal_characteristic_importance": float(importance[focal_index]),
        "curve_spearman_evaluator": float(spearmanr(truth, consensus).statistic),
        "curve_rmse_evaluator": float(np.sqrt(np.mean((truth - consensus) ** 2))),
        "warning_count": len(caught),
        "runtime_seconds": float(time.perf_counter() - started),
        "test_used_for_kink_selection": False,
        "oracle_used_for_kink_selection": False,
        "one_joint_multivariable_forest": True,
        "error": "",
    }
    curve_rows = [
        {
            "phase": phase, "dgp": dgp, "replication": replication,
            "master_seed": master_seed, "u": float(value),
            "truth_curve": float(truth_value), "hinge_curve": float(estimate),
        }
        for value, truth_value, estimate in zip(grid, truth, consensus, strict=True)
    ]
    return result, curve_rows


def paths(config_path: Path, config: dict[str, Any], phase: str) -> dict[str, Path]:
    output = (config_path.parent / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = phase.upper()
    return {
        "output": output,
        "results": output / f"KINK_RESOLUTION_V2_{prefix}_LONG.csv",
        "curves": output / f"KINK_RESOLUTION_V2_{prefix}_CURVES.csv",
    }


def append(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def load_inputs(config_path: Path, config: dict[str, Any]):
    product_path = (config_path.parent / config["product_config"]).resolve()
    product_config = json.loads(product_path.read_text(encoding="utf-8"))
    PRODUCT.validate_config(product_config)
    source_path = (product_path.parent / product_config["source_config"]).resolve()
    source = V2.load_json(source_path)
    base_config = json.loads((product_path.parent / product_config["base_config"]).resolve().read_text(encoding="utf-8"))
    backbone = V2.add_rank_columns(
        V2.v2.load_frozen_backbone(ROOT, source, source_path), source["features"]
    )
    return product_path, product_config, source_path, source, base_config, backbone


def run_phase(config_path: Path, config: dict[str, Any], phase: str) -> None:
    output = paths(config_path, config, phase)
    phase_config = config[phase]
    seeds = seed_grid(int(phase_config["master_seed"]), int(phase_config["replications"]))
    completed = set()
    if output["results"].exists():
        prior = pd.read_csv(output["results"])
        completed = set(zip(prior["dgp"], prior["replication"].astype(int), strict=False))
    _, product_config, _, source, base_config, backbone = load_inputs(config_path, config)
    total = len(phase_config["dgps"]) * int(phase_config["replications"])
    done = len(completed)
    for replication in range(int(phase_config["replications"])):
        for dgp in phase_config["dgps"]:
            if (dgp, replication) in completed:
                continue
            print(f"{phase.upper()} {done + 1}/{total} dgp={dgp} replication={replication}", flush=True)
            try:
                result, curves = run_cell(
                    backbone, source, base_config["shape_parameters"], product_config,
                    config, phase, dgp, replication, seeds[replication],
                )
            except Exception as error:  # noqa: BLE001 - failures must be retained
                result = {
                    "status": "failed", "phase": phase, "dgp": dgp,
                    "replication": replication, "master_seed": seeds[replication],
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
                curves = []
            append(output["results"], [result])
            if curves:
                append(output["curves"], curves)
            done += 1


def create_freeze(config_path: Path, config: dict[str, Any]) -> Path:
    validate_config(config)
    development = paths(config_path, config, "development")["results"]
    if not development.exists() or len(pd.read_csv(development)) != 15:
        raise RuntimeError("15-cell development batch incomplete")
    product_path, _, source_path, _, _, _ = load_inputs(config_path, config)
    freeze_path = (config_path.parent / config["freeze_path"]).resolve()
    payload = {
        "status": "KINK_V2_FROZEN_BEFORE_FRESH_CONFIRMATION",
        "frozen_at": "2026-10-22",
        "development_cells": 15,
        "confirmation_seed_grid": seed_grid(
            int(config["confirmation"]["master_seed"]), int(config["confirmation"]["replications"])
        ),
        "hashes": {
            "config_sha256": sha256_file(config_path),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "product_config_sha256": sha256_file(product_path),
            "product_runner_sha256": sha256_file(PRODUCT_PATH),
            "source_config_sha256": sha256_file(source_path),
            "development_results_sha256": sha256_file(development),
        },
    }
    freeze_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return freeze_path


def verify_freeze(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    freeze_path = (config_path.parent / config["freeze_path"]).resolve()
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    product_path, _, source_path, _, _, _ = load_inputs(config_path, config)
    development = paths(config_path, config, "development")["results"]
    current = {
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "product_config_sha256": sha256_file(product_path),
        "product_runner_sha256": sha256_file(PRODUCT_PATH),
        "source_config_sha256": sha256_file(source_path),
        "development_results_sha256": sha256_file(development),
    }
    if current != payload["hashes"]:
        raise RuntimeError("kink v2 freeze mismatch")
    return payload


def summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    completed = results[results["status"] == "completed"]
    for dgp, group in completed.groupby("dgp", sort=False):
        rows.append({
            "dgp": dgp,
            "replications": len(group),
            "classification_accuracy": float(group["classification_correct_evaluator"].mean()),
            "kink_call_rate": float((group["predicted_kink_state"] == "kinked").mean()),
            "median_candidate_center": float(group["candidate_center"].median()),
            "median_center_abs_error": float(group["center_abs_error_evaluator"].median()),
            "median_slope_ratio": float(group["median_slope_ratio"].median()),
            "median_slope_change": float(group["median_slope_change"].median()),
            "median_curve_spearman": float(group["curve_spearman_evaluator"].median()),
            "focal_top1_rate": float((group["focal_characteristic_rank"] == 1).mean()),
        })
    return pd.DataFrame(rows)


def render_svg(output: Path, results: pd.DataFrame, curves: pd.DataFrame) -> None:
    width, height = 1280, 520
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#20252b}.title{font-size:25px;font-weight:700}.sub{font-size:13px;fill:#5c6670}.panel{fill:#fff;stroke:#d6dce1}.truth{fill:none;stroke:#222;stroke-width:2;stroke-dasharray:6 4}.fit{fill:none;stroke:#238b57;stroke-width:2.5}.point{fill:#247b7b;fill-opacity:.75}</style>',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="35" y="38" class="title">Focused kink-resolution v2</text>',
        '<text x="35" y="62" class="sub">black dashed = simulation truth; green = median TRAIN hinge; points = nested-fold candidate centers</text>',
    ]
    for index, dgp in enumerate(TARGET_DGPS):
        x0 = 30 + index * 415
        y0, plot_x, plot_y, plot_w, plot_h = 85, x0 + 45, 135, 330, 240
        frame = curves[curves["dgp"] == dgp]
        grouped = frame.groupby("u", as_index=False).agg(
            truth=("truth_curve", "mean"), fit=("hinge_curve", "median")
        )
        low = float(grouped[["truth", "fit"]].min().min())
        high = float(grouped[["truth", "fit"]].max().max())
        margin = max(.05, .08 * (high - low))
        low, high = low - margin, high + margin
        truth_points = " ".join(
            f"{plot_x + (float(u) - .1) / .8 * plot_w:.1f},"
            f"{plot_y + plot_h - (float(y) - low) / (high - low) * plot_h:.1f}"
            for u, y in zip(grouped["u"], grouped["truth"], strict=True)
        )
        fit_points = " ".join(
            f"{plot_x + (float(u) - .1) / .8 * plot_w:.1f},"
            f"{plot_y + plot_h - (float(y) - low) / (high - low) * plot_h:.1f}"
            for u, y in zip(grouped["u"], grouped["fit"], strict=True)
        )
        lines.extend([
            f'<rect x="{x0}" y="{y0}" width="395" height="370" class="panel"/>',
            f'<text x="{x0 + 15}" y="{y0 + 26}" font-size="15" font-weight="700">{dgp}</text>',
            f'<polyline points="{truth_points}" class="truth"/>',
            f'<polyline points="{fit_points}" class="fit"/>',
        ])
        cell = results[results["dgp"] == dgp]
        for offset, center in enumerate(cell["candidate_center"].to_numpy(float)):
            cx = plot_x + (center - .1) / .8 * plot_w
            lines.append(f'<circle cx="{cx:.1f}" cy="{plot_y + plot_h + 28 + (offset % 2) * 7}" r="4" class="point"/>')
        rate = float((cell["predicted_kink_state"] == "kinked").mean())
        lines.append(f'<text x="{x0 + 15}" y="{y0 + 345}" class="sub">kink call rate: {rate:.0%}</text>')
    lines.append("</svg>")
    (output / "KINK_RESOLUTION_V2_FIGURE.svg").write_text("\n".join(lines), encoding="utf-8")


def write_report(config_path: Path, config: dict[str, Any]) -> None:
    output_paths = paths(config_path, config, "confirmation")
    results = pd.read_csv(output_paths["results"])
    curves = pd.read_csv(output_paths["curves"])
    table = summary(results)
    table.to_csv(output_paths["output"] / "KINK_RESOLUTION_V2_SUMMARY.csv", index=False)
    render_svg(output_paths["output"], results, curves)
    kink = table[table["dgp"] == "kinked"].iloc[0]
    controls = table[table["dgp"] != "kinked"]
    test_selection_flags = results["test_used_for_kink_selection"].map(
        lambda value: str(value).strip().lower() == "true"
    )
    oracle_selection_flags = results["oracle_used_for_kink_selection"].map(
        lambda value: str(value).strip().lower() == "true"
    )
    acceptance = {
        "kink_at_least_9_of_10": bool(kink["kink_call_rate"] >= .90),
        "each_smooth_control_false_kink_at_most_1_of_10": bool((controls["kink_call_rate"] <= .10).all()),
        "kink_median_location_error_at_most_0_05": bool(kink["median_center_abs_error"] <= .05),
        "failures_zero": bool((results["status"] == "failed").sum() == 0),
        "leakage_flags_zero": bool(
            (~test_selection_flags).all() and (~oracle_selection_flags).all()
        ),
    }
    rows = [
        f"| {row.dgp} | {row.kink_call_rate:.0%} | {row.classification_accuracy:.0%} | {row.median_candidate_center:.3f} | {row.median_center_abs_error:.3f} | {row.median_slope_ratio:.2f} |"
        for row in table.itertuples(index=False)
    ]
    report = f"""# 连续折点局部确认 v2

## 为什么只跑这一项

Morphology Product v1 的其余情境已经达到预设门槛，本轮不重复运行。这里只修复最后一步的连续折点判别，并保留普通平滑非线性与剧烈连续两个最接近的负对照，防止用放宽阈值制造假成功。

## 方法变化

旧规则要求 hinge 在 TRAIN 预测损失上显著优于 tanh。有限样本中两者可以互相近似，因此即使折点位置和左右斜率已经恢复，检验仍然低功效。v2 在 TRAIN 内进行五折 outer/三折 inner 的嵌套判断：inner folds 定位候选折点，outer fold 检查冻结 hinge 相对单一直线的正交损失改善；同时要求候选位置在分布内部、跨 folds/refits 稳定，且左右斜率同方向但幅度发生实质变化。untouched TEST 仍只确认 TRAIN 选出的非恒定候选，不参与定位或阈值选择。

## Fresh confirmation

| 情境 | kink调用率 | kink/非kink分类正确率 | 候选位置中位数 | 折点位置误差中位数 | 左右斜率比中位数 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

![折点局部确认结果](KINK_RESOLUTION_V2_FIGURE.svg)

图中黑虚线是 evaluator-only 真值，绿色是每个 fresh replication 的 TRAIN hinge 重构中位曲线，横轴下方的点是候选折点位置。真正折点应在内部同一位置集中；平滑负对照的 hinge 近似通常退到支持边缘或跨 folds 不稳定。

## Acceptance

```json
{json.dumps(acceptance, indent=2)}
```

本轮只证明当前预定义折点情境在该 `n=3000` operating region 下能否稳定恢复，不把这一结果外推为所有未知折点或所有样本量的普遍保证。
"""
    (output_paths["output"] / "KINK_RESOLUTION_V2_REPORT_ZH.md").write_text(report, encoding="utf-8")
    freeze = verify_freeze(config_path, config)
    manifest = {
        "status": "completed",
        "scope": "focused kink resolution only; Product v1 remains immutable",
        "cells_expected": 30,
        "cells_completed": int((results["status"] == "completed").sum()),
        "cells_failed": int((results["status"] == "failed").sum()),
        "acceptance": acceptance,
        "freeze": freeze,
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "package_versions": V2.package_versions(),
    }
    (output_paths["output"] / "KINK_RESOLUTION_V2_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


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
