"""Same-data nine-shape morphology selection and held-out confirmation pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "execution" / "run_nine_shape_v2_unified_pilot_v1.py"
DEFAULT_CONFIG = ROOT / "configs" / "nine_shape_same_data_hierarchical_pilot_v1.json"
SHAPES = (
    "homogeneous", "linear-monotone", "nonlinear-monotone", "steep-continuous",
    "inverted-u", "u-shaped", "kinked", "threshold-step", "sign-reversal",
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_PATH, "same_data_hierarchical_base")
V2 = BASE.V2


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


def inclusive_grid(specification: dict[str, float]) -> list[float]:
    start = float(specification["minimum"])
    stop = float(specification["maximum"])
    step = float(specification["step"])
    count = round((stop - start) / step)
    return [round(start + step * index, 12) for index in range(count + 1)]


def validate_config(config: dict[str, Any]) -> None:
    if config["schema_version"] != "nine-shape-same-data-hierarchical-pilot-v1":
        raise ValueError("wrong config schema")
    if int(config["n_per_shape"]) != 1500 or int(config["same_data_refits"]) != 3:
        raise ValueError("pilot must remain n=1500 with three same-data refits")
    if not np.isclose(float(config["test_size"]), 0.20):
        raise ValueError("external split changed")
    expected = {
        "class": "econml.dml.CausalForestDML", "n_estimators": 1000,
        "min_samples_leaf": 5, "max_samples": 0.45, "cv": 2,
        "honest": True, "inference": False, "n_jobs": 2,
    }
    if config["primary_forest"] != expected:
        raise ValueError("primary forest contract changed")
    if int(config["nuisance_crossfit_folds"]) != 5 or int(config["shape_cv_folds"]) != 5:
        raise ValueError("cross-fitting contract changed")
    if float(config["truth_contract_correction"]["steep-continuous"]["steepness"]) != 25.0:
        raise ValueError("frozen W4 k=25 was not restored")
    if config["selection"]["structured_families"] != [
        "constant", "linear", "quadratic", "tanh", "kink", "step"
    ]:
        raise ValueError("structured family library changed")
    if config["test_used_for_family_selection"] is not False:
        raise ValueError("TEST cannot select a family")
    if config["oracle_used_only_for_evaluation"] is not True:
        raise ValueError("oracle separation changed")


def family_grid(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grid = config["family_grid"]
    return {
        "constant": [{}],
        "linear": [{}],
        "quadratic": [{"c": c} for c in inclusive_grid(grid["quadratic_center"])],
        "tanh": [
            {"k": float(k), "c": c}
            for k in grid["tanh_k"] for c in inclusive_grid(grid["tanh_center"])
        ],
        "kink": [{"c": c} for c in inclusive_grid(grid["kink_location"])],
        "step": [{"c": c} for c in inclusive_grid(grid["step_boundary"])],
        "spline": [
            {"knots": int(knots), "alpha": float(alpha)}
            for knots in grid["spline_knots"] for alpha in grid["spline_alpha"]
        ],
    }


def basis(family: str, u: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    value = np.asarray(u, dtype=float)
    if family == "constant":
        return np.ones((len(value), 1))
    if family == "linear":
        return np.column_stack([np.ones(len(value)), value])
    if family == "quadratic":
        return np.column_stack([np.ones(len(value)), (value - float(params["c"])) ** 2])
    if family == "tanh":
        return np.column_stack([
            np.ones(len(value)),
            np.tanh(float(params["k"]) * (value - float(params["c"]))),
        ])
    if family == "kink":
        c = float(params["c"])
        return np.column_stack([np.ones(len(value)), value, np.maximum(value - c, 0.0)])
    if family == "step":
        return np.column_stack([
            np.ones(len(value)), (value >= float(params["c"])).astype(float)
        ])
    raise ValueError(family)


def fit_model(
    family: str,
    u: np.ndarray,
    y_residual: np.ndarray,
    t_residual: np.ndarray,
    params: dict[str, Any],
) -> dict[str, Any]:
    value = np.asarray(u, dtype=float)
    if family == "spline":
        transformer = SplineTransformer(
            n_knots=int(params["knots"]) + 2,
            degree=3,
            knots="quantile",
            include_bias=True,
            extrapolation="linear",
        )
        matrix = transformer.fit_transform(value.reshape(-1, 1))
        design = np.asarray(t_residual, dtype=float)[:, None] * matrix
        fitted = Ridge(alpha=float(params["alpha"]), fit_intercept=False).fit(
            design, y_residual
        )
        return {
            "family": family, "params": dict(params), "transformer": transformer,
            "coefficient": np.asarray(fitted.coef_, dtype=float),
        }
    matrix = basis(family, value, params)
    design = np.asarray(t_residual, dtype=float)[:, None] * matrix
    coefficient, *_ = np.linalg.lstsq(design, np.asarray(y_residual, dtype=float), rcond=None)
    return {"family": family, "params": dict(params), "coefficient": coefficient}


def predict_model(model: dict[str, Any], u: np.ndarray) -> np.ndarray:
    value = np.asarray(u, dtype=float)
    if model["family"] == "spline":
        matrix = model["transformer"].transform(value.reshape(-1, 1))
    else:
        matrix = basis(model["family"], value, model["params"])
    return np.asarray(matrix @ model["coefficient"], dtype=float).reshape(-1)


def observation_loss(
    y_residual: np.ndarray, t_residual: np.ndarray, prediction: np.ndarray
) -> np.ndarray:
    return (
        np.asarray(y_residual, dtype=float)
        - np.asarray(t_residual, dtype=float) * np.asarray(prediction, dtype=float)
    ) ** 2


def fit_library(
    u: np.ndarray,
    y_residual: np.ndarray,
    t_residual: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    splitter = KFold(
        n_splits=int(config["shape_cv_folds"]), shuffle=True, random_state=int(seed)
    )
    folds = list(splitter.split(u))
    family_best: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for family, candidates in family_grid(config).items():
        candidates_evaluated: list[dict[str, Any]] = []
        for params in candidates:
            fold_losses: list[float] = []
            for fit_index, validation_index in folds:
                model = fit_model(
                    family, u[fit_index], y_residual[fit_index],
                    t_residual[fit_index], params,
                )
                prediction = predict_model(model, u[validation_index])
                fold_losses.append(float(np.mean(observation_loss(
                    y_residual[validation_index], t_residual[validation_index], prediction
                ))))
            record = {
                "family": family,
                "params": dict(params),
                "cv_mean_r_loss": float(np.mean(fold_losses)),
                "cv_se_r_loss": float(np.std(fold_losses, ddof=1) / math.sqrt(len(fold_losses))),
                "fold_losses": fold_losses,
            }
            records.append(record)
            candidates_evaluated.append(record)
        best = min(
            candidates_evaluated,
            key=lambda item: (item["cv_mean_r_loss"], json.dumps(item["params"], sort_keys=True)),
        )
        model = fit_model(family, u, y_residual, t_residual, best["params"])
        model.update({
            "cv_mean_r_loss": best["cv_mean_r_loss"],
            "cv_se_r_loss": best["cv_se_r_loss"],
            "fold_losses": best["fold_losses"],
        })
        family_best[family] = model
    structured = config["selection"]["structured_families"]
    selected = min(
        structured,
        key=lambda family: (family_best[family]["cv_mean_r_loss"], structured.index(family)),
    )
    flexible = family_best[config["selection"]["flexible_family"]]
    simple = family_best[selected]
    flexible_clear_win = (
        flexible["cv_mean_r_loss"] + flexible["cv_se_r_loss"]
        < simple["cv_mean_r_loss"]
    )
    final = "ABSTAIN_FLEXIBLE_STRUCTURE" if flexible_clear_win else selected
    return {
        "models": family_best,
        "records": records,
        "selected": final,
        "best_structured": selected,
        "flexible_clear_win": flexible_clear_win,
    }


def first_crossing(grid: np.ndarray, curve: np.ndarray) -> float:
    hits = np.flatnonzero(curve[:-1] * curve[1:] <= 0)
    if not len(hits):
        return float("nan")
    index = int(hits[0])
    left, right = float(curve[index]), float(curve[index + 1])
    if np.isclose(left, right):
        return float((grid[index] + grid[index + 1]) / 2)
    return float(grid[index] - left * (grid[index + 1] - grid[index]) / (right - left))


def classify_model(model: dict[str, Any], config: dict[str, Any]) -> tuple[str, float, dict[str, float]]:
    family = model["family"]
    coefficient = np.asarray(model["coefficient"], dtype=float)
    support = config["selection"]["crossing_support"]
    grid = np.linspace(float(support[0]), float(support[1]), 801)
    curve = predict_model(model, grid)
    crossing = first_crossing(grid, curve)
    if family == "constant":
        return "homogeneous", float("nan"), {"level": float(coefficient[0])}
    if family == "linear":
        details = {"intercept": float(coefficient[0]), "slope": float(coefficient[1]), "crossing": crossing}
        if np.isfinite(crossing):
            return "sign-reversal", crossing, details
        return "linear-monotone", float("nan"), details
    if family == "quadratic":
        label = "u-shaped" if coefficient[1] > 0 else "inverted-u"
        return label, float(model["params"]["c"]), {
            "turning_point": float(model["params"]["c"]),
            "curvature": float(2 * coefficient[1]),
        }
    if family == "tanh":
        details = {
            "center": float(model["params"]["c"]), "k": float(model["params"]["k"]),
            "amplitude": float(coefficient[1]), "crossing": crossing,
        }
        if np.isfinite(crossing):
            return "sign-reversal", crossing, details
        if float(model["params"]["k"]) >= float(config["selection"]["steep_tanh_k_minimum"]):
            return "steep-continuous", float(model["params"]["c"]), details
        return "nonlinear-monotone", float("nan"), details
    if family == "kink":
        return "kinked", float(model["params"]["c"]), {
            "kink": float(model["params"]["c"]),
            "left_slope": float(coefficient[1]),
            "right_slope": float(coefficient[1] + coefficient[2]),
            "slope_change": float(coefficient[2]),
        }
    if family == "step":
        return "threshold-step", float(model["params"]["c"]), {
            "boundary": float(model["params"]["c"]), "jump": float(coefficient[1]),
            "left_effect": float(coefficient[0]),
            "right_effect": float(coefficient[0] + coefficient[1]),
        }
    raise ValueError(f"family {family} has no direct taxonomy label")


def confirm_against_constant(
    candidate: dict[str, Any],
    constant: dict[str, Any],
    u_test: np.ndarray,
    y_residual_test: np.ndarray,
    t_residual_test: np.ndarray,
    alpha: float,
) -> dict[str, float | bool]:
    candidate_loss = observation_loss(
        y_residual_test, t_residual_test, predict_model(candidate, u_test)
    )
    constant_loss = observation_loss(
        y_residual_test, t_residual_test, predict_model(constant, u_test)
    )
    difference = candidate_loss - constant_loss
    mean = float(np.mean(difference))
    standard_error = float(np.std(difference, ddof=1) / math.sqrt(len(difference)))
    upper = mean + float(norm.ppf(1 - alpha)) * standard_error
    return {
        "test_loss_difference": mean,
        "test_loss_difference_se": standard_error,
        "test_loss_upper_one_sided": upper,
        "test_confirmed_vs_constant": bool(upper < 0.0),
    }


def refit_seeds(master: int, refit: int) -> dict[str, int]:
    children = np.random.SeedSequence([int(master), 1701, int(refit)]).spawn(3)
    values = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    return {"forest_refit_seed": values[0], "nuisance_refit_seed": values[1], "shape_cv_refit_seed": values[2]}


def truth_curve(u: np.ndarray, tau: np.ndarray, grid: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"u": u, "tau": tau}).groupby("u", as_index=False).mean().sort_values("u")
    return np.interp(grid, frame["u"].to_numpy(float), frame["tau"].to_numpy(float))


def pairwise_curve_spearman(curves: list[np.ndarray]) -> float:
    correlations: list[float] = []
    for left, right in combinations(curves, 2):
        if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
            correlations.append(1.0 if np.allclose(left, right) else 0.0)
        else:
            correlations.append(float(spearmanr(left, right).statistic))
    return float(np.median(correlations)) if correlations else 0.0


def practical_state(
    modal_label: str,
    label_agreement: float,
    curve_rank_stability: float,
    parameter_range: float,
    confirmation_rate: float,
    config: dict[str, Any],
) -> str:
    if modal_label == "ABSTAIN_FLEXIBLE_STRUCTURE":
        return "INCONCLUSIVE"
    support = config["supportability"]
    stable_parameter = not np.isfinite(parameter_range) or parameter_range <= float(
        support["maximum_parameter_range"]
    )
    stable = (
        label_agreement >= float(support["minimum_label_agreement"])
        and curve_rank_stability >= float(support["minimum_pairwise_curve_spearman"])
        and stable_parameter
    )
    confirmed = confirmation_rate >= float(config["confirmation"]["minimum_confirmation_rate"])
    if modal_label == "homogeneous":
        return "NO_THEORY_CLAIM" if stable and not confirmed else "INCONCLUSIVE"
    return "SUPPORTED_CANDIDATE" if stable and confirmed else "INCONCLUSIVE"


def create_freeze(config_path: Path, config: dict[str, Any]) -> Path:
    validate_config(config)
    base_path = (config_path.parent / config["base_config"]).resolve()
    source_path = (config_path.parent / config["source_config"]).resolve()
    freeze_path = (config_path.parent / config["freeze_path"]).resolve()
    payload = {
        "status": "METHOD_FROZEN_BEFORE_FRESH_RESULTS",
        "frozen_at": "2026-10-19",
        "method": "same fixed data multi-refit stability; TRAIN family selection; untouched TEST candidate-versus-constant confirmation",
        "scientific_correction": "steep-continuous restores previously frozen W4 k=25",
        "practical_state_uses_oracle": False,
        "test_selects_family": False,
        "hashes": {
            "config_sha256": sha256_file(config_path),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "base_runner_sha256": sha256_file(BASE_PATH),
            "base_config_sha256": sha256_file(base_path),
            "source_config_sha256": sha256_file(source_path),
        },
    }
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return freeze_path


def verify_freeze(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    freeze_path = (config_path.parent / config["freeze_path"]).resolve()
    if not freeze_path.exists():
        raise RuntimeError("method freeze missing; run --freeze-only before results")
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    current = {
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "base_runner_sha256": sha256_file(BASE_PATH),
        "base_config_sha256": sha256_file((config_path.parent / config["base_config"]).resolve()),
        "source_config_sha256": sha256_file((config_path.parent / config["source_config"]).resolve()),
    }
    if payload.get("hashes") != current:
        raise RuntimeError("method freeze hash mismatch")
    return payload


def run_refit(
    shape: str,
    refit: int,
    data: dict[str, Any],
    train: np.ndarray,
    test: np.ndarray,
    u: np.ndarray,
    grid: np.ndarray,
    source: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    started = time.perf_counter()
    seeds = refit_seeds(int(config["master_seed"]), refit)
    common = dict(source["common"])
    common.update({
        "n_estimators": int(config["primary_forest"]["n_estimators"]),
        "min_samples_leaf": int(config["primary_forest"]["min_samples_leaf"]),
        "max_samples": float(config["primary_forest"]["max_samples"]),
        "cv": int(config["primary_forest"]["cv"]),
        "honest": bool(config["primary_forest"]["honest"]),
        "n_jobs": int(config["primary_forest"]["n_jobs"]),
    })
    estimator = V2.v2._new_m1(common, seeds["forest_refit_seed"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(data["Y"][train], data["T"][train], X=data["X"][train], W=None)
    forest_test = np.asarray(estimator.effect(data["X"][test]), dtype=float).reshape(-1)
    forest_train = np.asarray(estimator.effect(data["X"][train]), dtype=float).reshape(-1)
    importance = np.asarray(estimator.feature_importances_, dtype=float)
    key_index = source["features"].index(config["key_moderator"])
    key_rank = int(np.sum(importance > importance[key_index]) + 1)
    nuisance = V2.crossfit_nuisance(
        data["X"][train], data["Y"][train], data["T"][train], data["X"][test],
        seeds["nuisance_refit_seed"], int(config["nuisance_crossfit_folds"]),
    )
    library = fit_library(
        u[train], nuisance["y_residual_train"], nuisance["t_residual_train"],
        config, seeds["shape_cv_refit_seed"],
    )
    selected = library["selected"]
    if selected == "ABSTAIN_FLEXIBLE_STRUCTURE":
        label = selected
        parameter = float("nan")
        details: dict[str, float] = {}
        candidate = library["models"][library["best_structured"]]
    else:
        candidate = library["models"][selected]
        label, parameter, details = classify_model(candidate, config)
    constant = library["models"]["constant"]
    y_residual_test = data["Y"][test] - nuisance["m_test"]
    t_residual_test = data["T"][test] - nuisance["e_test"]
    if candidate["family"] == "constant":
        confirmation = {
            "test_loss_difference": 0.0,
            "test_loss_difference_se": 0.0,
            "test_loss_upper_one_sided": 0.0,
            "test_confirmed_vs_constant": False,
        }
    else:
        confirmation = confirm_against_constant(
            candidate, constant, u[test], y_residual_test, t_residual_test,
            float(config["confirmation"]["candidate_vs_constant_one_sided_alpha"]),
        )
    curve = predict_model(candidate, grid)
    raw_curve = np.interp(grid, V2.GRID, V2.train_forest_curve(u[train], forest_train))
    truth_test = data["tau_true"][test]
    candidate_test = predict_model(candidate, u[test])
    evaluator_spearman = (
        float(spearmanr(truth_test, candidate_test).statistic)
        if np.std(candidate_test) > 1e-12 and np.std(truth_test) > 1e-12 else float("nan")
    )
    row = {
        "world": shape, "refit": refit, "selected_family": selected,
        "best_structured_family": library["best_structured"],
        "predicted_label": label, "parameter_estimate": parameter,
        "parameter_details_json": json.dumps(details, sort_keys=True),
        "test_confirmed_vs_constant": confirmation["test_confirmed_vs_constant"],
        "forest_key_moderator_rank": key_rank,
        "forest_key_moderator_importance": float(importance[key_index]),
        "forest_test_cate_rmse": float(np.sqrt(np.mean((forest_test - truth_test) ** 2))),
        "reconstruction_test_cate_rmse": float(np.sqrt(np.mean((candidate_test - truth_test) ** 2))),
        "reconstruction_test_spearman": evaluator_spearman,
        "exact_label_correct": label == shape,
        "warning_count": len(caught),
        "warning_messages": "\n".join(str(item.message) for item in caught),
        "runtime_seconds": time.perf_counter() - started,
        "test_used_for_family_selection": False,
        "oracle_used_for_practical_state": False,
        **confirmation, **seeds,
    }
    family_rows = []
    for family, model in library["models"].items():
        family_rows.append({
            "world": shape, "refit": refit, "family": family,
            "params_json": json.dumps(model["params"], sort_keys=True),
            "cv_mean_r_loss": model["cv_mean_r_loss"],
            "cv_se_r_loss": model["cv_se_r_loss"],
            "selected": selected == family,
        })
    curve_rows = [
        {
            "world": shape, "refit": refit, "u": float(value),
            "truth_curve": float(truth), "raw_forest_curve": float(raw),
            "selected_curve": float(estimate), "predicted_label": label,
        }
        for value, truth, raw, estimate in zip(
            grid, truth_curve(u, data["tau_true"], grid), raw_curve, curve, strict=True
        )
    ]
    return row, family_rows, curve_rows


def summarize(
    rows: pd.DataFrame, curves: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for shape in SHAPES:
        frame = rows[rows["world"] == shape].copy()
        counts = frame["predicted_label"].value_counts()
        modal_label = str(counts.index[0])
        label_agreement = float(counts.iloc[0] / len(frame))
        modal_refits = frame.loc[frame["predicted_label"] == modal_label, "refit"].tolist()
        selected_curves = []
        for refit in modal_refits:
            selected_curves.append(
                curves[(curves["world"] == shape) & (curves["refit"] == refit)]
                .sort_values("u")["selected_curve"].to_numpy(float)
            )
        stability = pairwise_curve_spearman(selected_curves)
        parameters = frame.loc[
            (frame["predicted_label"] == modal_label) & frame["parameter_estimate"].notna(),
            "parameter_estimate",
        ].to_numpy(float)
        parameter_range = float(np.ptp(parameters)) if len(parameters) else float("nan")
        confirmation_rate = float(
            frame.loc[frame["predicted_label"] == modal_label, "test_confirmed_vs_constant"].mean()
        )
        state = practical_state(
            modal_label, label_agreement, stability, parameter_range,
            confirmation_rate, config,
        )
        output.append({
            "world": shape, "modal_predicted_label": modal_label,
            "practical_state": state, "label_agreement": label_agreement,
            "pairwise_curve_spearman": stability,
            "parameter_estimate_median": float(np.median(parameters)) if len(parameters) else float("nan"),
            "parameter_range": parameter_range,
            "test_confirmation_rate": confirmation_rate,
            "exact_label_correct": modal_label == shape,
            "forest_key_moderator_rank_median": float(frame["forest_key_moderator_rank"].median()),
            "forest_test_cate_rmse_median": float(frame["forest_test_cate_rmse"].median()),
            "reconstruction_test_cate_rmse_median": float(frame["reconstruction_test_cate_rmse"].median()),
            "reconstruction_test_spearman_median": float(frame["reconstruction_test_spearman"].median()),
            "runtime_seconds_total": float(frame["runtime_seconds"].sum()),
        })
    return pd.DataFrame(output)


def render_figure(curves: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    width, height = 1500, 1060
    panel_width, panel_height = 480, 305
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#20252b}.title{font-size:24px;font-weight:700}.sub{font-size:13px;fill:#5c6670}.panel{fill:#fff;stroke:#d6dce1}.name{font-size:14px;font-weight:700}.small{font-size:11px}.axis{stroke:#a7b0b8}.truth{fill:none;stroke:#20252b;stroke-width:2.2;stroke-dasharray:6 4}.forest{fill:none;stroke:#9aa4ad;stroke-width:1.4}.estimate{fill:none;stroke:#d95f02;stroke-width:2.6}</style>',
        '<text x="30" y="36" class="title">Same-data hierarchical recovery of nine CATE morphologies</text>',
        '<text x="30" y="60" class="sub">black dashed = truth (evaluation only); gray = raw forest; orange = TRAIN-selected reconstruction</text>',
    ]
    for index, shape in enumerate(SHAPES):
        row_index, column_index = divmod(index, 3)
        x0 = 25 + column_index * panel_width
        y0 = 82 + row_index * panel_height
        frame = curves[curves["world"] == shape]
        grid = np.sort(frame["u"].unique())
        truth = frame.groupby("u")["truth_curve"].mean().reindex(grid).to_numpy(float)
        raw = frame.groupby("u")["raw_forest_curve"].median().reindex(grid).to_numpy(float)
        estimate = frame.groupby("u")["selected_curve"].median().reindex(grid).to_numpy(float)
        summary_row = summary[summary["world"] == shape].iloc[0]
        all_values = np.concatenate([truth, raw, estimate])
        low, high = float(np.nanmin(all_values)), float(np.nanmax(all_values))
        margin = max(0.03, 0.12 * (high - low))
        low, high = low - margin, high + margin

        def points(
            values: np.ndarray,
            *,
            panel_x: float = x0,
            panel_y: float = y0,
            panel_grid: np.ndarray = grid,
            minimum: float = low,
            maximum: float = high,
        ) -> str:
            return " ".join(
                f"{panel_x + 38 + (float(u) - panel_grid.min()) / (panel_grid.max() - panel_grid.min()) * 400:.1f},"
                f"{panel_y + 250 - (float(value) - minimum) / (maximum - minimum) * 178:.1f}"
                for u, value in zip(panel_grid, values, strict=True)
            )

        zero_y = y0 + 250 - (0.0 - low) / (high - low) * 178
        lines.extend([
            f'<rect x="{x0}" y="{y0}" width="458" height="286" class="panel"/>',
            f'<text x="{x0 + 14}" y="{y0 + 22}" class="name">{shape}</text>',
            f'<text x="{x0 + 14}" y="{y0 + 42}" class="small">pred: {summary_row["modal_predicted_label"]}; {summary_row["practical_state"]}</text>',
            f'<line x1="{x0 + 38}" y1="{zero_y:.1f}" x2="{x0 + 438}" y2="{zero_y:.1f}" class="axis"/>',
            f'<polyline points="{points(truth)}" class="truth"/>',
            f'<polyline points="{points(raw)}" class="forest"/>',
            f'<polyline points="{points(estimate)}" class="estimate"/>',
        ])
    lines.extend([
        '<line x1="430" y1="1025" x2="470" y2="1025" class="truth"/><text x="478" y="1029" class="small">truth</text>',
        '<line x1="610" y1="1025" x2="650" y2="1025" class="forest"/><text x="658" y="1029" class="small">raw forest</text>',
        '<line x1="805" y1="1025" x2="845" y2="1025" class="estimate"/><text x="853" y="1029" class="small">hierarchical reconstruction</text>',
        '</svg>',
    ])
    (output / "NINE_SHAPE_SAME_DATA_HIERARCHICAL_FIGURE.svg").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_report(summary: pd.DataFrame, output: Path, freeze_hash: str) -> None:
    rows = []
    for row in summary.itertuples(index=False):
        parameter = "-" if pd.isna(row.parameter_estimate_median) else f"{row.parameter_estimate_median:.3f}"
        rows.append(
            f"| {row.world} | {row.modal_predicted_label} | {row.practical_state} | "
            f"{row.label_agreement:.0%} | {row.test_confirmation_rate:.0%} | "
            f"{row.reconstruction_test_spearman_median:.3f} | {parameter} |"
        )
    exact = int(summary["exact_label_correct"].sum())
    supported = int((summary["practical_state"] == "SUPPORTED_CANDIDATE").sum())
    no_claim = int((summary["practical_state"] == "NO_THEORY_CLAIM").sum())
    inconclusive = int((summary["practical_state"] == "INCONCLUSIVE").sum())
    report = f"""# 九形态同数据分层识别 Pilot v1

## 方法修正

旧稳定性表比较的是不同企业样本、处理分配和结果噪声下的五个模拟数据集，适合衡量方法的重复抽样成功率，但不等同于一项现实研究在固定数据上的估计稳定性。本轮固定企业、T、Y 和 80/20 划分，只改变森林随机种子与 TRAIN 内 nuisance/shape folds。TRAIN 选择函数家族，untouched TEST 只确认已选候选是否优于常数效应。真值不参与选择、确认或 practical state。

方法冻结文件 SHA256：`{freeze_hash}`。

## 结果

| 已知情境 | 实际识别 | practical state | 同数据标签一致率 | TEST确认率 | TEST排序相关 | 关键位置中位数 |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

九种已知形态中，最终标签准确 `{exact}/9`；`SUPPORTED_CANDIDATE` 为 `{supported}/9`，`NO_THEORY_CLAIM` 为 `{no_claim}/9`，仍为 `INCONCLUSIVE` 的是 `{inconclusive}/9`。这里的 TEST 排序相关和标签正确率仅为 simulation evaluator 指标，没有反馈进入实际识别器。

## 图的含义

`NINE_SHAPE_SAME_DATA_HIERARCHICAL_FIGURE.svg` 中，黑色虚线是真值（只用于评估），灰线是原始 causal forest 的中位曲线，橙线是分层正交重构。图直接回答两件事：原始森林是否压平或扰动形态，以及 TRAIN 选择、TEST 确认后的候选函数是否恢复了理论关心的方向、转折点、尖锐位置或边界。

## 解释边界

本轮是同一数据上的估计稳定性与九形态识别 pilot，不是 headline Monte Carlo。不同 DGP replication 仍需在后续用来估计总体成功概率，但不再被错误地直接当成单一现实研究的 abstention 条件。W4 和 W5 继续分别解释：W4 报告连续尖锐转变的位置和排序，W5 报告候选边界及边界两侧效应差异；不把二者互相分类作为论文成功标准。
"""
    (output / "NINE_SHAPE_SAME_DATA_HIERARCHICAL_REPORT_V1_ZH.md").write_text(
        report, encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if args.freeze_only:
        print(create_freeze(config_path, config), flush=True)
        return 0
    freeze = verify_freeze(config_path, config)
    base_config = json.loads(
        (config_path.parent / config["base_config"]).resolve().read_text(encoding="utf-8")
    )
    source_path = (config_path.parent / config["source_config"]).resolve()
    source = V2.load_json(source_path)
    backbone = V2.add_rank_columns(
        V2.v2.load_frozen_backbone(ROOT, source, source_path), source["features"]
    )
    parameters = json.loads(json.dumps(base_config["shape_parameters"]))
    parameters["steep-continuous"]["steepness"] = 25.0
    output = (config_path.parent / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    grid_spec = config["grid"]
    grid = np.linspace(
        float(grid_spec["minimum"]), float(grid_spec["maximum"]), int(grid_spec["points"])
    )
    base_info = BASE.lineage(int(config["master_seed"]), int(config["base_replication"]))
    all_rows: list[dict[str, Any]] = []
    all_family_rows: list[dict[str, Any]] = []
    all_curve_rows: list[dict[str, Any]] = []
    dataset_hashes: dict[str, str] = {}
    for shape in SHAPES:
        data = BASE.build_data(
            backbone, source, int(config["n_per_shape"]), shape, base_info, parameters
        )
        train, test = V2.v2.split_once(
            data["T"], data["unit_id"], float(config["test_size"]), base_info["split_seed"]
        )
        u = BASE.reference_u(backbone, data["unit_id"], config["key_moderator"])
        dataset_hash = sha256_arrays(data["X"], data["T"], data["Y"], train, test)
        dataset_hashes[shape] = dataset_hash
        for refit in range(int(config["same_data_refits"])):
            print(f"RUN shape={shape} same-data-refit={refit + 1}/{config['same_data_refits']}", flush=True)
            row, family_rows, curve_rows = run_refit(
                shape, refit, data, train, test, u, grid, source, config
            )
            row["dataset_sha256"] = dataset_hash
            all_rows.append(row)
            all_family_rows.extend(family_rows)
            all_curve_rows.extend(curve_rows)
    rows = pd.DataFrame(all_rows)
    if rows.groupby("world")["dataset_sha256"].nunique().max() != 1:
        raise AssertionError("same-data refits did not keep the dataset fixed")
    family_frame = pd.DataFrame(all_family_rows)
    curve_frame = pd.DataFrame(all_curve_rows)
    summary = summarize(rows, curve_frame, config)
    rows.to_csv(output / "NINE_SHAPE_SAME_DATA_REFIT_LONG.csv", index=False)
    family_frame.to_csv(output / "NINE_SHAPE_SAME_DATA_FAMILY_CV.csv", index=False)
    curve_frame.to_csv(output / "NINE_SHAPE_SAME_DATA_CURVES.csv", index=False)
    summary.to_csv(output / "NINE_SHAPE_SAME_DATA_SUMMARY.csv", index=False)
    render_figure(curve_frame, summary, output)
    freeze_hash = sha256_file((config_path.parent / config["freeze_path"]).resolve())
    write_report(summary, output, freeze_hash)
    manifest = {
        "status": "completed",
        "shapes": list(SHAPES),
        "n": int(config["n_per_shape"]),
        "same_data_refits": int(config["same_data_refits"]),
        "forest_fits": int(len(SHAPES) * config["same_data_refits"]),
        "one_joint_multivariable_forest_per_refit": True,
        "dataset_fixed_within_shape": True,
        "test_used_for_family_selection": False,
        "oracle_used_for_practical_state": False,
        "w4_w5_cross_classification_required": False,
        "freeze_sha256": freeze_hash,
        "freeze_payload": freeze,
        "dataset_sha256_by_shape": dataset_hashes,
        "package_versions": V2.package_versions(),
    }
    (output / "NINE_SHAPE_SAME_DATA_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
