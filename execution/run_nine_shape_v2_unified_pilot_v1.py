"""Unified nine-shape pilot using the frozen v2 orthogonal reconstruction.

The nine cells share the same sampled firms, treatment assignment, untreated
outcome and 80/20 split within each replication. Only the simulated CATE shape
changes. Each cell fits one joint CausalForestDML, discovers the candidate
moderator from TRAIN importance, fits the morphology library on TRAIN
orthogonal residuals, and evaluates accuracy on untouched TEST.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
V2_PATH = ROOT / "morphology_reconstruction_v2_stage_success" / "src" / "morphology_v2_runner.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V2 = load_module(V2_PATH, "morphology_v2_for_nine_shapes")
V2_CONFIG = ROOT / "morphology_reconstruction_v2_stage_success" / "config" / "morphology_v2_confirmatory.json"
SOURCE_CONFIG = ROOT / "codex_simulation_runner_v2" / "config" / "pilot_b.json"

SHAPES = (
    "homogeneous", "linear-monotone", "nonlinear-monotone", "steep-continuous",
    "inverted-u", "u-shaped", "kinked", "threshold-step", "sign-reversal",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standardize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    centered = value - float(np.mean(value))
    scale = float(np.std(centered))
    if scale <= 1e-12:
        raise ValueError("shape driver is constant")
    return centered / scale


def make_tau(shape: str, u: np.ndarray, target: float, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Make a bounded shape on the real-backbone rank scale, preserving ATE."""
    u = np.asarray(u, dtype=float)
    z = (u - 0.5) / 0.25
    if shape == "homogeneous":
        tau = np.full(len(u), target)
    elif shape == "linear-monotone":
        tau = target + float(params[shape]["scale"]) * standardize(z)
    elif shape == "nonlinear-monotone":
        raw = np.tanh(float(params[shape]["steepness"]) * z)
        tau = target + float(params[shape]["scale"]) * standardize(raw)
    elif shape == "steep-continuous":
        raw = np.tanh(float(params[shape]["steepness"]) * (u - float(params[shape]["center"])))
        tau = target + float(params[shape]["scale"]) * standardize(raw)
    elif shape == "inverted-u":
        tau = target + float(params[shape]["scale"]) * standardize(-(z**2))
    elif shape == "u-shaped":
        tau = target + float(params[shape]["scale"]) * standardize(z**2)
    elif shape == "kinked":
        raw = np.where(z < 0.0, 0.40 * z, 1.80 * z)
        tau = target + float(params[shape]["scale"]) * standardize(raw)
    elif shape == "threshold-step":
        raw = (u >= float(params[shape]["boundary"])).astype(float)
        tau = target + float(params[shape]["scale"]) * standardize(raw)
    elif shape == "sign-reversal":
        tau = target + float(params[shape]["scale"]) * standardize(0.75 - z)
    else:
        raise ValueError(shape)
    tau += target - float(np.mean(tau))
    return tau, {
        "true_morphology": shape,
        "true_ate": float(np.mean(tau)),
        "true_cate_sd": float(np.std(tau)),
        "true_cate_min": float(np.min(tau)),
        "true_cate_max": float(np.max(tau)),
        "true_negative_share": float(np.mean(tau < 0.0)),
        "true_positive_share": float(np.mean(tau > 0.0)),
        "true_boundary": (float(params[shape].get("boundary", 0.5)) if shape == "threshold-step" else None),
        "true_turning_point": (0.5 if shape in {"inverted-u", "u-shaped"} else None),
    }


def lineage(master: int, rep: int) -> dict[str, int]:
    root = np.random.SeedSequence([int(master), int(rep)])
    children = root.spawn(8)
    names = ["backbone_sample_seed", "treatment_seed", "outcome_noise_seed", "split_seed", "forest_seed", "nuisance_seed", "shape_cv_seed", "artifact_seed"]
    result = {name: int(child.generate_state(1, dtype=np.uint32)[0]) for name, child in zip(names, children)}
    result.update(master_seed=int(master), replication=int(rep))
    return result


def reference_u(backbone: pd.DataFrame, unit_id: np.ndarray, feature: str) -> np.ndarray:
    lookup = backbone.assign(__unit=backbone["gvkey"].astype(str)).set_index("__unit")
    values = lookup.loc[np.asarray(unit_id, dtype=str), feature].to_numpy(float)
    if feature.endswith("_rank_year_currency"):
        return np.clip(values, 1e-6, 1 - 1e-6)
    rank = pd.Series(values).rank(method="average", pct=True).to_numpy(float)
    return np.clip(rank, 1e-6, 1 - 1e-6)


def metric_row(shape: str, route: str, moderator_mode: str, selected_moderator: str, selected_family: str, tau: np.ndarray, estimate: np.ndarray, truth_u: np.ndarray, true_info: dict[str, Any], train_n: int, test_n: int, runtime: float, lineage_info: dict[str, int], raw_curve: np.ndarray, theory_curve: np.ndarray) -> dict[str, Any]:
    rho = float(spearmanr(tau, estimate).statistic) if np.std(tau) > 1e-12 and np.std(estimate) > 1e-12 else float("nan")
    result = {
        "status": "completed", "shape": shape, "route": route, "moderator_mode": moderator_mode,
        "selected_moderator": selected_moderator, "selected_theory_family": selected_family,
        "cate_rmse": float(np.sqrt(np.mean((estimate - tau) ** 2))),
        "cate_mae": float(np.mean(np.abs(estimate - tau))), "spearman": rho,
        "sign_region_accuracy": float(np.mean(np.sign(estimate) == np.sign(tau))),
        "predicted_sd": float(np.std(estimate)), "true_sd": float(np.std(tau)),
        "train_n": train_n, "test_n": test_n, "runtime_seconds": runtime, **lineage_info,
    }
    if shape == "sign-reversal":
        def crossing(x: np.ndarray, y: np.ndarray) -> float:
            order = np.argsort(x); x, y = x[order], y[order]
            hit = np.flatnonzero(y[:-1] * y[1:] <= 0)
            if not len(hit): return float("nan")
            i = int(hit[0]); return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
        true_cross = crossing(truth_u, tau); est_cross = crossing(truth_u, estimate)
        result["zero_crossing_abs_error"] = abs(est_cross - true_cross) if np.isfinite(est_cross) and np.isfinite(true_cross) else float("nan")
    if shape in {"inverted-u", "u-shaped"}:
        result["turning_point_abs_error"] = abs(float(truth_u[int(np.argmax(estimate))]) - 0.5)
    if shape in {"threshold-step", "steep-continuous"}:
        # A descriptive sharp-transition locator, not a discontinuity claim.
        order = np.argsort(truth_u); x, y = truth_u[order], estimate[order]
        result["sharp_transition_location"] = float(x[np.argmax(np.abs(np.diff(y, prepend=y[0])))])
        result["true_boundary"] = true_info.get("true_boundary")
    result["raw_curve_span"] = float(np.ptp(raw_curve)) if np.isfinite(raw_curve).all() else float("nan")
    result["theory_curve_span"] = float(np.ptp(theory_curve)) if np.isfinite(theory_curve).all() else float("nan")
    return result


def build_data(backbone: pd.DataFrame, source: dict[str, Any], n: int, shape: str, info: dict[str, int], params: dict[str, Any]) -> dict[str, Any]:
    base = V2.v2.generate_world_dataset(backbone, "W2", n, info, source)
    tau, truth = make_tau(shape, base["u_m1_true"], float(source["dgp"]["target_ate"]), params)
    y0 = base["Y"] - base["T"] * base["tau_true"]
    outcome = y0 + base["T"] * tau
    return {**base, "Y": outcome, "tau_true": tau, "truth": truth}


def run_one(shape: str, rep: int, config: dict[str, Any], source: dict[str, Any], backbone: pd.DataFrame, v2_config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    started = time.perf_counter()
    info = lineage(int(config["master_seed"]), rep)
    data = build_data(backbone, source, int(config["n_per_replication"]), shape, info, config["shape_parameters"])
    train, test = V2.v2.split_once(data["T"], data["unit_id"], float(config["test_size"]), info["split_seed"])
    common = dict(source["common"]); common["n_jobs"] = int(config["primary_forest"]["n_jobs"])
    estimator = V2.v2._new_m1(common, info["forest_seed"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(data["Y"][train], data["T"][train], X=data["X"][train], W=None)
    forest_train = np.asarray(estimator.effect(data["X"][train]), float).reshape(-1)
    forest_test = np.asarray(estimator.effect(data["X"][test]), float).reshape(-1)
    importance = np.asarray(estimator.feature_importances_, float)
    selected_index = int(np.argmax(importance)); selected = source["features"][selected_index]
    nuisance = V2.crossfit_nuisance(data["X"][train], data["Y"][train], data["T"][train], data["X"][test], info["nuisance_seed"], int(config["nuisance_crossfit_folds"]))
    rows: list[dict[str, Any]] = []; families: list[dict[str, Any]] = []; curves: list[dict[str, Any]] = []
    u_matrix = {feature: reference_u(backbone, data["unit_id"], feature) for feature in source["features"]}
    for mode, feature in [("forest_selected", selected), ("oracle_key_moderator_diagnostic", config["key_moderator"])]:
        u_all = u_matrix[feature]; u_train, u_test = u_all[train], u_all[test]
        theory = V2.fit_theory_library(u_train, nuisance["y_residual_train"], nuisance["t_residual_train"], v2_config, info["shape_cv_seed"] + source["features"].index(feature))
        if theory["selected"] == "ABSTAIN_FLEXIBLE_STRUCTURE":
            theory_test = np.full(len(test), np.nan); theory_grid = np.full(201, np.nan)
        else:
            model = theory["models"][theory["selected"]]
            theory_test = V2.predict_candidate(model, u_test); theory_grid = V2.predict_candidate(model, V2.GRID)
        raw_grid = V2.train_forest_curve(u_train, forest_train)
        # The curve on TEST is binned by the route's moderator for display only.
        truth_test = data["tau_true"][test]
        base = {"world": shape, "replication": rep, "moderator_mode": mode, "forest_selected_moderator": selected, "forest_selected_top1": selected == config["key_moderator"], "true_moderator_rank": int(np.sum(importance > importance[source["features"].index(config["key_moderator"])]) + 1), "warning_count": len(caught), "warning_messages": "\n".join(str(w.message) for w in caught), "test_used_for_selection": False, "external_split_count": 1, "dataset_sha256": V2.sha256_arrays(data["X"], data["T"], data["Y"]), "true_ate": float(np.mean(data["tau_true"])), "true_morphology": shape, "true_cate_sd": float(np.std(data["tau_true"])), "importance_json": json.dumps(dict(zip(source["features"], importance.tolist())), sort_keys=True)}
        rows.append({**base, **metric_row(shape, "raw_forest", mode, feature, "raw", data["tau_true"][test], forest_test, u_test, data["truth"], len(train), len(test), time.perf_counter() - started, info, raw_grid, raw_grid)})
        rows.append({**base, **metric_row(shape, "theory_reconstruction", mode, feature, theory["selected"], truth_test, theory_test, u_test, data["truth"], len(train), len(test), time.perf_counter() - started, info, raw_grid, theory_grid)})
        for record in theory["records"]:
            families.append({"world": shape, "replication": rep, "moderator_mode": mode, "family": record["family"], "params_json": json.dumps(record["params"], sort_keys=True), "cv_mean_r_loss": record["cv_mean_r_loss"], "cv_se_r_loss": record["cv_se_r_loss"], "selected_final": record["family"] == theory["selected"] and record["params"] == theory["models"][record["family"]]["params"], "best_overall": record["family"] == theory["best_overall"] and record["params"] == theory["models"][record["family"]]["params"]})
        for x, raw, recon in zip(V2.GRID, raw_grid, theory_grid, strict=True):
            curves.append({"world": shape, "replication": rep, "moderator_mode": mode, "u": float(x), "raw_forest_curve": float(raw), "theory_curve": float(recon) if np.isfinite(recon) else float("nan")})
    return rows, families, curves


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=ROOT / "configs" / "nine_shape_v2_unified_pilot_v1.json")
    args = parser.parse_args(); config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    source = V2.load_json((args.config.parent / config["source_config"]).resolve())
    backbone = V2.add_rank_columns(V2.v2.load_frozen_backbone(ROOT, source, (args.config.parent / config["source_config"]).resolve()), source["features"])
    v2_config = V2.load_json(V2_CONFIG)
    output = (args.config.parent / config["output_dir"]).resolve(); output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []; family_rows: list[dict[str, Any]] = []; curve_rows: list[dict[str, Any]] = []
    for rep in range(int(config["replications"])):
        for shape in SHAPES:
            print(f"RUN shape={shape} rep={rep + 1}/{config['replications']}", flush=True)
            rows, families, curves = run_one(shape, rep, config, source, backbone, v2_config)
            all_rows.extend(rows); family_rows.extend(families); curve_rows.extend(curves)
    pd.DataFrame(all_rows).to_csv(output / "NINE_SHAPE_V2_LONG.csv", index=False)
    pd.DataFrame(family_rows).to_csv(output / "NINE_SHAPE_V2_FAMILY_SELECTION.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output / "NINE_SHAPE_V2_CURVES.csv", index=False)
    frame = pd.DataFrame(all_rows); recon = frame[frame["route"] == "theory_reconstruction"]
    summary = recon.groupby(["world", "moderator_mode"], as_index=False).agg(replications=("replication", "nunique"), cate_rmse_mean=("cate_rmse", "mean"), spearman_mean=("spearman", "mean"), sign_region_accuracy_mean=("sign_region_accuracy", "mean"), top1_rate=("forest_selected_top1", "mean"), family_mode=("selected_theory_family", lambda x: x.mode().iat[0] if len(x) else ""))
    summary.to_csv(output / "NINE_SHAPE_V2_SUMMARY.csv", index=False)
    manifest = {"status": "completed", "algorithm": "frozen morphology v2 orthogonal residual library plus prototype nine-shape readout", "shapes": list(SHAPES), "n": config["n_per_replication"], "replications": config["replications"], "one_joint_forest_per_shape_replication": True, "train_nuisance_crossfit": config["nuisance_crossfit_folds"], "shape_cv_folds": config["shape_cv_folds"], "one_se_rule": config["one_se_rule"], "test_used_for_selection": False, "oracle_used_for_practical_selection": False, "source_config_sha256": sha256_file((args.config.parent / config["source_config"]).resolve()), "v2_runner_sha256": sha256_file(V2_PATH), "runner_sha256": sha256_file(Path(__file__).resolve())}
    (output / "NINE_SHAPE_V2_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
