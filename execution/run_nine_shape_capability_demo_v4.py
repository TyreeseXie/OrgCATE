"""Nine-shape capability confirmation with complete prototype hierarchy."""

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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "execution" / "run_nine_shape_same_data_hierarchical_pilot_v1.py"
DEFAULT_CONFIG = ROOT / "configs" / "nine_shape_capability_demo_v4.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HELPER = load_module(HELPER_PATH, "capability_demo_helpers")
BASE = HELPER.BASE
V2 = HELPER.V2
SHAPES = HELPER.SHAPES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def prototype_tau(shape: str, u: np.ndarray) -> np.ndarray:
    value = np.asarray(u, dtype=float)
    x = 5.0 * (value - 0.5)
    if shape == "homogeneous":
        return np.ones(len(value), dtype=float)
    if shape == "linear-monotone":
        return 1.7 + 0.6 * x
    if shape == "nonlinear-monotone":
        return 2.0 + np.tanh(1.2 * x)
    if shape == "steep-continuous":
        return 0.2 + 2.8 / (1.0 + np.exp(-4.0 * x))
    if shape == "inverted-u":
        return 2.0 - 0.55 * x**2
    if shape == "u-shaped":
        return -2.0 + 0.55 * x**2
    if shape == "kinked":
        return np.where(x < 0.0, 1.6 + 0.4 * x, 1.6 + 1.8 * x)
    if shape == "threshold-step":
        return np.where(x < 0.0, 0.3, 2.3)
    if shape == "sign-reversal":
        return 0.75 - x
    raise ValueError(shape)


def validate_config(config: dict[str, Any]) -> None:
    if config["schema_version"] != "nine-shape-capability-demo-v4":
        raise ValueError("wrong capability-demo schema")
    if int(config["n"]) != 3000 or int(config["same_data_nuisance_refits"]) != 3:
        raise ValueError("capability pilot size changed")
    if not np.isclose(float(config["test_size"]), 0.20):
        raise ValueError("external split changed")
    expected = {
        "class": "econml.dml.CausalForestDML", "n_estimators": 1000,
        "min_samples_leaf": 5, "max_samples": 0.45, "cv": 2,
        "honest": True, "inference": False, "n_jobs": 2,
    }
    if config["primary_forest"] != expected:
        raise ValueError("primary forest changed")
    if config["prototype_truth"] != {
        "moderator_transform": "x=5*(u-0.5)",
        "homogeneous": "1.0",
        "linear-monotone": "1.7+0.6*x",
        "nonlinear-monotone": "2.0+tanh(1.2*x)",
        "steep-continuous": "0.2+2.8/(1+exp(-4*x))",
        "inverted-u": "2.0-0.55*x^2",
        "u-shaped": "-2.0+0.55*x^2",
        "kinked": "x<0 ? 1.6+0.4*x : 1.6+1.8*x",
        "threshold-step": "x<0 ? 0.3 : 2.3",
        "sign-reversal": "0.75-x",
    }:
        raise ValueError("teacher prototype truth changed")
    if int(config["nuisance_crossfit_folds"]) != 5 or int(config["shape_cv_folds"]) != 5:
        raise ValueError("cross-fitting contract changed")
    if config["test_used_for_shape_selection"] is not False:
        raise ValueError("TEST cannot select morphology")
    if config["oracle_used_only_for_evaluation"] is not True:
        raise ValueError("oracle separation changed")


def build_capability_data(
    backbone: pd.DataFrame,
    source: dict[str, Any],
    config: dict[str, Any],
    shape: str,
    lineage: dict[str, int],
    base_parameters: dict[str, Any],
) -> dict[str, Any]:
    base = BASE.build_data(
        backbone, source, int(config["n"]), "homogeneous", lineage, base_parameters
    )
    tau = prototype_tau(shape, base["u_m1_true"])
    y0 = base["Y"] - base["T"] * base["tau_true"]
    outcome = y0 + base["T"] * tau
    if not np.allclose(outcome, y0 + base["T"] * tau):
        raise AssertionError("observed outcome identity failed")
    return {**base, "Y": outcome, "tau_true": tau, "y0_true": y0}


def aggregate_family_losses(libraries: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for family in HELPER.family_grid(libraries[0]["config"]):
        losses = []
        for library in libraries:
            losses.extend(library["models"][family]["fold_losses"])
        rows.append({
            "family": family,
            "mean_loss": float(np.mean(losses)),
            "se_loss": float(np.std(losses, ddof=1) / math.sqrt(len(losses))),
            "fold_losses": np.asarray(losses, dtype=float),
        })
    return pd.DataFrame(rows)


def paired_difference(
    table: pd.DataFrame, left: str, right: str
) -> tuple[float, float]:
    left_loss = table.loc[table["family"] == left, "fold_losses"].iloc[0]
    right_loss = table.loc[table["family"] == right, "fold_losses"].iloc[0]
    difference = np.asarray(left_loss, dtype=float) - np.asarray(right_loss, dtype=float)
    return (
        float(np.mean(difference)),
        float(np.std(difference, ddof=1) / math.sqrt(len(difference))),
    )


def model_curves(
    libraries: list[dict[str, Any]], family: str, grid: np.ndarray
) -> list[np.ndarray]:
    return [HELPER.predict_model(item["models"][family], grid) for item in libraries]


def monotone_share(curve: np.ndarray, grid: np.ndarray) -> float:
    derivative = np.gradient(curve, grid)
    positive = float(np.mean(derivative >= 0.0))
    negative = float(np.mean(derivative <= 0.0))
    return max(positive, negative)


def normalized_deviation_from_linear(curve: np.ndarray, grid: np.ndarray) -> float:
    """Scale-free curvature diagnostic computed from TRAIN reconstruction only."""
    design = np.column_stack([np.ones(len(grid)), grid])
    coefficient = np.linalg.lstsq(design, curve, rcond=None)[0]
    residual_rmse = float(np.sqrt(np.mean((curve - design @ coefficient) ** 2)))
    curve_span = float(np.ptp(curve))
    return residual_rmse / curve_span if curve_span > 1e-12 else 0.0


def crossing(grid: np.ndarray, curve: np.ndarray) -> float:
    return HELPER.first_crossing(grid, curve)


def same_sign_slope_ratio(model: dict[str, Any]) -> tuple[bool, float]:
    coefficient = np.asarray(model["coefficient"], dtype=float)
    left = float(coefficient[1])
    right = float(coefficient[1] + coefficient[2])
    same_sign = bool(left * right > 0.0)
    small = min(abs(left), abs(right))
    ratio = max(abs(left), abs(right)) / small if small > 1e-12 else float("inf")
    return same_sign, float(ratio)


def interior_turn_supported(
    curve: np.ndarray,
    grid: np.ndarray,
    center: float,
    support: list[float],
    minimum_tail_share: float,
) -> bool:
    """Require a central extremum with material movement on both sides."""
    if not (float(support[0]) <= center <= float(support[1])):
        return False
    center_effect = float(np.interp(center, grid, curve))
    left_effect = float(curve[0])
    right_effect = float(curve[-1])
    span = float(np.ptp(curve))
    if span <= 1e-12:
        return False
    is_minimum = center_effect <= min(left_effect, right_effect)
    is_maximum = center_effect >= max(left_effect, right_effect)
    if is_minimum:
        tail_contrasts = [left_effect - center_effect, right_effect - center_effect]
    elif is_maximum:
        tail_contrasts = [center_effect - left_effect, center_effect - right_effect]
    else:
        return False
    return min(tail_contrasts) >= minimum_tail_share * span


def select_hierarchical_family(
    libraries: list[dict[str, Any]], config: dict[str, Any], grid: np.ndarray
) -> dict[str, Any]:
    table = aggregate_family_losses(libraries)
    lookup = table.set_index("family")["mean_loss"].to_dict()
    continuous = ["linear", "tanh", "kink", "quadratic"]
    best_continuous = min(continuous, key=lambda family: lookup[family])
    if lookup["step"] < lookup[best_continuous]:
        return {"family": "step", "label": "threshold-step", "reason": "continuity layer selected level jump"}
    tanh_k = float(np.median([
        item["models"]["tanh"]["params"]["k"] for item in libraries
    ]))
    step_difference, step_se = paired_difference(table, "step", "tanh")
    if (
        tanh_k >= float(config["hierarchy"]["step_saturation_k"])
        and step_difference
        <= float(config["hierarchy"]["step_near_tie_z"]) * step_se
    ):
        return {
            "family": "step",
            "label": "threshold-step",
            "reason": "smooth transition saturated at the frozen grid limit and step loss was statistically tied",
        }
    best_monotone = min(["linear", "tanh", "kink"], key=lambda family: lookup[family])
    if lookup["quadratic"] < lookup[best_monotone]:
        quadratic_models = [item["models"]["quadratic"] for item in libraries]
        coefficient = float(np.median([
            model["coefficient"][1] for model in quadratic_models
        ]))
        center = float(np.median([model["params"]["c"] for model in quadratic_models]))
        quadratic_curve = np.median(
            [HELPER.predict_model(model, grid) for model in quadratic_models], axis=0
        )
        if interior_turn_supported(
            quadratic_curve,
            grid,
            center,
            config["hierarchy"]["turning_point_support"],
            float(config["hierarchy"]["minimum_two_sided_tail_contrast_share"]),
        ):
            label = "u-shaped" if coefficient > 0 else "inverted-u"
            return {"family": "quadratic", "label": label, "reason": "turning-point layer found a supported interior extremum"}

    candidate_curves = model_curves(libraries, best_monotone, grid)
    consensus = np.median(candidate_curves, axis=0)
    cross = crossing(grid, consensus)
    if (
        monotone_share(consensus, grid)
        >= float(config["hierarchy"]["minimum_monotone_derivative_share"])
        and np.isfinite(cross)
    ):
        return {"family": best_monotone, "label": "sign-reversal", "reason": "sign layer found an interior zero crossing"}

    linear_deviation = normalized_deviation_from_linear(consensus, grid)
    linear_tolerance = float(
        config["hierarchy"]["maximum_normalized_deviation_from_linear"]
    )
    if linear_deviation <= linear_tolerance:
        return {
            "family": "linear",
            "label": "linear-monotone",
            "reason": (
                "TRAIN reconstruction is near-linear: normalized deviation "
                f"{linear_deviation:.4f} <= frozen {linear_tolerance:.4f}"
            ),
        }

    best_nonlinear = min(["tanh", "kink"], key=lambda family: lookup[family])
    mean_difference, se_difference = paired_difference(table, "linear", best_nonlinear)
    z = float(config["hierarchy"]["train_paired_loss_z"])
    if mean_difference <= z * se_difference:
        return {"family": "linear", "label": "linear-monotone", "reason": "nonlinearity did not improve paired TRAIN CV loss"}

    kink_difference, kink_se = paired_difference(table, "kink", "tanh")
    kink_models = [item["models"]["kink"] for item in libraries]
    geometry = [same_sign_slope_ratio(model) for model in kink_models]
    geometry_pass = (
        sum(item[0] for item in geometry) >= 2
        and float(np.median([item[1] for item in geometry]))
        >= float(config["hierarchy"]["kink_minimum_slope_ratio"])
    )
    if kink_difference <= z * kink_se and geometry_pass:
        return {"family": "kink", "label": "kinked", "reason": "continuous same-direction slope break supported"}

    label = (
        "steep-continuous"
        if tanh_k >= float(config["hierarchy"]["steep_tanh_k_cutpoint"])
        else "nonlinear-monotone"
    )
    return {"family": "tanh", "label": label, "reason": "continuous monotone nonlinear branch"}


def parameter_record(
    label: str, models: list[dict[str, Any]], grid: np.ndarray
) -> dict[str, Any]:
    parameters: list[float] = []
    details: list[dict[str, float]] = []
    for model in models:
        coefficient = np.asarray(model["coefficient"], dtype=float)
        curve = HELPER.predict_model(model, grid)
        if label == "homogeneous":
            parameter = float("nan")
            detail = {"level": float(np.mean(curve))}
        elif label == "linear-monotone":
            parameter = float("nan")
            detail = {"slope": float(coefficient[1])}
        elif label in {"nonlinear-monotone", "steep-continuous"}:
            parameter = float(model["params"]["c"])
            detail = {
                "center": parameter, "k": float(model["params"]["k"]),
                "amplitude": float(coefficient[1]),
            }
        elif label in {"inverted-u", "u-shaped"}:
            parameter = float(model["params"]["c"])
            detail = {"turning_point": parameter, "curvature": float(2 * coefficient[1])}
        elif label == "kinked":
            parameter = float(model["params"]["c"])
            detail = {
                "kink": parameter, "left_slope": float(coefficient[1]),
                "right_slope": float(coefficient[1] + coefficient[2]),
            }
        elif label == "threshold-step":
            parameter = float(model["params"]["c"])
            detail = {
                "boundary": parameter, "jump": float(coefficient[1]),
                "left_effect": float(coefficient[0]),
                "right_effect": float(coefficient[0] + coefficient[1]),
            }
        elif label == "sign-reversal":
            parameter = crossing(grid, curve)
            detail = {"crossing": parameter}
        else:
            raise ValueError(label)
        parameters.append(parameter)
        details.append(detail)
    finite = np.asarray([item for item in parameters if np.isfinite(item)], dtype=float)
    return {
        "parameter_median": float(np.median(finite)) if len(finite) else float("nan"),
        "parameter_range": float(np.ptp(finite)) if len(finite) else float("nan"),
        "details_json": json.dumps(details, sort_keys=True),
    }


def pairwise_curve_spearman(curves: list[np.ndarray]) -> float:
    values = []
    for left, right in combinations(curves, 2):
        values.append(float(spearmanr(left, right).statistic))
    return float(np.median(values))


def create_freeze(config_path: Path, config: dict[str, Any]) -> Path:
    validate_config(config)
    freeze_path = (config_path.parent / config["freeze_path"]).resolve()
    payload = {
        "status": "METHOD_FROZEN_BEFORE_FRESH_V4_CONFIRMATORY_RESULTS",
        "frozen_at": "2026-10-19",
        "calibration_disclosure": "v3 seed 2026101904 exposed that an edge quadratic vertex was incorrectly treated as the prototype's interior extremum; central-support and two-sided-tail requirements were frozen before seed 2026101905",
        "benchmark_role": "capability demonstration; existing matched-ATE configuration remains the stress benchmark",
        "test_selects_shape": False,
        "oracle_selects_shape": False,
        "hashes": {
            "config_sha256": sha256_file(config_path),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "helper_sha256": sha256_file(HELPER_PATH),
            "source_config_sha256": sha256_file((config_path.parent / config["source_config"]).resolve()),
        },
    }
    freeze_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return freeze_path


def verify_freeze(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    freeze_path = (config_path.parent / config["freeze_path"]).resolve()
    if not freeze_path.exists():
        raise RuntimeError("capability method freeze missing")
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    current = {
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "helper_sha256": sha256_file(HELPER_PATH),
        "source_config_sha256": sha256_file((config_path.parent / config["source_config"]).resolve()),
    }
    if payload["hashes"] != current:
        raise RuntimeError("capability freeze hash mismatch")
    return payload


def render_svg(curves: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    width, height = 1500, 1050
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#20252b}.title{font-size:24px;font-weight:700}.sub{font-size:13px;fill:#5c6670}.panel{fill:#fff;stroke:#d6dce1}.name{font-size:14px;font-weight:700}.small{font-size:11px}.truth{fill:none;stroke:#20252b;stroke-width:2.2;stroke-dasharray:6 4}.estimate{fill:none;stroke:#d95f02;stroke-width:2.7}.band{fill:#d95f02;fill-opacity:.12;stroke:none}</style>',
        '<text x="30" y="36" class="title">Nine-shape capability demo v4 on the real firm-covariate backbone</text>',
        '<text x="30" y="60" class="sub">black dashed = simulation truth (evaluation only); orange = TRAIN reconstruction; shaded = same-data nuisance-refit range</text>',
    ]
    for index, shape in enumerate(SHAPES):
        row_index, column_index = divmod(index, 3)
        x0, y0 = 25 + column_index * 480, 82 + row_index * 300
        frame = curves[curves["world"] == shape]
        grouped = frame.groupby("u", as_index=False).agg(
            truth=("truth_curve", "mean"), estimate=("estimate_curve", "median"),
            low=("estimate_curve", "min"), high=("estimate_curve", "max"),
        )
        columns = ["truth", "estimate", "low", "high"]
        low_y = float(grouped[columns].min().min())
        high_y = float(grouped[columns].max().max())
        margin = max(0.05, 0.10 * (high_y - low_y))
        low_y, high_y = low_y - margin, high_y + margin
        grid = grouped["u"].to_numpy(float)

        def point(
            value: float,
            effect: float,
            *,
            panel_x: float = x0,
            panel_y: float = y0,
            panel_grid: np.ndarray = grid,
            minimum: float = low_y,
            maximum: float = high_y,
        ) -> tuple[float, float]:
            return (
                panel_x + 38 + (value - panel_grid.min()) / (panel_grid.max() - panel_grid.min()) * 400,
                panel_y + 247 - (effect - minimum) / (maximum - minimum) * 175,
            )

        def polyline(column: str, *, panel_frame: pd.DataFrame = grouped) -> str:
            return " ".join(
                f"{point(float(u), float(effect))[0]:.1f},{point(float(u), float(effect))[1]:.1f}"
                for u, effect in zip(panel_frame["u"], panel_frame[column], strict=True)
            )

        upper = [point(float(u), float(effect)) for u, effect in zip(grouped["u"], grouped["high"], strict=True)]
        lower = [point(float(u), float(effect)) for u, effect in zip(grouped["u"][::-1], grouped["low"][::-1], strict=True)]
        polygon = " ".join(f"{x:.1f},{y:.1f}" for x, y in [*upper, *lower])
        row = summary[summary["world"] == shape].iloc[0]
        lines.extend([
            f'<rect x="{x0}" y="{y0}" width="458" height="282" class="panel"/>',
            f'<text x="{x0 + 14}" y="{y0 + 22}" class="name">{shape}</text>',
            f'<text x="{x0 + 14}" y="{y0 + 42}" class="small">identified: {row["predicted_label"]}; {row["practical_state"]}</text>',
            f'<polygon points="{polygon}" class="band"/>',
            f'<polyline points="{polyline("truth")}" class="truth"/>',
            f'<polyline points="{polyline("estimate")}" class="estimate"/>',
        ])
    lines.append('</svg>')
    (output / "NINE_SHAPE_CAPABILITY_DEMO_V4_FIGURE.svg").write_text("\n".join(lines), encoding="utf-8")


def write_report(summary: pd.DataFrame, output: Path, freeze_hash: str) -> None:
    table = []
    for row in summary.itertuples(index=False):
        parameter = "-" if pd.isna(row.parameter_median) else f"{row.parameter_median:.3f}"
        table.append(
            f"| {row.world} | {row.predicted_label} | {row.practical_state} | "
            f"{row.test_confirmation_rate:.0%} | {row.curve_spearman:.3f} | "
            f"{row.oracle_curve_spearman:.3f} | {parameter} |"
        )
    correct = int(summary["exact_label_correct"].sum())
    report = f"""# 九形态 Capability Demo v4

## 定位

本演示把老师 prototype 中预先存在的九个 treatment-effect 函数映射到冻结的真实 Compustat 企业协变量支持上。它回答“实现链在可识别信号下能否恢复形态”，不替代既有 matched-ATE 弱信号压力测试。旧压力测试允许 abstain；本演示要求识别。

方法冻结 SHA256：`{freeze_hash}`。v3 seed `2026101904` 暴露了一个定义性遗漏：靠近支持边缘的二次顶点不应被当成 prototype 所说的“内部极值”。中部支持区间与两侧实质 tail contrast 要求随后被冻结。以下为全新 confirmatory seed `2026101905` 的结果，不能用本轮结果反向修改规则。

## Fresh 结果

| 真实形态 | 识别形态 | practical state | TEST确认率 | 同数据曲线稳定性 | evaluator排序相关 | 关键位置 |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(table)}

准确识别 `{correct}/9`。`TEST确认率` 是三个 TRAIN nuisance 重拟合下，预先选定形态相对常数效应的 untouched-TEST 单侧确认比例；`同数据曲线稳定性` 不改变企业、T、Y 或 split。evaluator 排序相关只在模拟评估层计算，不参与选型或 practical state。

## 为什么这版不再大面积 INCONCLUSIVE

上一版同时犯了两个错误：把不同模拟数据集间的差异当成单一现实研究的稳定性；把最低 R-loss 的基函数名字直接当成形态名字。本版固定同一数据做稳定性，并严格按“连续性 -> 单调/转向 -> 与 ATE 的符号关系”读取性质。比如一条曲线即使被 kink 基函数近似，只要它保持单调并在内部跨零，最终仍归为 sign reversal；这才符合老师 prototype 的三层逻辑。

## 图的 implication

`NINE_SHAPE_CAPABILITY_DEMO_V4_FIGURE.svg` 中黑虚线是真值，橙线是 practical TRAIN 重构，浅橙带是同一数据三次 nuisance 重拟合范围。橙线是否贴合黑线说明恢复精度；带宽说明算法稳定性。图与表必须一起读：曲线可能排序准确但幅度有偏，形态标签正确也不等于机制已经得到因果确认。
"""
    (output / "NINE_SHAPE_CAPABILITY_DEMO_REPORT_V4_ZH.md").write_text(report, encoding="utf-8")


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
    source_path = (config_path.parent / config["source_config"]).resolve()
    source = V2.load_json(source_path)
    base_config = json.loads((config_path.parent / config["base_config"]).resolve().read_text(encoding="utf-8"))
    backbone = V2.add_rank_columns(
        V2.v2.load_frozen_backbone(ROOT, source, source_path), source["features"]
    )
    lineage = BASE.lineage(int(config["master_seed"]), 0)
    grid_spec = config["grid"]
    grid = np.linspace(float(grid_spec["minimum"]), float(grid_spec["maximum"]), int(grid_spec["points"]))
    output = (config_path.parent / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    long_rows = []
    curve_rows = []
    family_rows = []
    for shape in SHAPES:
        print(f"CAPABILITY shape={shape}", flush=True)
        started = time.perf_counter()
        data = build_capability_data(
            backbone, source, config, shape, lineage, base_config["shape_parameters"]
        )
        train, test = V2.v2.split_once(
            data["T"], data["unit_id"], float(config["test_size"]), lineage["split_seed"]
        )
        u = BASE.reference_u(backbone, data["unit_id"], config["key_moderator"])
        common = dict(source["common"])
        common["n_jobs"] = int(config["primary_forest"]["n_jobs"])
        estimator = V2.v2._new_m1(common, lineage["forest_seed"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(data["Y"][train], data["T"][train], X=data["X"][train], W=None)
        forest_test = np.asarray(estimator.effect(data["X"][test]), dtype=float)
        importance = np.asarray(estimator.feature_importances_, dtype=float)
        key_index = source["features"].index(config["key_moderator"])
        key_rank = int(np.sum(importance > importance[key_index]) + 1)
        libraries = []
        nuisances = []
        for refit in range(int(config["same_data_nuisance_refits"])):
            seeds = HELPER.refit_seeds(int(config["master_seed"]), refit)
            nuisance = V2.crossfit_nuisance(
                data["X"][train], data["Y"][train], data["T"][train], data["X"][test],
                seeds["nuisance_refit_seed"], int(config["nuisance_crossfit_folds"]),
            )
            library_config = dict(config)
            library_config["selection"] = {
                "structured_families": ["constant", "linear", "quadratic", "tanh", "kink", "step"],
                "flexible_family": "spline",
            }
            library = HELPER.fit_library(
                u[train], nuisance["y_residual_train"], nuisance["t_residual_train"],
                library_config, seeds["shape_cv_refit_seed"],
            )
            library["config"] = library_config
            libraries.append(library)
            nuisances.append(nuisance)
        selection = select_hierarchical_family(libraries, config, grid)
        selected_family = selection["family"]
        selected_models = [item["models"][selected_family] for item in libraries]
        estimates = [HELPER.predict_model(model, grid) for model in selected_models]
        parameter = parameter_record(selection["label"], selected_models, grid)
        confirmations = []
        for refit, (library, nuisance, model) in enumerate(zip(libraries, nuisances, selected_models, strict=True)):
            y_test = data["Y"][test] - nuisance["m_test"]
            t_test = data["T"][test] - nuisance["e_test"]
            confirm = HELPER.confirm_against_constant(
                model, library["models"]["constant"], u[test], y_test, t_test,
                float(config["confirmation"]["candidate_vs_constant_one_sided_alpha"]),
            )
            confirmations.append(bool(confirm["test_confirmed_vs_constant"]))
            long_rows.append({
                "world": shape, "refit": refit, "selected_family": selected_family,
                "predicted_label": selection["label"], "selection_reason": selection["reason"],
                **confirm,
            })
            for family, family_model in library["models"].items():
                family_rows.append({
                    "world": shape, "refit": refit, "family": family,
                    "cv_mean_r_loss": family_model["cv_mean_r_loss"],
                    "cv_se_r_loss": family_model["cv_se_r_loss"],
                    "params_json": json.dumps(family_model["params"], sort_keys=True),
                })
        confirmation_rate = float(np.mean(confirmations))
        curve_stability = pairwise_curve_spearman(estimates)
        predicted_label = selection["label"]
        if confirmation_rate < float(config["confirmation"]["minimum_confirmation_rate"]):
            predicted_label = "homogeneous"
            state = "NO_THEORY_CLAIM"
            selected_family = "constant"
            estimates = [
                HELPER.predict_model(item["models"]["constant"], grid) for item in libraries
            ]
            parameter = parameter_record("homogeneous", [item["models"]["constant"] for item in libraries], grid)
        else:
            stable_parameter = (
                not np.isfinite(parameter["parameter_range"])
                or parameter["parameter_range"] <= float(config["supportability"]["maximum_parameter_range"])
            )
            state = (
                "SUPPORTED_CANDIDATE"
                if curve_stability >= float(config["supportability"]["minimum_pairwise_curve_spearman"])
                and stable_parameter
                else "INCONCLUSIVE"
            )
        consensus = np.median(estimates, axis=0)
        truth_grid = prototype_tau(shape, grid)
        oracle_spearman = (
            float(spearmanr(truth_grid, consensus).statistic)
            if np.std(truth_grid) > 1e-12 and np.std(consensus) > 1e-12 else float("nan")
        )
        truth_test = data["tau_true"][test]
        summary_rows.append({
            "world": shape, "predicted_label": predicted_label,
            "practical_state": state, "selected_family": selected_family,
            "selection_reason": selection["reason"],
            "test_confirmation_rate": confirmation_rate,
            "curve_spearman": curve_stability,
            "oracle_curve_spearman": oracle_spearman,
            "curve_rmse": float(np.sqrt(np.mean((consensus - truth_grid) ** 2))),
            "parameter_median": parameter["parameter_median"],
            "parameter_range": parameter["parameter_range"],
            "parameter_details_json": parameter["details_json"],
            "exact_label_correct": predicted_label == shape,
            "forest_key_moderator_rank": key_rank,
            "forest_test_cate_rmse": float(np.sqrt(np.mean((forest_test - truth_test) ** 2))),
            "forest_test_cate_spearman": (
                float(spearmanr(truth_test, forest_test).statistic)
                if np.std(truth_test) > 1e-12 else float("nan")
            ),
            "true_ate": float(np.mean(data["tau_true"])),
            "warning_count": len(caught),
            "runtime_seconds": time.perf_counter() - started,
            "test_used_for_shape_selection": False,
            "oracle_used_for_practical_state": False,
        })
        for refit, estimate in enumerate(estimates):
            for value, truth, fitted in zip(grid, truth_grid, estimate, strict=True):
                curve_rows.append({
                    "world": shape, "refit": refit, "u": float(value),
                    "truth_curve": float(truth), "estimate_curve": float(fitted),
                })
    summary = pd.DataFrame(summary_rows)
    long_frame = pd.DataFrame(long_rows)
    curve_frame = pd.DataFrame(curve_rows)
    family_frame = pd.DataFrame(family_rows)
    summary.to_csv(output / "NINE_SHAPE_CAPABILITY_V4_SUMMARY.csv", index=False)
    long_frame.to_csv(output / "NINE_SHAPE_CAPABILITY_V4_CONFIRMATION_LONG.csv", index=False)
    curve_frame.to_csv(output / "NINE_SHAPE_CAPABILITY_V4_CURVES.csv", index=False)
    family_frame.to_csv(output / "NINE_SHAPE_CAPABILITY_V4_FAMILY_CV.csv", index=False)
    render_svg(curve_frame, summary, output)
    freeze_hash = sha256_file((config_path.parent / config["freeze_path"]).resolve())
    write_report(summary, output, freeze_hash)
    manifest = {
        "status": "completed", "benchmark_role": "capability_demo",
        "stress_benchmark_unchanged": True, "shapes": list(SHAPES),
        "n": int(config["n"]), "forest_fits": len(SHAPES),
        "one_joint_multivariable_forest_per_shape": True,
        "same_data_nuisance_refits": int(config["same_data_nuisance_refits"]),
        "test_used_for_shape_selection": False,
        "oracle_used_for_practical_state": False,
        "freeze_sha256": freeze_hash, "freeze": freeze,
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "package_versions": V2.package_versions(),
    }
    (output / "NINE_SHAPE_CAPABILITY_V4_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
