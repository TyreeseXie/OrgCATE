"""Compare conventional constant-effect OLS with DML across nine HTE forms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .public_contract import conventional_ate_diagnostics
from .public_demo import _contract
from .release import NINE_SITUATIONS, package_versions, repository_root, sha256_file


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def summarize_estimator_comparison(long: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bias, RMSE, coverage, and OLS-DML deviation by HTE form."""

    required = {
        "situation",
        "true_ate",
        "ols_estimate",
        "ols_ci_covers_true_ate",
        "dml_estimate",
        "dml_ci_covers_true_ate",
        "ols_dml_relative_deviation",
    }
    missing = required.difference(long.columns)
    if missing:
        raise ValueError(f"comparison table missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for situation, frame in long.groupby("situation", sort=False):
        ols_error = frame["ols_estimate"] - frame["true_ate"]
        dml_error = frame["dml_estimate"] - frame["true_ate"]
        deviation = frame["ols_dml_relative_deviation"].dropna()
        rows.append(
            {
                "situation": situation,
                "replications": len(frame),
                "mean_true_ate": float(frame["true_ate"].mean()),
                "mean_ols_estimate": float(frame["ols_estimate"].mean()),
                "ols_bias": float(ols_error.mean()),
                "ols_rmse": float(np.sqrt(np.mean(np.square(ols_error)))),
                "ols_95ci_coverage": float(frame["ols_ci_covers_true_ate"].mean()),
                "mean_dml_estimate": float(frame["dml_estimate"].mean()),
                "dml_bias": float(dml_error.mean()),
                "dml_rmse": float(np.sqrt(np.mean(np.square(dml_error)))),
                "dml_95ci_coverage": float(frame["dml_ci_covers_true_ate"].mean()),
                "mean_ols_minus_dml": float(
                    (frame["ols_estimate"] - frame["dml_estimate"]).mean()
                ),
                "median_ols_dml_relative_deviation": (
                    float(deviation.median()) if len(deviation) else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _report(summary: pd.DataFrame, config: dict[str, Any]) -> str:
    lines = [
        "# Conventional constant-effect OLS and DML-ATE comparison",
        "",
        "This diagnostic precedes CATE morphology analysis. Each of the nine",
        "one-dimensional HTE generators is evaluated separately. The conventional",
        "estimator is the single treatment coefficient from `Y ~ 1 + T + X`, with",
        "all 12 pre-treatment characteristics and an HC3 standard error. It is called",
        "the conventional benchmark because each firm contributes one treatment and",
        "one outcome observation. HC3 standard errors quantify its uncertainty.",
        "",
        "The comparison reports bias, RMSE, and interval coverage for both estimators.",
        "Simulation truth is opened only in the evaluator report for these calculations.",
        "",
        f"Sample size is `{config['sample_size']}` with an 80/20 firm split; results",
        f"use the TRAIN partition and `{config['replications_per_situation']}` replications",
        "per situation.",
        "",
        "`ATE_ESTIMATOR_COMPARISON.svg` compares mean true ATE with mean OLS",
        "and DML estimates across the nine HTE forms.",
        "",
        "| HTE form | OLS bias | OLS RMSE | OLS coverage | DML bias | DML RMSE | DML coverage | Median relative OLS-DML deviation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.situation} | {row.ols_bias:.4f} | {row.ols_rmse:.4f} | "
            f"{row.ols_95ci_coverage:.1%} | {row.dml_bias:.4f} | {row.dml_rmse:.4f} | "
            f"{row.dml_95ci_coverage:.1%} | {row.median_ols_dml_relative_deviation:.1%} |"
        )
    lines.extend(
        [
            "",
            "This comparison establishes the average-effect reference used before the",
            "causal-forest and orthogonal-reconstruction stages estimate HTE morphology.",
            "",
        ]
    )
    return "\n".join(lines)


def write_estimator_comparison_figure(summary: pd.DataFrame, path: Path) -> None:
    """Write a dependency-free SVG comparing mean ATE estimates."""

    width, height = 1200, 620
    left, right, top, bottom = 90, 40, 70, 135
    values = summary[["mean_true_ate", "mean_ols_estimate", "mean_dml_estimate"]]
    lower = float(min(values.min().min(), 0.0))
    upper = float(max(values.max().max(), 0.0))
    padding = max((upper - lower) * 0.08, 0.1)
    lower -= padding
    upper += padding
    plot_height = height - top - bottom
    plot_width = width - left - right

    def y_position(value: float) -> float:
        return top + (upper - value) / (upper - lower) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="90" y="34" font-family="Arial" font-size="23" font-weight="700" fill="#202124">Average-effect estimators across one-dimensional HTE forms</text>',
    ]
    for tick in np.linspace(lower, upper, 6):
        y_tick = y_position(float(tick))
        svg.append(
            f'<line x1="{left}" y1="{y_tick:.2f}" x2="{width-right}" y2="{y_tick:.2f}" stroke="#e5e7eb"/>'
            f'<text x="{left-10}" y="{y_tick+5:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{tick:.2f}</text>'
        )
    zero_y = y_position(0.0)
    svg.append(
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" y2="{zero_y:.2f}" stroke="#374151" stroke-width="1.4"/>'
    )
    fields = (
        ("mean_true_ate", -13, "#202124"),
        ("mean_ols_estimate", 0, "#c96c3b"),
        ("mean_dml_estimate", 13, "#1f7a7a"),
    )
    step = plot_width / len(summary)
    for index, row in enumerate(summary.itertuples(index=False)):
        center = left + (index + 0.5) * step
        for field, offset, color in fields:
            value = float(getattr(row, field))
            svg.append(
                f'<circle cx="{center+offset:.2f}" cy="{y_position(value):.2f}" r="5.5" fill="{color}"/>'
            )
        svg.append(
            f'<text x="{center:.2f}" y="{height-bottom+28}" text-anchor="middle" font-family="Arial" font-size="10" fill="#374151">{row.situation}</text>'
        )
    legend = (
        ("True ATE", "#202124"),
        ("Constant-effect OLS", "#c96c3b"),
        ("DML-ATE", "#1f7a7a"),
    )
    for index, (label, color) in enumerate(legend):
        x_value = 750 + index * 150
        svg.append(
            f'<circle cx="{x_value}" cy="48" r="5.5" fill="{color}"/>'
            f'<text x="{x_value+10}" y="53" font-family="Arial" font-size="12" fill="#374151">{label}</text>'
        )
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def run_ate_estimator_comparison(
    *,
    config_path: str | Path = "configs/ate_estimator_comparison_v1.json",
    output: str | Path = "artifacts/ate_estimator_comparison",
    root: Path | None = None,
) -> dict[str, Any]:
    """Run the nine-form comparison without fitting a causal forest."""

    root = repository_root(root)
    config_path = (root / config_path).resolve() if not Path(config_path).is_absolute() else Path(config_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = _read_json(config_path)
    situations = tuple(config["situations"])
    if situations != NINE_SITUATIONS:
        raise ValueError("ATE comparison must contain the nine situations in canonical order")

    product, method, _, source, base_parameters, inputs = _contract(root)
    if int(method["n"]) != int(config["sample_size"]):
        raise ValueError("comparison sample size must match the frozen main workflow")
    backbone_path = (
        root
        / "execution"
        / "phase2_real_backbone_v1"
        / "headline_one_anchor_per_firm_L3.pkl.gz"
    )
    backbone = product.V2.add_rank_columns(
        product.V2.v2.load_frozen_backbone(root, source, inputs["source_config"]),
        source["features"],
    )
    seed_sequence = np.random.SeedSequence(int(config["master_seed"]))
    children = seed_sequence.spawn(len(situations) * int(config["replications_per_situation"]))
    rows: list[dict[str, Any]] = []
    child_index = 0
    for situation in situations:
        for replication in range(int(config["replications_per_situation"])):
            master_seed = int(children[child_index].generate_state(1)[0])
            child_index += 1
            lineage = product.BASE.lineage(master_seed, 0)
            data = product.build_data(
                backbone, source, method, situation, lineage, base_parameters
            )
            train, _ = product.V2.v2.split_once(
                data["T"],
                data["unit_id"],
                float(method["test_size"]),
                lineage["split_seed"],
            )
            dml = product.estimate_ate(
                data["X"][train],
                data["Y"][train],
                data["T"][train],
                lineage["nuisance_seed"],
                method,
            )
            conventional = conventional_ate_diagnostics(
                data["X"][train],
                data["Y"][train],
                data["T"][train],
                float(dml["ate_dml"]),
            )
            true_ate = float(np.mean(data["tau_true"][train]))
            rows.append(
                {
                    "situation": situation,
                    "replication": replication,
                    "master_seed": master_seed,
                    "train_n": len(train),
                    "true_ate": true_ate,
                    "ols_estimate": float(conventional["adjusted_ols"]),
                    "ols_standard_error_hc3": float(
                        conventional["adjusted_ols_standard_error_hc3"]
                    ),
                    "ols_ci_low": float(conventional["adjusted_ols_ci_low"]),
                    "ols_ci_high": float(conventional["adjusted_ols_ci_high"]),
                    "ols_ci_covers_true_ate": bool(
                        conventional["adjusted_ols_ci_low"]
                        <= true_ate
                        <= conventional["adjusted_ols_ci_high"]
                    ),
                    "dml_estimate": float(dml["ate_dml"]),
                    "dml_standard_error": float(dml["ate_standard_error"]),
                    "dml_ci_low": float(dml["ate_ci_low"]),
                    "dml_ci_high": float(dml["ate_ci_high"]),
                    "dml_ci_covers_true_ate": bool(
                        dml["ate_ci_low"] <= true_ate <= dml["ate_ci_high"]
                    ),
                    "ols_minus_dml": float(conventional["adjusted_ols_minus_dml"]),
                    "ols_dml_relative_deviation": conventional[
                        "adjusted_ols_dml_relative_deviation"
                    ],
                }
            )

    long = pd.DataFrame(rows)
    summary = summarize_estimator_comparison(long)
    long_path = output / "ATE_ESTIMATOR_COMPARISON_LONG.csv"
    summary_path = output / "ATE_ESTIMATOR_COMPARISON_SUMMARY.csv"
    report_path = output / "ATE_ESTIMATOR_COMPARISON_REPORT.md"
    figure_path = output / "ATE_ESTIMATOR_COMPARISON.svg"
    long.to_csv(long_path, index=False)
    summary.to_csv(summary_path, index=False)
    report_path.write_text(_report(summary, config), encoding="utf-8")
    write_estimator_comparison_figure(summary, figure_path)
    manifest = {
        "schema_version": "orgcate-ate-estimator-comparison-manifest-v1",
        "input_backbone": str(backbone_path.relative_to(root)).replace("\\", "/"),
        "input_backbone_sha256": sha256_file(backbone_path),
        "config": str(config_path.relative_to(root)).replace("\\", "/"),
        "config_sha256": _config_hash(config_path),
        "situations": list(situations),
        "replications_per_situation": int(config["replications_per_situation"]),
        "total_replications": len(long),
        "causal_forests_fitted": 0,
        "package_versions": package_versions(),
        "outputs": {
            path.name: sha256_file(path)
            for path in (long_path, summary_path, report_path, figure_path)
        },
    }
    _write_json(output / "RUN_MANIFEST.json", manifest)
    return {"summary": summary.to_dict(orient="records"), "manifest": manifest}
