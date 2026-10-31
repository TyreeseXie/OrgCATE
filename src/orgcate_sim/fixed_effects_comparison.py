"""Add a panel fixed-effects benchmark to frozen ATE and CATE results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .public_demo import _contract
from .release import NINE_SITUATIONS, package_versions, repository_root, sha256_file


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _versioned_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def fit_two_way_fixed_effects(
    outcome: np.ndarray,
    treated_firm: np.ndarray,
    post_period: np.ndarray,
) -> dict[str, float]:
    """Estimate a balanced-panel firm and period fixed-effects coefficient."""

    y = np.asarray(outcome, dtype=float)
    treated = np.asarray(treated_firm, dtype=float).reshape(-1)
    post = np.asarray(post_period, dtype=float).reshape(-1)
    if y.ndim != 2:
        raise ValueError("outcome must be a firm-by-period matrix")
    if y.shape != (len(treated), len(post)):
        raise ValueError("outcome, firm treatment, and period indicators disagree")
    treatment = treated[:, None] * post[None, :]
    y_within = y - y.mean(axis=1, keepdims=True) - y.mean(axis=0, keepdims=True) + y.mean()
    treatment_within = (
        treatment
        - treatment.mean(axis=1, keepdims=True)
        - treatment.mean(axis=0, keepdims=True)
        + treatment.mean()
    )
    denominator = float(np.sum(np.square(treatment_within)))
    if denominator <= 1e-12:
        raise ValueError("fixed-effects treatment has no within-panel variation")
    estimate = float(np.sum(treatment_within * y_within) / denominator)
    residual = y_within - estimate * treatment_within
    firm_scores = np.sum(treatment_within * residual, axis=1)
    firms, periods = y.shape
    observations = firms * periods
    residual_df = observations - firms - periods
    if firms <= 1 or residual_df <= 0:
        raise ValueError("panel is too small for clustered uncertainty")
    cr1 = firms / (firms - 1) * (observations - 1) / residual_df
    variance = cr1 * float(np.sum(np.square(firm_scores))) / denominator**2
    standard_error = float(np.sqrt(max(variance, 0.0)))
    critical = 1.959963984540054
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_low": estimate - critical * standard_error,
        "ci_high": estimate + critical * standard_error,
    }


def build_auxiliary_panel(
    firm_effect: np.ndarray,
    treatment: np.ndarray,
    cate: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand frozen firm-level truth into a six-period outcome panel."""

    alpha = np.asarray(firm_effect, dtype=float).reshape(-1)
    treated = np.asarray(treatment, dtype=float).reshape(-1)
    tau = np.asarray(cate, dtype=float).reshape(-1)
    if not (len(alpha) == len(treated) == len(tau)):
        raise ValueError("firm effect, treatment, and CATE lengths disagree")
    periods = np.asarray(config["relative_periods"], dtype=int)
    period_effects = np.asarray(config["period_effects"], dtype=float)
    if len(periods) != len(period_effects):
        raise ValueError("relative periods and period effects must have equal length")
    post = periods >= int(config["post_period_starts_at"])
    rng = np.random.default_rng(int(seed))
    sigma = float(config["panel_noise_standard_deviation"])
    rho = float(config["panel_noise_ar1"])
    if sigma < 0 or not 0 <= rho < 1:
        raise ValueError("panel noise parameters are outside their valid range")
    innovation = rng.normal(size=(len(alpha), len(periods)))
    noise = np.empty_like(innovation)
    noise[:, 0] = sigma * innovation[:, 0]
    innovation_scale = sigma * np.sqrt(1.0 - rho**2)
    for column in range(1, len(periods)):
        noise[:, column] = (
            rho * noise[:, column - 1] + innovation_scale * innovation[:, column]
        )
    outcome = (
        alpha[:, None]
        + period_effects[None, :]
        + treated[:, None] * post[None, :] * tau[:, None]
        + noise
    )
    return outcome, post


def summarize_fixed_effects_comparison(long: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the four requested quantities and FE operating characteristics."""

    required = {
        "situation",
        "true_ate",
        "true_ate_train",
        "true_att_train",
        "true_ate_test",
        "fixed_effects_estimate",
        "fixed_effects_ci_covers_ate",
        "fixed_effects_ci_covers_att",
        "dml_ate",
        "cate_test_mean",
    }
    missing = required.difference(long.columns)
    if missing:
        raise ValueError(f"fixed-effects comparison is missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for situation, frame in long.groupby("situation", sort=False):
        fe_ate_error = frame["fixed_effects_estimate"] - frame["true_ate_train"]
        fe_att_error = frame["fixed_effects_estimate"] - frame["true_att_train"]
        dml_error = frame["dml_ate"] - frame["true_ate"]
        cate_error = frame["cate_test_mean"] - frame["true_ate_test"]
        rows.append(
            {
                "situation": situation,
                "replications": len(frame),
                "mean_true_ate": float(frame["true_ate"].mean()),
                "mean_true_att_train": float(frame["true_att_train"].mean()),
                "mean_fixed_effects_estimate": float(
                    frame["fixed_effects_estimate"].mean()
                ),
                "fixed_effects_bias_vs_ate": float(fe_ate_error.mean()),
                "fixed_effects_rmse_vs_ate": float(
                    np.sqrt(np.mean(np.square(fe_ate_error)))
                ),
                "fixed_effects_95ci_coverage_ate": float(
                    frame["fixed_effects_ci_covers_ate"].mean()
                ),
                "fixed_effects_bias_vs_att": float(fe_att_error.mean()),
                "fixed_effects_rmse_vs_att": float(
                    np.sqrt(np.mean(np.square(fe_att_error)))
                ),
                "fixed_effects_95ci_coverage_att": float(
                    frame["fixed_effects_ci_covers_att"].mean()
                ),
                "mean_dml_ate": float(frame["dml_ate"].mean()),
                "dml_bias_vs_ate": float(dml_error.mean()),
                "dml_rmse_vs_ate": float(np.sqrt(np.mean(np.square(dml_error)))),
                "mean_cate_test": float(frame["cate_test_mean"].mean()),
                "cate_bias_vs_test_ate": float(cate_error.mean()),
                "cate_rmse_vs_test_ate": float(
                    np.sqrt(np.mean(np.square(cate_error)))
                ),
            }
        )
    return pd.DataFrame(rows)


def write_fixed_effects_figure(summary: pd.DataFrame, path: Path) -> None:
    """Write an SVG with True ATE, fixed effects, DML-ATE, and mean CATE."""

    width, height = 1240, 620
    left, right, top, bottom = 90, 40, 70, 135
    fields = (
        ("mean_true_ate", -21, "#202124", "True ATE"),
        ("mean_fixed_effects_estimate", -7, "#c96c3b", "Firm + period FE"),
        ("mean_dml_ate", 7, "#1f7a7a", "DML-ATE"),
        ("mean_cate_test", 21, "#4f6fb3", "CATE"),
    )
    values = summary[[field[0] for field in fields]]
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
        '<text x="90" y="34" font-family="Arial" font-size="23" font-weight="700" fill="#202124">Average-effect estimates with panel fixed effects</text>',
    ]
    for tick in np.linspace(lower, upper, 6):
        y_tick = y_position(float(tick))
        svg.append(
            f'<line x1="{left}" y1="{y_tick:.2f}" x2="{width-right}" y2="{y_tick:.2f}" stroke="#e5e7eb"/>'
            f'<text x="{left-10}" y="{y_tick+5:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{tick:.2f}</text>'
        )
    svg.append(
        f'<line x1="{left}" y1="{y_position(0.0):.2f}" x2="{width-right}" y2="{y_position(0.0):.2f}" stroke="#374151" stroke-width="1.4"/>'
    )
    step = plot_width / len(summary)
    for index, row in enumerate(summary.itertuples(index=False)):
        center = left + (index + 0.5) * step
        for field, offset, color, _ in fields:
            value = float(getattr(row, field))
            svg.append(
                f'<circle cx="{center+offset:.2f}" cy="{y_position(value):.2f}" r="5.5" fill="{color}"/>'
            )
        svg.append(
            f'<text x="{center:.2f}" y="{height-bottom+28}" text-anchor="middle" font-family="Arial" font-size="10" fill="#374151">{row.situation}</text>'
        )
    legend_x = (600, 750, 915, 1045)
    for x_value, (_, _, color, label) in zip(legend_x, fields, strict=True):
        svg.append(
            f'<circle cx="{x_value}" cy="48" r="5.5" fill="{color}"/>'
            f'<text x="{x_value+10}" y="53" font-family="Arial" font-size="12" fill="#374151">{label}</text>'
        )
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def _report(summary: pd.DataFrame, config: dict[str, Any]) -> str:
    lines = [
        "# Panel fixed-effects comparison",
        "",
        "The fixed-effects benchmark expands each frozen firm-level replication",
        "into six relative periods. Firm and period effects are absorbed, and the",
        "treatment coefficient is estimated from `T_i x Post_t` with firm-clustered",
        "CR1 uncertainty. The existing True ATE, DML-ATE, and CATE are read",
        "from the frozen confirmation evidence.",
        "",
        f"The comparison contains `{config['replications_per_situation']}` replications",
        "for each of the nine one-dimensional HTE forms.",
        "",
        "| HTE form | True ATE | Firm + period FE | DML-ATE | CATE | FE bias vs ATE | FE bias vs ATT |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.situation} | {row.mean_true_ate:.4f} | "
            f"{row.mean_fixed_effects_estimate:.4f} | {row.mean_dml_ate:.4f} | "
            f"{row.mean_cate_test:.4f} | {row.fixed_effects_bias_vs_ate:.4f} | "
            f"{row.fixed_effects_bias_vs_att:.4f} |"
        )
    lines.extend(
        [
            "",
            "The ATT columns identify the estimand represented by the conventional",
            "fixed-effects coefficient when treatment selection and effect",
            "heterogeneity are associated. The DML and CATE values are unchanged",
            "frozen outputs from the unified morphology workflow.",
            "",
        ]
    )
    return "\n".join(lines)


def run_fixed_effects_comparison(
    *,
    config_path: str | Path = "configs/fixed_effects_comparison_v1.json",
    output: str | Path = "artifacts/fixed_effects_comparison",
    root: Path | None = None,
) -> dict[str, Any]:
    """Calculate only the auxiliary fixed-effects estimate for frozen cells."""

    root = repository_root(root)
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = _read_json(config_path)
    evidence_path = root / str(config["frozen_estimate_input"])
    if config.get("frozen_estimate_input_hash_mode") != "normalized_text_lf":
        raise ValueError("unsupported frozen ATE/CATE evidence hash mode")
    observed_hash = _versioned_hash(evidence_path)
    if observed_hash != str(config["frozen_estimate_input_sha256"]):
        raise ValueError("frozen ATE/CATE evidence hash mismatch")
    evidence = pd.read_csv(evidence_path)
    if tuple(evidence["dgp"].drop_duplicates()) != NINE_SITUATIONS:
        raise ValueError("frozen evidence does not contain the nine situations in order")
    counts = evidence.groupby("dgp", sort=False).size()
    expected_replications = int(config["replications_per_situation"])
    if not (counts == expected_replications).all():
        raise ValueError("frozen evidence replication counts changed")

    product, method, _, source, base_parameters, inputs = _contract(root)
    if int(method["n"]) != int(config["sample_size"]):
        raise ValueError("fixed-effects comparison must match the frozen sample size")
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
    rows: list[dict[str, Any]] = []
    namespace = int(config["panel_seed_namespace"])
    for record in evidence.itertuples(index=False):
        master_seed = int(record.master_seed)
        lineage = product.BASE.lineage(master_seed, 0)
        data = product.build_data(
            backbone, source, method, str(record.dgp), lineage, base_parameters
        )
        train, test = product.V2.v2.split_once(
            data["T"],
            data["unit_id"],
            float(method["test_size"]),
            lineage["split_seed"],
        )
        reconstructed_true_ate = float(np.mean(data["tau_true"]))
        if not np.isclose(reconstructed_true_ate, float(record.true_ate), atol=1e-9):
            raise ValueError(
                f"frozen DGP reconstruction changed for {record.dgp} replication {record.replication}"
            )
        panel_seed = int(
            np.random.SeedSequence([master_seed, namespace]).generate_state(1)[0]
        )
        panel_y, post = build_auxiliary_panel(
            data["y0_true"][train],
            data["T"][train],
            data["tau_true"][train],
            config,
            panel_seed,
        )
        fixed_effects = fit_two_way_fixed_effects(
            panel_y, data["T"][train], post
        )
        true_ate_train = float(np.mean(data["tau_true"][train]))
        true_ate_test = float(np.mean(data["tau_true"][test]))
        treated_train = data["T"][train] == 1
        true_att_train = float(np.mean(data["tau_true"][train][treated_train]))
        rows.append(
            {
                "situation": str(record.dgp),
                "replication": int(record.replication),
                "master_seed": master_seed,
                "panel_seed": panel_seed,
                "train_firms": len(train),
                "test_firms": len(test),
                "panel_periods": len(post),
                "true_ate": float(record.true_ate),
                "true_ate_train": true_ate_train,
                "true_att_train": true_att_train,
                "true_ate_test": true_ate_test,
                "fixed_effects_estimate": fixed_effects["estimate"],
                "fixed_effects_standard_error": fixed_effects["standard_error"],
                "fixed_effects_ci_low": fixed_effects["ci_low"],
                "fixed_effects_ci_high": fixed_effects["ci_high"],
                "fixed_effects_ci_covers_ate": bool(
                    fixed_effects["ci_low"]
                    <= true_ate_train
                    <= fixed_effects["ci_high"]
                ),
                "fixed_effects_ci_covers_att": bool(
                    fixed_effects["ci_low"]
                    <= true_att_train
                    <= fixed_effects["ci_high"]
                ),
                "dml_ate": float(record.train_ate_dml),
                "dml_standard_error": float(record.train_ate_standard_error),
                "cate_test_mean": float(record.cate_test_mean),
                "practical_state": str(record.practical_state),
                "predicted_label": str(record.predicted_label),
            }
        )

    long = pd.DataFrame(rows)
    summary = summarize_fixed_effects_comparison(long)
    long_path = output / "PANEL_FIXED_EFFECTS_COMPARISON_LONG.csv"
    summary_path = output / "PANEL_FIXED_EFFECTS_COMPARISON_SUMMARY.csv"
    report_path = output / "PANEL_FIXED_EFFECTS_COMPARISON_REPORT.md"
    figure_path = output / "PANEL_FIXED_EFFECTS_COMPARISON.svg"
    long.to_csv(long_path, index=False)
    summary.to_csv(summary_path, index=False)
    report_path.write_text(_report(summary, config), encoding="utf-8")
    write_fixed_effects_figure(summary, figure_path)
    manifest = {
        "schema_version": "orgcate-fixed-effects-comparison-manifest-v1",
        "config": str(config_path.relative_to(root)).replace("\\", "/"),
        "config_sha256": _versioned_hash(config_path),
        "frozen_estimate_input": str(evidence_path.relative_to(root)).replace(
            "\\", "/"
        ),
        "frozen_estimate_input_hash_mode": "normalized_text_lf",
        "frozen_estimate_input_sha256": observed_hash,
        "input_backbone": str(backbone_path.relative_to(root)).replace("\\", "/"),
        "input_backbone_sha256": sha256_file(backbone_path),
        "situations": list(NINE_SITUATIONS),
        "replications_per_situation": expected_replications,
        "total_replications": len(long),
        "new_causal_forests_fitted": 0,
        "frozen_dml_and_cate_rows_reused": len(evidence),
        "new_quantity": "firm and relative-period fixed-effects coefficient",
        "package_versions": package_versions(),
        "outputs": {
            path.name: sha256_file(path)
            for path in (long_path, summary_path, report_path, figure_path)
        },
    }
    _write_json(output / "RUN_MANIFEST.json", manifest)
    return {"summary": summary.to_dict(orient="records"), "manifest": manifest}
