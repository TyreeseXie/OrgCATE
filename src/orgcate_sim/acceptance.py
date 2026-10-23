"""End-to-end acceptance analysis for the bundled real covariate backbone."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .public_demo import run_public_demo
from .release import repository_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_state(root: Path) -> dict[str, Any]:
    commit = os.environ.get("GITHUB_SHA")
    if not commit:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        commit = completed.stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "working_tree_dirty": bool(status.strip())}


def _scale(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    if maximum - minimum <= 1e-12:
        return np.full(len(values), (lower + upper) / 2)
    return lower + (values - minimum) * (upper - lower) / (maximum - minimum)


def _curve_figure(curve: pd.DataFrame, path: Path) -> None:
    width, height = 920, 560
    left, right, top, bottom = 90, 30, 80, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    x = curve["u"].to_numpy(float)
    estimate = curve["estimated_cate"].to_numpy(float)
    truth = curve["true_cate"].to_numpy(float)
    ate = float(curve["dml_ate"].iloc[0])
    y_all = np.concatenate([estimate, truth, np.array([ate])])
    x_px = _scale(x, left, left + plot_width)
    y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
    padding = max(0.05, 0.08 * (y_max - y_min))
    y_min, y_max = y_min - padding, y_max + padding

    def y_pixel(value: np.ndarray | float) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        return top + (y_max - array) * plot_height / (y_max - y_min)

    estimated_points = " ".join(
        f"{px:.2f},{py:.2f}" for px, py in zip(x_px, y_pixel(estimate), strict=True)
    )
    truth_points = " ".join(
        f"{px:.2f},{py:.2f}" for px, py in zip(x_px, y_pixel(truth), strict=True)
    )
    grid = []
    for value in np.linspace(y_min, y_max, 6):
        py = float(y_pixel(value))
        grid.append(
            f'<line x1="{left}" y1="{py:.2f}" x2="{left + plot_width}" '
            f'y2="{py:.2f}" stroke="#d8dde3" stroke-width="1"/>'
            f'<text x="{left - 12}" y="{py + 5:.2f}" text-anchor="end" '
            f'font-size="13" fill="#4b5563">{value:.2f}</text>'
        )
    ate_y = float(y_pixel(ate))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="30" font-family="Arial" font-size="22" font-weight="700" fill="#202124">CATE reconstruction on the uploaded firm backbone</text>
{''.join(grid)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#374151" stroke-width="1.5"/>
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#374151" stroke-width="1.5"/>
<line x1="{left}" y1="{ate_y:.2f}" x2="{left + plot_width}" y2="{ate_y:.2f}" stroke="#6b7280" stroke-width="2" stroke-dasharray="8 6"/>
<polyline points="{truth_points}" fill="none" stroke="#ca6b3c" stroke-width="3"/>
<polyline points="{estimated_points}" fill="none" stroke="#1f7a7a" stroke-width="3"/>
<text x="{left + plot_width / 2}" y="{height - 28}" text-anchor="middle" font-family="Arial" font-size="15" fill="#374151">Relative firm size rank</text>
<text x="24" y="{top + plot_height / 2}" transform="rotate(-90 24 {top + plot_height / 2})" text-anchor="middle" font-family="Arial" font-size="15" fill="#374151">Treatment effect</text>
<line x1="{left + 475}" y1="55" x2="{left + 510}" y2="55" stroke="#1f7a7a" stroke-width="3"/><text x="{left + 518}" y="60" font-family="Arial" font-size="13">Estimated CATE</text>
<line x1="{left + 620}" y1="55" x2="{left + 655}" y2="55" stroke="#ca6b3c" stroke-width="3"/><text x="{left + 663}" y="60" font-family="Arial" font-size="13">Simulation truth</text>
<text x="{left + plot_width - 4}" y="{ate_y - 7:.2f}" text-anchor="end" font-family="Arial" font-size="13" fill="#4b5563">DML ATE = {ate:.3f}</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def _importance_figure(ranking: pd.DataFrame, path: Path) -> None:
    frame = ranking.sort_values("importance", ascending=True).reset_index(drop=True)
    width, height = 980, 690
    left, right, top, bottom = 265, 50, 60, 55
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(float(frame["importance"].max()), 1e-12)
    bar_height = plot_height / len(frame) * 0.62
    rows = []
    for index, row in frame.iterrows():
        center = top + (index + 0.5) * plot_height / len(frame)
        bar_width = float(row["importance"]) / maximum * plot_width
        color = "#1f7a7a" if row["rank"] == 1 else "#7aa6a6"
        rows.append(
            f'<text x="{left - 12}" y="{center + 5:.2f}" text-anchor="end" '
            f'font-family="Arial" font-size="13" fill="#30343b">{escape(str(row["feature"]))}</text>'
            f'<rect x="{left}" y="{center - bar_height / 2:.2f}" width="{bar_width:.2f}" '
            f'height="{bar_height:.2f}" fill="{color}"/>'
            f'<text x="{left + bar_width + 8:.2f}" y="{center + 5:.2f}" '
            f'font-family="Arial" font-size="12" fill="#4b5563">{float(row["importance"]):.3f}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="32" font-family="Arial" font-size="22" font-weight="700" fill="#202124">Causal-forest feature importance</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#374151" stroke-width="1.5"/>
{''.join(rows)}
<text x="{left + plot_width / 2}" y="{height - 17}" text-anchor="middle" font-family="Arial" font-size="14" fill="#374151">Importance (ranking information, not moderator confirmation)</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def _report_text(
    *,
    config: dict[str, Any],
    input_path: str,
    input_rows: int,
    input_columns: int,
    statistics: dict[str, Any],
) -> str:
    return f"""# OrgCATE data-to-results acceptance report

## What was analyzed

The analysis read `{input_path}`, the uploaded and hash-verified L=3 firm
covariate backbone ({input_rows:,} rows and {input_columns} columns). A fixed
sample of {statistics['analysis_sample_n']:,} firms supplied the observed
pre-treatment covariates. Treatment, potential outcomes, observed outcomes and
the true CATE were generated during this run with master seed
`{config['analysis']['master_seed']}`. No pre-existing result table or figure
was used as an analysis input.

The frozen workflow estimated a five-fold cross-fitted DML/AIPW ATE, fitted one
joint 12-characteristic honest `CausalForestDML`, reconstructed the CATE shape
on TRAIN, and assessed the selected candidate on untouched TEST.

## Results

| Quantity | Recomputed value |
|---|---:|
| DML ATE | {statistics['dml_ate']:.4f} |
| DML ATE 95% CI | [{statistics['dml_ate_ci_low']:.4f}, {statistics['dml_ate_ci_high']:.4f}] |
| Simulated true ATE | {statistics['true_ate']:.4f} |
| ATE error | {statistics['ate_error']:.4f} |
| Estimated morphology | `{statistics['estimated_morphology']}` |
| Practical state | `{statistics['practical_state']}` |
| TEST CATE rank correlation | {statistics['cate_rank_correlation']:.4f} |
| TEST CATE RMSE | {statistics['cate_rmse']:.4f} |
| ATE-CATE standardized gap | {statistics['ate_cate_standardized_gap']:.4f} |
| Focal-characteristic rank | {statistics['focal_characteristic_rank']} of 12 |

Shape summary: {statistics['shape_interpretation']}

## Figures

- `figures/cate_curve.svg` compares the recomputed CATE curve with simulation
  truth and the DML ATE.
- `figures/feature_importance.svg` displays the 12-variable forest ranking.

## Data boundary

The downloadable artifact contains aggregate statistics, curves, diagnostics,
figures and reports only. It excludes firm identifiers, firm-level covariates,
treatment assignments, outcomes, potential outcomes and unit-level oracle
truth. These results demonstrate executable semi-synthetic reproduction for
one fixed scenario; they are not empirical causal estimates for real firms.
"""


def run_data_to_results_acceptance(
    *,
    config_path: str | Path,
    output: str | Path,
    root: Path | None = None,
) -> dict[str, Any]:
    """Run and package one fresh semi-synthetic acceptance analysis."""

    started = time.perf_counter()
    root = repository_root(root)
    config_path = Path(config_path).resolve()
    output = Path(output).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "orgcate-data-to-results-acceptance-v1":
        raise ValueError("wrong acceptance schema_version")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"acceptance output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    relative_input = Path(config["input"]["path"])
    input_path = (root / relative_input).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    observed_input_hash = _sha256(input_path)
    if observed_input_hash != config["input"]["sha256"]:
        raise ValueError("bundled backbone SHA256 does not match acceptance config")
    input_frame = pd.read_pickle(input_path, compression="gzip")
    if len(input_frame) != int(config["input"]["expected_rows"]):
        raise ValueError("bundled backbone row count does not match acceptance config")
    if len(input_frame.columns) != int(config["input"]["expected_columns"]):
        raise ValueError("bundled backbone column count does not match acceptance config")

    scenario = str(config["analysis"]["scenario"])
    seed = int(config["analysis"]["master_seed"])
    result = run_public_demo(
        scenario=scenario,
        output=output,
        seed=seed,
        use_bundled_backbone=True,
        save_row_level=False,
        root=root,
    )
    practical = result["practical"]
    evaluation = result["evaluation"]
    run_manifest = result["manifest"]

    curve = pd.read_csv(output / "practical" / "cate_curve.csv")
    truth = pd.read_csv(output / "evaluator" / "truth_curve.csv")
    curve = curve.merge(truth, on="u", how="inner", validate="one_to_one")
    curve = curve.rename(
        columns={"estimate_curve": "estimated_cate", "truth_curve": "true_cate"}
    )
    curve["dml_ate"] = float(practical["ate"]["estimate"])
    curve_path = output / "tables" / "cate_curve_with_ate.csv"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(curve_path, index=False)

    ranking = pd.read_csv(output / "practical" / "feature_ranking.csv")
    statistics = {
        "schema_version": "orgcate-acceptance-statistics-v1",
        "analysis_sample_n": int(run_manifest["analysis_rows"]),
        "train_n": int(run_manifest["train_rows"]),
        "test_n": int(run_manifest["test_rows"]),
        "dml_ate": float(practical["ate"]["estimate"]),
        "dml_ate_standard_error": float(practical["ate"]["standard_error"]),
        "dml_ate_ci_low": float(practical["ate"]["ci_low"]),
        "dml_ate_ci_high": float(practical["ate"]["ci_high"]),
        "true_ate": float(evaluation["true_ate"]),
        "ate_error": float(evaluation["ate_error"]),
        "estimated_morphology": str(practical["morphology"]["label"]),
        "practical_state": str(practical["practical_state"]),
        "cate_rank_correlation": float(evaluation["curve_rank_correlation"]),
        "cate_rmse": float(evaluation["curve_rmse"]),
        "ate_cate_standardized_gap": float(
            practical["ate_cate_coherence"]["standardized_absolute_gap"]
        ),
        "ate_cate_coherence_passed": bool(practical["ate_cate_coherence"]["passed"]),
        "test_confirmation_rate": float(
            practical["confirmation"]["test_confirmation_rate"]
        ),
        "focal_characteristic": str(
            practical["moderator"]["focal_characteristic"]
        ),
        "focal_characteristic_rank": int(practical["moderator"]["focal_rank"]),
        "shape_interpretation": str(
            practical["morphology"]["interpretation_en"]
        ),
    }
    if statistics["analysis_sample_n"] != int(config["analysis"]["sample_size"]):
        raise AssertionError("actual analysis sample size differs from acceptance config")
    statistics_path = output / "statistics" / "summary_statistics.json"
    _write_json(statistics_path, statistics)

    diagnostics = {
        "schema_version": "orgcate-acceptance-diagnostics-v1",
        "input_validation": {
            "path": relative_input.as_posix(),
            "sha256": observed_input_hash,
            "bytes": input_path.stat().st_size,
            "rows_read": len(input_frame),
            "columns_read": len(input_frame.columns),
        },
        "data_roles": {
            "observed_real_data": "pre-treatment firm covariates and history",
            "generated_during_run": [
                "treatment",
                "potential_outcomes",
                "observed_outcome",
                "true_cate",
            ],
            "preexisting_results_used_as_analysis_input": False,
        },
        "eligibility": practical["eligibility"],
        "ate_cate_coherence": practical["ate_cate_coherence"],
        "forest": practical["forest"],
        "seed_lineage": practical["audit"]["seed_lineage"],
        "warnings": {"count": practical["audit"]["warning_count"]},
        "package_versions": practical["audit"]["package_versions"],
        "analyst_oracle_separated": run_manifest["analyst_oracle_separated"],
        "row_level_outputs_written": run_manifest["row_level_outputs_written"],
    }
    diagnostics_path = output / "diagnostics" / "run_diagnostics.json"
    _write_json(diagnostics_path, diagnostics)

    _curve_figure(curve, output / "figures" / "cate_curve.svg")
    _importance_figure(ranking, output / "figures" / "feature_importance.svg")

    report = _report_text(
        config=config,
        input_path=relative_input.as_posix(),
        input_rows=len(input_frame),
        input_columns=len(input_frame.columns),
        statistics=statistics,
    )
    report_path = output / "ANALYSIS_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    excluded = {"ACCEPTANCE_MANIFEST.json", "ACCEPTANCE_MANIFEST.sha256"}
    output_records = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        if relative in excluded:
            continue
        output_records.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    git_state = _git_state(root)
    manifest = {
        "schema_version": "orgcate-data-to-results-acceptance-manifest-v1",
        "status": "completed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": (
            "orgcate acceptance --config "
            f"{config_path.relative_to(root).as_posix()} --output artifacts/ci_acceptance"
        ),
        "git": git_state,
        "input": {
            "path": relative_input.as_posix(),
            "classification": "uploaded authorized real pre-treatment covariate backbone",
            "sha256": observed_input_hash,
            "bytes": input_path.stat().st_size,
            "rows_read": len(input_frame),
            "columns_read": len(input_frame.columns),
            "analysis_sample_n": int(run_manifest["analysis_rows"]),
        },
        "generated_data": {
            "treatment": "simulated",
            "potential_outcomes": "simulated",
            "observed_outcome": "simulated",
            "true_cate": "simulated",
            "row_level_generated_data_in_artifact": False,
        },
        "analysis": {
            "scenario": scenario,
            "external_split": "80% TRAIN / 20% untouched TEST",
            "ate_method": practical["ate"]["method"],
            "learner": practical["forest"]["learner"],
            "model_configuration": practical["forest"]["configuration"],
            "candidate_characteristics": practical["forest"][
                "candidate_characteristics"
            ],
            "master_seed": seed,
            "seed_lineage": practical["audit"]["seed_lineage"],
        },
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": _sha256(config_path),
        },
        "runtime_seconds": time.perf_counter() - started,
        "outputs": output_records,
        "security": {
            "contains_firm_level_rows": False,
            "contains_credentials": False,
            "contains_raw_microdata_copy": False,
        },
    }
    manifest_path = output / "ACCEPTANCE_MANIFEST.json"
    _write_json(manifest_path, manifest)
    (output / "ACCEPTANCE_MANIFEST.sha256").write_text(
        f"{_sha256(manifest_path)}  ACCEPTANCE_MANIFEST.json\n", encoding="ascii"
    )
    return {"statistics": statistics, "manifest": manifest}
