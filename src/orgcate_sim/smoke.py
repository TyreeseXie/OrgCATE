"""Generate truth-audited public synthetic smoke datasets for W0/W1/W2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_pilot_config
from .estimators import fit_causal_forest_dml
from .experiment import simulate_one
from .manifest import build_run_manifest, write_manifest
from .metrics import evaluate_cate_predictions


def write_synthetic_smoke(config_path: str | Path, output_dir: str | Path) -> pd.DataFrame:
    """Write separate analyst/oracle CSVs and evaluator-side manifests for W0/W1/W2."""
    pilot = load_pilot_config(config_path)
    expected = ("constant", "monotone", "smooth_flip")
    if tuple(pilot.worlds) != expected:
        raise ValueError(f"smoke config must declare exactly {expected}")

    root = np.random.SeedSequence(pilot.master_seed)
    child_sequences = root.spawn(len(pilot.worlds))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for morphology, child in zip(pilot.worlds, child_sequences, strict=True):
        seed = int(child.generate_state(1, dtype=np.uint32)[0])
        cfg = pilot.config_for_world(morphology)
        output = simulate_one(cfg, seed=seed)
        world_dir = destination / morphology
        world_dir.mkdir(exist_ok=True)
        analyst_path = world_dir / "analyst.csv"
        oracle_path = world_dir / "oracle.csv"
        manifest_path = world_dir / "RUN_MANIFEST.json"
        output.analyst.to_csv(analyst_path, index=False)
        output.oracle.to_csv(oracle_path, index=False)

        dgp_metadata = dict(output.metadata)
        feature_roles = dgp_metadata.pop("oracle_feature_roles")
        manifest = build_run_manifest(
            config=cfg,
            master_seed=pilot.master_seed,
            simulation_seed=seed,
            seed_spawn_key=child.spawn_key,
            output_paths={
                "analyst": analyst_path.as_posix(),
                "oracle": oracle_path.as_posix(),
                "manifest": manifest_path.as_posix(),
            },
            dgp_metadata=dgp_metadata,
            feature_roles=feature_roles,
            estimator_status="not_run_dependency_check_pending",
        )
        write_manifest(manifest_path, manifest)
        summary_rows.append(
            {
                "world": morphology,
                "seed": seed,
                "n": output.metadata["n"],
                "realized_ate": output.metadata["realized_ate"],
                "realized_treatment_rate": output.metadata["realized_treatment_rate"],
                "propensity_min": output.metadata["propensity_min"],
                "propensity_max": output.metadata["propensity_max"],
                "truth_audit_passed": output.metadata["truth_audit_passed"],
                "active_x": json.dumps(output.metadata["active_x"]),
                "realized_flip_u": output.metadata.get("realized_flip_u"),
                "config_sha256": manifest["config_sha256"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(destination / "smoke_summary.csv", index=False)
    return summary


def run_causal_forest_smoke(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    n_estimators: int = 100,
    min_samples_leaf: int = 10,
) -> pd.DataFrame:
    """Fit one multivariable forest per generated world and save evaluator outputs.

    The function consumes the already separated ``analyst.csv`` and ``oracle.csv``
    files.  ``predictions.csv`` contains only unit IDs, split labels, and fitted
    CATE values.  Oracle-dependent metrics are written separately and are never
    passed to the estimator.
    """
    pilot = load_pilot_config(config_path)
    expected = ("constant", "monotone", "smooth_flip")
    if tuple(pilot.worlds) != expected:
        raise ValueError(f"smoke config must declare exactly {expected}")
    destination = Path(output_dir)
    rows: list[dict[str, object]] = []

    for morphology in expected:
        world_dir = destination / morphology
        analyst = pd.read_csv(world_dir / "analyst.csv")
        oracle = pd.read_csv(world_dir / "oracle.csv")
        x_cols = [column for column in analyst.columns if column.startswith("X_")]
        w_cols = [column for column in analyst.columns if column.startswith("W_")]
        manifest_path = world_dir / "RUN_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        simulation_seed = int(manifest["simulation_seed"])
        estimator_seed = simulation_seed + int(pilot.estimator["random_state_offset"])
        estimator, cate_hat = fit_causal_forest_dml(
            analyst,
            x_cols=x_cols,
            w_cols=w_cols,
            random_state=estimator_seed,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
        )
        prediction_path = world_dir / "predictions.csv"
        metrics_path = world_dir / "estimator_metrics.json"
        pd.DataFrame(
            {
                "unit_id": analyst["unit_id"],
                "split_label": analyst["split_label"],
                "cate_hat": cate_hat,
            }
        ).to_csv(prediction_path, index=False)
        metrics = evaluate_cate_predictions(analyst, oracle, cate_hat)
        metrics.update(
            {
                "world": morphology,
                "n_estimators": int(n_estimators),
                "min_samples_leaf": int(min_samples_leaf),
                "x_columns": x_cols,
                "w_columns": w_cols,
                "oracle_evaluation_only": True,
            }
        )
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        manifest.update(
            {
                "estimator_status": "completed",
                "estimator": {
                    "name": type(estimator).__name__,
                    "n_estimators": int(n_estimators),
                    "min_samples_leaf": int(min_samples_leaf),
                    "random_state": estimator_seed,
                    "x_columns": x_cols,
                    "w_columns": w_cols,
                },
                "output_paths": {
                    **manifest.get("output_paths", {}),
                    "predictions": prediction_path.as_posix(),
                    "estimator_metrics": metrics_path.as_posix(),
                },
            }
        )
        write_manifest(manifest_path, manifest)
        rows.append({"world": morphology, **metrics})

    summary = pd.DataFrame(rows)
    summary.to_csv(destination / "estimator_smoke_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--output-dir", default="data/synthetic_public/w0_w1_w2_smoke")
    parser.add_argument("--fit-estimator", action="store_true", help="also fit one CausalForestDML per world")
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()
    summary = write_synthetic_smoke(args.config, args.output_dir)
    print(summary.to_string(index=False))
    if args.fit_estimator:
        estimator_summary = run_causal_forest_smoke(
            args.config,
            args.output_dir,
            n_estimators=args.n_estimators,
        )
        print(estimator_summary.to_string(index=False))


if __name__ == "__main__":
    main()
