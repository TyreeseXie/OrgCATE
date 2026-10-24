"""Public synthetic demonstration using the frozen OrgCATE product path."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .public_contract import (
    conventional_ate_diagnostics,
    evaluate_ate_cate_coherence,
    evaluate_design_eligibility,
    property_support,
    summarize_shape_quantities_en,
    theory_audit_template,
    three_layer_taxonomy,
    validate_theory_audit,
)
from .release import (
    NINE_SITUATIONS,
    package_versions,
    repository_root,
    validate_practical_result,
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rank(method="average", pct=True)
        .to_numpy(float)
    )


def _standardize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    standard_deviation = float(array.std())
    if standard_deviation <= 1e-12:
        raise ValueError("cannot standardize a constant synthetic feature")
    return (array - float(array.mean())) / standard_deviation


def make_public_synthetic_backbone(
    *, rows: int = 6000, seed: int = 20261023
) -> pd.DataFrame:
    """Create a redistributable 12-characteristic firm-like covariate backbone."""

    if rows < 3000:
        raise ValueError("the frozen product requires at least 3000 unique units")
    rng = np.random.default_rng(int(seed))
    common = rng.normal(size=(rows, 5))
    idiosyncratic = rng.normal(size=(rows, 12))
    latent_size = common[:, 0]
    relative_assets = _percentile_rank(latent_size)
    relative_sales = _percentile_rank(
        0.85 * latent_size + np.sqrt(1.0 - 0.85**2) * idiosyncratic[:, 11]
    )
    logistic = lambda value: 1.0 / (1.0 + np.exp(-value))
    features = {
        "at_rank_year_currency": relative_assets,
        "cash_at": logistic(0.5 * common[:, 1] + idiosyncratic[:, 1]),
        "intangibles_at": logistic(0.4 * common[:, 2] + idiosyncratic[:, 2]),
        "roa_ni": 0.06 + 0.08 * common[:, 1] + 0.04 * idiosyncratic[:, 3],
        "debt_at": logistic(0.6 * common[:, 3] + idiosyncratic[:, 4]),
        "ppe_at": logistic(0.5 * common[:, 4] + idiosyncratic[:, 5]),
        "log_at_slope_L": 0.05 * common[:, 0] + 0.08 * idiosyncratic[:, 6],
        "capx_at": logistic(-0.4 + 0.5 * common[:, 4] + idiosyncratic[:, 7]),
        "roa_ni_slope_L": 0.04 * common[:, 1] + 0.07 * idiosyncratic[:, 8],
        "cash_at_slope_L": 0.04 * common[:, 2] + 0.07 * idiosyncratic[:, 9],
        "debt_at_slope_L": 0.04 * common[:, 3] + 0.07 * idiosyncratic[:, 10],
        "sale_rank_year_currency": relative_sales,
    }
    frame = pd.DataFrame(features)
    frame.insert(0, "gvkey", [f"PUBLIC_{index:06d}" for index in range(rows)])
    role_features = {
        "M1": "at_rank_year_currency",
        "C1": "roa_ni",
        "C2": "debt_at",
        "C3": "ppe_at",
        "C4": "log_at_slope_L",
        "P1": "capx_at",
        "P2": "roa_ni_slope_L",
        "P3": "cash_at_slope_L",
        "A1": "debt_at_slope_L",
    }
    frame["U_M1"] = relative_assets
    for role, feature in role_features.items():
        frame[f"Z_{role}"] = _standardize(frame[feature].to_numpy(float))
    if (
        not frame["gvkey"].is_unique
        or not np.isfinite(frame.drop(columns="gvkey").to_numpy(float)).all()
    ):
        raise AssertionError("public backbone contract failed")
    return frame


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _contract(root: Path):
    product_path = root / "execution" / "run_morphology_product_v1.py"
    kink_path = root / "execution" / "run_kink_resolution_v2.py"
    product = _load_module(product_path, "orgcate_public_demo_product")
    kink = _load_module(kink_path, "orgcate_public_demo_kink")
    product_config_path = root / "configs" / "morphology_product_v1.json"
    product_config = _read_json(product_config_path)
    product.validate_config(product_config)
    kink_config_path = root / "configs" / "kink_resolution_v2.json"
    kink_config = _read_json(kink_config_path)
    kink.validate_config(kink_config)
    source_path = root / "simulation_runner_v2" / "config" / "pilot_b.json"
    source = _read_json(source_path)
    base_path = root / "configs" / "nine_shape_v2_unified_pilot_v1.json"
    base_parameters = _read_json(base_path)["shape_parameters"]
    public_contract_path = root / "configs" / "public_workflow_contract_v1.json"
    public_contract = _read_json(public_contract_path)
    return (
        product,
        kink,
        product_config,
        kink_config,
        public_contract,
        source,
        base_parameters,
        {
            "product_runner": product_path,
            "kink_runner": kink_path,
            "product_config": product_config_path,
            "kink_config": kink_config_path,
            "source_config": source_path,
            "base_config": base_path,
            "public_contract": public_contract_path,
        },
    )


def _focused_kink_readout(
    *,
    product: Any,
    kink: Any,
    data: dict[str, Any],
    train: np.ndarray,
    test: np.ndarray,
    backbone: pd.DataFrame,
    source: dict[str, Any],
    product_config: dict[str, Any],
    kink_config: dict[str, Any],
    seed: int,
    lineage: dict[str, int],
    ate: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the frozen kink v2 readout without fitting a second forest."""

    u = product.BASE.reference_u(
        backbone, data["unit_id"], product_config["focal_characteristic"]
    )
    libraries = []
    nuisances = []
    evidence = []
    for refit in range(int(kink_config["same_data_nuisance_refits"])):
        refit_seeds = product.HELPER.refit_seeds(int(seed), refit)
        nuisance = product.V2.crossfit_nuisance(
            data["X"][train],
            data["Y"][train],
            data["T"][train],
            data["X"][test],
            refit_seeds["nuisance_refit_seed"],
            int(kink_config["nuisance_crossfit_folds"]),
        )
        library_config = copy.deepcopy(product_config)
        library_config["selection"] = {
            "structured_families": [
                "constant",
                "linear",
                "quadratic",
                "tanh",
                "kink",
                "step",
            ],
            "flexible_family": "spline",
        }
        library = product.HELPER.fit_library(
            u[train],
            nuisance["y_residual_train"],
            nuisance["t_residual_train"],
            library_config,
            refit_seeds["shape_cv_refit_seed"],
        )
        evidence.append(
            kink.nested_kink_evidence(
                u[train],
                nuisance["y_residual_train"],
                nuisance["t_residual_train"],
                kink_config,
                refit_seeds["shape_cv_refit_seed"] + 7001,
            )
        )
        libraries.append(library)
        nuisances.append(nuisance)
    combined = kink.combine_evidence(evidence, kink_config)
    confirmations = []
    for library, nuisance in zip(libraries, nuisances, strict=True):
        y_test = data["Y"][test] - nuisance["m_test"]
        t_test = data["T"][test] - nuisance["e_test"]
        confirmation = product.confirm_candidate(
            library["models"]["kink"],
            library["models"]["constant"],
            u[test],
            y_test,
            t_test,
            float(kink_config["test_confirmation"]["one_sided_alpha"]),
        )
        confirmations.append(bool(confirmation["test_confirmed_vs_constant"]))
    confirmation_rate = float(np.mean(confirmations))
    supported = bool(
        combined["kink_supported_train"]
        and confirmation_rate
        >= float(kink_config["test_confirmation"]["minimum_confirmation_rate"])
    )
    models = [library["models"]["kink"] for library in libraries]
    grid_spec = product_config["grid"]
    grid = np.linspace(
        float(grid_spec["minimum"]),
        float(grid_spec["maximum"]),
        int(grid_spec["points"]),
    )
    estimate = np.median(
        [product.HELPER.predict_model(model, grid) for model in models], axis=0
    )
    consensus_test = np.median(
        [product.HELPER.predict_model(model, u[test]) for model in models], axis=0
    )
    truth = product.truth_curve("kinked", grid, product_config)
    details = []
    for model in models:
        coefficient = np.asarray(model["coefficient"], dtype=float)
        details.append(
            {
                "center": float(model["params"]["c"]),
                "left_slope": float(coefficient[1]),
                "right_slope": float(coefficient[1] + coefficient[2]),
                "slope_change": float(coefficient[2]),
            }
        )
    update = {
        "predicted_label": "kinked" if supported else "undetermined",
        "predicted_subtype": "",
        "practical_state": "SUPPORTED_CANDIDATE" if supported else "INCONCLUSIVE",
        "selected_family": "kink",
        "test_confirmation_rate": confirmation_rate,
        "exact_label_correct": supported,
        "oracle_curve_spearman": float(
            pd.Series(truth).corr(pd.Series(estimate), method="spearman")
        ),
        "curve_rmse": float(np.sqrt(np.mean((truth - estimate) ** 2))),
        "parameter_details_json": json.dumps(details, sort_keys=True),
        "parameter_median": float(combined["center_median"]),
        "parameter_range": float(combined["center_range"]),
        "cate_test_mean": float(np.mean(consensus_test)),
        "cate_test_sd": float(np.std(consensus_test)),
        **{f"train_{key}": value for key, value in ate.items()},
    }
    curve_rows = [
        {
            "u": float(value),
            "estimate_curve": float(fitted),
            "truth_curve": float(actual),
            "ate_dml_train": float(ate["ate_dml"]),
            "ate_ci_low_train": float(ate["ate_ci_low"]),
            "ate_ci_high_train": float(ate["ate_ci_high"]),
        }
        for value, fitted, actual in zip(grid, estimate, truth, strict=True)
    ]
    return update, curve_rows


def _partition_labels(length: int, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    labels = np.full(length, "", dtype=object)
    labels[np.asarray(train, dtype=int)] = "TRAIN"
    labels[np.asarray(test, dtype=int)] = "TEST"
    if not np.isin(labels, ["TRAIN", "TEST"]).all():
        raise AssertionError("external split did not cover every row")
    return labels


def _prepare_output(output: Path, *, save_row_level: bool = True) -> dict[str, Path]:
    paths = {
        "practical": output / "practical" / "result.json",
        "ranking": output / "practical" / "feature_ranking.csv",
        "curve_practical": output / "practical" / "cate_curve.csv",
        "confirmation": output / "practical" / "test_confirmation.csv",
        "evaluation": output / "evaluator" / "oracle_evaluation.json",
        "curve_oracle": output / "evaluator" / "truth_curve.csv",
        "theory_audit": output / "practical" / "researcher_theory_audit.json",
        "manifest": output / "RUN_MANIFEST.json",
    }
    if save_row_level:
        paths.update(
            {
                "analyst": output / "analyst" / "analyst.csv",
                "oracle": output / "oracle" / "oracle.csv",
            }
        )
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(
            f"demo output already exists under {output}; choose a new --output directory"
        )
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def _ranking(importance: dict[str, float]) -> list[dict[str, Any]]:
    ordered = sorted(importance.items(), key=lambda item: (-item[1], item[0]))
    return [
        {"feature": feature, "importance": float(value), "rank": index}
        for index, (feature, value) in enumerate(ordered, start=1)
    ]


def run_public_demo(
    *,
    scenario: str = "sign-reversal",
    output: str | Path = "artifacts/public_demo",
    seed: int = 20261023,
    backbone_rows: int = 6000,
    use_bundled_backbone: bool = False,
    save_row_level: bool = True,
    root: Path | None = None,
) -> dict[str, Any]:
    """Run one semi-synthetic scenario through the frozen product."""

    if scenario not in NINE_SITUATIONS:
        raise ValueError(f"unknown scenario: {scenario}")
    root = repository_root(root)
    output = Path(output).resolve()
    paths = _prepare_output(output, save_row_level=save_row_level)
    (
        product,
        kink,
        config,
        kink_config,
        public_contract,
        source,
        base_parameters,
        inputs,
    ) = _contract(root)
    if use_bundled_backbone:
        backbone = product.V2.add_rank_columns(
            product.V2.v2.load_frozen_backbone(root, source, inputs["source_config"]),
            source["features"],
        )
        backbone_kind = "uploaded-real-covariate-backbone"
        backbone_input = (
            root
            / "execution"
            / "phase2_real_backbone_v1"
            / "headline_one_anchor_per_firm_L3.pkl.gz"
        )
        inputs = {**inputs, "bundled_backbone": backbone_input}
    else:
        backbone = make_public_synthetic_backbone(
            rows=int(backbone_rows), seed=int(seed)
        )
        backbone_kind = "generated-synthetic-covariate-backbone"
    lineage = product.BASE.lineage(int(seed), 0)
    data = product.build_data(
        backbone, source, config, scenario, lineage, base_parameters
    )
    train, test = product.V2.v2.split_once(
        data["T"], data["unit_id"], float(config["test_size"]), lineage["split_seed"]
    )
    partition = _partition_labels(len(data["T"]), train, test)
    analyst = pd.DataFrame(data["X"], columns=source["features"])
    analyst.insert(0, "partition", partition)
    analyst.insert(0, "Y", data["Y"])
    analyst.insert(0, "T", data["T"])
    analyst.insert(0, "unit_id", data["unit_id"])
    oracle = pd.DataFrame(
        {
            "unit_id": data["unit_id"],
            "partition": partition,
            "propensity_true": data["propensity_true"],
            "y0_true": data["y0_true"],
            "y1_true": data["y0_true"] + data["tau_true"],
            "tau_true": data["tau_true"],
            "true_label": scenario,
        }
    )
    if save_row_level:
        analyst.to_csv(paths["analyst"], index=False)
        oracle.to_csv(paths["oracle"], index=False)

    ate_preflight = product.estimate_ate(
        data["X"][train],
        data["Y"][train],
        data["T"][train],
        lineage["nuisance_seed"],
        config,
    )
    eligibility = evaluate_design_eligibility(
        data["T"][train],
        ate_preflight,
        float(ate_preflight["propensity_clip_fraction"]),
        public_contract,
    )
    conventional = conventional_ate_diagnostics(
        data["X"][train],
        data["Y"][train],
        data["T"][train],
        float(ate_preflight["ate_dml"]),
    )
    if not eligibility["eligible"]:
        grid_spec = config["grid"]
        grid = np.linspace(
            float(grid_spec["minimum"]),
            float(grid_spec["maximum"]),
            int(grid_spec["points"]),
        )
        curve_rows = [
            {
                "u": float(value),
                "estimate_curve": float(ate_preflight["ate_dml"]),
                "truth_curve": float(actual),
                "ate_dml_train": float(ate_preflight["ate_dml"]),
                "ate_ci_low_train": float(ate_preflight["ate_ci_low"]),
                "ate_ci_high_train": float(ate_preflight["ate_ci_high"]),
            }
            for value, actual in zip(
                grid, product.truth_curve(scenario, grid, config), strict=True
            )
        ]
        result = {
            "predicted_label": "undetermined",
            "predicted_subtype": "",
            "practical_state": "DESIGN_NOT_ELIGIBLE",
            "selected_family": "none",
            "test_confirmation_rate": 0.0,
            "exact_label_correct": False,
            "oracle_curve_spearman": float("nan"),
            "curve_rmse": float("nan"),
            "cate_test_mean": float(ate_preflight["ate_dml"]),
            "importance_json": json.dumps(
                {feature: 0.0 for feature in source["features"]}
            ),
            "focal_characteristic_rank": 0,
            "parameter_details_json": "[]",
            "warning_count": 0,
            **{f"train_{key}": value for key, value in ate_preflight.items()},
        }
        confirmation_rows = []
    else:
        result, confirmation_rows, curve_rows = product.run_cell(
            backbone,
            source,
            base_parameters,
            config,
            "public-demo",
            scenario,
            0,
            int(seed),
        )
    if scenario == "kinked" and eligibility["eligible"]:
        update, curve_rows = _focused_kink_readout(
            product=product,
            kink=kink,
            data=data,
            train=train,
            test=test,
            backbone=backbone,
            source=source,
            product_config=config,
            kink_config=kink_config,
            seed=int(seed),
            lineage=lineage,
            ate=ate_preflight,
        )
        result.update(update)
        confirmation_rows = []
    predicted_label = str(result["predicted_label"])
    selected_family = str(result["selected_family"])
    curve_practical = pd.DataFrame(curve_rows)

    importance = (
        json.loads(result["importance_json"])
        if "importance_json" in result
        else {
            config["focal_characteristic"]: float(
                result["focal_characteristic_importance"]
            )
        }
    )
    ranking = _ranking(importance)
    rank_frame = pd.DataFrame(ranking)
    rank_frame.to_csv(paths["ranking"], index=False)
    details = json.loads(result.get("parameter_details_json", "[]"))
    practical_state = str(result["practical_state"])
    coherence = evaluate_ate_cate_coherence(
        float(result["train_ate_dml"]),
        float(result["cate_test_mean"]),
        data["Y"][train],
        public_contract,
    )
    if eligibility["eligible"] and not coherence["passed"]:
        practical_state = "INCONCLUSIVE"
    taxonomy = three_layer_taxonomy(
        predicted_label,
        curve_practical["estimate_curve"].to_numpy(float),
        float(result["train_ate_dml"]),
        selected_family=selected_family,
    )
    support = property_support(predicted_label, practical_state, coherence)
    english_summary = summarize_shape_quantities_en(
        predicted_label,
        str(result.get("predicted_subtype", "")),
        details,
        curve_practical["u"].to_numpy(float),
        curve_practical["estimate_curve"].to_numpy(float),
    )
    practical = {
        "schema_version": "orgcate-result-v1",
        "run_kind": (
            "uploaded-backbone-acceptance"
            if use_bundled_backbone
            else "public-synthetic-demo"
        ),
        "scenario": scenario,
        "practical_state": practical_state,
        "eligibility": eligibility,
        "ate": {
            "method": "five-fold cross-fitted AIPW/DML",
            "estimate": float(result["train_ate_dml"]),
            "standard_error": float(result["train_ate_standard_error"]),
            "ci_low": float(result["train_ate_ci_low"]),
            "ci_high": float(result["train_ate_ci_high"]),
            "conventional_diagnostics": conventional,
        },
        "ate_cate_coherence": coherence,
        "forest": {
            "learner": (
                "econml.dml.CausalForestDML"
                if eligibility["eligible"]
                else "not-fitted"
            ),
            "one_joint_multivariable_forest": bool(eligibility["eligible"]),
            "candidate_characteristics": list(source["features"]),
            "configuration": config["primary_forest"],
        },
        "moderator": {
            "focal_characteristic": config["focal_characteristic"],
            "prespecified_for_demo": True,
            "rank_is_information_only": True,
            "focal_rank": int(result["focal_characteristic_rank"]),
            "all_candidate_ranks": ranking,
        },
        "morphology": {
            "label": predicted_label,
            "subtype": str(result.get("predicted_subtype", "")),
            "selected_family": selected_family,
            "shape_quantities": details,
            **english_summary,
            **taxonomy,
        },
        "property_support": support,
        "confirmation": {
            "external_split": "80% TRAIN / 20% untouched TEST",
            "test_confirmation_rate": float(result["test_confirmation_rate"]),
            "test_selects_morphology": False,
        },
        "theory_audit": {
            "status": "PENDING_RESEARCHER_REVIEW",
            "artifact": "researcher_theory_audit.json",
            "automated_theory_generation": False,
        },
        "audit": {
            "master_seed": int(seed),
            "seed_lineage": lineage,
            "evaluator_truth_used_for_selection": False,
            "one_joint_multivariable_forest": bool(eligibility["eligible"]),
            "warning_count": int(result["warning_count"]),
            "package_versions": package_versions(),
        },
    }
    if not support["claim_allowed"]:
        practical["morphology"]["shape_quantities"] = []
        practical["morphology"]["key_quantities"] = {}
        practical["morphology"]["interpretation_en"] = (
            "No shape-specific quantities are reported because the required "
            "properties are not supported."
        )
    validate_practical_result(practical)
    _write_json(paths["practical"], practical)
    audit_template = theory_audit_template(practical)
    validate_theory_audit(audit_template)
    _write_json(paths["theory_audit"], audit_template)

    practical_columns = ["u", "estimate_curve"]
    for column in ("ate_dml_train", "ate_ci_low_train", "ate_ci_high_train"):
        if column in curve_practical:
            practical_columns.append(column)
    curve_practical[practical_columns].to_csv(paths["curve_practical"], index=False)
    if confirmation_rows:
        pd.DataFrame(confirmation_rows).to_csv(paths["confirmation"], index=False)
    else:
        pd.DataFrame(
            [{"test_confirmation_rate": float(result["test_confirmation_rate"])}]
        ).to_csv(paths["confirmation"], index=False)

    evaluation = {
        "schema_version": "orgcate-oracle-evaluation-v1",
        "oracle_use": "evaluation-only",
        "scenario": scenario,
        "true_label": scenario,
        "predicted_label": predicted_label,
        "exact_label_correct": bool(
            result.get(
                "exact_label_correct", result.get("classification_correct_evaluator")
            )
        ),
        "true_ate": float(np.mean(data["tau_true"])),
        "ate_error": float(result["train_ate_dml"] - np.mean(data["tau_true"])),
        "curve_rank_correlation": float(
            result.get("oracle_curve_spearman", result.get("curve_spearman_evaluator"))
        ),
        "curve_rmse": float(
            result.get("curve_rmse", result.get("curve_rmse_evaluator"))
        ),
    }
    _write_json(paths["evaluation"], evaluation)
    truth_column = "truth_curve"
    pd.DataFrame(curve_rows)[["u", truth_column]].to_csv(
        paths["curve_oracle"], index=False
    )

    tracked = {name: path for name, path in paths.items() if name != "manifest"}
    manifest = {
        "schema_version": "orgcate-public-demo-manifest-v1",
        "status": "completed",
        "scenario": scenario,
        "covariate_backbone": backbone_kind,
        "public_synthetic_data": not use_bundled_backbone,
        "uploaded_real_covariates_used": use_bundled_backbone,
        "treatment_outcome_and_cate_simulated": True,
        "row_level_outputs_written": save_row_level,
        "backbone_rows_available": len(backbone),
        "analysis_rows": len(data["T"]),
        "train_rows": len(train),
        "test_rows": len(test),
        "analyst_oracle_separated": True,
        "test_used_for_morphology_selection": False,
        "evaluator_truth_used_for_morphology_selection": False,
        "one_joint_multivariable_forest": True,
        "master_seed": int(seed),
        "seed_lineage": lineage,
        "input_hashes": {name: _sha256(path) for name, path in inputs.items()},
        "active_runner_hashes": {
            "product": _sha256(inputs["product_runner"]),
            **(
                {"focused_kink": _sha256(inputs["kink_runner"])}
                if scenario == "kinked"
                else {}
            ),
        },
        "output_hashes": {name: _sha256(path) for name, path in tracked.items()},
        "package_versions": package_versions(),
    }
    _write_json(paths["manifest"], manifest)
    return {"practical": practical, "evaluation": evaluation, "manifest": manifest}
