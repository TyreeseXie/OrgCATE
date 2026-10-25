"""Public release evidence, schema, and reproducibility checks."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

NINE_SITUATIONS = (
    "homogeneous",
    "linear-monotone",
    "nonlinear-monotone",
    "steep-continuous",
    "inverted-u",
    "u-shaped",
    "kinked",
    "threshold-step",
    "sign-reversal",
)
PRACTICAL_STATES = {
    "SUPPORTED_CANDIDATE",
    "INCONCLUSIVE",
    "NO_THEORY_CLAIM",
    "DESIGN_NOT_ELIGIBLE",
}
ORACLE_MARKERS = {
    "tau_true",
    "propensity_true",
    "y0_true",
    "y1_true",
    "true_label",
    "true_ate",
    "oracle",
}
EXPECTED_VERSIONS = {
    "econml": "0.17.0",
    "scikit-learn": "1.6.1",
    "numpy": "2.5.2",
    "pandas": "3.0.5",
    "scipy": "1.18.1",
    "pyarrow": "25.0.1",
}
METHOD_CONTRACT_STATUSES = {"PASS", "PARTIAL", "MISSING", "HUMAN_DECISION"}
PUBLIC_PRODUCT_CONFIG_SHA256 = (
    "c7a6e188da1cfbcb9014da9be3a6efd58b5c135320cfc3a34baf85c22c365571"
)


def repository_root(start: Path | None = None) -> Path:
    candidates = []
    if start is not None:
        candidates.extend([start.resolve(), *start.resolve().parents])
    module_root = Path(__file__).resolve().parents[2]
    candidates.extend([module_root, *module_root.parents, Path.cwd().resolve()])
    for candidate in candidates:
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "execution").is_dir()
            and (candidate / "configs").is_dir()
        ):
            return candidate
    raise FileNotFoundError("OrgCATE repository root was not found")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_versioned_text_file(path: Path) -> str:
    """Hash repository text using Git's LF-normalized representation."""

    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def sha256_portable_file_candidates(path: Path) -> set[str]:
    """Hash a file under the byte representations permitted by Git checkout."""

    content = path.read_bytes()
    candidates = {hashlib.sha256(content).hexdigest()}
    if path.suffix.lower() in {".csv", ".json", ".md", ".svg", ".txt", ".yaml", ".yml"}:
        lf_content = content.replace(b"\r\n", b"\n")
        crlf_content = lf_content.replace(b"\n", b"\r\n")
        candidates.add(hashlib.sha256(lf_content).hexdigest())
        candidates.add(hashlib.sha256(crlf_content).hexdigest())
    return candidates


def validate_simulation_path_migration(root: Path | None = None) -> dict[str, Any]:
    """Validate the public dependency layout and its hash-pinned inputs."""

    root = repository_root(root)
    product_freeze = _read_json(
        root / "reports" / "MORPHOLOGY_PRODUCT_V1_METHOD_FREEZE.json"
    )
    kink_freeze = _read_json(
        root / "reports" / "KINK_RESOLUTION_V2_METHOD_FREEZE.json"
    )
    product_config = root / "configs" / "morphology_product_v1.json"
    source_config = root / "simulation_runner_v2" / "config" / "pilot_b.json"
    public_config_hash = sha256_versioned_text_file(product_config)
    checks = {
        "public_directory_present": (root / "simulation_runner_v2").is_dir(),
        "public_product_config_hash": public_config_hash
        == PUBLIC_PRODUCT_CONFIG_SHA256,
        "frozen_product_source_hash_agrees": product_freeze["hashes"][
            "source_config_sha256"
        ]
        == kink_freeze["hashes"]["source_config_sha256"],
        "source_config_content_preserved": source_config.is_file()
        and sha256_versioned_text_file(source_config)
        == product_freeze["hashes"]["source_config_sha256"],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "current_path": "simulation_runner_v2",
        "public_product_config_sha256": public_config_hash,
    }


def package_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in EXPECTED_VERSIONS:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = "NOT_INSTALLED"
    return result


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_method_contract(root: Path | None = None) -> dict[str, Any]:
    """Audit proposal/prototype requirements separately from frozen evidence.

    ``verify_release`` answers whether the frozen result files are complete and
    hash-consistent. This function answers the distinct question of whether the
    GitHub workflow implements every methodological and release gate promised in
    the proposal. A blocker remains a blocker even when old evidence verifies.
    """

    root = repository_root(root)
    contract_path = root / "configs" / "github_release_method_contract_v1.json"
    contract = _read_json(contract_path)
    requirements = contract.get("requirements", [])
    if not requirements:
        raise ValueError("method contract has no requirements")
    ids = [item.get("id") for item in requirements]
    if len(ids) != len(set(ids)) or any(not item for item in ids):
        raise ValueError("method contract requirement ids must be unique and nonempty")
    bad_statuses = {
        item.get("status")
        for item in requirements
        if item.get("status") not in METHOD_CONTRACT_STATUSES
    }
    if bad_statuses:
        raise ValueError(f"unknown method-contract statuses: {sorted(bad_statuses)}")
    blockers = [
        item
        for item in requirements
        if item.get("release_blocking") is True and item.get("status") != "PASS"
    ]
    return {
        "schema_version": "orgcate-method-contract-audit-v1",
        "status": "READY" if not blockers else "BLOCKED",
        "frozen_evidence_status": verify_release(root)["status"],
        "contract_path": str(contract_path),
        "requirements_total": len(requirements),
        "requirements_passed": sum(
            item.get("status") == "PASS" for item in requirements
        ),
        "release_blocker_count": len(blockers),
        "release_blockers": [
            {
                "id": item["id"],
                "stage": item["stage"],
                "status": item["status"],
                "required_action": item["required_action"],
            }
            for item in blockers
        ],
        "note": (
            "Frozen evidence verification and method-contract readiness are "
            "different claims; both must pass before public release."
        ),
    }


def load_integrated_evidence(root: Path | None = None) -> pd.DataFrame:
    """Load the unified nine-situation recovery evidence."""

    root = repository_root(root)
    summary_path = (
        root
        / "reports"
        / "single_scenario_release_candidate_v1"
        / "SINGLE_SCENARIO_UNIFIED_SUMMARY.csv"
    )
    product_path = (
        root
        / "reports"
        / "morphology_product_v1"
        / "MORPHOLOGY_PRODUCT_V1_SUMMARY.csv"
    )
    summary = pd.read_csv(summary_path)
    product = pd.read_csv(product_path).set_index("dgp")
    if tuple(summary["scenario"]) != NINE_SITUATIONS:
        raise ValueError("unified summary does not contain the ordered nine situations")
    rows: list[dict[str, Any]] = []
    for scenario in NINE_SITUATIONS:
        current = summary.loc[summary["scenario"] == scenario].iloc[0]
        evidence_version = "morphology-product-v1"
        if scenario == "kinked":
            evidence_version = "kink-resolution-v2"
        elif scenario == "nonlinear-monotone":
            evidence_version = "nonlinear-monotone-resolution-v1"
        rows.append(
            {
                "scenario": scenario,
                "evidence_version": evidence_version,
                "replications": int(current["replications"]),
                "unified_recovery_rate": float(current["recovery_success_rate"]),
                "blind_evidence_rate": float(current["blind_evidence_rate"]),
                "exact_recovery_rate": float(current["exact_morphology_rate"]),
                "curve_fidelity_rate": float(current["curve_fidelity_rate"]),
                "key_quantities_rate": float(current["key_quantities_rate"]),
                "scenario_pass": bool(current["scenario_pass"]),
                "baseline_product_v1_rate": float(
                    product.loc[scenario, "exact_label_rate"]
                ),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 9 or result["scenario"].duplicated().any():
        raise AssertionError("integrated evidence must contain nine unique situations")
    rates = result[
        [
            "unified_recovery_rate",
            "blind_evidence_rate",
            "exact_recovery_rate",
            "curve_fidelity_rate",
            "key_quantities_rate",
        ]
    ]
    if not ((rates >= 0.0) & (rates <= 1.0)).all().all():
        raise ValueError("evidence rates must be probabilities")
    return result


def _hash_check(checks: dict[str, bool], name: str, path: Path, expected: str) -> None:
    checks[name] = path.is_file() and sha256_versioned_text_file(path) == expected


def _raw_hash_check(
    checks: dict[str, bool], name: str, path: Path, expected: str
) -> None:
    checks[name] = path.is_file() and expected in sha256_portable_file_candidates(path)


def verify_release(root: Path | None = None) -> dict[str, Any]:
    """Verify frozen code/config hashes and aggregate evidence contracts."""

    root = repository_root(root)
    product_manifest = _read_json(
        root
        / "reports"
        / "morphology_product_v1"
        / "MORPHOLOGY_PRODUCT_V1_MANIFEST.json"
    )
    kink_manifest = _read_json(
        root / "reports" / "kink_resolution_v2" / "KINK_RESOLUTION_V2_MANIFEST.json"
    )
    checks: dict[str, bool] = {
        "product_cells_complete": product_manifest.get("cells_completed") == 100
        and product_manifest.get("cells_failed") == 0,
        "product_one_joint_forest": product_manifest.get(
            "one_joint_multivariable_forest_per_cell"
        )
        is True,
        "product_no_test_or_oracle_selection": product_manifest.get(
            "test_used_for_shape_selection"
        )
        is False
        and product_manifest.get("oracle_used_for_shape_selection") is False,
        "kink_cells_complete": kink_manifest.get("cells_completed") == 30
        and kink_manifest.get("cells_failed") == 0,
        "kink_acceptance_passed": all(kink_manifest.get("acceptance", {}).values()),
    }
    product_config_path = root / "configs" / "morphology_product_v1.json"
    migration = validate_simulation_path_migration(root)
    checks["product_config_hash"] = (
        sha256_versioned_text_file(product_config_path)
        == product_manifest["config_sha256"]
        or migration["status"] == "PASS"
    )
    _hash_check(
        checks,
        "product_runner_hash",
        root / "execution" / "run_morphology_product_v1.py",
        product_manifest["runner_sha256"],
    )
    _hash_check(
        checks,
        "product_freeze_hash",
        root / "reports" / "MORPHOLOGY_PRODUCT_V1_METHOD_FREEZE.json",
        product_manifest["freeze_sha256"],
    )
    _hash_check(
        checks,
        "kink_config_hash",
        root / "configs" / "kink_resolution_v2.json",
        kink_manifest["config_sha256"],
    )
    _hash_check(
        checks,
        "kink_runner_hash",
        root / "execution" / "run_kink_resolution_v2.py",
        kink_manifest["runner_sha256"],
    )
    evidence = load_integrated_evidence(root)
    checks["nine_situations_present"] = tuple(evidence["scenario"]) == NINE_SITUATIONS
    checks["all_nine_meet_90_percent"] = bool(
        (evidence["unified_recovery_rate"] >= 0.90).all()
    )
    single_dir = root / "reports" / "single_scenario_release_candidate_v1"
    single_manifest = _read_json(
        single_dir / "SINGLE_SCENARIO_RELEASE_CANDIDATE_MANIFEST.json"
    )
    checks["single_scenario_manifest_passed"] = bool(
        single_manifest.get("decision", {}).get("all_nine_pass") is True
        and single_manifest.get("decision", {}).get("minimum_success_rate", 0.0)
        >= 0.90
    )
    for name, expected in single_manifest.get("artifact_hashes", {}).items():
        _raw_hash_check(
            checks, f"single_artifact_{name}", single_dir / name, expected
        )
    source_paths = {
        "baseline_summary_sha256": root
        / "reports"
        / "unified_recovery_acceptance_v1"
        / "UNIFIED_SINGLE_SUMMARY.csv",
        "nonlinear_summary_sha256": root
        / "reports"
        / "nonlinear_monotone_resolution_v1"
        / "confirmation"
        / "NONLINEAR_CONFIRMATION_SUMMARY.csv",
    }
    for name, path in source_paths.items():
        _raw_hash_check(
            checks,
            f"single_source_{name}",
            path,
            single_manifest["source_hashes"][name],
        )
    nonlinear_dir = (
        root / "reports" / "nonlinear_monotone_resolution_v1" / "confirmation"
    )
    nonlinear_manifest = _read_json(
        nonlinear_dir / "NONLINEAR_CONFIRMATION_MANIFEST.json"
    )
    checks["nonlinear_cells_complete"] = bool(
        nonlinear_manifest.get("cells") == 30
        and nonlinear_manifest.get("failed_cells") == 0
        and nonlinear_manifest.get("decision", {}).get("nonlinear_monotone_pass")
        is True
    )
    for name, expected in nonlinear_manifest.get("artifact_hashes", {}).items():
        _raw_hash_check(
            checks, f"nonlinear_artifact_{name}", nonlinear_dir / name, expected
        )
    _raw_hash_check(
        checks,
        "nonlinear_freeze_hash",
        root / "reports" / "NONLINEAR_MONOTONE_RESOLUTION_V1_METHOD_FREEZE.json",
        nonlinear_manifest["freeze_sha256"],
    )
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema_version": "orgcate-release-verification-v1",
        "status": status,
        "checks": checks,
        "path_migration": migration,
        "package_versions": package_versions(),
        "repository": str(root),
    }


def doctor(root: Path | None = None) -> dict[str, Any]:
    root = repository_root(root)
    versions = package_versions()
    version_checks = {
        package: versions.get(package) == expected
        for package, expected in EXPECTED_VERSIONS.items()
    }
    source_config = _read_json(
        root / "simulation_runner_v2" / "config" / "pilot_b.json"
    )
    source_path = root / "simulation_runner_v2" / "config"
    backbone = (source_path / source_config["backbone_path"]).resolve()
    expected_backbone_sha256 = (
        "91c963e02cbdc17a19de85868ba44f5f4fd65791bec83906980f307a6cfd3174"
    )
    observed_backbone_sha256 = sha256_file(backbone) if backbone.is_file() else None
    return {
        "schema_version": "orgcate-doctor-v1",
        "python_supported": sys.version_info >= (3, 10),
        "versions": versions,
        "release_lock_matches": version_checks,
        "aggregate_evidence_available": (
            root / "reports" / "kink_resolution_v2" / "KINK_RESOLUTION_V2_SUMMARY.csv"
        ).is_file(),
        "licensed_backbone_available_for_full_reproduction": backbone.is_file(),
        "licensed_backbone_path": str(backbone),
        "bundled_backbone_sha256": observed_backbone_sha256,
        "bundled_backbone_hash_matches": (
            observed_backbone_sha256 == expected_backbone_sha256
        ),
        "public_demo_requires_licensed_data": False,
    }


def _contains_oracle_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if (
                normalized in ORACLE_MARKERS
                or normalized.startswith("true_")
                or "oracle" in normalized
            ):
                return True
            if _contains_oracle_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_oracle_key(item) for item in value)
    return False


def validate_practical_result(record: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "run_kind",
        "scenario",
        "practical_state",
        "eligibility",
        "ate",
        "ate_cate_coherence",
        "forest",
        "moderator",
        "property_support",
        "morphology",
        "confirmation",
        "theory_audit",
        "audit",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"practical result missing fields: {sorted(missing)}")
    extra = set(record).difference(required)
    if extra:
        raise ValueError(f"practical result has unknown fields: {sorted(extra)}")
    if record["schema_version"] != "orgcate-result-v1":
        raise ValueError("wrong practical result schema")
    if record["scenario"] not in NINE_SITUATIONS:
        raise ValueError("unknown nine-situation label")
    if record["practical_state"] not in PRACTICAL_STATES:
        raise ValueError("invalid practical state")
    eligibility = record["eligibility"]
    ate = record["ate"]
    coherence = record["ate_cate_coherence"]
    support = record["property_support"]
    morphology = record["morphology"]
    blocks = {
        "eligibility": (
            eligibility,
            {
                "eligible",
                "treatment_rate",
                "propensity_q01_oof",
                "propensity_q99_oof",
                "propensity_clip_fraction",
                "thresholds",
                "reasons",
            },
        ),
        "ate": (
            ate,
            {
                "method",
                "estimate",
                "standard_error",
                "ci_low",
                "ci_high",
                "conventional_diagnostics",
            },
        ),
        "ate_cate_coherence": (
            coherence,
            {
                "passed",
                "dml_ate_train",
                "final_cate_test_mean",
                "signed_gap",
                "outcome_sd_train",
                "standardized_absolute_gap",
                "threshold",
                "sensitivity_thresholds",
                "rule",
                "failure_state",
            },
        ),
        "property_support": (
            support,
            {
                "ordering",
                "magnitude",
                "curvature",
                "boundaries",
                "shape_requirements",
                "claim_allowed",
                "reason",
            },
        ),
        "morphology": (
            morphology,
            {
                "label",
                "subtype",
                "selected_family",
                "shape_quantities",
                "key_quantities",
                "interpretation_en",
                "continuity",
                "monotonicity",
                "sign_relation",
                "taxonomy_cell",
                "requires_abduction",
                "classification_basis",
            },
        ),
        "theory_audit": (
            record["theory_audit"],
            {"status", "artifact", "automated_theory_generation"},
        ),
    }
    for block_name, (block, fields) in blocks.items():
        if not isinstance(block, dict):
            raise TypeError(f"{block_name} must be an object")
        block_missing = fields.difference(block)
        if block_missing:
            raise ValueError(f"{block_name} missing fields: {sorted(block_missing)}")
    if not isinstance(eligibility.get("eligible"), bool):
        raise TypeError("eligibility must expose a boolean eligible field")
    if (
        not eligibility["eligible"]
        and record["practical_state"] != "DESIGN_NOT_ELIGIBLE"
    ):
        raise ValueError("failed eligibility must route to DESIGN_NOT_ELIGIBLE")
    if eligibility["eligible"] and record["practical_state"] == "DESIGN_NOT_ELIGIBLE":
        raise ValueError("eligible data cannot route to DESIGN_NOT_ELIGIBLE")
    if not isinstance(coherence.get("passed"), bool):
        raise TypeError("ATE-CATE coherence must expose a boolean passed field")
    if not coherence["passed"] and record["practical_state"] == "SUPPORTED_CANDIDATE":
        raise ValueError("failed ATE-CATE coherence cannot support a candidate")
    properties = {"ordering", "magnitude", "curvature", "boundaries"}
    if any(not isinstance(support.get(name), bool) for name in properties):
        raise ValueError("property support declarations must be boolean")
    requirements = set(support.get("shape_requirements", []))
    if not requirements.issubset(properties):
        raise ValueError("unknown required morphology property")
    if support.get("claim_allowed") and not all(support[name] for name in requirements):
        raise ValueError("morphology claim exceeds declared property support")
    for field in (
        "continuity",
        "monotonicity",
        "sign_relation",
        "taxonomy_cell",
        "requires_abduction",
    ):
        if field not in morphology:
            raise ValueError(f"morphology missing three-layer field: {field}")
    if int(morphology["taxonomy_cell"]) not in range(13):
        raise ValueError("taxonomy_cell must be 0 (withheld) or 1..12")
    if record["theory_audit"].get("automated_theory_generation") is not False:
        raise ValueError("the interpretation record must be completed by a researcher")
    if _contains_oracle_key(record):
        raise ValueError("oracle/evaluator field leaked into practical result")
