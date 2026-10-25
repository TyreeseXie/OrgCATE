from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from orgcate_sim.cli import build_parser
from orgcate_sim.public_demo import (
    _apply_tanh_transition_rule,
    make_public_synthetic_backbone,
)
from orgcate_sim.release import (
    NINE_SITUATIONS,
    audit_method_contract,
    load_integrated_evidence,
    repository_root,
    sha256_portable_file_candidates,
    sha256_versioned_text_file,
    validate_practical_result,
    validate_simulation_path_migration,
    verify_release,
)


def test_repository_root_does_not_require_private_workspace_instructions(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    (tmp_path / "execution").mkdir()
    (tmp_path / "configs").mkdir()

    assert repository_root(tmp_path) == tmp_path


def test_versioned_text_hash_is_independent_of_checkout_line_endings(
    tmp_path: Path,
) -> None:
    lf_path = tmp_path / "lf.json"
    crlf_path = tmp_path / "crlf.json"
    lf_path.write_bytes(b'{\n  "frozen": true\n}\n')
    crlf_path.write_bytes(b'{\r\n  "frozen": true\r\n}\r\n')

    assert sha256_versioned_text_file(lf_path) == sha256_versioned_text_file(crlf_path)


def test_frozen_evidence_hash_accepts_only_line_ending_variants(tmp_path: Path) -> None:
    lf_path = tmp_path / "evidence.csv"
    lf_path.write_bytes(b"name,value\nalpha,1\n")
    crlf_hash = hashlib.sha256(b"name,value\r\nalpha,1\r\n").hexdigest()
    changed_hash = hashlib.sha256(b"name,value\r\nalpha,2\r\n").hexdigest()

    candidates = sha256_portable_file_candidates(lf_path)
    assert crlf_hash in candidates
    assert changed_hash not in candidates


def test_integrated_evidence_uses_unified_success_definition() -> None:
    evidence = load_integrated_evidence()
    assert tuple(evidence["scenario"]) == NINE_SITUATIONS
    assert len(evidence) == 9
    assert (evidence["unified_recovery_rate"] >= 0.90).all()
    assert (evidence["scenario_pass"]).all()
    kink = evidence.loc[evidence["scenario"] == "kinked"].iloc[0]
    assert kink["evidence_version"] == "kink-resolution-v2"
    assert kink["unified_recovery_rate"] == 1.0
    assert kink["baseline_product_v1_rate"] == 0.1
    nonlinear = evidence.loc[evidence["scenario"] == "nonlinear-monotone"].iloc[0]
    assert nonlinear["evidence_version"] == "nonlinear-monotone-resolution-v1"
    assert nonlinear["unified_recovery_rate"] == 0.90
    assert nonlinear["blind_evidence_rate"] == 1.0


def test_release_verification_passes_frozen_hashes() -> None:
    result = verify_release()
    assert result["status"] == "PASS", json.dumps(result["checks"], indent=2)
    assert all(result["checks"].values())


def test_public_dependency_layout_is_hash_pinned() -> None:
    result = validate_simulation_path_migration()
    assert result["status"] == "PASS", json.dumps(result["checks"], indent=2)
    assert all(result["checks"].values())


def test_method_contract_distinguishes_verified_evidence_from_release_readiness() -> (
    None
):
    result = audit_method_contract()
    assert result["frozen_evidence_status"] == "PASS"
    assert result["status"] == "BLOCKED"
    blocker_ids = {item["id"] for item in result["release_blockers"]}
    assert blocker_ids == {"public_license_and_citation"}
    assert result["requirements_passed"] == 16


def test_public_backbone_is_finite_unique_and_contains_correlated_size_proxy() -> None:
    frame = make_public_synthetic_backbone(rows=3000, seed=17)
    assert frame["gvkey"].is_unique
    assert len(frame) == 3000
    assert np.isfinite(frame.drop(columns="gvkey").to_numpy(float)).all()
    correlation = (
        frame[["at_rank_year_currency", "sale_rank_year_currency"]].corr().iloc[0, 1]
    )
    assert 0.75 < correlation < 0.95


def test_tanh_transition_rule_uses_effective_gradient_not_raw_k() -> None:
    ordinary = {
        "selected_family": "tanh",
        "predicted_label": "steep-continuous",
        "parameter_details_json": json.dumps(
            [
                {"amplitude": 0.85, "k": 10.0},
                {"amplitude": 0.90, "k": 10.0},
                {"amplitude": 0.88, "k": 10.0},
            ]
        ),
    }
    steep = {
        "selected_family": "tanh",
        "predicted_label": "nonlinear-monotone",
        "parameter_details_json": json.dumps(
            [
                {"amplitude": 1.35, "k": 8.0},
                {"amplitude": 1.40, "k": 8.0},
                {"amplitude": 1.38, "k": 8.0},
            ]
        ),
    }
    assert _apply_tanh_transition_rule(ordinary)["predicted_label"] == "nonlinear-monotone"
    assert _apply_tanh_transition_rule(steep)["predicted_label"] == "steep-continuous"


def test_practical_schema_rejects_oracle_leakage() -> None:
    valid = {
        "schema_version": "orgcate-result-v1",
        "run_kind": "unit-test",
        "scenario": "homogeneous",
        "practical_state": "NO_THEORY_CLAIM",
        "eligibility": {
            "eligible": True,
            "treatment_rate": 0.5,
            "propensity_q01_oof": 0.2,
            "propensity_q99_oof": 0.8,
            "propensity_clip_fraction": 0.0,
            "thresholds": {},
            "reasons": [],
        },
        "ate": {
            "method": "test",
            "estimate": 0.2,
            "standard_error": 0.1,
            "ci_low": 0.0,
            "ci_high": 0.4,
            "conventional_diagnostics": {},
        },
        "ate_cate_coherence": {
            "passed": True,
            "dml_ate_train": 0.2,
            "final_cate_test_mean": 0.2,
            "signed_gap": 0.0,
            "outcome_sd_train": 1.0,
            "standardized_absolute_gap": 0.0,
            "threshold": 0.1,
            "sensitivity_thresholds": [0.05, 0.15],
            "rule": "test",
            "failure_state": "INCONCLUSIVE",
        },
        "forest": {},
        "moderator": {},
        "property_support": {
            "ordering": False,
            "magnitude": False,
            "curvature": False,
            "boundaries": False,
            "shape_requirements": [],
            "claim_allowed": True,
            "reason": "test",
        },
        "morphology": {
            "label": "homogeneous",
            "subtype": "",
            "selected_family": "constant",
            "shape_quantities": [],
            "key_quantities": {},
            "interpretation_en": "test",
            "continuity": "continuous",
            "monotonicity": "linear",
            "sign_relation": "same",
            "taxonomy_cell": 1,
            "requires_abduction": False,
            "classification_basis": "test",
        },
        "confirmation": {},
        "theory_audit": {
            "status": "PENDING_RESEARCHER_REVIEW",
            "artifact": "audit.json",
            "automated_theory_generation": False,
        },
        "audit": {},
    }
    validate_practical_result(valid)
    valid["audit"]["tau_true"] = [0.2]
    with pytest.raises(ValueError, match="oracle/evaluator"):
        validate_practical_result(valid)


def test_cli_exposes_current_product_commands_and_nine_names() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    for command in (
        "doctor",
        "demo",
        "acceptance",
        "evidence",
        "verify",
        "contract",
        "theory-audit",
        "reproduce",
    ):
        assert command in help_text
    parsed = parser.parse_args(["demo", "--scenario", "kinked", "--output", "unused"])
    assert parsed.scenario == "kinked"
