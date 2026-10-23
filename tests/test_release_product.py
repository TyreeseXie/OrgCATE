from __future__ import annotations

import json

import numpy as np
import pytest

from orgcate_sim.cli import build_parser
from orgcate_sim.public_demo import make_public_synthetic_backbone
from orgcate_sim.release import (
    NINE_SITUATIONS,
    audit_method_contract,
    load_integrated_evidence,
    validate_practical_result,
    verify_release,
)


def test_integrated_evidence_uses_focused_kink_version() -> None:
    evidence = load_integrated_evidence()
    assert tuple(evidence["scenario"]) == NINE_SITUATIONS
    assert len(evidence) == 9
    assert (evidence["exact_recovery_rate"] >= 0.90).all()
    kink = evidence.loc[evidence["scenario"] == "kinked"].iloc[0]
    assert kink["evidence_version"] == "kink-resolution-v2"
    assert kink["exact_recovery_rate"] == 1.0
    assert kink["baseline_product_v1_rate"] == 0.1


def test_release_verification_passes_frozen_hashes() -> None:
    result = verify_release()
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
        "evidence",
        "verify",
        "contract",
        "theory-audit",
        "reproduce",
    ):
        assert command in help_text
    parsed = parser.parse_args(["demo", "--scenario", "kinked", "--output", "unused"])
    assert parsed.scenario == "kinked"
