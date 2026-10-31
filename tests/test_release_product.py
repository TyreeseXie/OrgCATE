from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from orgcate_sim.cli import build_parser
from orgcate_sim.public_demo import make_public_synthetic_backbone
from orgcate_sim.release import (
    NINE_SITUATIONS,
    audit_method_contract,
    load_integrated_evidence,
    repository_root,
    sha256_versioned_text_file,
    validate_practical_result,
    verify_release,
)


def test_repository_root_does_not_require_private_agent_instructions(
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


def test_integrated_evidence_uses_one_unified_version() -> None:
    evidence = load_integrated_evidence()
    assert tuple(evidence["scenario"]) == NINE_SITUATIONS
    assert len(evidence) == 9
    assert (evidence["exact_recovery_rate"] >= 0.90).all()
    assert set(evidence["evidence_version"]) == {"unified-morphology-v2"}


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
            "final_cate_train_target_mean": 0.2,
            "descriptive_cate_test_mean": 0.2,
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
        "fixed-effects-comparison",
    ):
        assert command in help_text
    parsed = parser.parse_args(["demo", "--scenario", "kinked", "--output", "unused"])
    assert parsed.scenario == "kinked"


def test_ci_uploads_one_complete_analysis_artifact() -> None:
    root = repository_root()
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert workflow.count("uses: actions/upload-artifact@v4") == 1
    assert "name: orgcate-complete-analysis-${{ github.run_id }}" in workflow
    assert "orgcate-data-to-results-${{ github.run_id }}" not in workflow
    assert "orgcate-additive-coexistence-${{ github.run_id }}" not in workflow


def test_complete_artifact_contract_includes_provenance_files() -> None:
    root = repository_root()
    assembler = (root / "tools" / "assemble_complete_analysis_artifact.py").read_text(
        encoding="utf-8"
    )
    verifier = (root / "tools" / "verify_complete_analysis_artifact.py").read_text(
        encoding="utf-8"
    )

    for required in (
        "RUN_CONTEXT.json",
        "configuration/data_to_results_acceptance_v1.json",
        "configuration/ate_estimator_comparison_v1.json",
        "configuration/fixed_effects_comparison_v1.json",
    ):
        assert required in verifier
    assert '"schema_version": "orgcate-analysis-run-context-v1"' in assembler
    assert '"schema_version": "orgcate-complete-analysis-artifact-v2"' in assembler
