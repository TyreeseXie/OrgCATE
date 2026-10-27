"""Contracts for the separately reported additive coexistence examples."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "additive_coexistence"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_and_three_examples_are_separate_inside_one_module() -> None:
    for label, count in (("two_component", 2), ("three_component", 3)):
        config = json.loads((MODULE / label / "config.json").read_text(encoding="utf-8"))
        assert set(config["scenarios"]) == {
            f"active_count_{count}",
            f"null_count_{count}",
        }
        assert {item["component_count"] for item in config["scenarios"].values()} == {
            count
        }


def test_recorded_coexistence_evidence_passes() -> None:
    verifier = load(MODULE / "verify.py", "additive_coexistence_verifier")
    result = verifier.verify()
    assert result["status"] == "PASS", json.dumps(result["checks"], indent=2)


def test_downloadable_coexistence_artifact_is_complete() -> None:
    verifier = load(
        ROOT / "tools" / "verify_additive_coexistence_artifact.py",
        "additive_coexistence_artifact_verifier",
    )
    result = verifier.verify(MODULE)
    assert result["status"] == "PASS", json.dumps(result["checks"], indent=2)


def test_shared_runner_accepts_each_separate_config() -> None:
    runner = load(MODULE / "src" / "run.py", "additive_coexistence_runner")
    for label in ("two_component", "three_component"):
        config_path = MODULE / label / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        _, _, method, _ = runner.validate_config(config_path, config)
        assert method["primary_forest"]["class"] == "econml.dml.CausalForestDML"
        assert method["primary_forest"]["n_estimators"] == 1000


def test_two_component_report_figures_render_separately(tmp_path: Path) -> None:
    runner = load(MODULE / "src" / "run.py", "additive_coexistence_figure_runner")
    config = json.loads(
        (MODULE / "two_component" / "config.json").read_text(encoding="utf-8")
    )
    results = pd.DataFrame(
        [
            {
                "scenario_role": "active",
                "component_count": 2,
                "total_cate_fidelity_pass": True,
            },
            {
                "scenario_role": "all-component-null",
                "component_count": 2,
                "any_false_confirmation": False,
            },
        ]
    )
    components = pd.DataFrame(
        [
            {
                "scenario_role": "active",
                "scenario": "active_count_2",
                "component_count": 2,
                "feature": feature,
                "truth_label": label,
                "component_recovery_success": True,
            }
            for feature, label in (("ppe_at", "inverted-u"), ("intangibles_at", "u-shaped"))
        ]
    )
    curves = pd.DataFrame(
        [
            {
                "scenario": "active_count_2",
                "feature": feature,
                "truth_label": label,
                "u": u,
                "truth_curve": value,
                "estimate_curve": value,
            }
            for feature, label, sign in (
                ("ppe_at", "inverted-u", 1.0),
                ("intangibles_at", "u-shaped", -1.0),
            )
            for u, value in ((0.1, -sign), (0.5, sign), (0.9, -sign))
        ]
    )
    runner.render_componentwise_figures(config, results, components, curves, tmp_path)
    figures = sorted(tmp_path.glob("FIGURE_*.svg"))
    assert len(figures) == 3
    assert all(path.stat().st_size > 0 for path in figures)
