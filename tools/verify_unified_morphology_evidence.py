"""Verify the unified single-morphology evidence package."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from orgcate_sim.release import NINE_SITUATIONS, verify_release

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports" / "unified_morphology_v2"


def verify() -> dict[str, object]:
    summary = pd.read_csv(EVIDENCE / "UNIFIED_MORPHOLOGY_V2_ACCEPTANCE_SUMMARY.csv")
    decision = json.loads(
        (EVIDENCE / "UNIFIED_MORPHOLOGY_V2_ACCEPTANCE.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (EVIDENCE / "UNIFIED_MORPHOLOGY_V2_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    required_files = (
        "UNIFIED_MORPHOLOGY_V2_SUMMARY.csv",
        "UNIFIED_MORPHOLOGY_V2_ACCEPTANCE_SUMMARY.csv",
        "UNIFIED_MORPHOLOGY_V2_ACCEPTANCE.json",
        "UNIFIED_MORPHOLOGY_V2_CONFUSION_MATRIX.csv",
        "UNIFIED_MORPHOLOGY_V2_EXAMPLE_OUTPUT.json",
        "UNIFIED_MORPHOLOGY_V2_MANIFEST.json",
        "UNIFIED_MORPHOLOGY_V2_REPORT_ZH.md",
        "FIGURE_1_CATE_ATE_AND_MORPHOLOGY.svg",
        "FIGURE_2_RECOVERY_RATES.svg",
        "FIGURE_3_KINK_AND_BOUNDARY_DIAGNOSTICS.svg",
    )
    checks = {
        "release_verification": verify_release(ROOT)["status"] == "PASS",
        "nine_situations": set(summary["scenario"]) == set(NINE_SITUATIONS),
        "ten_replications_each": summary["replications"].eq(10).all(),
        "common_standard_passed": summary["scenario_pass"].all(),
        "decision_passed": decision.get("all_nine_pass") is True,
        "decision_complete": decision.get("all_scenarios_complete") is True,
        "one_joint_forest": manifest.get("one_joint_multivariable_forest_per_cell")
        is True,
        "blind_common_path": manifest.get("scenario_specific_dispatch") is False
        and manifest.get("dgp_label_used_for_shape_selection") is False,
        "one_hundred_cells": manifest.get("cells_completed") == 100
        and manifest.get("cells_failed") == 0,
        "required_outputs_nonempty": all(
            (EVIDENCE / name).is_file() and (EVIDENCE / name).stat().st_size > 0
            for name in required_files
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": {name: bool(value) for name, value in checks.items()},
    }


def main() -> int:
    result = verify()
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
