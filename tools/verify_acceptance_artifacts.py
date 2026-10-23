"""Verify the downloadable data-to-results acceptance artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_BACKBONE = {
    "path": "execution/phase2_real_backbone_v1/headline_one_anchor_per_firm_L3.pkl.gz",
    "sha256": "91c963e02cbdc17a19de85868ba44f5f4fd65791bec83906980f307a6cfd3174",
    "rows": 8648,
    "analysis_sample_n": 3000,
}
REQUIRED_FILES = {
    "ACCEPTANCE_MANIFEST.json",
    "ACCEPTANCE_MANIFEST.sha256",
    "ANALYSIS_REPORT.md",
    "RUN_MANIFEST.json",
    "diagnostics/run_diagnostics.json",
    "evaluator/oracle_evaluation.json",
    "evaluator/truth_curve.csv",
    "figures/cate_curve.svg",
    "figures/feature_importance.svg",
    "practical/cate_curve.csv",
    "practical/feature_ranking.csv",
    "practical/researcher_theory_audit.json",
    "practical/result.json",
    "practical/test_confirmation.csv",
    "statistics/summary_statistics.json",
    "tables/cate_curve_with_ate.csv",
}
FORBIDDEN_ARTIFACT_SUFFIXES = {
    ".gz",
    ".joblib",
    ".parquet",
    ".pkl",
    ".sav",
    ".zip",
}
FORBIDDEN_ROW_FILES = {"analyst.csv", "oracle.csv"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_acceptance_artifacts(output: str | Path) -> dict[str, Any]:
    root = Path(output).resolve()
    _require(root.is_dir(), f"acceptance artifact directory is missing: {root}")
    actual_files = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    }
    missing = sorted(REQUIRED_FILES.difference(actual_files))
    _require(not missing, f"required acceptance files are missing: {missing}")
    empty = sorted(name for name, path in actual_files.items() if path.stat().st_size == 0)
    _require(not empty, f"acceptance files are empty: {empty}")

    forbidden = sorted(
        name
        for name, path in actual_files.items()
        if path.suffix.lower() in FORBIDDEN_ARTIFACT_SUFFIXES
        or path.name.lower() in FORBIDDEN_ROW_FILES
    )
    _require(not forbidden, f"row-level or binary data entered the artifact: {forbidden}")

    manifest_path = actual_files["ACCEPTANCE_MANIFEST.json"]
    manifest = _read_json(manifest_path)
    _require(manifest.get("status") == "completed", "acceptance manifest is incomplete")
    input_record = manifest.get("input", {})
    _require(input_record.get("path") == EXPECTED_BACKBONE["path"], "wrong input path")
    _require(
        input_record.get("sha256") == EXPECTED_BACKBONE["sha256"],
        "wrong input SHA256",
    )
    _require(input_record.get("rows_read") == EXPECTED_BACKBONE["rows"], "wrong input row count")
    _require(
        input_record.get("analysis_sample_n") == EXPECTED_BACKBONE["analysis_sample_n"],
        "wrong analysis sample size",
    )
    generated = manifest.get("generated_data", {})
    _require(generated.get("row_level_generated_data_in_artifact") is False, "row data policy failed")
    security = manifest.get("security", {})
    _require(not any(security.values()), "artifact security declaration failed")
    analysis = manifest.get("analysis", {})
    _require(analysis.get("learner") == "econml.dml.CausalForestDML", "wrong learner")
    _require(len(analysis.get("candidate_characteristics", [])) == 12, "wrong feature count")

    recorded_outputs = {item["path"]: item for item in manifest.get("outputs", [])}
    for name, record in recorded_outputs.items():
        path = root / name
        _require(path.is_file(), f"manifest output is missing: {name}")
        _require(path.stat().st_size == record["bytes"], f"output byte count changed: {name}")
        _require(_sha256(path) == record["sha256"], f"output hash changed: {name}")
    expected_recorded = set(actual_files).difference(
        {"ACCEPTANCE_MANIFEST.json", "ACCEPTANCE_MANIFEST.sha256"}
    )
    _require(set(recorded_outputs) == expected_recorded, "manifest output inventory is incomplete")

    checksum_line = actual_files["ACCEPTANCE_MANIFEST.sha256"].read_text(
        encoding="ascii"
    ).strip()
    expected_checksum = f"{_sha256(manifest_path)}  ACCEPTANCE_MANIFEST.json"
    _require(checksum_line == expected_checksum, "acceptance manifest checksum failed")

    statistics = _read_json(actual_files["statistics/summary_statistics.json"])
    numeric_fields = (
        "dml_ate",
        "dml_ate_standard_error",
        "true_ate",
        "ate_error",
        "cate_rank_correlation",
        "cate_rmse",
        "ate_cate_standardized_gap",
    )
    _require(
        all(math.isfinite(float(statistics[field])) for field in numeric_fields),
        "summary statistics contain non-finite values",
    )
    _require(statistics.get("analysis_sample_n") == 3000, "statistics sample size failed")

    curve = pd.read_csv(actual_files["tables/cate_curve_with_ate.csv"])
    _require(len(curve) >= 100, "aggregate CATE curve is unexpectedly short")
    _require(
        {"u", "estimated_cate", "true_cate", "dml_ate"}.issubset(curve.columns),
        "aggregate CATE curve columns are incomplete",
    )
    ranking = pd.read_csv(actual_files["practical/feature_ranking.csv"])
    _require(len(ranking) == 12, "feature ranking must contain 12 characteristics")
    _require(
        {"feature", "importance", "rank"}.issubset(ranking.columns),
        "feature ranking columns are incomplete",
    )

    for name in ("figures/cate_curve.svg", "figures/feature_importance.svg"):
        text = actual_files[name].read_text(encoding="utf-8")
        _require("<svg" in text and "</svg>" in text, f"invalid SVG figure: {name}")
        _require(len(text) > 1000, f"SVG figure is unexpectedly small: {name}")
    report = actual_files["ANALYSIS_REPORT.md"].read_text(encoding="utf-8")
    for phrase in ("What was analyzed", "DML ATE", "Data boundary"):
        _require(phrase in report, f"analysis report is incomplete: {phrase}")

    return {
        "status": "PASS",
        "files_verified": len(actual_files),
        "input_sha256": input_record["sha256"],
        "analysis_sample_n": input_record["analysis_sample_n"],
        "estimated_morphology": statistics["estimated_morphology"],
        "practical_state": statistics["practical_state"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = verify_acceptance_artifacts(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
