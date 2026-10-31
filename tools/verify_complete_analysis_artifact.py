"""Verify the reviewer-facing complete analysis package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED = {
    "README.md",
    "COMPLETE_ANALYSIS_MANIFEST.json",
    "nine_situation_analysis/ANALYSIS_REPORT.md",
    "nine_situation_analysis/figures/nine_scenario_cate_overview.svg",
    "average_effect_comparison/ATE_ESTIMATOR_COMPARISON_REPORT.md",
    "average_effect_comparison/ATE_ESTIMATOR_COMPARISON.svg",
    "fixed_effects_comparison/PANEL_FIXED_EFFECTS_COMPARISON_REPORT.md",
    "fixed_effects_comparison/PANEL_FIXED_EFFECTS_COMPARISON.svg",
    "additive_coexistence/ADDITIVE_COEXISTENCE_REPORT.md",
    "optional_theory_development/PROMPT.md",
    "optional_theory_development/AI_Strategic_Decision_Making_Theory_Development.txt",
    "documentation/TECHNICAL_APPENDIX.docx",
}
FORBIDDEN_SUFFIXES = {
    ".gz",
    ".joblib",
    ".parquet",
    ".pkl",
    ".pyc",
    ".sav",
    ".zip",
}
FORBIDDEN_NAMES = {"analyst.csv", "oracle.csv"}

CONTENT_REQUIREMENTS = {
    "README.md": (
        "## Start here",
        "## Headline results",
        "optional_theory_development/PROMPT.md",
    ),
    "nine_situation_analysis/ANALYSIS_REPORT.md": (
        "## Nine-situation results",
        "The DML-ATE/CATE gap test passed in 9 of 9 runs",
        "### homogeneous",
        "### linear-monotone",
        "### nonlinear-monotone",
        "### steep-continuous",
        "### inverted-u",
        "### u-shaped",
        "### kinked",
        "### threshold-step",
        "### sign-reversal",
    ),
    "average_effect_comparison/ATE_ESTIMATOR_COMPARISON_REPORT.md": (
        "# Conventional constant-effect OLS and DML-ATE comparison",
        "Median relative OLS-DML deviation",
    ),
    "fixed_effects_comparison/PANEL_FIXED_EFFECTS_COMPARISON_REPORT.md": (
        "# Panel fixed-effects comparison",
        "Firm + period FE",
        "DML-ATE",
        "CATE",
    ),
    "additive_coexistence/ADDITIVE_COEXISTENCE_REPORT.md": (
        "two-component",
        "three-component",
    ),
    "optional_theory_development/PROMPT.md": (
        "## 2. Worked example",
        "## 3. Reusable prompt template with example entries",
        "### Tasks",
        "## 4. Suggested output structure",
    ),
    "optional_theory_development/AI_Strategic_Decision_Making_Theory_Development.txt": (
        "EMPIRICAL PHENOMENON",
        "COMPETING ORGANIZATIONAL MECHANISMS",
        "TESTABLE THEORETICAL PROPOSITIONS",
        "NEXT EVIDENCE",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify(output: Path) -> int:
    files = {
        path.relative_to(output).as_posix(): path
        for path in output.rglob("*")
        if path.is_file()
    }
    _require(REQUIRED.issubset(files), f"missing files: {sorted(REQUIRED - files.keys())}")
    _require(all(path.stat().st_size > 0 for path in files.values()), "empty output file")
    forbidden = [
        name
        for name, path in files.items()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.lower() in FORBIDDEN_NAMES
    ]
    _require(not forbidden, f"forbidden files in artifact: {forbidden}")
    _require(
        not any("__pycache__" in Path(name).parts for name in files),
        "cache directory entered the artifact",
    )
    for name, phrases in CONTENT_REQUIREMENTS.items():
        content = files[name].read_text(encoding="utf-8")
        missing = [phrase for phrase in phrases if phrase not in content]
        _require(not missing, f"missing required content in {name}: {missing}")
    for name in (
        "nine_situation_analysis/figures/nine_scenario_cate_overview.svg",
        "average_effect_comparison/ATE_ESTIMATOR_COMPARISON.svg",
        "fixed_effects_comparison/PANEL_FIXED_EFFECTS_COMPARISON.svg",
    ):
        _require("<svg" in files[name].read_text(encoding="utf-8"), f"invalid SVG: {name}")
    manifest = json.loads(files["COMPLETE_ANALYSIS_MANIFEST.json"].read_text(encoding="utf-8"))
    records = {item["path"]: item for item in manifest["files"]}
    expected = set(files) - {"COMPLETE_ANALYSIS_MANIFEST.json"}
    _require(set(records) == expected, "complete manifest inventory mismatch")
    for name, record in records.items():
        path = files[name]
        _require(path.stat().st_size == record["bytes"], f"byte count mismatch: {name}")
        _require(_sha256(path) == record["sha256"], f"hash mismatch: {name}")
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    count = verify(args.output.resolve())
    print(f"Complete analysis artifact verified: {count} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
