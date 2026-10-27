"""Verify that a downloadable additive-coexistence artifact is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_ROOT_FILES = (
    "README.md",
    "README.zh-CN.md",
    "METHOD.md",
    "ADDITIVE_COEXISTENCE_REPORT.md",
    "ADDITIVE_COEXISTENCE_REPORT.zh-CN.md",
    "verify.py",
)
REQUIRED_EVIDENCE_FILES = (
    "COMPONENT_SUMMARY.csv",
    "SCENARIO_SUMMARY.csv",
    "NULL_SUMMARY.csv",
    "REPORT.md",
    "REPORT.zh-CN.md",
    "FIGURE_1_RECOVERY_AND_NULL_CONTROL.svg",
    "FIGURE_2_COMPONENT_CURVES.svg",
    "EVIDENCE_MANIFEST.json",
)


def required_files(root: Path) -> list[Path]:
    paths = [root / relative for relative in REQUIRED_ROOT_FILES]
    for label in ("two_component", "three_component"):
        paths.append(root / label / "config.json")
        paths.extend(root / label / "evidence" / name for name in REQUIRED_EVIDENCE_FILES)
    return paths


def verify(root: Path) -> dict[str, object]:
    checks: dict[str, bool] = {}
    for path in required_files(root):
        checks[str(path.relative_to(root))] = path.is_file() and path.stat().st_size > 0
    for label in ("two_component", "three_component"):
        manifest_path = root / label / "evidence" / "EVIDENCE_MANIFEST.json"
        if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks[f"{label}_manifest_decision"] = manifest.get("decision") == "PASS"
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "required_file_count": len(checks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = verify(args.root.resolve())
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
