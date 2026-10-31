"""Assemble one reviewer-facing analysis package from verified outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _copy(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--ate", type=Path, required=True)
    parser.add_argument("--fixed-effects", type=Path, required=True)
    parser.add_argument("--additive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    _copy(args.acceptance.resolve(), output / "nine_situation_analysis")
    _copy(args.ate.resolve(), output / "average_effect_comparison")
    _copy(args.fixed_effects.resolve(), output / "fixed_effects_comparison")
    _copy(args.additive.resolve(), output / "additive_coexistence")
    theory = output / "optional_theory_development"
    theory.mkdir()
    for name in ("PROMPT.md", "AI_Strategic_Decision_Making_Theory_Development.txt"):
        shutil.copy2(ROOT / "optional_theory_development" / name, theory / name)
    documentation = output / "documentation"
    documentation.mkdir()
    shutil.copy2(ROOT / "docs" / "TECHNICAL_APPENDIX.docx", documentation / "TECHNICAL_APPENDIX.docx")

    (output / "README.md").write_text(
        """# OrgCATE complete analysis package

This package contains the outputs produced or verified by the public workflow.

## Start here

| Question | Open this file |
|---|---|
| What did the nine-situation analysis find? | [`nine_situation_analysis/ANALYSIS_REPORT.md`](nine_situation_analysis/ANALYSIS_REPORT.md) |
| What do the nine fitted CATE curves look like? | [`nine_situation_analysis/figures/nine_scenario_cate_overview.svg`](nine_situation_analysis/figures/nine_scenario_cate_overview.svg) |
| How do conventional OLS and DML-ATE compare? | [`average_effect_comparison/ATE_ESTIMATOR_COMPARISON_REPORT.md`](average_effect_comparison/ATE_ESTIMATOR_COMPARISON_REPORT.md) and [`ATE_ESTIMATOR_COMPARISON.svg`](average_effect_comparison/ATE_ESTIMATOR_COMPARISON.svg) |
| How do fixed effects, DML-ATE, and CATE compare? | [`fixed_effects_comparison/PANEL_FIXED_EFFECTS_COMPARISON_REPORT.md`](fixed_effects_comparison/PANEL_FIXED_EFFECTS_COMPARISON_REPORT.md) and [`PANEL_FIXED_EFFECTS_COMPARISON.svg`](fixed_effects_comparison/PANEL_FIXED_EFFECTS_COMPARISON.svg) |
| Can two or three HTE components be recovered together? | [`additive_coexistence/ADDITIVE_COEXISTENCE_REPORT.md`](additive_coexistence/ADDITIVE_COEXISTENCE_REPORT.md) |
| What is supplied to the optional theory-development model? | [`optional_theory_development/PROMPT.md`](optional_theory_development/PROMPT.md) |
| What does the worked theory-development output look like? | [`optional_theory_development/AI_Strategic_Decision_Making_Theory_Development.txt`](optional_theory_development/AI_Strategic_Decision_Making_Theory_Development.txt) |
| Where are the formulas and implementation details? | [`documentation/TECHNICAL_APPENDIX.docx`](documentation/TECHNICAL_APPENDIX.docx) |

## Headline results

- All nine fresh analyses passed the prespecified DML-ATE/CATE failure test. The largest standardized gap was `0.0746`, below the `0.10` threshold.
- The unified morphology procedure recovered the correct form in all nine fresh analyses and reported the associated slopes, turning points, curvature, crossings, kinks, or boundaries.
- The two-component example recovered both component morphologies at rates of `90%` and `100%`; the three-component example recovered its three component morphologies at `95%`, `100%`, and `90%`.
- The complete English theory-development prompt and its worked output are included as separate, directly readable files.

| Directory | Contents |
|---|---|
| `nine_situation_analysis/` | Fresh analysis of all nine HTE forms, including statistics, diagnostics, CATE curves, feature rankings, figures, and reports |
| `average_effect_comparison/` | Conventional constant-effect OLS and cross-fitted DML-ATE results across the nine forms |
| `fixed_effects_comparison/` | True ATE, firm-and-period fixed effects, DML-ATE, and CATE comparison |
| `additive_coexistence/` | Verified two- and three-component additive coexistence evidence |
| `optional_theory_development/` | Reusable Organization Science prompt and worked model output |
| `documentation/` | Locked technical appendix with derivations, implementation details, results, and the worked theory-development example |

The manifest records the Git commit, byte count, and SHA256 hash of every file
in this package. No row-level generated analyst or oracle data are included.
""",
        encoding="utf-8",
    )

    records = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "COMPLETE_ANALYSIS_MANIFEST.json":
            records.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        "schema_version": "orgcate-complete-analysis-artifact-v1",
        "git_commit": _commit(),
        "components": [
            "nine_situation_analysis",
            "average_effect_comparison",
            "fixed_effects_comparison",
            "additive_coexistence",
            "optional_theory_development",
            "documentation",
        ],
        "files": records,
    }
    (output / "COMPLETE_ANALYSIS_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Assembled complete analysis artifact: {output} ({len(records)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
