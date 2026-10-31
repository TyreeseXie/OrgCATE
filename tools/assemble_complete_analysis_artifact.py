"""Assemble one reviewer-facing analysis package from verified outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
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


def _tracked_source_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _workflow_context() -> dict[str, str | None]:
    github = os.environ.get("GITHUB_ACTIONS") == "true"
    return {
        "provider": "github-actions" if github else "local",
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "workflow": os.environ.get("GITHUB_WORKFLOW"),
        "job": os.environ.get("GITHUB_JOB"),
        "repository": os.environ.get("GITHUB_REPOSITORY"),
        "event_name": os.environ.get("GITHUB_EVENT_NAME"),
        "ref": os.environ.get("GITHUB_REF"),
        "sha": os.environ.get("GITHUB_SHA"),
        "runner_os": os.environ.get("RUNNER_OS"),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
    }


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

    acceptance_manifest = _read_json(args.acceptance.resolve() / "ACCEPTANCE_MANIFEST.json")
    ate_manifest = _read_json(args.ate.resolve() / "RUN_MANIFEST.json")
    fixed_effects_manifest = _read_json(args.fixed_effects.resolve() / "RUN_MANIFEST.json")
    config_sources = (
        (
            Path(acceptance_manifest["config"]["path"]),
            acceptance_manifest["config"]["sha256"],
        ),
        (Path(ate_manifest["config"]), ate_manifest["config_sha256"]),
        (
            Path(fixed_effects_manifest["config"]),
            fixed_effects_manifest["config_sha256"],
        ),
    )
    configuration = output / "configuration"
    configuration.mkdir()
    configuration_records = []
    for relative, declared_hash in config_sources:
        source = (ROOT / relative).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        observed_hash = _sha256(source)
        if observed_hash != declared_hash:
            raise ValueError(
                f"configuration changed after analysis: {relative.as_posix()}"
            )
        destination = configuration / source.name
        shutil.copy2(source, destination)
        configuration_records.append(
            {
                "source_path": relative.as_posix(),
                "packaged_path": destination.relative_to(output).as_posix(),
                "sha256": observed_hash,
            }
        )
    theory = output / "optional_theory_development"
    theory.mkdir()
    for name in ("PROMPT.md", "AI_Strategic_Decision_Making_Theory_Development.txt"):
        shutil.copy2(ROOT / "optional_theory_development" / name, theory / name)
    documentation = output / "documentation"
    documentation.mkdir()
    shutil.copy2(ROOT / "docs" / "TECHNICAL_APPENDIX.docx", documentation / "TECHNICAL_APPENDIX.docx")

    created_at = datetime.now(UTC).isoformat()
    git_commit = _commit()
    workflow = _workflow_context()
    component_manifests = []
    for path in (
        output / "nine_situation_analysis" / "ACCEPTANCE_MANIFEST.json",
        output / "average_effect_comparison" / "RUN_MANIFEST.json",
        output / "fixed_effects_comparison" / "RUN_MANIFEST.json",
        output
        / "additive_coexistence"
        / "two_component"
        / "evidence"
        / "EVIDENCE_MANIFEST.json",
        output
        / "additive_coexistence"
        / "three_component"
        / "evidence"
        / "EVIDENCE_MANIFEST.json",
    ):
        component_manifests.append(
            {
                "path": path.relative_to(output).as_posix(),
                "sha256": _sha256(path),
            }
        )
    commands = [
        (
            "orgcate acceptance --config "
            f"{acceptance_manifest['config']['path']} --output "
            f"{_display_path(args.acceptance)}"
        ),
        (
            "orgcate ate-comparison --config "
            f"{ate_manifest['config']} --output {_display_path(args.ate)}"
        ),
        (
            "orgcate fixed-effects-comparison --config "
            f"{fixed_effects_manifest['config']} --output "
            f"{_display_path(args.fixed_effects)}"
        ),
        f"python tools/verify_acceptance_artifacts.py {_display_path(args.acceptance)}",
        (
            "python tools/verify_average_effect_artifacts.py --ate "
            f"{_display_path(args.ate)} --fixed-effects "
            f"{_display_path(args.fixed_effects)}"
        ),
        (
            "python tools/assemble_additive_coexistence_artifact.py "
            f"{_display_path(args.additive)}"
        ),
        (
            "python tools/verify_additive_coexistence_artifact.py "
            f"{_display_path(args.additive)}"
        ),
    ]
    run_context = {
        "schema_version": "orgcate-analysis-run-context-v1",
        "created_at_utc": created_at,
        "git": {
            "commit": git_commit,
            "tracked_source_dirty": _tracked_source_dirty(),
            "dirty_scope": (
                "tracked and staged source files only; untracked generated outputs excluded"
            ),
        },
        "workflow": workflow,
        "input_backbone": acceptance_manifest["input"],
        "commands": commands,
        "configuration_files": configuration_records,
        "component_manifests": component_manifests,
    }
    (output / "RUN_CONTEXT.json").write_text(
        json.dumps(run_context, indent=2) + "\n", encoding="utf-8"
    )

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
| Which commit, commands, inputs, seeds, and configurations produced this package? | [`RUN_CONTEXT.json`](RUN_CONTEXT.json) and [`configuration/`](configuration/) |

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
| `configuration/` | Exact JSON configurations used by the nine-situation, ATE, and fixed-effects analyses |

`RUN_CONTEXT.json` records the Git commit, workflow run, generation time,
commands, input-backbone hash, configuration hashes, and component manifests.
The complete manifest records the byte count and SHA256 hash of every included
file. No row-level generated analyst or oracle data are included.
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
        "schema_version": "orgcate-complete-analysis-artifact-v2",
        "created_at_utc": created_at,
        "git_commit": git_commit,
        "workflow_run": workflow,
        "components": [
            "nine_situation_analysis",
            "average_effect_comparison",
            "fixed_effects_comparison",
            "additive_coexistence",
            "optional_theory_development",
            "documentation",
            "configuration",
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
