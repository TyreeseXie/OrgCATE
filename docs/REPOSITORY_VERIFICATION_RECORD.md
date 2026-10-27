# Repository verification record

Verification date: 2026-10-26

Repository baseline before this verification update:
`bf1aaf73b29288446db9d9ed21d5ee8027889a67`

The files added or revised in this update are hash-listed in
`PUBLIC_RELEASE_MANIFEST.json`.

## Scope

This record covers the OrgCATE repository prepared for public review. It
verifies the Python package, frozen single-scenario evidence, bundled
firm-covariate backbone, executable data-to-results workflow, and the separate
two- and three-component additive coexistence examples. Broader component-count
and interaction studies remain outside the verified public claim.

## Input data

The executable acceptance run used:

`execution/phase2_real_backbone_v1/headline_one_anchor_per_firm_L3.pkl.gz`

- Classification: authorized Compustat-derived pre-treatment firm-covariate
  backbone.
- Available rows: 8,648.
- SHA256:
  `91c963e02cbdc17a19de85868ba44f5f4fd65791bec83906980f307a6cfd3174`.
- Analysis sample: 3,000 firms for each functional form.
- Treatment, potential outcomes, observed outcomes, CATE, and morphology truth
  were simulated during the verification run.

No pre-existing result table was used as an analysis input.

## Commands and results

| Check | Result |
|---|---|
| `ruff check src tests execution tools additive_coexistence` | PASS |
| `python -m compileall -q src tests execution tools additive_coexistence` | PASS |
| `pytest -q -p no:cacheprovider` | 48 passed |
| `orgcate doctor --json` | PASS; package lock and backbone hash matched |
| `orgcate verify` | PASS; all frozen and unified evidence checks passed |
| `python tools/verify_single_scenario_evidence.py` | PASS |
| `python additive_coexistence/verify.py` | PASS; separate two/three evidence and hashes verified |
| `python tools/assemble_additive_coexistence_artifact.py artifacts/additive_coexistence` | PASS; module package assembled |
| `python tools/verify_additive_coexistence_artifact.py artifacts/additive_coexistence` | PASS; all module files nonempty |
| `python tools/audit_public_release.py --allow-metadata-pending` | PASS; 147 files, zero findings |
| Staged LF checkout verification | PASS; tests and frozen hashes reproduced |

The executable acceptance command was:

```powershell
orgcate acceptance `
  --config configs/data_to_results_acceptance_v1.json `
  --output artifacts/final_preupload_acceptance
```

The independently executed output verifier reported:

- status: `PASS`;
- situations verified: 9;
- exact morphology labels: 9/9;
- analysis sample per situation: 3,000;
- nonempty files verified: 123;
- input SHA256 matched the authorized backbone.

## Fresh analysis results

| Functional form | DML ATE | True ATE | Absolute error | Recovered form |
|---|---:|---:|---:|---|
| homogeneous | 1.0332 | 1.0000 | 0.0332 | homogeneous |
| linear-monotone | 1.6323 | 1.6006 | 0.0318 | linear-monotone |
| nonlinear-monotone | 1.9453 | 1.9082 | 0.0371 | nonlinear-monotone |
| steep-continuous | 1.4986 | 1.4622 | 0.0364 | steep-continuous |
| inverted-u | 0.9361 | 0.9124 | 0.0237 | inverted-u |
| u-shaped | -0.9002 | -0.9124 | 0.0122 | u-shaped |
| kinked | 2.3146 | 2.2652 | 0.0494 | kinked |
| threshold-step | 1.2452 | 1.1973 | 0.0478 | threshold-step |
| sign-reversal | 0.9176 | 0.9157 | 0.0019 | sign-reversal |

All nine standardized DML-ATE/final-CATE coherence checks passed. The largest
absolute DML-ATE error in this fixed-seed acceptance run was 0.0494.

## Output completeness

The acceptance package contains:

- one nine-form aggregate statistics table;
- one nine-panel CATE overview figure;
- two SVG figures for each functional form;
- practical CATE curves and feature rankings;
- run diagnostics and summary statistics;
- one analysis report;
- run-level and package-level SHA256 manifests.

The verifier fails if any required table, figure, report, or manifest entry is
missing, empty, or hash-inconsistent. Firm-level simulation rows are excluded
from the downloadable package.

The coexistence artifact is uploaded separately as
`orgcate-additive-coexistence-<run_id>`. It contains the integrated report and
the independent two- and three-component evidence packages; it does not contain
firm-level simulation rows.

## Repository presentation audit

Tracked filenames and text were checked for development-only coordination
files, credentials, private paths, and unpublished data artifacts. No such
material is included in the public commit.

## Decision

The single-scenario workflow and the separate two/three-component coexistence
module are ready for a standard GitHub push. Publication
metadata (`LICENSE` and `CITATION.cff`) remains a separate research-team
decision and is not required for the current code-verification upload.
