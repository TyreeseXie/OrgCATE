# Repository verification record

Verification date: 2026-10-27

## Scope

This verification covers the installable package, the unified nine-situation
morphology workflow, the authorized firm-covariate backbone, the executable
data-to-results acceptance path, and the separate two- and three-component
additive examples.

## Input data

The executable acceptance run used
`execution/phase2_real_backbone_v1/headline_one_anchor_per_firm_L3.pkl.gz`.
The file contains 8,648 rows of pre-treatment firm covariates and has SHA256
`91c963e02cbdc17a19de85868ba44f5f4fd65791bec83906980f307a6cfd3174`.
Each situation sampled 3,000 firms. Treatment, potential outcomes, observed
outcomes, CATE, and morphology truth were generated during the run.

## Checks

| Check | Result |
|---|---|
| `ruff check src tests execution tools additive_coexistence` | PASS |
| `python -m compileall -q src tests execution tools additive_coexistence` | PASS |
| `pytest -q -p no:cacheprovider` | 44 passed |
| `orgcate verify` | PASS; unified code, configuration, freeze, and evidence hashes matched |
| `orgcate doctor --json` | PASS; dependency lock and backbone hash matched |
| `python tools/verify_unified_morphology_evidence.py` | PASS; 100/100 cells and nine common-standard results verified |
| `python additive_coexistence/verify.py` | PASS; two/three-component evidence verified |
| `python tools/audit_public_release.py --allow-metadata-pending` | PASS; zero findings |

## Fresh data-to-results run

The command

```powershell
orgcate acceptance --config configs/data_to_results_acceptance_v1.json --output artifacts/unified_acceptance_check
```

analyzed all nine situations from the bundled backbone. Independent artifact
verification reported `PASS`, 9/9 exact labels, 3,000 firms per situation, 123
nonempty output files, and the expected input hash. The package contains
aggregate statistics, diagnostics, figures, curves, reports, and manifests; it
does not contain firm-level generated treatment, outcomes, or oracle rows.

## Decision

The active single-morphology repository path is unified: it has no
situation-specific runtime dispatch and no separate kink or nonlinear runner.
The separate additive module reuses the same unified estimator components.
Software license and citation metadata remain a research-team decision and are
not part of this technical verification.
