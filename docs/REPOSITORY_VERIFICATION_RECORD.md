# Repository verification record

Verification date: 2026-10-31

## Scope

This verification covers the installable package, the unified nine-situation
morphology workflow, the authorized firm-covariate backbone, the executable
data-to-results acceptance path, the nine-form average-effect comparison, and
the auxiliary panel fixed-effects comparison, and the separate two- and
three-component additive examples.

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
| `pytest -q -p no:cacheprovider` | 50 passed |
| `orgcate verify` | PASS; unified code, configuration, freeze, and evidence hashes matched |
| `orgcate doctor --json` | PASS; dependency lock and backbone hash matched |
| `python tools/verify_unified_morphology_evidence.py` | PASS; 100/100 cells and nine common-standard results verified |
| `python additive_coexistence/verify.py` | PASS; two/three-component evidence verified |
| `python tools/audit_public_release.py --allow-metadata-pending` | PASS; zero findings |
| publication-prose blocklist in the public-release audit | PASS; zero findings |

## Average-effect comparison

The command

```powershell
orgcate ate-comparison --config configs/ate_estimator_comparison_v1.json --output artifacts/ate_estimator_comparison_v1
```

completed 20 replications for each of the nine one-dimensional HTE forms at
`n=3000` (180 replications in total). It compared five-fold cross-fitted
DML/AIPW with the constant treatment coefficient from `Y ~ 1 + T + X`, using
all 12 pre-treatment characteristics and HC3 uncertainty. This pre-CATE
experiment evaluates average-effect estimation for one treatment-outcome
observation per firm.

DML absolute bias ranged from 0.0013 to 0.0236 across the nine forms and its
95% interval coverage ranged from 90% to 100%. Conventional OLS absolute bias
ranged from 0.0498 to 0.3469 and coverage ranged from 0% to 75%. The aggregate
report, figure, long results, summary, and manifest are stored in
`reports/ate_estimator_comparison_v1/`.

The public workflow applies the frozen DML-ATE/CATE failure test. The absolute
gap is divided by `SD(Y_train)` and compared with the `.10` threshold; failure
returns `INCONCLUSIVE`.

The separate panel fixed-effects comparison completed 10 replications for each
of the nine forms (90 replications in total). It expanded the frozen TRAIN
firms into six relative periods, absorbed firm and period effects, and used
firm-clustered CR1 uncertainty. True ATE, DML-ATE, and CATE
were reused from frozen evidence; no causal forest was refitted. The fixed-
effects estimate closely recovered ATT in all nine forms. It recovered ATE in
the homogeneous form, while differences from ATE appeared when treatment
selection was associated with heterogeneous effects. The aggregate table,
figure, long results, method freeze, and manifest are stored in
`reports/fixed_effects_comparison_v1/`.

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

The end-to-end audit identified and corrected a target-population mismatch in
the DML-ATE/CATE failure test. DML-ATE targeted the TRAIN firms while the former
CATE mean described the TEST firms. The corrected test standardizes the final
reconstructed curve to the TRAIN focal-characteristic distribution, so both
quantities target the same firms; untouched TEST outcomes remain reserved for
independent confirmation. All nine gap tests then passed. The largest gap was
`.07457` in the kinked situation, below the unchanged `.10` threshold. Exact
morphology labels remained 9/9.

## Complete reviewer artifact

The CI workflow now recalculates the nine-situation analysis, the 180-cell
OLS/DML comparison, and the 90-cell panel fixed-effects comparison. It verifies
the additive coexistence evidence and then assembles these outputs together
with the optional English theory-development prompt and worked output. The
complete package contains 165 nonempty files and excludes row-level generated
data, compressed data, caches, and environments. Its entry page links directly
to the nine-situation overview, both average-effect comparisons, the additive
results, the complete Prompt, the worked model output, and the technical
appendix. The package verifier checks these required sections and SVG contents,
then checks the byte count and SHA256 hash of every file against the manifest.

## Decision

The active single-morphology repository path uses one unified runner and
classifier for all nine situations. The additive module reuses the same
estimator components.
Software license and citation metadata remain a research-team decision and are
not part of this technical verification.
