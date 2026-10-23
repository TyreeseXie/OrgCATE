# OrgCATE

[简体中文](README.zh-CN.md)

OrgCATE is a Python package for studying treatment-effect heterogeneity in
organizational data. The package estimates an average treatment effect with
cross-fitted DML/AIPW, fits one multivariable `CausalForestDML`, and reconstructs
the CATE function along a focal firm characteristic. The final output reports
the functional form and its main quantities, such as slopes, turning points,
zero crossings, and threshold locations.

The empirical sequence is:

```text
design and overlap checks
  -> cross-fitted DML/AIPW ATE
  -> ATE-CATE coherence check
  -> one multivariable CausalForestDML
  -> held-out assessment of CATE properties
  -> functional-form classification and parameter estimates
  -> interpretation record completed by the researcher
```

Estimation, held-out confirmation, and simulation evaluation use separate data
objects. Simulation truth is used to evaluate the method and is not available
to the estimation or classification routines.

## Installation and basic checks

The reference environment uses Python 3.13 and EconML 0.17.0. Package versions
are recorded in `requirements-lock.txt`.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps

orgcate doctor
orgcate verify
orgcate contract
orgcate evidence
```

`doctor` checks the environment and bundled backbone. `verify` checks the
recorded code, configuration, and result hashes. `contract` checks whether the
method and release requirements are complete. `evidence` prints the integrated
nine-situation results.

## Synthetic example

The example below runs the full procedure with simulated covariates, treatment,
and outcomes. It uses `n=3000` and the same 1,000-tree forest specification as
the reported experiments.

```powershell
orgcate demo --scenario sign-reversal --output artifacts/public_demo
```

The output is divided into `analyst/`, `practical/`, `oracle/`, and
`evaluator/`. The first two contain the estimation inputs and reported result;
the latter two contain simulation truth and accuracy statistics. The example
also creates a blank interpretation record. After completing that record, its
schema can be checked with:

```powershell
orgcate theory-audit --input artifacts/public_demo/practical/researcher_theory_audit.json
```

## Functional forms

| Label | Reported information |
|---|---|
| homogeneous | constant effect estimate |
| linear-monotone | direction and slope |
| nonlinear-monotone | effect ordering and nonlinear intensity |
| steep-continuous | ordering and location of a sharp continuous transition |
| inverted-u | turning point, peak, and tail contrasts |
| u-shaped | turning point, trough, and tail contrasts |
| kinked | kink location and left/right slopes |
| threshold-step | boundary location and effects on each side |
| sign-reversal | zero crossing, endpoint effects, and slopes |

Multiple ordered thresholds are reported as a subtype of `threshold-step`.
They are not treated as a separate high-level class.

## Validation results

The table combines Morphology Product v1 with the separately developed
kink-resolution v2 experiment. Each proportion is based on ten confirmation
replications at `n=3000`.

| Functional form | Exact recovery |
|---|---:|
| homogeneous | 10/10 |
| linear-monotone | 10/10 |
| nonlinear-monotone | 9/10 |
| steep-continuous | 10/10 |
| inverted-u | 10/10 |
| u-shaped | 10/10 |
| kinked | 10/10 |
| threshold-step | 10/10 |
| sign-reversal | 10/10 |

The kinked row reports the final independently confirmed kink-resolution v2
result.

## Semi-synthetic reproduction

The repository includes the authorized Compustat-derived L=3 pre-treatment
firm-panel backbone used in the experiments:

`execution/phase2_real_backbone_v1/headline_one_anchor_per_firm_L3.pkl.gz`

The backbone supplies firm covariates only. Treatment, potential outcomes,
observed outcomes, true CATE, and morphology labels are simulated by the
versioned code. Full runs use the following commands:

```powershell
orgcate reproduce --target morphology-product --phase confirmation
orgcate reproduce --target morphology-product --phase report
orgcate reproduce --target kink-resolution --phase confirmation
orgcate reproduce --target kink-resolution --phase report
```

The runners preserve completed replications and record failures. `--freeze-only`
is intended for the documented development-before-confirmation procedure.

## Documentation

- `docs/METHOD_CONTRACT_AND_STOP_RULES.md`: estimation order and stopping rules.
- `docs/OUTPUT_SCHEMA.md`: result fields and state definitions.
- `docs/SHAPE_OUTPUT_REFERENCE.md`: quantities reported for each functional form.
- `reports/morphology_product_v1/`: main experiment results and figures.
- `reports/kink_resolution_v2/`: focused kink experiment.
- `DATA_NOTICE.md`: contents and authorization scope of the L=3 backbone.
- `PUBLIC_RELEASE_MANIFEST.json`: file inventory and SHA256 hashes.

Only the backbone identified in `DATA_NOTICE.md` is included. Raw downloads,
other processed panels, access records, credentials, and generated firm-level
simulation outputs are outside the release.

## Development checks

```powershell
.venv\Scripts\ruff.exe check src tests execution tools
.venv\Scripts\python.exe -m compileall -q src tests execution tools
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
python tools/audit_public_release.py --allow-metadata-pending
```

The metadata exception is removed after `LICENSE` and `CITATION.cff` are added.
