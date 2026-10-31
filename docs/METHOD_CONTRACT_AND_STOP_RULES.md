# OrgCATE method contract and stop rules

This document records two complementary verification layers:

1. **Frozen-evidence verification:** result files exist, their hashes match, and
   the recorded benchmark recovered the nine high-level situations.
2. **Method-contract readiness:** the public workflow implements every gate
   promised in the proposal and teacher prototype.

`orgcate verify` checks the first claim. `orgcate contract` checks the second.
Together they establish frozen-evidence integrity and method-contract readiness.

## Intended execution order

```text
data/treatment eligibility
-> cross-fitted DML ATE and conventional comparison
-> one joint honest multivariable CausalForestDML
-> DML-ATE/CATE gap failure test
-> no-signal, reproducibility, and stability checks
-> supportable properties: ordering / magnitude / curvature / boundaries
-> continuity / monotonicity / sign relation / taxonomy cell
-> shape-specific quantities
-> researcher-owned theory audit or explicit abstention
```

Failed design eligibility is terminal for estimation. A failed DML-ATE/CATE
test is terminal for downstream interpretation and returns `INCONCLUSIVE`.

## Average-effect comparison and DML-ATE/CATE failure test

The conventional comparison is the constant treatment coefficient in
`Y ~ 1 + T + X`, with all 12 pre-treatment characteristics and an HC3 standard
error. It is compared with cross-fitted DML-ATE across the nine one-dimensional
HTE forms. In simulation, evaluator-only truth is used to report bias, RMSE,
and interval coverage for both estimators.

The gate uses the final orthogonal reconstructed CATE:

```text
abs(mean final reconstructed CATE standardized to TRAIN - TRAIN DML ATE)
-------------------------------------------------------------------------
                              SD(Y_train)
```

The primary threshold is `0.10`; `0.05` and `0.15` are recorded as sensitivity
values. A failure returns `INCONCLUSIVE` and prevents a supported morphology
claim. It does not imply homogeneous treatment effects.

Both quantities target the TRAIN firms. The reconstructed curve is averaged
over the TRAIN focal-characteristic distribution; values outside the frozen
reported curve support use the nearest supported endpoint. The untouched TEST
sample remains reserved for independent morphology confirmation. Its mean CATE
is retained as a descriptive output but does not enter this consistency test.

## Eligibility and public output integration

`configs/public_workflow_contract_v1.json` freezes the public eligibility
thresholds and the DML-ATE/CATE rule. The eligibility gate runs after a TRAIN-only
cross-fitted DML preflight and before the unified morphology workflow. It requires a
binary treatment with both arms, treatment prevalence in `[0.10, 0.90]`, OOF
propensity 1st/99th percentiles inside `[0.01, 0.99]`, and no more than 10%
propensity clipping. Failure skips the forest and returns
`DESIGN_NOT_ELIGIBLE`.

The public result includes conventional ATE diagnostics, property-support
declarations, the proposal's three-layer/12-cell fields, and a blank
researcher-owned Step 4 theory-audit artifact. Shape quantities are suppressed
when required properties are unsupported. All eligible inputs use one unified
candidate library and property-based classifier; neither the DGP label nor a
scenario-specific branch can select the morphology method. The DGPs, primary
forest, and confirmation split remain unchanged.

## DGP and taxonomy distinction

The benchmark uses the same real pre-treatment firm-covariate backbone with
separately simulated treatment and potential outcomes. It has nine high-level
morphology generators; multi-threshold-step is an additional subtype of the
threshold-step generator. These generators are test cases, not nine unrelated
empirical datasets.

The proposal's three-layer taxonomy is a different object. Continuity (2) x
monotonicity class (3) x sign relation (2) defines 12 feasible classification
cells. The nine generators sample representative shapes from that property
space. A public result should report both its readable morphology label and its
three-layer property/cell representation.

## Authoritative machine-readable status

`configs/github_release_method_contract_v1.json` records each requirement,
evidence, current status, release-blocking flag, and required action. All 16
machine-resolvable requirements are implemented. Public release remains blocked
until the research team resolves the license/citation/authorship item; that
human decision must not be auto-completed by code.
