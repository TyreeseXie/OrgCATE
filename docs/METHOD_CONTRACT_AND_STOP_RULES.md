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
-> ATE-CATE aggregate-coherence gate
-> one joint honest multivariable CausalForestDML
-> no-signal, reproducibility, and stability checks
-> supportable properties: ordering / magnitude / curvature / boundaries
-> continuity / monotonicity / sign relation / taxonomy cell
-> shape-specific quantities
-> researcher-owned theory audit or explicit abstention
```

Any failed upstream gate is terminal for downstream interpretation. Diagnostic
estimates remain available on disk, and the practical state records that the
candidate was not supported.

## ATE-CATE point-tolerance rule is frozen for integration

The human research lead selected the final orthogonal reconstructed CATE, not
raw forest CATE, as the result-bearing quantity for aggregate coherence. The
approved lightweight gate is:

```text
abs(mean final reconstructed CATE_TEST - TRAIN cross-fitted DML ATE)
------------------------------------------------------------------- <= 0.10
                           SD(Y_train)
```

The denominator is the TRAIN outcome scale rather than absolute DML ATE so the
gate remains defined under cancellation and near-zero ATE. The `0.05` and
`0.15` ratios are sensitivity analyses only. A failure routes to upstream
`INCONCLUSIVE`; it does not imply homogeneous treatment effects and does not
route to `DESIGN_NOT_ELIGIBLE`.

A pre-frozen fresh ten-cell confirmation passed 10/10 at the 10% rule, with a
median ratio of 0.058 and maximum of 0.096. The earlier paired-equivalence
pilot produced 10/10 INCONCLUSIVE and 0/10 DRIFT because its TEST DR-score
intervals were too wide; it remains a documented negative result. Therefore
the public claim is practical point tolerance, not statistical equivalence.
The point gate is now wired into the public runtime state machine. Failure
overrides any downstream supported-candidate state with `INCONCLUSIVE`.

## Eligibility and public output integration

`configs/public_workflow_contract_v1.json` freezes the public eligibility
thresholds and the ATE-CATE rule. The eligibility gate runs after a TRAIN-only
cross-fitted DML preflight and before the unified morphology workflow. It requires a
binary treatment with both arms, treatment prevalence in `[0.10, 0.90]`, OOF
propensity 1st/99th percentiles inside `[0.01, 0.99]`, and no more than 10%
propensity clipping. Failure skips the forest and returns
`DESIGN_NOT_ELIGIBLE`.

The public result now includes conventional ATE diagnostics, property-support
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
