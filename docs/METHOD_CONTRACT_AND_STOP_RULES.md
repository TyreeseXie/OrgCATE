# Estimation sequence and stopping rules

OrgCATE reports two forms of verification:

1. `orgcate verify` checks the hashes and completeness of the recorded
   simulation evidence.
2. `orgcate contract` checks whether the procedures and release metadata
   required by the project specification are present.

A release is complete only when both checks pass.

## Estimation sequence

```text
data and treatment checks
-> cross-fitted DML/AIPW ATE and conventional estimates
-> ATE-CATE coherence check
-> one multivariable CausalForestDML
-> held-out loss and repeated-refit stability
-> support for ordering, magnitude, curvature, and boundaries
-> continuity, monotonicity, and sign classification
-> shape-specific parameter estimates
-> researcher interpretation record
```

If a check fails, subsequent quantities may be retained for diagnosis but are
not reported as a supported result.

## ATE-CATE coherence

The mean of the final reconstructed CATE on TEST is compared with the
cross-fitted DML ATE estimated on TRAIN:

```text
abs(mean reconstructed CATE_TEST - DML ATE_TRAIN)
-------------------------------------------------- <= 0.10
                    SD(Y_TRAIN)
```

The outcome standard deviation is used because a denominator based on the ATE
is unstable when the average effect is close to zero. Ratios of 0.05 and 0.15
are reported as sensitivity checks. A ratio above 0.10 produces
`INCONCLUSIVE`; it is not interpreted as evidence of homogeneous effects.

The rule was evaluated in ten new simulation cells after its configuration and
seeds were recorded. All ten were below 0.10; the median ratio was 0.058 and
the maximum was 0.096. An earlier confidence-interval equivalence rule produced
ten inconclusive results because the TEST DR-score intervals were too wide.
Accordingly, the current procedure is a point-tolerance check rather than a
statistical equivalence test.

## Eligibility checks

Before fitting the forest, the program requires:

- a binary treatment with observations in both arms;
- treatment prevalence between 0.10 and 0.90;
- OOF propensity-score 1st and 99th percentiles within 0.01 and 0.99;
- no more than 10% propensity clipping.

Failure produces `DESIGN_NOT_ELIGIBLE` and skips the forest. Thresholds are
stored in `configs/public_workflow_contract_v1.json`.

## Validation and reporting

The TRAIN sample is used for functional-form selection. TEST is used only to
compare the selected form with a constant-effect model. Three nuisance-model
refits provide a stability check. The result also states whether the data
support effect ordering, magnitude, curvature, and boundary estimation; shape
parameters are reported only when their required properties are supported.

The practical output includes continuity, monotonicity, sign relation, a
12-cell property index, and one of nine readable functional-form labels. The
nine labels are representative forms within the property classification, not
nine separate datasets. Multi-threshold-step is a subtype of threshold-step.

Simulation truth is stored separately and is used only to calculate accuracy.

## Single-scenario simulation acceptance

The nine single-scenario simulations use one success definition. A supported
heterogeneous result must pass blind TEST confirmation in at least two of three
fixed nuisance refits; the three fitted curves must have pairwise Spearman at
least `.80`; and the range of the primary location estimate may not exceed
`.20` on the frozen ECDF scale. The evaluator then requires the correct
operational morphology, curve Spearman at least `.95`, normalized curve RMSE at
most `.20`, and all required key quantities within tolerance.

Location and width error tolerance is `.10 ECDF`. Effect levels, slopes,
curvature, contrasts, and amplitudes are compared through their implied effect
over observed support and may differ by at most 20% of the true CATE range. A
scenario passes when at least 9 of 10 frozen replications satisfy every
condition. Full details are in
`docs/SINGLE_SCENARIO_ACCEPTANCE_STANDARD_V1.md`.

## Machine-readable status

`configs/github_release_method_contract_v1.json` lists each requirement, its
evidence, status, and any remaining action. At present, all methodological
requirements pass; publication metadata remains to be supplied by the research
team.
