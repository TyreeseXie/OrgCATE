# Single-scenario acceptance standard v1

This is the only success definition for the nine single-component simulation
situations. Practical estimation is blind to simulation truth. The evaluator
opens truth only after the practical label, curve, quantities, and evidence
state are fixed.

## Upstream eligibility

A cell reaches morphology analysis only when treatment is binary, propensity
support is acceptable, the cross-fitted DML/AIPW ATE is available, and the
standardized difference between DML ATE and mean final reconstructed CATE is
at most `0.10 SD(Y_train)`. Failure ends morphology interpretation and returns
the corresponding practical stop state.

## One practical workflow

Each cell uses one 80/20 firm-level TRAIN/untouched-TEST split and one honest
1,000-tree `CausalForestDML` containing all 12 candidate firm characteristics.
TRAIN chooses the family and every shape quantity. Three five-fold nuisance
refits must give pairwise curve Spearman at least `.80`; the primary location
range must not exceed `.20` ECDF. Untouched TEST does not move a parameter or
choose a family. At least two of three fixed-candidate comparisons must improve
one-sided orthogonal loss relative to the relevant no-heterogeneity model.

Homogeneous truth passes only when the workflow makes no supported HTE claim.
For a heterogeneous situation, blind practical evidence must support the
fixed candidate.

## Evaluator accuracy

For each heterogeneous replication:

```text
RECOVERY_SUCCESS
  = BLIND_EVIDENCE_PASS
  AND EXACT_OPERATIONAL_MORPHOLOGY
  AND CURVE_FIDELITY_PASS
  AND ALL_REQUIRED_KEY_QUANTITIES_PASS
```

Curve fidelity requires Spearman >= `.95` and normalized curve RMSE <= `.20`.
Location and width errors may not exceed `.10` on the frozen ECDF scale.
Effect-level, slope, curvature, contrast, and amplitude errors are translated
to their effect over observed support and may not exceed 20% of the true CATE
range. A situation passes when at least 9 of 10 frozen replications succeed.

## Required quantities

| Situation | Required evaluator quantities |
|---|---|
| homogeneous | constant effect level and no supported HTE claim |
| linear-monotone | direction and slope |
| nonlinear-monotone | direction, center, amplitude, and 25%-75% central transition width |
| steep-continuous | direction, center, amplitude, and 25%-75% central transition width |
| inverted-u | turning point, curvature, and peak-to-tail contrast |
| u-shaped | turning point, curvature, and tail-to-trough contrast |
| kinked | kink position, left slope, right slope, and slope change |
| threshold-step | every boundary, band levels, and jumps |
| sign-reversal | zero crossing, endpoint effects/signs, global slope, and slopes on both sides |

For tanh shapes, raw `k` remains a reported diagnostic. It is not the sole
acceptance quantity because amplitude and `k` can compensate while leaving the
substantive CATE transition nearly unchanged. Effective center gradient
`|amplitude x k|` determines whether the transition is ordinary nonlinear or
steep; transition width reports the interpretable support interval over which
the central half of the effect change occurs.

Variable rank and forest importance are information outputs only. They do not
replace blind confirmation, curve accuracy, or key-quantity accuracy.
