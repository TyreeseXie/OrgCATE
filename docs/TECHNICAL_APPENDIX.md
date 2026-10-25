# Technical appendix

This appendix states the operational method implemented in the repository. It
uses the same core information as the README so that the estimation and
validation contract can be audited without relying on a separate manuscript.

## Data and estimands

The bundled L=3 Compustat-derived backbone supplies pre-treatment firm
characteristics and firm identifiers. Treatment, potential outcomes, observed
outcomes, and true CATE are simulated. The practical workflow receives only
analyst fields; oracle truth is stored and evaluated separately.

The average estimand is

```text
ATE = E[Y(1) - Y(0)]
```

and the heterogeneous estimand is

```text
CATE(x) = E[Y(1) - Y(0) | X = x].
```

## DML ATE

The ATE stage uses cross-fitted nuisance predictions for the outcome regressions
`m1(X)`, `m0(X)` and propensity `e(X)`. For each held-out observation it forms
the augmented inverse-propensity score

```text
m1(X) - m0(X)
+ T * (Y - m1(X)) / e(X)
- (1-T) * (Y - m0(X)) / (1-e(X)).
```

The mean score is the DML/AIPW ATE. Cross-fitting keeps each observation out of
the nuisance models used to score it. This ATE is an independently constructed
average-effect reference; it is not obtained by averaging the same fitted
causal forest.

## CATE estimation and reconstruction

Each dataset uses one multivariable `CausalForestDML` with all 12 candidate firm
characteristics, `cv=2`, `honest=True`, 1,000 trees, minimum leaf size 5, and
maximum subsample fraction `.45`. The external split is 80% TRAIN and 20%
untouched TEST. Forest fitting and functional-family selection use TRAIN.
Untouched TEST is used only for prespecified confirmation.

The practical procedure follows three property layers:

1. continuity versus a threshold-like discontinuity;
2. monotone versus an interior extremum within continuous forms;
3. same-sign versus sign-reversing effects within the monotone branch.

These properties map to nine operational forms: homogeneous,
linear-monotone, nonlinear-monotone, steep-continuous, inverted-U, U-shaped,
kinked, threshold-step, and sign-reversal. Multiple ordered thresholds are a
subtype of threshold-step.

## ATE-CATE coherence and stopping

The mean final reconstructed CATE on TEST is compared with the TRAIN DML ATE:

```text
abs(mean reconstructed CATE_TEST - DML ATE_TRAIN) / SD(Y_TRAIN) <= 0.10.
```

Failure yields `INCONCLUSIVE`; it is not evidence for homogeneous effects.
The workflow also stops when treatment is not binary, treatment prevalence is
outside `.10`--`.90`, propensity support is unacceptable, or propensity
clipping exceeds 10%.

## Blind confirmation and stability

TRAIN fixes the family and all shape quantities. Three five-fold nuisance
refits produce fixed candidate curves. At least two of three one-sided
orthogonal-loss comparisons on TEST must favor the candidate over the relevant
no-heterogeneity model. Pairwise curve Spearman must be at least `.80`, and the
range of the primary location estimate must not exceed `.20` ECDF.

ECDF is the empirical cumulative distribution rank of the focal firm
characteristic. It runs from 0 to 1. Thus, `.10 ECDF` means an absolute error
of at most ten percentile points, not a 10% relative error.

## Simulation accuracy standard

After practical outputs are frozen, the evaluator checks truth. A heterogeneous
replication succeeds only when all four conditions hold:

```text
blind evidence
AND exact operational morphology
AND curve fidelity
AND all required shape quantities
```

Curve fidelity requires Spearman >= `.95` and normalized RMSE <= `.20`.
Locations and widths must be within `.10 ECDF`. Errors in effect levels,
slopes, curvature, contrasts, and amplitudes are converted to their implied
effect over observed support and must be no more than 20% of the true CATE
range. For example, a true CATE range of 2.0 permits an effect-scale error of
0.4. A scenario passes when at least 9 of 10 frozen replications succeed.

For nonlinear and steep tanh transitions, raw steepness `k` is diagnostic.
The primary quantities are direction, center, amplitude, 25%-75% central
transition width, and effective center gradient `|amplitude x k|`. This avoids
treating compensating changes in amplitude and `k` as substantively different
CATE transitions.

## Shape-specific outputs

| Form | Required information |
|---|---|
| homogeneous | constant effect and no supported HTE claim |
| linear-monotone | direction and slope |
| nonlinear-monotone | direction, center, amplitude, transition width, effective gradient, and local slopes |
| steep-continuous | direction, center, amplitude, transition width, effective gradient, and local slopes |
| inverted-u | turning point, curvature, peak effect, and peak-to-tail contrasts |
| u-shaped | turning point, curvature, trough effect, and tail-to-trough contrasts |
| kinked | kink position, left and right slopes, and slope change |
| threshold-step | every boundary, band effect, and adjacent jump |
| sign-reversal | zero crossing, endpoint effects and signs, global slope, and side-specific slopes |

Feature importance and variable rank are reported as descriptive information.
Neither is a substitute for held-out confirmation or shape recovery.

## Reproducibility and public audit

Configuration files and hierarchical seeds determine every run. The release
verifier checks configuration, code, result, and source hashes. GitHub Actions
runs linting, compilation, tests, frozen-evidence verification, a fresh demo,
and nine fresh data-to-results analyses. The artifact verifier requires every
expected table, figure, report, and manifest entry before CI can pass.

The downloadable CI artifact contains aggregate results only. It excludes raw
firm-level simulation rows, oracle microdata, credentials, and environment
directories.
