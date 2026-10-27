# Unified blind morphology classification v2

## Scope

The active single-component workflow uses one classifier for every eligible
dataset. A simulation scenario name is used to generate treatment-effect truth
and to score the final output, but it is never passed to morphology selection.
There is no situation-specific runtime dispatch.

## Common input and candidate library

After the design and ATE gates, one 12-characteristic honest
`CausalForestDML` is fitted. For the prespecified focal characteristic, TRAIN
data then provide five-fold cross-fitted nuisance residuals. Three nuisance
refits are used to fit the same candidate library:

- constant effect;
- linear effect;
- quadratic effect with a candidate interior center;
- smooth monotone tanh effect with a pooled center and steepness;
- continuous hinge with a candidate kink;
- zero-, one-, two-, or three-boundary piecewise-constant effects.

All candidates minimize the same orthogonal R-loss. Tanh center and steepness
are pooled across the three refits using TRAIN cross-validation so that a
single unstable refit cannot determine the smooth/steep label.

## Property-based decision order

The TRAIN-only classifier applies the same ordered rules to every curve:

1. Select the smallest supported piecewise boundary count under the frozen
   one-standard-error rule. A supported piecewise improvement over the best
   continuous family yields `threshold-step`.
2. A supported interior quadratic extremum yields `u-shaped` or
   `inverted-u`, according to curvature direction.
3. A monotone curve that crosses zero relative to the DML ATE sign yields
   `sign-reversal`.
4. A curve within the frozen near-linear tolerance, or without a supported
   nonlinear R-loss improvement, yields `linear-monotone`.
5. The nested five-fold/three-fold hinge diagnostic tests every remaining
   nonlinear curve. It requires an interior and stable candidate kink,
   same-direction slopes, a material slope change, and out-of-fold improvement
   over a straight line. Passing curves yield `kinked`.
6. Remaining smooth monotone curves are split into `nonlinear-monotone` and
   `steep-continuous` using the frozen effective center gradient
   `abs(amplitude * k)`, not raw `k` alone.

Untouched TEST never selects a family, boundary, center, or threshold. It only
checks whether the fixed nonconstant candidate improves orthogonal loss over a
constant effect. Failure yields `NO_THEORY_CLAIM`; unstable repeated fits yield
`INCONCLUSIVE`.

## Shape-specific readout

The classifier is common; only the quantities read after classification differ:

| Label | Required readout |
|---|---|
| homogeneous | level and span |
| linear-monotone | direction, slope, endpoint effects |
| nonlinear-monotone | center, amplitude, effective gradient, transition width |
| steep-continuous | center, amplitude, effective gradient, transition width |
| inverted-u / u-shaped | turning point, curvature, center-to-tail contrasts |
| kinked | kink, left slope, right slope, slope change |
| threshold-step | boundary count and locations, band effects, jumps |
| sign-reversal | zero crossing, side signs, endpoint effects, slopes |

## Simulation evaluation

Oracle truth is opened only after the practical output is fixed. A replication
passes only when blind practical evidence passes, the high-level label is
exact, curve rank and magnitude fidelity pass, and all prespecified key
quantities meet their tolerances. Real empirical applications do not have the
oracle stage and therefore report the practical state rather than recovery
accuracy.

## Active files

- configuration: `configs/unified_morphology_v2.json`;
- runner: `execution/run_unified_morphology_v2.py`;
- method freeze: `reports/UNIFIED_MORPHOLOGY_V2_METHOD_FREEZE.json`;
- fresh evidence: `reports/unified_morphology_v2/`;
- public execution path: `src/orgcate_sim/public_demo.py`.

The public repository contains this unified path rather than separate
shape-specific runners. Development history remains in Git history, not in the
active package.
