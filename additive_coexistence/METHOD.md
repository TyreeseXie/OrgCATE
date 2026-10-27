# Method Contract for Additive Coexistence

## Data and learner

- Real Compustat-derived L=3 pre-treatment firm histories provide the joint
  covariate distribution.
- Treatment, outcomes, active components, and causal truth are simulated.
- Every replication samples `n=3000` firms and makes one 80/20
  TRAIN/untouched-TEST split.
- One honest 1,000-tree `CausalForestDML` jointly uses all 12 candidate firm
  characteristics (`cv=2`, `min_samples_leaf=5`, `max_samples=.45`).
- The active component set is prespecified for decomposition. Oracle truth and
  TEST outcomes cannot select variables, shapes, or parameters.

## Reconstruction and confirmation

The frozen additive target is

```text
tau(X) = tau_0 + f_1(X_1) + ... + f_J(X_J),  J in {2, 3}.
```

The joint forest estimates the complete CATE from all 12 characteristics; it
is not refitted once per component. The additive decomposition is a subsequent
reconstruction layer for the prespecified active set.

TRAIN nuisance residuals are obtained with five-fold cross-fitting. The
component curves are reconstructed through six additive backfitting cycles and
three nuisance refits. Functional families and parameters are fixed on TRAIN.
Untouched TEST estimates joint orthogonal component scores, with Holm
adjustment across the components examined in that example.

## Acceptance

For component `j` in replication `r`, complete recovery requires all of:

1. blind practical evidence and refit stability;
2. the exact morphology label;
3. component-curve Spearman at least `.95`;
4. normalized component-curve RMSE at most `.20`;
5. all shape-defining locations within `.10` on the reference ECDF scale;
6. all effect-scale quantities within 20% of the true component CATE range.

An example passes when every component's success rate is at least 90%, the
total-CATE fidelity rate is at least 90%, and the matched all-null probability
of confirming any component is at most 10%. The method does not require all
components to succeed in the same replication at least 90% of the time, because
that conjunction becomes mechanically harder as component count grows.

This preserves comparability with the single-characteristic benchmark: adding
components does not loosen the accuracy required of any one component. The
additional total-CATE and family-wise checks address errors that exist only in
the coexistence setting.

## Recorded design

The clear-signal demonstration uses a common scale declared before candidate
screening: linear half-range `.90` and absolute quadratic amplitude `2.00`.
Candidate selection used five development seeds and prioritized the weakest
component, total-CATE fidelity, lower active-feature correlation, and fixed
candidate order. The selected examples were evaluated with 20 previously
unused confirmation seeds. All screened candidates and original-strength
failures remain in the private audit record.
