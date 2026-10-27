# 2-Component Additive Coexistence Example

## Decision

**PASS** under the frozen componentwise standard.

This example uses the real L=3 firm-covariate backbone, `n=3000`, one external
80/20 TRAIN/untouched-TEST split, and one 1,000-tree `CausalForestDML` containing
all 12 candidate characteristics. The active set is prespecified for this
capability example; it is not selected using oracle truth or TEST outcomes.

## Component results

| Firm characteristic | Morphology | Successful replications | Complete recovery rate |
|---|---|---:|---:|
| Intangible-asset intensity | u-shaped | 20/20 | 100% |
| Fixed-asset intensity | inverted-u | 18/20 | 90% |

Complete recovery requires practical evidence, the exact morphology, curve
Spearman at least `.95`, normalized RMSE at most `.20`, and all prespecified
key quantities within the unchanged single-component tolerances.

## Overall and null results

- Total-CATE fidelity: 100%
- Median reconstructed-CATE Spearman: 0.997
- Median normalized RMSE: 0.025
- Matched all-null family-wise false confirmation: 0%

`FIGURE_1_RECOVERY_AND_NULL_CONTROL.svg` displays each component separately,
alongside overall CATE fidelity and the all-null error rate.
`FIGURE_2_COMPONENT_CURVES.svg` displays the truth, median estimate, and
10%-90% interval across 20 fresh replications.

The result supports this clear, prespecified, additive example. Active-set
discovery, interactions, and broader combinations are separate capability
questions.

## Research implication

The two-component result is the first extension beyond the
single-characteristic benchmark. It shows that the complete morphology and
key-quantity standard can be retained for both components after their effects
are combined in one CATE surface. Passing total-CATE and all-null checks means
the separation is not obtained by trading away overall accuracy or null
control.
