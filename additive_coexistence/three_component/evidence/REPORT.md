# 3-Component Additive Coexistence Example

## Decision

**PASS** under the frozen componentwise standard.

This example uses the real L=3 firm-covariate backbone, `n=3000`, one external
80/20 TRAIN/untouched-TEST split, and one 1,000-tree `CausalForestDML` containing
all 12 candidate characteristics. The active set is prespecified for this
capability example; it is not selected using oracle truth or TEST outcomes.

## Component results

| Firm characteristic | Morphology | Successful replications | Complete recovery rate |
|---|---|---:|---:|
| Change in cash holdings | sign-reversal | 19/20 | 95% |
| Change in leverage | u-shaped | 20/20 | 100% |
| Intangible-asset intensity | inverted-u | 18/20 | 90% |

Complete recovery requires practical evidence, the exact morphology, curve
Spearman at least `.95`, normalized RMSE at most `.20`, and all prespecified
key quantities within the unchanged single-component tolerances.

## Overall and null results

- Total-CATE fidelity: 100%
- Median reconstructed-CATE Spearman: 0.993
- Median normalized RMSE: 0.025
- Matched all-null family-wise false confirmation: 5%

`FIGURE_1_RECOVERY_AND_NULL_CONTROL.svg` displays each component separately,
alongside overall CATE fidelity and the all-null error rate.
`FIGURE_2_COMPONENT_CURVES.svg` displays the truth, median estimate, and
10%-90% interval across 20 fresh replications.

The result supports this clear, prespecified, additive example. It does not
establish automatic active-set discovery, interactions, or universal recovery
for arbitrary combinations.

## Research implication

The three-component result extends the demonstrated boundary to three
different functional forms in one CATE surface. All components retain the
single-characteristic recovery standard, total-CATE fidelity remains complete,
and family-wise false confirmation remains within the prespecified ceiling.
The claim is therefore component separation under this frozen additive design,
not merely accurate prediction of the combined effect.
