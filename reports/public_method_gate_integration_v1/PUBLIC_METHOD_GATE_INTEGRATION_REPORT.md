# Method integration check

Date: 2026-10-22

This check evaluated the complete estimation sequence without changing the DGP,
the 12 candidate characteristics, the 80/20 split, the primary
`CausalForestDML`, or the recorded confirmation experiments.

```text
treatment and overlap checks
-> five-fold cross-fitted DML/AIPW ATE and conventional estimates
-> 10% DML-ATE/CATE coherence rule
-> one multivariable honest CausalForestDML
-> TEST comparison and property assessment
-> property classification and functional form
-> researcher interpretation record
```

## Checks added to the public command

| Component | Implementation |
|---|---|
| Eligibility | Binary treatment, both treatment arms, prevalence, OOF propensity tails, and clipping are checked before the forest is fitted. |
| ATE | Five-fold cross-fitted DML/AIPW is the primary estimate; difference in means and adjusted OLS are reported for comparison. |
| ATE-CATE | `abs(mean(CATE_TEST)-DML ATE_TRAIN)/SD(Y_TRAIN)` must not exceed 0.10. |
| CATE properties | The output states whether ordering, magnitude, curvature, and boundary estimates are supported. |
| Classification | Continuity, monotonicity, sign relation, 12-cell index, and a readable functional form are reported. |
| Interpretation | The command creates a blank record for mechanisms, sources, rejections, and the final research decision. |
| Shape parameters | Refits and final summaries are reported through `shape_quantities`, `key_quantities`, and `interpretation_en`. |

Eligibility thresholds are stored in
`configs/public_workflow_contract_v1.json`. Threshold-step reports an
operational boundary and two-side contrast; it is not a formal discontinuity
test.

## End-to-end example

A sign-reversal example produced the following output:

| Quantity | Result |
|---|---:|
| Treatment prevalence | 0.3042 |
| DML/AIPW ATE | 0.7141 |
| Difference in means | 0.6079 |
| Adjusted OLS | 0.5786 |
| Mean reconstructed CATE on TEST | 0.6611 |
| Standardized ATE-CATE difference | 0.0416 |
| Threshold | 0.1000 |
| Result state | `SUPPORTED_CANDIDATE` |
| Property classification | continuous / linear / opposite, cell 2 |

True ATE and CATE were used only in the separate evaluator output.

## Software checks at integration

```text
targeted tests:       12 passed
full test suite:     110 passed
ruff:                PASS
compileall:          PASS
orgcate verify:      PASS 12/12
orgcate contract:    BLOCKED 16/17
```

The remaining contract item is publication metadata: license, authors,
affiliations, contact details, and citation text.
