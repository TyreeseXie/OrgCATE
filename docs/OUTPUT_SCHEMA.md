# Result schema

`schemas/orgcate-result-v1.schema.json` defines the practical result produced
by the public command. Its principal sections are:

1. `eligibility`: treatment prevalence, propensity tails, and clipping;
2. `ate`: cross-fitted DML/AIPW estimate, standard error, confidence interval,
   difference in means, and conventional constant-effect OLS with HC3 uncertainty;
3. `ate_cate_coherence`: the gap between TRAIN DML-ATE and the reconstructed
   CATE standardized to the same TRAIN target firms, divided by TRAIN outcome
   variation, together with its threshold and result;
4. `forest`: specification of the joint multivariable `CausalForestDML`;
5. `moderator`: focal characteristic and ranks of all candidate variables;
6. `property_support`: support for ordering, magnitude, curvature, and
   boundaries;
7. `morphology`: functional-form label, property classification, refit-level
   estimates, final `key_quantities`, and `interpretation_en`;
8. `confirmation`: TEST comparison and final state;
9. `theory_audit`: path to the researcher-completed interpretation record.

An eligibility failure yields `DESIGN_NOT_ELIGIBLE`. Failure of
`abs(CATE_TRAIN_TARGET-DML_ATE_TRAIN)/SD(Y_train) <= 0.10` yields
`INCONCLUSIVE` and withholds
downstream morphology claims.

The property fields are `continuity`, `monotonicity`, `sign_relation`,
`taxonomy_cell`, and `requires_abduction`. These fields describe the estimated
properties directly; the nine morphology labels provide a shorter summary.

True CATE, propensity, potential outcomes, and true labels are excluded from
the practical result. In the synthetic example, truth and accuracy statistics
are written to `oracle/` and `evaluator/` rather than the analyst-facing output.

## Result states

- `SUPPORTED_CANDIDATE`: stability and TEST confirmation criteria were met;
- `INCONCLUSIVE`: an estimated pattern did not meet the stability or
  specificity requirements;
- `NO_THEORY_CLAIM`: the data did not support a nonconstant effect;
- `DESIGN_NOT_ELIGIBLE`: treatment or overlap requirements failed.

Variable rank summarizes the forest's ordering of candidate characteristics;
TEST confirmation supplies the evidence state for the selected morphology.

The interpretation template is saved as
`practical/researcher_theory_audit.json`. It records proposed explanations,
sources, rejected explanations, unresolved readings, and the final decision.
`orgcate theory-audit --input <path>` checks that the completed file follows
the schema.

See `docs/SHAPE_OUTPUT_REFERENCE.md` for the parameters reported by each
functional form.
