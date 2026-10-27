# OrgCATE Technical Appendix

This appendix records the technical contract behind the public workflow. The
README repeats the essential information so that the repository can be
evaluated without treating this appendix as required reading. The appendix adds
implementation details and maps each methodological claim to auditable files.

## 1. Research target

The estimand is the conditional average treatment effect

```text
tau(x) = E[Y(1) - Y(0) | X=x].
```

The objective is to evaluate how adequately a causally identified scalar ATE
represents treatment effects and, where heterogeneity is supported, to recover
the functional properties needed for theory development.

The workflow is:

```text
causal-design eligibility
  -> cross-fitted DML/AIPW ATE anchor
  -> one joint multivariable causal forest
  -> ATE-CATE coherence check
  -> candidate characteristic and rank information
  -> TRAIN-only morphology reconstruction
  -> untouched-TEST confirmation
  -> researcher-owned theory audit or abstention
```

## 2. Semi-synthetic data architecture

Real Compustat-derived pre-treatment firm histories supply covariate levels,
changes, slopes, support, dependence, mixed distributions, and group
structure. The simulation supplies treatment propensity, binary treatment,
`Y(0)`, `Y(1)`, observed `Y`, true CATE, active characteristics, and true shape
parameters. Real historical treatment and outcomes are not treated as causal
truth.

The analyst data contain observed covariates, treatment, outcome, firm ID, and
split assignment. Potential outcomes, true propensity, true CATE, active-set
labels, morphology labels, and true parameters remain evaluator-only. The
practical learner cannot read them.

## 3. Design and execution states

The practical state is one of:

- `DESIGN_INELIGIBLE`: treatment, overlap, support, or required-data checks fail;
- `SUPPORTED_CANDIDATE`: held-out evidence supports the reported property;
- `INCONCLUSIVE`: estimation ran, but a required coherence or confirmation check fails;
- `NO_CLAIM`: the workflow deliberately abstains from a theory claim.

An executable result is therefore not automatically an interpretable result.
The detailed state fields are defined in `docs/OUTPUT_SCHEMA.md` and
`schemas/orgcate-result-v1.schema.json`.

## 4. Cross-fitted DML/AIPW ATE anchor

The ATE anchor is estimated before causal-forest interpretation. Five-fold
stratified cross-fitting estimates:

- `e(X) = P(T=1 | X)` with `HistGradientBoostingClassifier`;
- `m0(X) = E[Y | T=0, X]` and `m1(X) = E[Y | T=1, X]` with separate
  `HistGradientBoostingRegressor` models.

For observation `i`, the augmented inverse-probability weighted score is

```text
psi_i = m1(X_i) - m0(X_i)
        + T_i * (Y_i - m1(X_i)) / e(X_i)
        - (1 - T_i) * (Y_i - m0(X_i)) / (1 - e(X_i)).
```

Estimated propensities are clipped to `[0.05, 0.95]` for this score. The ATE is
`mean(psi)` and its standard error is `sd(psi) / sqrt(n)`. Because every score
uses nuisance predictions from a fold that did not train those nuisance
models, flexible prediction error is separated from evaluation of that
observation. This supplies an independently estimated population-effect
anchor. Causal identification remains a property of the research design.

Implementation: `src/orgcate_sim/ate.py`.

## 5. ATE-CATE coherence stop rule

The practical comparison uses the final reconstructed CATE on untouched TEST,
not raw forest predictions:

```text
gap = abs(mean(final_CATE_test) - DML_ATE_train) / SD(Y_train).
```

The primary rule requires `gap <= 0.10`; `.05` and `.15` are recorded as
sensitivity thresholds. Failure produces `INCONCLUSIVE` and stops substantive
shape interpretation. This is a scale-adjusted coherence diagnostic between
two empirical estimates. In the frozen
single-morphology acceptance run, all 10/10 cells passed at `.10`, with median
gap `.058` and maximum gap `.096`.

Configuration: `configs/public_workflow_contract_v1.json`.

## 6. Primary CATE learner

Each dataset/outcome horizon fits one multivariable
`econml.dml.CausalForestDML` containing all 12 approved candidate firm
characteristics. The workflow does not fit one forest per characteristic.

| Setting | Frozen value |
|---|---:|
| external split | 80% TRAIN / 20% untouched TEST |
| nuisance cross-fitting | `cv=2` |
| honesty | `honest=True` |
| trees | 1,000 |
| minimum leaf size | 5 |
| forest subsample | `.45` |

The external split supplies final confirmation data. `cv=2` fits treatment and
outcome nuisance functions inside TRAIN; forest honesty is an internal tree
construction property. These mechanisms are not substitutes for one another.

Configuration: `configs/unified_morphology_v2.json`.

## 7. Orthogonal morphology reconstruction

Raw forest CATE targets heterogeneous-effect prediction and ranking. The
reconstruction layer adds smooth, readable estimates of slopes, turning
points, crossings, kinks, and jumps using TRAIN-fitted nuisance
residuals and orthogonal loss to compare prespecified theoretical shape
families. Candidate families and parameter grids are fixed before confirmation.
The untouched TEST set assesses the selected candidate; TEST outcomes do not
select the family, tune the grid, or move a boundary.

Every input uses the same candidate library and ordered decision rules. The
runtime does not inspect a simulation label to choose a shape-specific route.
Nested kink evidence is computed for every remaining nonlinear curve, and the
ordinary-versus-steep smooth distinction uses effective center gradient rather
than raw tanh steepness. Shape-specific functions are used only to read the
appropriate quantities after the common classifier has selected a label.

This two-stage, estimand-preserving design keeps `CausalForestDML` as the
primary heterogeneity learner while improving recovery of theory-relevant
functional form.

## 8. Three-layer classification and reported quantities

The classification follows three questions: continuity, monotonicity within
the continuous family, and sign relation within the monotone branch. The nine
named situations and their required outputs are:

| Situation | Required practical information |
|---|---|
| homogeneous | no supported heterogeneity claim |
| linear-monotone | direction and slope |
| nonlinear-monotone | effect ordering and nonlinear intensity |
| steep-continuous | ordering and sharp continuous-transition location |
| inverted-u | turning point, peak effect, curvature, and tail contrasts |
| u-shaped | turning point, trough effect, curvature, and tail contrasts |
| kinked | kink location and left/right slopes |
| threshold-step | boundary location, effects on both sides, and jump size |
| sign-reversal | zero crossing, endpoint effects and signs, global slope, and slopes on both sides |

Multiple ordered boundaries are a `threshold-step` subtype. Exact output field
names and English narrative templates are in `docs/SHAPE_OUTPUT_REFERENCE.md`.

## 9. Frozen single-morphology evidence

The active evidence is generated by one hash-frozen unified classifier. The
public package contains no separate shape-specific runtime branch.

| Situation | Unified recovery |
|---|---:|
| homogeneous | 9/10 |
| linear-monotone | 10/10 |
| nonlinear-monotone | 10/10 |
| steep-continuous | 10/10 |
| inverted-u | 9/10 |
| u-shaped | 10/10 |
| kinked | 9/10 |
| threshold-step | 10/10 |
| sign-reversal | 10/10 |

These results are controlled semi-synthetic recovery evidence. Empirical data
receive a shape label only when the practical evidence conditions pass. Run
`orgcate evidence` for the machine-readable evidence and `orgcate verify` for
frozen hashes.

## 10. Multiple coexisting HTE components

The coexistence branch asks whether additive components can be separated after
one joint forest has estimated the full CATE surface. The active component set
is prespecified for decomposition. TRAIN performs additive backfitting and
family selection. Untouched
TEST evaluates the full candidate and each component through ablation.

The repository includes independently confirmed clear-signal examples at
`n=3000`, with 20 active and 20 matched all-null replications for each component
count. The two-component recovery rates are 90% and 100%; the three-component
rates are 95%, 90%, and 100%. Total-CATE fidelity is 100% in both cases and
the matched all-null family-wise false-confirmation rates are 0% and 5%.
All rates use the unchanged single-component definition: blind practical
evidence, exact morphology, curve Spearman >= `.95`, normalized curve RMSE <=
`.20`, and every prespecified key quantity within tolerance. The examples use
uniformly declared clear signals (`.90` linear half-range and `2.00` quadratic
amplitude) and fresh confirmation seeds. The current evidence covers these
prespecified two- and three-component additive examples. Unknown active sets,
interactions, and higher component counts define subsequent capability studies.
Evidence, reports, and reproduction commands are under `additive_coexistence/`.

## 11. Leakage and reproducibility controls

- Configuration is frozen and hashed before fresh confirmation results.
- Hierarchical seeds distinguish backbone sampling, treatment, outcome noise,
  split, nuisance models, and estimator randomness.
- The same firm cannot appear in both TRAIN and TEST within a replication.
- Oracle fields cannot enter feature ranking, tuning, family selection,
  boundary selection, or practical status.
- Failed replications remain visible and are not silently dropped.
- Manifests record code, configuration, data hashes, versions, seed lineage,
  commands, and output hashes.

## 12. Evidence-to-file map

| Claim or contract | Primary file |
|---|---|
| ATE estimator | `src/orgcate_sim/ate.py` |
| workflow stop rules | `configs/public_workflow_contract_v1.json` |
| primary forest and unified classifier contract | `configs/unified_morphology_v2.json` |
| practical output schema | `docs/OUTPUT_SCHEMA.md` |
| reported shape quantities | `docs/SHAPE_OUTPUT_REFERENCE.md` |
| unified classifier specification | `docs/UNIFIED_MORPHOLOGY_V2_METHOD.md` |
| single-shape frozen evidence | `reports/unified_morphology_v2/` |
| confirmed two/three-component examples | `additive_coexistence/` |
| method-contract audit | `docs/METHOD_CONTRACT_AND_STOP_RULES.md` |
| repository verification | `docs/REPOSITORY_VERIFICATION_RECORD.md` |

## 13. Research responsibilities and current scope

OrgCATE provides an auditable path from a credible causal design to qualified
heterogeneity and morphology candidates. The software handles estimation,
reconstruction, confirmation, and provenance. The researcher is responsible
for design validity, construct meaning, mechanism arguments, literature
integration, and the final decision to advance or abstain from a theoretical
claim.
