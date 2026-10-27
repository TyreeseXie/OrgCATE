# OrgCATE

[简体中文](README.zh-CN.md)

OrgCATE is a researcher-directed workflow for developing organizational theory
from treatment-effect heterogeneity. Starting from a credible causal design,
it combines a cross-fitted ATE estimate, multivariable causal forests,
held-out confirmation, and interpretable CATE morphology reconstruction.

```text
credible causal design
  -> treatment and overlap eligibility
  -> cross-fitted DML/AIPW ATE anchor
  -> naive/adjusted ATE diagnostics
  -> DML-ATE/final-CATE 10% coherence stop rule
  -> one joint multivariable CausalForestDML
  -> held-out validation and supportable CATE properties
  -> prespecified focal characteristic + all-variable rank information
  -> three-layer classification + readable functional morphology
  -> supported candidate / inconclusive / no claim / design ineligible
  -> researcher-owned theory audit or abstention
```

The causal design identifies the effect, the forest estimates heterogeneity,
held-out data assess the candidate pattern, and the researcher develops and
evaluates the theoretical explanation.

## Method in one page

The estimand is the conditional average treatment effect
`tau(x) = E[Y(1) - Y(0) | X=x]`. The benchmark is semi-synthetic: real
pre-treatment Compustat-derived firm histories supply the joint covariate
distribution, while treatment, potential outcomes, observed outcomes, and
oracle CATE are simulated. Oracle fields are never available to the practical
workflow.

| Stage | What is estimated or checked | Role in the workflow |
|---|---|---|
| Design eligibility | binary treatment, overlap, support, and required variables | stops analysis when the causal design is not usable |
| ATE anchor | five-fold cross-fitted DML/AIPW estimate | provides an independently estimated population-effect reference |
| HTE estimation | one joint 12-characteristic `CausalForestDML` | estimates multivariable CATE without fitting one forest per characteristic |
| Coherence check | `abs(mean(CATE_test) - ATE_train) / SD(Y_train) <= 0.10` | stops interpretation when the final CATE surface is inconsistent with the ATE anchor |
| Morphology reconstruction | TRAIN-only orthogonal shape fitting | converts a noisy forest surface into explicit slopes, turning points, crossings, kinks, or thresholds |
| Confirmation | untouched TEST orthogonal-loss and component-ablation checks | separates construction from evidential confirmation |
| Theory audit | researcher records mechanisms, literature, exclusions, and decision | keeps theory development human-owned and permits abstention |

For observation `i`, the AIPW score is

```text
psi_i = m1(X_i) - m0(X_i)
        + T_i * (Y_i - m1(X_i)) / e(X_i)
        - (1 - T_i) * (Y_i - m0(X_i)) / (1 - e(X_i)).
```

The nuisance functions are fitted out of fold; the score clips estimated
propensities to `[0.05, 0.95]`. The primary forest uses an external 80/20
TRAIN/TEST split, `cv=2`, `honest=True`, 1,000 trees,
`min_samples_leaf=5`, and `max_samples=0.45`. Every eligible curve enters the
same constant, linear, quadratic, tanh, hinge, and piecewise candidate library.
Shape families and parameters are selected on TRAIN only; there is no
scenario-name dispatch. TEST is reserved for confirmation and oracle truth is
reserved for simulation evaluation. The full implementation
contract, formulas, output fields, and evidence-to-claim map are repeated in
[`docs/TECHNICAL_APPENDIX.md`](docs/TECHNICAL_APPENDIX.md).

## Quick start

The verified release environment is Python 3.10+ with EconML 0.17.0. Exact
versions are listed in `requirements-lock.txt`.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps

orgcate doctor
orgcate verify
orgcate contract
orgcate evidence
```

Run one fully public synthetic demonstration. It uses the frozen `n=3000`,
1,000-tree product contract, so it is a real workflow smoke rather than a toy
ten-row example.

```powershell
orgcate demo --scenario sign-reversal --output artifacts/public_demo
```

The demo writes separate `analyst/`, `oracle/`, `practical/`, and `evaluator/`
directories plus a hash manifest. Any of the nine named situations may be used
with `--scenario`.

The practical directory also contains a blank researcher-owned Step 4 record.
After completing it, validate its structure without asking the software to
invent a mechanism:

```powershell
orgcate theory-audit --input artifacts/public_demo/practical/researcher_theory_audit.json
```

## Nine-situation morphology layer

| Situation | Substantive output |
|---|---|
| homogeneous | no supported heterogeneity claim |
| linear-monotone | direction and slope |
| nonlinear-monotone | effect ordering and nonlinear intensity |
| steep-continuous | ordering and sharp continuous-transition location |
| inverted-u | turning point, peak, and tail contrasts |
| u-shaped | turning point, trough, and tail contrasts |
| kinked | kink location and left/right slopes |
| threshold-step | boundary location and two-side effect contrast |
| sign-reversal | zero crossing, endpoint effects/signs, and global/two-side slopes |

Multiple ordered thresholds are reported as a subtype of `threshold-step`, not
as a tenth high-level situation.

## Frozen evidence

The active result view is produced by one unified blind classifier. Every
situation follows the same runtime path and the same acceptance standard.

| Situation | Fresh unified recovery |
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

Use `orgcate evidence --format json` for the machine-readable table,
`orgcate verify` to verify frozen code/config/result hashes, and
`orgcate contract` to audit the proposal-level stop rules. Every row is
produced by the same classifier and differs only in the shape-specific
quantities read after classification.

## Multiple coexisting HTE components

The single-morphology benchmark asks whether one focal characteristic's shape
can be recovered. Separate capability pilots ask the harder question: when
several prespecified characteristics contribute additively to CATE, can the
same joint forest and orthogonal reconstruction separate their contributions?
These experiments map the workflow's initial capability under additive
coexistence.

Two repository examples pass the same component-level standard used for a
single morphology. At `n=3000` and `R=20`, the two-component example recovers
fixed-asset intensity's inverted U in 90% of replications and
intangible-asset intensity's U shape in 100%. The three-component example
recovers change in cash holdings' sign reversal in 95%, intangible-asset
intensity's inverted U in 90%, and change in leverage's U shape in 100%.
Total-CATE fidelity is 100% in both examples; matched all-null family-wise
false confirmation is 0% and 5%. Reproduction instructions, figures, reports,
and hashes are in [`additive_coexistence/`](additive_coexistence/). The current
evidence covers these prespecified two- and three-component additive designs;
active-set discovery, interactions, and higher component counts define the
next capability studies.

## Full semi-synthetic reproduction

The GitHub upload candidate includes the research-team-authorized frozen
Compustat-derived L=3 pre-treatment firm-panel backbone used by the final
benchmark: `execution/phase2_real_backbone_v1/headline_one_anchor_per_firm_L3.pkl.gz`.
It contains real covariates only; treatment, potential outcomes, observed
outcomes, true CATE, and all oracle labels are freshly simulated. Verify the
bundled data and environment with `orgcate doctor`, then use the unified router:

```powershell
orgcate reproduce --target unified-morphology --phase confirmation
orgcate reproduce --target unified-morphology --phase report
orgcate reproduce --target unified-morphology --phase acceptance
```

Existing completed rows are checkpointed, so the runners do not silently drop
failures or refit completed cells. Do not use `--freeze-only` unless following
the documented development-before-confirmation protocol.

## Output and audit contracts

- `schemas/orgcate-result-v1.schema.json`: practical output schema.
- `docs/OUTPUT_SCHEMA.md`: field semantics and practical states.
- `docs/SHAPE_OUTPUT_REFERENCE.md`: English key quantities reported for every
  morphology, including turning points, curvature, kinks, boundaries, slopes,
  band effects, jumps, and zero crossings.
- `docs/UNIFIED_MORPHOLOGY_V2_METHOD.md`: common candidate library, decision order, and readouts.
- `reports/unified_morphology_v2/`: unified fresh evidence, confusion matrix, and figures.
- `docs/REPOSITORY_VERIFICATION_RECORD.md`: public verification record.

Oracle truth never enters practical feature ranking, model tuning, morphology
selection, boundary selection, or practical theory states.

## Data and release safety

The research team has explicitly authorized only the hash-pinned processed L=3
backbone documented in the public candidate's `DATA_NOTICE.md`. Do not add any
other raw download, transformed firm panel, access note, credential,
machine-specific provenance, or generated analyst/oracle row-level output.

Before the first public release, the research team must approve a software
license, citation metadata, authorship, and the exact aggregate artifacts to be
tracked. All machine-resolvable proposal-level method gates are integrated;
their authoritative status is in `docs/METHOD_CONTRACT_AND_STOP_RULES.md` and
`configs/github_release_method_contract_v1.json`. See `docs/RELEASE_CHECKLIST.md`.

## Verification

```powershell
.venv\Scripts\ruff.exe check src tests
.venv\Scripts\python.exe -m compileall -q src tests
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp reports/.pytest_release
```
