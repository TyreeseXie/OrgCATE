# Additive Heterogeneity Coexistence Report

## Research question

The single-characteristic benchmark asks whether a CATE relationship can be
recovered as a functional morphology with accurate shape-defining quantities.
This module asks the next question: when several firm characteristics jointly
produce treatment-effect heterogeneity, can their separate additive
contributions still be reconstructed and confirmed?

The module extends the single-characteristic workflow. Every replication fits
one `CausalForestDML` using all
12 candidate characteristics. The forest estimates the joint CATE; a
TRAIN-only additive reconstruction then separates a prespecified active set,
and untouched TEST observations confirm the fixed components jointly.

## Contribution

The module advances the workflow in three ways:

1. It moves from recovering one focal CATE curve to separating two or three
   curves that coexist in the same data-generating process.
2. It retains the full single-component morphology and key-quantity standard
   for every component rather than accepting good total-CATE prediction alone.
3. It adds a matched all-null experiment that records a family-wise error
   whenever any candidate component is falsely confirmed.

Two-component and three-component results are kept separate. They share code
and thresholds but have independent configurations, reports, figures, tables,
and evidence manifests.

## Frozen design and acceptance

- Real Compustat-derived L=3 histories supply the joint distribution of firm
  characteristics; treatment and causal outcomes are simulated.
- Each replication uses `n=3000`, one external 80/20 TRAIN/untouched-TEST
  split, and one honest 1,000-tree multivariable `CausalForestDML`.
- The active set is prespecified for this capability experiment. TEST outcomes
  and oracle truth do not select variables, shapes, or parameters.
- Each confirmation uses 20 fresh active replications and 20 matched all-null
  replications.
- A component succeeds only if practical evidence, exact morphology, curve
  Spearman of at least `.95`, normalized RMSE of at most `.20`, and all
  prespecified key quantities pass together.
- An experiment passes only if every component reaches 90%, total-CATE
  fidelity reaches 90%, and all-null family-wise false confirmation is at most
  10%.

## Two-component confirmation

The two components are fixed-asset intensity with an inverted-U effect and
intangible-asset intensity with a U-shaped effect.

| Component | Morphology | Complete recovery |
|---|---|---:|
| Fixed-asset intensity | inverted U | 18/20 = 90% |
| Intangible-asset intensity | U-shaped | 20/20 = 100% |

Total-CATE fidelity is 100%; median total-CATE Spearman is `.997`, median
normalized RMSE is `.025`, and all-null family-wise false confirmation is 0%.

![Two-component recovery and null control](two_component/evidence/FIGURE_1_RECOVERY_AND_NULL_CONTROL.svg)

The result establishes that both constituent morphologies and their required
quantities remain recoverable when they coexist in this additive design. The
total-effect and null checks also pass, so the component result is not obtained
by sacrificing overall CATE accuracy or false-confirmation control.

## Three-component confirmation

The three components are a sign reversal for change in cash holdings, an
inverted U for intangible-asset intensity, and a U shape for change in
leverage.

| Component | Morphology | Complete recovery |
|---|---|---:|
| Change in cash holdings | sign reversal | 19/20 = 95% |
| Intangible-asset intensity | inverted U | 18/20 = 90% |
| Change in leverage | U-shaped | 20/20 = 100% |

Total-CATE fidelity is 100%; median total-CATE Spearman is `.993`, median
normalized RMSE is `.025`, and all-null family-wise false confirmation is 5%.

![Three-component recovery and null control](three_component/evidence/FIGURE_1_RECOVERY_AND_NULL_CONTROL.svg)

The result shows that three distinct functional forms can be separated under
the same component-level standard. The 5% all-null result means that one of 20
matched null replications falsely confirmed at least one component; this
remains below the prespecified 10% family-wise ceiling.

## Interpretation and scope

Together, the two confirmations establish a limited but substantive extension
of the single-characteristic result: clear, prespecified, additive
heterogeneity from two or three firm characteristics can be separated while
preserving component accuracy, total-CATE fidelity, and family-wise null
control.

The current result concerns a prespecified active set and additive effects.
Active-set discovery, interactions, broader variable-and-shape combinations,
and higher component counts define the next capability experiments.

## Reproduction and audit

Run the two confirmations separately from the repository root:

```powershell
python additive_coexistence/src/run.py --config additive_coexistence/two_component/config.json
python additive_coexistence/src/run.py --config additive_coexistence/three_component/config.json
python additive_coexistence/verify.py
```

The verifier checks the shared method contract, separate configurations,
acceptance results, expected files, and recorded SHA256 hashes. See
[METHOD.md](METHOD.md), the
[two-component report](two_component/evidence/REPORT.md), and the
[three-component report](three_component/evidence/REPORT.md).
