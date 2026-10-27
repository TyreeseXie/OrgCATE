# Additive Coexistence

This module is the repository's extension from single-characteristic
morphology recovery to coexisting treatment-effect heterogeneity. Two or three
prespecified firm characteristics contribute additively to the same CATE
surface. One multivariable `CausalForestDML` is fitted to all 12 candidate
characteristics. TRAIN-only additive reconstruction separates the components;
untouched TEST data evaluates each fixed component jointly with the others.

The methodological question is whether each constituent relationship retains
its functional form and key quantities after the other active relationships
are controlled. The module applies the complete single-component standard to
every component and adds total-CATE and all-null family-wise checks.

The module has one shared method and two separately reported examples:

```text
additive_coexistence/
|-- shared/                  common method configuration
|-- src/                     shared estimation and reconstruction code
|-- two_component/           separate config, report, tables, figures, manifest
`-- three_component/         separate config, report, tables, figures, manifest
```

## Recorded results

| Example | Complete component recovery | Total-CATE fidelity | All-null family-wise false confirmation |
|---|---:|---:|---:|
| Two components | 90%, 100% | 100% | 0% |
| Three components | 95%, 90%, 100% | 100% | 5% |

The two-component example combines an inverted-U effect of fixed-asset
intensity with a U-shaped effect of intangible-asset intensity. The
three-component example combines a sign reversal for change in cash holdings,
an inverted U for intangible-asset intensity, and a U shape for change in
leverage.

![Two-component recovery](two_component/evidence/FIGURE_1_RECOVERY_AND_NULL_CONTROL.svg)

![Three-component recovery](three_component/evidence/FIGURE_1_RECOVERY_AND_NULL_CONTROL.svg)

A component succeeds only when practical evidence, exact morphology, curve
Spearman of at least `.95`, normalized RMSE of at most `.20`, and every
prespecified key quantity pass together. Each component must reach 90% across
20 fresh replications. Total-CATE fidelity must reach 90%, and matched all-null
family-wise false confirmation must remain at or below 10%.

## Reproduce each example separately

From the repository root:

```powershell
python additive_coexistence/src/run.py --config additive_coexistence/two_component/config.json
python additive_coexistence/src/run.py --config additive_coexistence/three_component/config.json
python additive_coexistence/verify.py
```

The two commands write to separate directories under `artifacts/`. Completed
cells are checkpointed. The full runs are intentionally not part of routine CI;
CI verifies code/config contracts and every recorded aggregate evidence hash.

GitHub Actions uploads the recorded module evidence as
`orgcate-additive-coexistence-<run_id>`. The downloadable package contains the
integrated module report and the separate two- and three-component reports,
figures, aggregate tables, configs, and manifests. It contains no firm-level
records.

## Current scope

These are clear-signal, prespecified, additive examples at `n=3000`. They show
that two- and three-component coexistence can be separated under the stated
conditions. Active-set discovery, interactions, and broader variable-and-shape
combinations define the next capability studies.

See the [integrated module report](ADDITIVE_COEXISTENCE_REPORT.md),
[METHOD.md](METHOD.md),
[the two-component report](two_component/evidence/REPORT.md), and
[the three-component report](three_component/evidence/REPORT.md).
