# Scientific code provenance

The public repository was assembled from the private research repository after
the single-morphology method was consolidated into one common classifier.

The public-path changes are limited to:

- replacing private directory names with repository-relative paths;
- moving common acceptance calculations into the installable package;
- excluding private packs, row-level generated outputs, and local logs;
- retaining only aggregate evidence and the authorized firm-covariate backbone;
- adding public verification, artifact packaging, and CI checks.

The public consolidation does not change the DGP formulas, primary
`CausalForestDML` settings, 80/20 split, DML ATE estimator, candidate variables,
classification thresholds, or acceptance tolerances. Configuration, runner,
acceptance helper, and method-freeze hashes are checked by `orgcate verify`.

The internal `run_nine_shape_*` modules are shared numerical dependencies of
the unified runner. They do not route observations by a known simulation label.
The public entry point is `execution/run_unified_morphology_v2.py`.
