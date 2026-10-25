# Code provenance

At assembly, the selected Python files in the following paths were compared
with their source files in the private research repository using SHA256:

- `src/orgcate_sim/`
- `execution/`
- `simulation_runner_v2/src/`
- `morphology_reconstruction_v2_stage_success/src/`

All 30 files matched. The assembly step did not change the DGP formulas,
estimator settings, sample split, classification rules, thresholds, or
experiment runners.

Changes made for publication are limited to documentation, ignore rules, CI,
the repository scan, publication-metadata placeholders, and copies of aggregate
figures. `PUBLIC_RELEASE_MANIFEST.json` records the SHA256 of every included
file except the manifest itself.

The lower-level dependency modules retain their original formatting so their
hashes remain comparable with the recorded experiments.

The public dependency is stored under `simulation_runner_v2`. The runner and
source-configuration file contents are byte-identical to the frozen versions.
The release verifier hash-checks the public configuration, dependency source,
and frozen evidence records separately.
