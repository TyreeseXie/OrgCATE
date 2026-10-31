# GitHub repository assembly

## Public scope

This directory is the GitHub repository. It contains one active
single-morphology workflow, one separate additive-coexistence module, the
authorized covariate backbone, aggregate evidence, tests, and documentation.
Private development packs and row-level simulation outputs are excluded.

## Repository structure

```text
OrgCATE/
|-- .github/workflows/ci.yml
|-- configs/                         # active method and audit contracts
|-- docs/                            # method, output, and verification notes
|-- execution/                       # unified runner and shared internals
|-- simulation_runner_v2/            # treatment/outcome simulation dependency
|-- morphology_reconstruction_v2_stage_success/
|                                      # reconstruction dependency
|-- additive_coexistence/             # separate two/three-component module
|-- reports/unified_morphology_v2/    # aggregate evidence and figures
|-- execution/phase2_real_backbone_v1/ # authorized L=3 covariate backbone
|-- schemas/                          # practical output contract
|-- src/orgcate_sim/                  # installable package and CLI
|-- tests/
|-- tools/
|-- README.md
|-- README.zh-CN.md
|-- DATA_NOTICE.md
|-- SECURITY.md
|-- PUBLIC_RELEASE_MANIFEST.json
|-- pyproject.toml
`-- requirements-lock.txt
```

The active single-morphology path is:

```text
cross-fitted DML ATE
-> one 12-characteristic honest CausalForestDML
-> one common TRAIN-only candidate library and classifier
-> untouched-TEST confirmation
-> practical state, morphology, and key quantities
```

All nine situations use one classifier and one runtime path. Multiple
thresholds are a subtype of `threshold-step`.

## Included data

The only row-level input is the research-team-authorized, hash-pinned L=3
pre-treatment covariate backbone. Treatment, potential outcomes, observed
outcomes, and CATE truth are generated during simulation. Generated row-level
analyst and oracle data are not committed.

## Verification

```powershell
python -m pip install -e . --no-deps
ruff check src tests execution tools additive_coexistence
python -m compileall -q src tests execution tools additive_coexistence
pytest -q -p no:cacheprovider
orgcate verify
orgcate contract
python tools/verify_unified_morphology_evidence.py
python additive_coexistence/verify.py
python tools/audit_public_release.py --allow-metadata-pending
```

`orgcate contract` remains blocked only while license and citation metadata are
awaiting research-team approval. That publication decision is separate from
code and evidence verification.
