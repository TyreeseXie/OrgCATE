# GitHub release assembly

## Release boundary

This directory is the only intended GitHub upload base. It was created beside
the private research repository by copying an explicit allowlist. The private
repository remains the complete provenance archive and must not be initialized
as, or pushed to, the public GitHub repository.

## Final directory structure

```text
OrgCATE_GitHub_Upload_v1/
|-- .github/workflows/ci.yml
|-- configs/                         # frozen public method rules
|-- docs/                            # method, outputs, safety, release guide
|-- execution/                       # five exact hash-verified frozen runners
|-- codex_simulation_runner_v2/      # exact low-level dependency, code/config only
|-- morphology_reconstruction_v2_stage_success/
|                                      # reconstruction dependency, code/config only
|-- reports/
|   |-- morphology_product_v1/       # aggregate table, manifest, report, SVGs
|   |-- kink_resolution_v2/          # aggregate table, manifest, report, figures
|   `-- public_method_gate_integration_v1/
|-- evidence/figures/                # three convenience copies for README/paper use
|-- execution/phase2_real_backbone_v1/ # authorized frozen L=3 covariate backbone
|-- schemas/                         # machine-readable practical output contract
|-- src/orgcate_sim/                 # installable public package and CLI
|-- tests/                           # selected public, leakage, and hash tests
|-- tools/audit_public_release.py    # disclosure and manifest audit
|-- README.md
|-- README.zh-CN.md
|-- SECURITY.md
|-- DATA_NOTICE.md
|-- PUBLICATION_METADATA_TEMPLATE.md
|-- PUBLIC_RELEASE_MANIFEST.json     # generated; inspect before staging
|-- pyproject.toml
`-- requirements-lock.txt
```

`LICENSE` and `CITATION.cff` are intentionally absent until the research team
approves them. Their absence blocks a public release but does not block testing
the candidate.

## Why the frozen dependency paths remain

The verified public demo reaches the current frozen morphology method through
five runner modules and two lower-level code dependencies. Their exact hashes
are part of the frozen evidence audit. For v1, the safer choice is to preserve
only those exact files and clearly mark them as internal runtime, rather than
rename/refactor them and silently invalidate the verified hashes.

No historical results, ZIPs, local outputs, raw downloads, or unused experimental
packs from those directories were copied. The only row-level data file is the
research-team-authorized, hash-pinned L=3 headline backbone required by the
frozen runners. A later major version may refactor
the internals after fixed-seed equivalence tests, but that is not required for
the v1 release.

## Assembly policy

- Include installable source code, operative configurations, schema, selected
  tests, aggregate evidence, public documentation, and the single authorized
  frozen L=3 real-covariate backbone.
- Exclude every other row-level dataset, including generated treatment,
  analyst, oracle, and evaluator tables.
- Exclude private Git history, handoffs, unpublished source documents, access notes,
  credentials, archives, caches, local outputs, and machine-specific paths.
- Retain evaluator/oracle *code* because simulation truth evaluation is part of
  the scientific method; never retain saved evaluator/oracle *row data*.
- Keep only the three selected aggregate SVGs under `evidence/figures`; these
  duplicate verified report figures for convenient public presentation.
- Build and review `PUBLIC_RELEASE_MANIFEST.json` before staging files.

The detailed decision table is in `PUBLIC_RELEASE_CONTENT_AUDIT.md`.

## Required checks

Candidate-stage check, while publication metadata is pending:

```powershell
python tools/audit_public_release.py --allow-metadata-pending
python -m pip install -e . --no-deps
ruff check src tests execution tools
python -m compileall -q src tests execution tools
pytest -q -p no:cacheprovider
orgcate verify
orgcate contract
```

`orgcate contract` must remain blocked only by license/citation metadata during
candidate assembly. After adding approved metadata, run:

```powershell
python tools/audit_public_release.py
```

Only a clean audit, clean tests, verified hashes, and research-team review of the final
staged paths authorize upload.

