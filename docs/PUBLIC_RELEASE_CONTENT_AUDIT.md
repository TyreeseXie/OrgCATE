# Repository content review

This directory was assembled from a file allowlist. Source files were copied;
the private research directory and its Git history were not used as the public
repository.

## Included material

| Area | Contents |
|---|---|
| Package | `src/orgcate_sim/`, CLI, package metadata |
| Runtime | five experiment runners and two required dependency modules |
| Configuration | method, product, kink, and release JSON files |
| Tests | DML, product, kink, leakage, result-schema, and release tests |
| Results | aggregate tables, development summaries, manifests, reports, and figures |
| Documentation | methods, output fields, data notice, and release instructions |
| Data | one authorized L=3 backbone and its dictionary, QC, correlations, and manifests |

## Excluded material

| Material | Reason |
|---|---|
| Raw Compustat downloads, L1/L5 registries, and other processed panels | Not required for the reported analysis |
| Generated analyst, oracle, or evaluator tables | Unit-level simulation output is regenerated as needed |
| Credentials, access notes, and `.env` files | Confidential information |
| ZIP files and historical execution packages | Redundant and difficult to review |
| Earlier exploratory branches and W-number reports | Not part of the current interface |
| Briefing documents and manuscripts | Separate research documents |
| Environments, caches, logs, and temporary files | Machine-generated material |
| Private Git history and handoff notes | Internal project records |

The terms `tau_true`, `propensity_true`, and `oracle` occur in simulation code
and tests. The package needs these objects to evaluate accuracy, but saved
unit-level truth is not included. The only binary data exception is the exact
path and hash listed in `DATA_NOTICE.md`.

## Pre-release procedure

1. Run `python tools/audit_public_release.py --allow-metadata-pending` during
   preparation.
2. Review `PUBLIC_RELEASE_MANIFEST.json` and its SHA256 file list.
3. Add the approved `LICENSE` and `CITATION.cff`.
4. Run `python tools/audit_public_release.py` without the exception.
5. Review the staged paths and diff summary before the first commit.
6. Push this directory, rather than the private research directory, to GitHub.
