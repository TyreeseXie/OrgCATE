# GitHub release checklist

## Automated and complete

- [x] One `orgcate` command exposes environment checks, public demo, integrated
  evidence, frozen-result verification, and full-run routing.
- [x] Public synthetic demo requires no licensed data.
- [x] Practical and evaluator outputs use separate files and directories.
- [x] One unified morphology runner, configuration, method freeze, and evidence
  manifest are hash-verified before results are presented as frozen.
- [x] Public execution contains no scenario-name dispatch; all nine situations
  enter the same candidate library and classifier.
- [x] Exact package versions are recorded in `requirements-lock.txt`.
- [x] English and Chinese entry documentation use the current nine-situation
  vocabulary rather than historical internal labels.
- [x] `orgcate verify` verifies frozen evidence and hashes without claiming that
  every proposal-level method gate is implemented.
- [x] `orgcate contract` separately reports method-contract readiness and exits
  nonzero while any release-blocking requirement is partial or missing.
- [x] The research-team-authorized frozen L=3 Compustat-derived backbone and
  its QC/hash metadata are bundled at the exact path used by the runners.

## Method contract required before public release

- [x] Freeze and integrate the standardized 10% point-tolerance rule using the
  mean final orthogonal reconstructed CATE on untouched TEST. This is a practical
  point-coherence rule, not a statistical-equivalence claim.
- [x] Connect causal-design and overlap diagnostics to `DESIGN_NOT_ELIGIBLE`
  before any morphology analysis runs.
- [x] Expose the approved no-signal, repeated-refit reproducibility, and
  property-support declarations in the public result contract.
- [x] Emit continuity, monotonicity, sign relation, taxonomy cell, and
  `requires_abduction` alongside the convenient nine-situation label.
- [x] Add the researcher-owned Step 4 decision/audit log to the public workflow.
- [x] Threshold-step is an operational candidate-boundary label reported with
  its location, two-side contrast, and supporting diagnostics.

The authoritative status of these items is machine readable in
`configs/github_release_method_contract_v1.json` and checked by
`orgcate contract`.

## Human approval required before public release

- [ ] Choose and add a software license.
- [ ] Confirm author order, affiliations, contact details, and citation text.
- [ ] Decide which non-disclosive aggregate reports and figures will be tracked.
- [x] Record the human research lead's authorization for the single frozen
  processed Compustat-derived backbone in the upload candidate's
  `DATA_NOTICE.md`.
- [ ] Confirm that the staged tree contains only that hash-pinned backbone and
  no raw extract, access note, credential, or additional firm-level dataset.
- [ ] Create a clean release commit/tag only after the staged-file audit passes.

The absence of a license means the repository is not ready for public reuse,
even when the code and tests run successfully.

## Final assembly route

The current research worktree must not be uploaded wholesale. Follow
`docs/GITHUB_RELEASE_ASSEMBLY.md`: consolidate the frozen runtime out of the
historical dynamic-import chain, prove fixed-seed equivalence, build a clean
candidate by allowlist, scan it for licensed rows and machine paths, and test it
from a clean installation before creating the public repository.
