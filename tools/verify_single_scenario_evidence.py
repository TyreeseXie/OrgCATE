"""Verify the unified nine-situation evidence package and its source hashes."""

from __future__ import annotations

import json
import sys

from orgcate_sim.release import load_integrated_evidence, verify_release


def main() -> int:
    evidence = load_integrated_evidence()
    result = verify_release()
    checks = {
        "nine_rows": len(evidence) == 9 and evidence["scenario"].nunique() == 9,
        "all_unified_rates_at_least_90_percent": bool(
            (evidence["unified_recovery_rate"] >= 0.90).all()
        ),
        "all_scenarios_pass": bool(evidence["scenario_pass"].all()),
        "all_hash_and_contract_checks": result["status"] == "PASS",
    }
    payload = {
        "schema_version": "orgcate-single-scenario-evidence-verification-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "release_checks": result["checks"],
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
