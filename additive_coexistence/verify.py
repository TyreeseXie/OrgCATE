"""Verify the recorded two- and three-component coexistence evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify() -> dict[str, object]:
    checks: dict[str, bool] = {}
    for name in ("ADDITIVE_COEXISTENCE_REPORT.md", "ADDITIVE_COEXISTENCE_REPORT.zh-CN.md"):
        path = ROOT / name
        checks[f"module_{name}"] = path.is_file() and path.stat().st_size > 0
    for label, count in (("two_component", 2), ("three_component", 3)):
        base = ROOT / label
        config = json.loads((base / "config.json").read_text(encoding="utf-8"))
        evidence = base / "evidence"
        manifest = json.loads(
            (evidence / "EVIDENCE_MANIFEST.json").read_text(encoding="utf-8")
        )
        checks[f"{label}_scenario_separation"] = (
            set(config["scenarios"]) == {f"active_count_{count}", f"null_count_{count}"}
            and {item["component_count"] for item in config["scenarios"].values()}
            == {count}
        )
        checks[f"{label}_method_contract"] = (
            config["n"] == 3000
            and config["test_size"] == 0.2
            and config["replications"] == 20
            and config["selection_policy"]["one_joint_multivariable_forest"] is True
            and config["selection_policy"]["interactions_included"] is False
        )
        checks[f"{label}_manifest_pass"] = manifest["decision"] == "PASS"
        for name, record in manifest["outputs"].items():
            path = evidence / name
            checks[f"{label}_{name}"] = (
                path.is_file()
                and path.stat().st_size == int(record["bytes"])
                and sha256(path) == record["sha256"]
            )
        components = pd.read_csv(evidence / "COMPONENT_SUMMARY.csv")
        scenario = pd.read_csv(evidence / "SCENARIO_SUMMARY.csv")
        null = pd.read_csv(evidence / "NULL_SUMMARY.csv")
        checks[f"{label}_component_gate"] = (
            len(components) == count
            and components["component_recovery_rate"].ge(0.90).all()
        )
        checks[f"{label}_total_cate_gate"] = scenario[
            "total_cate_fidelity_rate"
        ].ge(0.90).all()
        checks[f"{label}_null_gate"] = null[
            "familywise_false_confirmation_rate"
        ].le(0.10).all()
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": {name: bool(value) for name, value in checks.items()},
    }


def main() -> int:
    result = verify()
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
