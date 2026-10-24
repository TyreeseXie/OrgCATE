"""Unified command-line interface for the OrgCATE research artifact."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .acceptance import run_data_to_results_acceptance
from .public_contract import validate_theory_audit
from .public_demo import run_public_demo
from .release import (
    NINE_SITUATIONS,
    audit_method_contract,
    doctor,
    load_integrated_evidence,
    repository_root,
    verify_release,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orgcate",
        description="DML ATE, causal-forest CATE estimation, and functional-form analysis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="check environment and data availability"
    )
    doctor_parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )

    evidence = subparsers.add_parser(
        "evidence", help="show the integrated nine-situation evidence"
    )
    evidence.add_argument("--format", choices=("table", "json", "csv"), default="table")
    evidence.add_argument("--output", type=Path, help="optional JSON/CSV destination")

    verify = subparsers.add_parser(
        "verify", help="verify frozen hashes and result contracts"
    )
    verify.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )

    contract = subparsers.add_parser(
        "contract", help="audit proposal/prototype method gates for GitHub release"
    )
    contract.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )

    theory_audit = subparsers.add_parser(
        "theory-audit", help="validate a researcher-completed Step 4 theory audit"
    )
    theory_audit.add_argument("--input", type=Path, required=True)

    demo = subparsers.add_parser(
        "demo", help="run one public synthetic end-to-end demonstration"
    )
    demo.add_argument("--scenario", choices=NINE_SITUATIONS, default="sign-reversal")
    demo.add_argument("--output", type=Path, default=Path("artifacts/public_demo"))
    demo.add_argument("--seed", type=int, default=20261023)
    demo.add_argument("--backbone-rows", type=int, default=6000)

    acceptance = subparsers.add_parser(
        "acceptance",
        help="analyze the bundled real covariate backbone and package aggregate results",
    )
    acceptance.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_to_results_acceptance_v1.json"),
    )
    acceptance.add_argument(
        "--output", type=Path, default=Path("artifacts/data_to_results_acceptance")
    )

    reproduce = subparsers.add_parser(
        "reproduce",
        help="route to a frozen full runner; licensed backbone may be required",
    )
    reproduce.add_argument(
        "--target", choices=("morphology-product", "kink-resolution"), required=True
    )
    reproduce.add_argument(
        "--phase", choices=("development", "confirmation", "report"), required=True
    )
    reproduce.add_argument("--freeze-only", action="store_true")
    return parser


def _print_mapping(value: dict) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _run_reproduction(args: argparse.Namespace) -> int:
    root = repository_root()
    command = [
        sys.executable,
        str(root / "execution" / "run_frozen_reproduction.py"),
        "--target",
        args.target,
        "--phase",
        args.phase,
    ]
    if args.freeze_only:
        raise ValueError(
            "the public package preserves the existing method freeze; "
            "creating a replacement freeze is intentionally disabled"
        )
    completed = subprocess.run(command, cwd=root, check=False)
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        result = doctor()
        if args.json:
            _print_mapping(result)
        else:
            print(f"Python supported: {result['python_supported']}")
            print(
                f"Release lock matches: {all(result['release_lock_matches'].values())}"
            )
            print(
                f"Aggregate evidence available: {result['aggregate_evidence_available']}"
            )
            print(
                "Bundled backbone available for full reproduction: "
                f"{result['licensed_backbone_available_for_full_reproduction']}"
            )
            print(
                "Bundled backbone hash matches: "
                f"{result['bundled_backbone_hash_matches']}"
            )
            print("Public demo requires licensed data: False")
        return (
            0
            if result["python_supported"]
            and all(result["release_lock_matches"].values())
            else 1
        )
    if args.command == "evidence":
        frame = load_integrated_evidence()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.format == "json":
                args.output.write_text(
                    json.dumps(frame.to_dict(orient="records"), indent=2),
                    encoding="utf-8",
                )
            else:
                frame.to_csv(args.output, index=False)
        if args.format == "json":
            print(json.dumps(frame.to_dict(orient="records"), indent=2))
        elif args.format == "csv":
            print(frame.to_csv(index=False), end="")
        else:
            print(frame.to_string(index=False))
        return 0
    if args.command == "verify":
        result = verify_release()
        if args.json:
            _print_mapping(result)
        else:
            for name, passed in result["checks"].items():
                print(f"{'PASS' if passed else 'FAIL'}  {name}")
            print(f"Overall: {result['status']}")
        return 0 if result["status"] == "PASS" else 1
    if args.command == "contract":
        result = audit_method_contract()
        if args.json:
            _print_mapping(result)
        else:
            print(f"Frozen evidence: {result['frozen_evidence_status']}")
            for blocker in result["release_blockers"]:
                print(
                    f"BLOCK  {blocker['id']} [{blocker['status']}]: "
                    f"{blocker['required_action']}"
                )
            print(f"Method-contract readiness: {result['status']}")
        return 0 if result["status"] == "READY" else 2
    if args.command == "theory-audit":
        record = json.loads(args.input.read_text(encoding="utf-8"))
        validate_theory_audit(record)
        print(f"Theory audit valid: {args.input.resolve()}")
        return 0
    if args.command == "demo":
        result = run_public_demo(
            scenario=args.scenario,
            output=args.output,
            seed=args.seed,
            backbone_rows=args.backbone_rows,
        )
        print(f"Demo completed: {args.output.resolve()}")
        print(f"Scenario: {args.scenario}")
        print(f"DML ATE: {result['practical']['ate']['estimate']:.4f}")
        print(f"Morphology: {result['practical']['morphology']['label']}")
        print(f"Practical state: {result['practical']['practical_state']}")
        print(
            "Shape interpretation: "
            f"{result['practical']['morphology']['interpretation_en']}"
        )
        return 0
    if args.command == "acceptance":
        result = run_data_to_results_acceptance(
            config_path=args.config,
            output=args.output,
        )
        rows = result["scenario_summary"]
        print(f"Acceptance analysis completed: {args.output.resolve()}")
        print(f"Situations analyzed: {len(rows)}")
        print(
            "Exact morphology labels: "
            f"{sum(bool(row['exact_label_correct']) for row in rows)}/{len(rows)}"
        )
        return 0
    if args.command == "reproduce":
        return _run_reproduction(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
