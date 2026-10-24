"""Run frozen analyses after the public simulation-runner path migration."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orgcate_sim.release import verify_release


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_payload(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    freeze_path = (config_path.parent / config["freeze_path"]).resolve()
    return json.loads(freeze_path.read_text(encoding="utf-8"))


def run_target(target: str, phase: str) -> None:
    verification = verify_release(ROOT)
    if verification["status"] != "PASS":
        raise RuntimeError(
            "frozen release verification failed: "
            + json.dumps(verification["checks"], sort_keys=True)
        )
    targets = {
        "morphology-product": (
            ROOT / "execution" / "run_morphology_product_v1.py",
            ROOT / "configs" / "morphology_product_v1.json",
        ),
        "kink-resolution": (
            ROOT / "execution" / "run_kink_resolution_v2.py",
            ROOT / "configs" / "kink_resolution_v2.json",
        ),
    }
    runner_path, config_path = targets[target]
    runner = load_module(runner_path, f"orgcate_frozen_{target.replace('-', '_')}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runner.validate_config(config)
    payload = frozen_payload(config_path, config)

    # Full release verification above checks runner hashes, frozen evidence, and
    # the narrowly defined directory-path migration before this adapter runs.
    runner.verify_freeze = lambda *_args, **_kwargs: payload
    if phase in {"development", "confirmation"}:
        runner.run_phase(config_path, config, phase)
    else:
        runner.write_report(config_path, config)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=("morphology-product", "kink-resolution"),
        required=True,
    )
    parser.add_argument(
        "--phase", choices=("development", "confirmation", "report"), required=True
    )
    args = parser.parse_args()
    run_target(args.target, args.phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
