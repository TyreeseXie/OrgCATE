"""Verify fresh average-effect and panel fixed-effects result directories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SITUATIONS = (
    "homogeneous",
    "linear-monotone",
    "nonlinear-monotone",
    "steep-continuous",
    "inverted-u",
    "u-shaped",
    "kinked",
    "threshold-step",
    "sign-reversal",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _verify_output_hashes(root: Path, manifest: dict) -> None:
    for name, expected_hash in manifest["outputs"].items():
        path = root / name
        _require(path.is_file() and path.stat().st_size > 0, f"missing output: {path}")
        _require(_sha256(path) == expected_hash, f"output hash mismatch: {path}")


def verify(ate: Path, fixed_effects: Path) -> dict[str, int]:
    ate_manifest = json.loads((ate / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    fixed_manifest = json.loads(
        (fixed_effects / "RUN_MANIFEST.json").read_text(encoding="utf-8")
    )
    _verify_output_hashes(ate, ate_manifest)
    _verify_output_hashes(fixed_effects, fixed_manifest)

    ate_long = pd.read_csv(ate / "ATE_ESTIMATOR_COMPARISON_LONG.csv")
    ate_summary = pd.read_csv(ate / "ATE_ESTIMATOR_COMPARISON_SUMMARY.csv")
    fixed_long = pd.read_csv(fixed_effects / "PANEL_FIXED_EFFECTS_COMPARISON_LONG.csv")
    fixed_summary = pd.read_csv(
        fixed_effects / "PANEL_FIXED_EFFECTS_COMPARISON_SUMMARY.csv"
    )
    _require(len(ate_long) == 180, "ATE comparison must contain 180 replications")
    _require(len(fixed_long) == 90, "fixed-effects comparison must contain 90 replications")
    _require(tuple(ate_summary["situation"]) == SITUATIONS, "wrong ATE situations")
    _require(
        tuple(fixed_summary["situation"]) == SITUATIONS,
        "wrong fixed-effects situations",
    )
    _require(ate_manifest["causal_forests_fitted"] == 0, "ATE comparison fitted a forest")
    _require(
        fixed_manifest["new_causal_forests_fitted"] == 0,
        "fixed-effects comparison fitted a forest",
    )
    for path in (
        ate / "ATE_ESTIMATOR_COMPARISON.svg",
        fixed_effects / "PANEL_FIXED_EFFECTS_COMPARISON.svg",
    ):
        text = path.read_text(encoding="utf-8")
        _require("<svg" in text and "</svg>" in text and len(text) > 1000, f"invalid figure: {path}")
    return {"ate_replications": len(ate_long), "fixed_effects_replications": len(fixed_long)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ate", type=Path, required=True)
    parser.add_argument("--fixed-effects", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.ate.resolve(), args.fixed_effects.resolve())
    print(
        "Average-effect artifacts verified: "
        f"{result['ate_replications']} ATE cells, "
        f"{result['fixed_effects_replications']} fixed-effects cells"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
