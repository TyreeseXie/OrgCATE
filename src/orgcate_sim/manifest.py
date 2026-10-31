"""Reproducibility records for simulation and estimator runs."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

PACKAGE_NAMES = ("numpy", "pandas", "scipy", "scikit-learn", "econml", "PyYAML", "pyarrow")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def config_sha256(config: Any) -> str:
    payload = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def build_run_manifest(
    *,
    config: Any,
    master_seed: int,
    simulation_seed: int | None = None,
    seed_spawn_key: tuple[int, ...] | None = None,
    output_paths: dict[str, str],
    dgp_metadata: dict[str, Any],
    feature_roles: dict[str, list[str]],
    estimator_status: str = "not_run",
) -> dict[str, Any]:
    """Build an evaluator-side manifest; it never belongs in analyst input."""
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "packages": package_versions(),
        "config": _jsonable(config),
        "config_sha256": config_sha256(config),
        "master_seed": int(master_seed),
        "simulation_seed": int(simulation_seed) if simulation_seed is not None else int(master_seed),
        "seed_spawn_key": list(seed_spawn_key or ()),
        "dgp_metadata": _jsonable(dgp_metadata),
        "oracle_feature_roles": _jsonable(feature_roles),
        "output_paths": output_paths,
        "estimator_status": estimator_status,
        "oracle_data_is_evaluator_only": True,
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
