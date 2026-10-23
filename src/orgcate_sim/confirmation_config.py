"""Validation and hashing for a human-approved confirmation method freeze."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REQUIRED_WORLDS: frozenset[str] = frozenset({"W1", "W2", "W3", "W4", "W5"})


def canonical_config_json(config: Mapping[str, Any]) -> str:
    """Serialize a configuration deterministically for its scientific hash."""

    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def confirmation_config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def validate_frozen_confirmation_config(config: Mapping[str, Any]) -> None:
    """Reject draft, incomplete, or unaudited configurations before a run."""

    if config.get("status") != "FROZEN_APPROVED":
        raise ValueError("confirmation configuration is not FROZEN_APPROVED")
    approval = config.get("approval")
    if not isinstance(approval, Mapping):
        raise TypeError("frozen configuration requires an approval record")
    for field in ("approved_by", "approved_at", "decision_log_ref"):
        if not str(approval.get(field, "")).strip():
            raise ValueError(f"approval record missing {field}")

    delta = config.get("practical_delta")
    if not isinstance(delta, Mapping) or delta.get("value") is None:
        raise ValueError("practical_delta must be approved and non-null")
    if float(delta["value"]) < 0.0:
        raise ValueError("practical_delta must be non-negative")

    clip = config.get("propensity_clip")
    if not isinstance(clip, list) or len(clip) != 2 or not (0.0 < float(clip[0]) < float(clip[1]) < 1.0):
        raise ValueError("propensity_clip must satisfy 0 < low < high < 1")

    support = config.get("minimum_support")
    if not isinstance(support, Mapping):
        raise TypeError("minimum_support is required")
    if int(support.get("treated_per_region", 0)) < 1 or int(support.get("control_per_region", 0)) < 1:
        raise ValueError("minimum treatment/control support must be positive")

    confidence = float(config.get("confidence_level", 0.0))
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if not isinstance(config.get("require_ci_excludes_zero"), bool):
        raise TypeError("require_ci_excludes_zero must be explicitly approved as true or false")
    max_clip = float(config.get("maximum_propensity_clip_fraction", -1.0))
    if not 0.0 <= max_clip <= 1.0:
        raise ValueError("maximum_propensity_clip_fraction must be in [0, 1]")

    regions = config.get("regions")
    if not isinstance(regions, Mapping) or set(regions) != REQUIRED_WORLDS:
        raise ValueError("frozen configuration requires exact W1-W5 region specifications")
    for world, specification in regions.items():
        if not isinstance(specification, Mapping) or not specification:
            raise ValueError(f"region specification for {world} must be a non-empty object")
        if any(value is None for value in specification.values()):
            raise ValueError(f"region specification for {world} contains an unresolved null")

    if config.get("test_can_change_discovery") is not False:
        raise ValueError("test_can_change_discovery must be false")
    if config.get("oracle_allowed_in_practical_decision") is not False:
        raise ValueError("oracle_allowed_in_practical_decision must be false")
    nuisance = config.get("nuisance_models")
    if not isinstance(nuisance, Mapping) or set(nuisance) != {
        "propensity",
        "outcome_control",
        "outcome_treated",
        "seed_lineage",
    }:
        raise ValueError("frozen configuration requires an exact nuisance-model specification")


def load_frozen_confirmation_config(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load, validate, and return an approved configuration plus its hash."""

    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_frozen_confirmation_config(config)
    return config, confirmation_config_hash(config)
