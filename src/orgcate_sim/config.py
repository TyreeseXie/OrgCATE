"""Typed, strict loading for project YAML configuration files.

The loader validates configuration syntax and research-contract fields.  It does
not decide any open scientific question or run a simulation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import MORPHOLOGIES, SimulationConfig


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], *, field_name: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"unknown {field_name} field(s): {sorted(unknown)}")


@dataclass(frozen=True)
class PilotConfig:
    project: str
    master_seed: int
    n_replications: int
    simulation: SimulationConfig
    worlds: tuple[str, ...]
    estimator: dict[str, Any]
    output: dict[str, Any]

    def config_for_world(self, morphology: str) -> SimulationConfig:
        if morphology not in self.worlds:
            raise ValueError(f"{morphology!r} is not declared in this pilot config")
        if morphology not in MORPHOLOGIES:
            raise ValueError(f"unknown morphology: {morphology}")
        active_x = () if morphology == "constant" else self.simulation.active_x
        return SimulationConfig(
            **{**self.simulation.__dict__, "morphology": morphology, "active_x": active_x}
        )


@dataclass(frozen=True)
class PlanningConfig:
    """Typed representation of non-executable core/stress planning YAML."""

    kind: str
    values: dict[str, Any]


def load_yaml(path: str | Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load project YAML configuration files") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return _mapping(loaded, field_name="YAML root")


def load_pilot_config(path: str | Path) -> PilotConfig:
    raw = load_yaml(path)
    _reject_unknown(
        raw,
        {
            "project",
            "master_seed",
            "n_replications",
            "backbone",
            "causal_design",
            "worlds",
            "world_params",
            "splitting",
            "estimator",
            "output",
        },
        field_name="pilot config",
    )
    backbone = _mapping(raw["backbone"], field_name="backbone")
    causal = _mapping(raw["causal_design"], field_name="causal_design")
    splits = _mapping(raw["splitting"], field_name="splitting")
    params = _mapping(raw.get("world_params", {}), field_name="world_params")
    worlds = tuple(raw["worlds"])
    if not worlds or any(world not in MORPHOLOGIES for world in worlds):
        raise ValueError("pilot config contains an unknown or empty morphology list")

    _reject_unknown(
        backbone,
        {"type", "n_units", "n_x", "n_w", "rho", "candidate_moderator_index"},
        field_name="backbone",
    )
    _reject_unknown(
        causal,
        {
            "target_treatment_rate",
            "overlap_strength",
            "outcome_noise_sd",
            "target_ate",
            "substantive_delta",
        },
        field_name="causal_design",
    )
    _reject_unknown(splits, {"discovery", "confirmation", "replication"}, field_name="splitting")
    _reject_unknown(params, {"monotone", "smooth_flip"}, field_name="world_params")

    monotone = _mapping(params.get("monotone", {}), field_name="world_params.monotone")
    smooth_flip = _mapping(params.get("smooth_flip", {}), field_name="world_params.smooth_flip")
    _reject_unknown(monotone, {"heterogeneity_sd", "enforce_same_sign"}, field_name="world_params.monotone")
    _reject_unknown(smooth_flip, {"heterogeneity_sd", "cutoff_quantile", "steepness"}, field_name="world_params.smooth_flip")
    if monotone.get("enforce_same_sign", True) is not True:
        raise ValueError("W1 in this MVP must enforce a same-sign effect pattern")
    _reject_unknown(
        _mapping(raw["estimator"], field_name="estimator"),
        {"name", "discrete_treatment", "honest", "n_estimators", "min_samples_leaf", "random_state_offset"},
        field_name="estimator",
    )
    _reject_unknown(
        _mapping(raw["output"], field_name="output"),
        {"save_unit_level_oracle", "save_unit_level_analyst", "format"},
        field_name="output",
    )

    simulation = SimulationConfig(
        n_units=int(backbone["n_units"]),
        n_x=int(backbone["n_x"]),
        n_w=int(backbone["n_w"]),
        rho=float(backbone["rho"]),
        target_treatment_rate=float(causal["target_treatment_rate"]),
        overlap_strength=float(causal["overlap_strength"]),
        outcome_noise_sd=float(causal["outcome_noise_sd"]),
        target_ate=float(causal["target_ate"]),
        substantive_delta=float(causal["substantive_delta"]),
        active_x=(int(backbone["candidate_moderator_index"]),),
        heterogeneity_sd=float(monotone.get("heterogeneity_sd", 0.10)),
        cutoff_quantile=float(smooth_flip.get("cutoff_quantile", 0.35)),
        steepness=float(smooth_flip.get("steepness", 6.0)),
        discovery_share=float(splits["discovery"]),
        confirmation_share=float(splits["confirmation"]),
        replication_share=float(splits["replication"]),
    )
    simulation.validate()
    estimator = dict(_mapping(raw["estimator"], field_name="estimator"))
    output = dict(_mapping(raw["output"], field_name="output"))
    return PilotConfig(
        project=str(raw["project"]),
        master_seed=int(raw["master_seed"]),
        n_replications=int(raw["n_replications"]),
        simulation=simulation,
        worlds=worlds,
        estimator=estimator,
        output=output,
    )


def load_planning_config(path: str | Path) -> PlanningConfig:
    raw = dict(load_yaml(path))
    if "stress_tests" in raw:
        return PlanningConfig(kind="stress", values=raw)
    if "backbone" in raw and "sample_sizes" in _mapping(raw["backbone"], field_name="backbone"):
        worlds = raw.get("worlds", [])
        if any(world not in MORPHOLOGIES for world in worlds):
            raise ValueError("core config contains an unknown morphology")
        return PlanningConfig(kind="core", values=raw)
    raise ValueError("planning config is neither a core matrix nor a stress-test plan")
