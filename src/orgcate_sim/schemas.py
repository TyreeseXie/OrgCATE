from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

Morphology = Literal[
    "constant",
    "monotone",
    "smooth_flip",
    "steep_continuous",
    "inverted_u",
    "step",
    "additive_multi",
    "configuration",
    "complex_abstain",
]

MORPHOLOGIES: frozenset[str] = frozenset(
    {
        "constant",
        "monotone",
        "smooth_flip",
        "steep_continuous",
        "inverted_u",
        "step",
        "additive_multi",
        "configuration",
        "complex_abstain",
    }
)


@dataclass(frozen=True)
class SimulationConfig:
    n_units: int = 1000
    n_x: int = 10
    n_w: int = 10
    rho: float = 0.30
    target_treatment_rate: float = 0.30
    overlap_strength: float = 0.75
    target_ate: float = 0.20
    outcome_noise_sd: float = 1.0
    morphology: Morphology = "constant"
    active_x: tuple[int, ...] = (0,)
    heterogeneity_sd: float = 0.10
    cutoff_quantile: float = 0.35
    steepness: float = 6.0
    substantive_delta: float = 0.05
    discovery_share: float = 0.50
    confirmation_share: float = 0.25
    replication_share: float = 0.25

    def validate(self) -> None:
        if self.n_units <= 0:
            raise ValueError("n_units must be positive")
        if self.n_x < 1 or self.n_w < 0:
            raise ValueError("invalid feature dimensions")
        if not (-0.95 < self.rho < 0.95):
            raise ValueError("rho must be inside (-0.95, 0.95)")
        if self.morphology not in MORPHOLOGIES:
            raise ValueError(f"unknown morphology: {self.morphology}")
        if not (0 < self.target_treatment_rate < 1):
            raise ValueError("target_treatment_rate must be in (0,1)")
        if self.overlap_strength < 0:
            raise ValueError("overlap_strength must be non-negative")
        if self.outcome_noise_sd <= 0:
            raise ValueError("outcome_noise_sd must be positive")
        if self.heterogeneity_sd < 0:
            raise ValueError("heterogeneity_sd must be non-negative")
        if not (0 < self.cutoff_quantile < 1):
            raise ValueError("cutoff_quantile must be in (0,1)")
        if self.steepness <= 0:
            raise ValueError("steepness must be positive")
        if self.substantive_delta < 0:
            raise ValueError("substantive_delta must be non-negative")
        if abs(self.discovery_share + self.confirmation_share + self.replication_share - 1.0) > 1e-9:
            raise ValueError("split shares must sum to 1")
        for j in self.active_x:
            if j < 0 or j >= self.n_x:
                raise ValueError(f"active_x index {j} outside X dimension")


@dataclass
class SimulationOutput:
    analyst: pd.DataFrame
    oracle: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
