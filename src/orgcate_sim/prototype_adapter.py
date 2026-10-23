"""Compatibility boundary between OrgCATE simulation outputs and ``shapeht``.

The collaborator's prototype expects NumPy arrays and a moderator axis. The
main project stores the same information in an analyst/oracle pair of tables.
This module makes that mapping explicit without changing the existing DGP.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .schemas import SimulationOutput

ORACLE_FIELDS = frozenset(
    {
        "tau_true", "propensity_true", "Y0_true", "Y1_true", "y0_true",
        "y1_true", "mu_true", "true_morphology", "active_x_json",
        "active_moderator", "true_boundary", "true_flip_point",
    }
)


@dataclass(frozen=True)
class PrototypeInput:
    """Array view consumed by the reference prototype."""

    X: np.ndarray
    W: np.ndarray
    T: np.ndarray
    Y: np.ndarray
    moderator: np.ndarray
    covariate_names: tuple[str, ...]
    moderator_name: str
    adjustment_names: tuple[str, ...]
    moderator_source: Literal["oracle", "candidate"]

    def validate(self) -> None:
        n = len(self.Y)
        if not (len(self.X) == len(self.T) == len(self.moderator) == n):
            raise ValueError("prototype arrays must have the same number of rows")
        if self.X.ndim != 2 or self.X.shape[1] == 0:
            raise ValueError("prototype X must be a non-empty two-dimensional array")
        if self.W.ndim != 2 or self.W.shape[0] != n:
            raise ValueError("prototype W must be a two-dimensional array with matching rows")
        if not np.isfinite(self.X).all() or not np.isfinite(self.moderator).all():
            raise ValueError("prototype covariates must be finite")
        if not np.isfinite(self.Y).all() or not np.isfinite(self.T).all():
            raise ValueError("prototype outcomes and treatment must be finite")
        if not np.isin(self.T, [0, 1]).all():
            raise ValueError("prototype treatment must be binary")
        if self.moderator_name not in self.covariate_names:
            raise ValueError("moderator must be one of the prototype covariates")


@dataclass(frozen=True)
class PrototypeRun:
    """Serializable summary of a Steps 1-3 compatibility run."""

    ate: Any
    shape_estimate: Any
    label: Any
    backend_ate: str
    backend_cate: str
    moderator_name: str
    moderator_source: str

    def to_record(self) -> dict[str, Any]:
        return {
            "ate": float(self.ate.theta),
            "ate_ci": [float(self.ate.lo), float(self.ate.hi)],
            "ate_backend": self.backend_ate,
            "cate_backend": self.backend_cate,
            "moderator_name": self.moderator_name,
            "moderator_source": self.moderator_source,
            "supportable": list(self.shape_estimate.supportable),
            "shape": self.label.shape,
            "cell": int(self.label.cell),
            "refused": bool(self.label.refused),
            "refusal_reason": self.label.refusal_reason,
        }


def _columns(frame, prefix: str) -> list[str]:
    return [c for c in frame.columns if c.startswith(prefix)]


def from_simulation(
    simulation: SimulationOutput,
    *,
    moderator_column: str,
    moderator_source: Literal["oracle", "candidate"],
    include_adjustment_controls: bool = True,
) -> PrototypeInput:
    """Map an existing analyst table into the prototype's array contract.

    Candidate moderators enter ``X`` and adjustment controls enter ``W``.
    Keeping them separate is required by the prototype's causal design: the
    forest may split on candidate X while nuisance models condition on W.
    """

    analyst = simulation.analyst
    leaked = ORACLE_FIELDS.intersection(analyst.columns)
    if leaked:
        raise ValueError(f"oracle fields present in analyst table: {sorted(leaked)}")
    x_names = _columns(analyst, "X_")
    w_names = _columns(analyst, "W_")
    covariate_names = x_names
    adjustment_names = w_names if include_adjustment_controls else []
    missing = [c for c in covariate_names + adjustment_names + ["T", "Y", moderator_column] if c not in analyst]
    if missing:
        raise ValueError(f"analyst table missing prototype fields: {missing}")
    if moderator_column not in x_names:
        raise ValueError("moderator_column must be a candidate X_* feature")
    result = PrototypeInput(
        X=analyst[covariate_names].to_numpy(dtype=float),
        W=analyst[adjustment_names].to_numpy(dtype=float) if adjustment_names else np.empty((len(analyst), 0)),
        T=analyst["T"].to_numpy(dtype=int),
        Y=analyst["Y"].to_numpy(dtype=float),
        moderator=analyst[moderator_column].to_numpy(dtype=float),
        covariate_names=tuple(covariate_names),
        moderator_name=moderator_column,
        adjustment_names=tuple(adjustment_names),
        moderator_source=moderator_source,
    )
    result.validate()
    return result


def _load_shapeht():
    """Load the checked-in reference package without installing it globally."""

    root = Path(__file__).resolve().parents[2] / "new" / "prototype" / "src"
    if not (root / "shapeht").is_dir():
        raise FileNotFoundError(f"prototype source package not found: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from shapeht import classify  # type: ignore[import-not-found]
    from shapeht.dml import estimate_ate  # type: ignore[import-not-found]
    from shapeht.forest import estimate_cate  # type: ignore[import-not-found]
    return classify, estimate_ate, estimate_cate


def run_steps_1_to_3(
    view: PrototypeInput,
    *,
    seed: int,
    backend: str = "fallback",
    n_trees: int = 100,
    n_folds: int = 2,
    n_seeds: int = 2,
) -> PrototypeRun:
    """Run the prototype's ATE, CATE, and gated classifier on one view."""

    view.validate()
    classify, estimate_ate, estimate_cate = _load_shapeht()
    ate_features = np.column_stack([view.X, view.W]) if view.W.shape[1] else view.X
    ate = estimate_ate(ate_features, view.T, view.Y, n_folds=n_folds, seed=seed, backend=backend)
    shape_estimate = estimate_cate(
        view.X, view.T, view.Y, view.moderator, ate, W=view.W,
        n_trees=n_trees, n_folds=n_folds, n_seeds=n_seeds,
        seed=seed, backend=backend,
    )
    label = classify(shape_estimate)
    return PrototypeRun(
        ate=ate, shape_estimate=shape_estimate, label=label,
        backend_ate=ate.backend, backend_cate="prototype:" + backend,
        moderator_name=view.moderator_name, moderator_source=view.moderator_source,
    )
