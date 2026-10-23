"""Independent cross-fitted AIPW/DML estimator for a global ATE."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

ORACLE_FIELDS = frozenset({"tau_true", "propensity_true", "Y0_true", "Y1_true", "y0_true", "y1_true", "mu_true", "true_morphology", "active_x_json", "active_moderator", "true_boundary", "true_flip_point", "cate_hat", "forest_prediction"})

@dataclass(frozen=True)
class ATEAnchorConfig:
    n_splits: int = 5
    propensity_clip_low: float = .05
    propensity_clip_high: float = .95
    confidence_level: float = .95
    random_state: int = 20261015
    max_iter: int = 150
    learning_rate: float = .05
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    def validate(self):
        if self.n_splits < 2: raise ValueError("n_splits must be at least 2")
        if not (0 < self.propensity_clip_low < self.propensity_clip_high < 1): raise ValueError("invalid propensity clipping")
        if not (0 < self.confidence_level < 1): raise ValueError("invalid confidence level")

@dataclass(frozen=True)
class ATEAnchorResult:
    ate: float; standard_error: float; ci_low: float; ci_high: float
    n_observations: int; n_splits: int; treatment_rate: float
    propensity_clip_fraction: float; propensity_min: float; propensity_max: float
    propensity_q01: float; propensity_q50: float; propensity_q99: float
    score_sd: float; adjustment_columns: tuple[str, ...]
    fold_assignments: np.ndarray; dr_scores: np.ndarray; propensity_oof: np.ndarray
    def to_record(self) -> dict[str, Any]:
        return {"ate_dml": self.ate, "ate_standard_error": self.standard_error, "ate_ci_low": self.ci_low, "ate_ci_high": self.ci_high, "n_observations": self.n_observations, "n_splits": self.n_splits, "treatment_rate": self.treatment_rate, "propensity_clip_fraction": self.propensity_clip_fraction, "propensity_min_oof": self.propensity_min, "propensity_max_oof": self.propensity_max, "propensity_q01_oof": self.propensity_q01, "propensity_q50_oof": self.propensity_q50, "propensity_q99_oof": self.propensity_q99, "dr_score_sd": self.score_sd, "adjustment_columns": list(self.adjustment_columns)}

def resolve_adjustment_columns(analyst: pd.DataFrame) -> list[str]:
    cols = [c for c in analyst.columns if c.startswith(("X_", "W_"))]
    if not cols: raise ValueError("no X_* or W_* adjustment columns found")
    return cols

def _validate(analyst, cols):
    leaked = ORACLE_FIELDS.intersection(analyst.columns)
    if leaked: raise ValueError(f"oracle or forest fields detected in analyst data: {sorted(leaked)}")
    missing = sorted({"Y", "T", *cols}.difference(analyst.columns))
    if missing: raise ValueError(f"ATE inputs missing required columns: {missing}")
    if analyst.empty or not np.isfinite(analyst[["Y", *cols]].to_numpy(float)).all(): raise ValueError("invalid ATE inputs")
    t = analyst["T"].to_numpy(int)
    if not np.isin(t, [0, 1]).all() or np.unique(t).size != 2: raise ValueError("treatment must be binary with both arms")

def fit_cross_fitted_aipw_ate(analyst: pd.DataFrame, *, adjustment_columns: list[str] | None = None, config: ATEAnchorConfig | None = None) -> ATEAnchorResult:
    config = config or ATEAnchorConfig(); config.validate()
    cols = adjustment_columns or resolve_adjustment_columns(analyst); _validate(analyst, cols)
    x = analyst[cols].to_numpy(float); y = analyst["Y"].to_numpy(float); t = analyst["T"].to_numpy(int)
    if len(analyst) < config.n_splits * 2: raise ValueError("input too small for folds")
    folds = np.full(len(analyst), -1, int); e = np.full(len(analyst), np.nan); m0 = np.full(len(analyst), np.nan); m1 = np.full(len(analyst), np.nan)
    splitter = StratifiedKFold(config.n_splits, shuffle=True, random_state=config.random_state)
    for fold, (tr, te) in enumerate(splitter.split(x, t)):
        folds[te] = fold
        common = {
            "max_iter": config.max_iter,
            "learning_rate": config.learning_rate,
            "max_leaf_nodes": config.max_leaf_nodes,
            "min_samples_leaf": config.min_samples_leaf,
            "l2_regularization": config.l2_regularization,
        }
        with threadpool_limits(limits=1):
            p = HistGradientBoostingClassifier(**common, random_state=config.random_state + fold); p.fit(x[tr], t[tr]); e[te] = p.predict_proba(x[te])[:, 1]
            for val, target in ((0, m0), (1, m1)):
                arm = tr[t[tr] == val]; o = HistGradientBoostingRegressor(**common, random_state=config.random_state + 1000 + fold * 2 + val); o.fit(x[arm], y[arm]); target[te] = o.predict(x[te])
    if not np.isfinite(np.c_[e, m0, m1]).all(): raise RuntimeError("incomplete out-of-fold nuisance predictions")
    raw = e.copy(); e = np.clip(e, config.propensity_clip_low, config.propensity_clip_high)
    scores = m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
    ate = float(scores.mean()); sd = float(scores.std(ddof=1)); se = sd / sqrt(len(scores)); z = float(norm.ppf(.5 + config.confidence_level / 2))
    return ATEAnchorResult(ate, se, ate-z*se, ate+z*se, len(scores), config.n_splits, float(t.mean()), float(np.mean(raw != e)), float(raw.min()), float(raw.max()), float(np.quantile(raw,.01)), float(np.quantile(raw,.5)), float(np.quantile(raw,.99)), sd, tuple(cols), folds, scores, raw)
