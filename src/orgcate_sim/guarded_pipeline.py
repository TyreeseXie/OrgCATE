"""Estimation and sample-splitting utilities used by the prototype.

This module separates the three practical stages that must not be silently
collapsed:

1. TRAIN-only moderator discovery from a fitted multivariable forest;
2. train-fitted nuisance models and untouched-TEST doubly robust scores;
3. a compact three-layer morphology record supplied by the prototype's
   classifier.

No function in this module accepts oracle truth.  Evaluation code may compare
the returned records with oracle tables elsewhere, but the practical path
cannot do so accidentally.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from threadpoolctl import threadpool_limits

from .theory_confirmation import doubly_robust_score
from .theory_contract import ORACLE_ONLY_FIELDS, validate_no_oracle_fields

ORACLE_COLUMNS = frozenset(ORACLE_ONLY_FIELDS) | frozenset(
    {"cate_hat", "forest_prediction", "y0_true", "y1_true", "mu_true"}
)


def _validate_analyst_frame(frame: pd.DataFrame, *, required: Sequence[str]) -> None:
    leaked = ORACLE_COLUMNS.intersection(frame.columns)
    if leaked:
        raise ValueError(f"oracle fields present in practical frame: {sorted(leaked)}")
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"practical frame missing required columns: {missing}")
    values = frame[list(required)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("practical frame contains non-finite values")


@dataclass(frozen=True)
class FeatureRank:
    feature: str
    importance: float
    rank: int
    construct: str


@dataclass(frozen=True)
class ModeratorDiscovery:
    """TRAIN-only feature-ranking result.

    ``candidate_feature`` is the top TRAIN feature. ``candidate_construct``
    collapses correlated proxy indicators into the theory-facing construct
    declared in the data contract; this prevents assets and sales from being
    reported as two independent moderators.
    """

    candidate_feature: str
    candidate_construct: str
    ranked_features: tuple[FeatureRank, ...]
    source: str = "TRAIN_forest_feature_importance"

    def to_record(self) -> dict[str, Any]:
        record = {
            "candidate_feature": self.candidate_feature,
            "candidate_construct": self.candidate_construct,
            "source": self.source,
            "ranked_features": [
                {
                    "feature": item.feature,
                    "importance": item.importance,
                    "rank": item.rank,
                    "construct": item.construct,
                }
                for item in self.ranked_features
            ],
        }
        validate_no_oracle_fields(record)
        return record


def discover_moderator(
    feature_names: Sequence[str],
    feature_importances: Sequence[float],
    *,
    construct_by_feature: Mapping[str, str] | None = None,
    canonical_feature_by_construct: Mapping[str, str] | None = None,
) -> ModeratorDiscovery:
    """Discover a candidate from TRAIN forest importances only.

    The function deliberately accepts importances rather than a full dataset or
    oracle labels.  That makes it impossible for TEST outcomes or true CATE to
    enter the ranking routine through an accidental dataframe join.
    """

    names = tuple(str(name) for name in feature_names)
    values = np.asarray(feature_importances, dtype=float).reshape(-1)
    if len(names) == 0 or len(names) != len(values):
        raise ValueError("feature_names and feature_importances must have equal non-zero length")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("feature importances must be finite and non-negative")
    if len(set(names)) != len(names):
        raise ValueError("feature names must be unique")
    by_feature = dict(construct_by_feature or {name: name for name in names})
    missing_construct = sorted(set(names).difference(by_feature))
    if missing_construct:
        raise ValueError(f"construct mapping missing features: {missing_construct}")
    canonical = dict(canonical_feature_by_construct or {})
    order = np.argsort(-values, kind="stable")
    ranked = tuple(
        FeatureRank(
            feature=names[index],
            importance=float(values[index]),
            rank=rank,
            construct=by_feature[names[index]],
        )
        for rank, index in enumerate(order, start=1)
    )
    top = ranked[0]
    construct = top.construct
    candidate_feature = canonical.get(construct, top.feature)
    return ModeratorDiscovery(
        candidate_feature=candidate_feature,
        candidate_construct=construct,
        ranked_features=ranked,
    )


@dataclass(frozen=True)
class ExternalSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    train_ids: tuple[Any, ...]
    test_ids: tuple[Any, ...]

    def validate(self, *, id_column: str = "unit_id") -> None:
        if self.train.empty or self.test.empty:
            raise ValueError("external TRAIN/TEST split must contain both partitions")
        if set(self.train_ids).intersection(self.test_ids):
            raise ValueError("firm/unit IDs overlap across TRAIN and TEST")
        if (
            id_column in self.train
            and id_column in self.test
            and set(self.train[id_column]).intersection(self.test[id_column])
        ):
            raise ValueError("firm/unit IDs overlap across TRAIN and TEST frames")


def make_external_split(
    analyst: pd.DataFrame,
    *,
    test_share: float = 0.20,
    random_state: int = 20261017,
    id_column: str = "unit_id",
) -> ExternalSplit:
    """Create the external 80/20 split used by the analysis."""

    if not (0.0 < test_share < 1.0):
        raise ValueError("test_share must be between zero and one")
    if id_column not in analyst.columns:
        raise ValueError(f"missing split identity column: {id_column}")
    if analyst[id_column].duplicated().any():
        raise ValueError("the split identity must be unique within a dataset")
    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(analyst))
    n_test = max(1, round(len(analyst) * test_share))
    test_index = order[:n_test]
    train_index = order[n_test:]
    train = analyst.iloc[train_index].copy().reset_index(drop=True)
    test = analyst.iloc[test_index].copy().reset_index(drop=True)
    result = ExternalSplit(
        train=train,
        test=test,
        train_ids=tuple(train[id_column].tolist()),
        test_ids=tuple(test[id_column].tolist()),
    )
    result.validate(id_column=id_column)
    return result


@dataclass(frozen=True)
class TestDRResult:
    propensity: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    score: np.ndarray
    clip_fraction: float
    nuisance_train_rows: int
    feature_columns: tuple[str, ...]

    def practical_record(self) -> dict[str, Any]:
        record = {
            "nuisance_train_rows": self.nuisance_train_rows,
            "test_rows": len(self.score),
            "propensity_clip_fraction": self.clip_fraction,
            "propensity_min": float(np.min(self.propensity)),
            "propensity_max": float(np.max(self.propensity)),
            "dr_score_mean": float(np.mean(self.score)),
            "dr_score_sd": float(np.std(self.score, ddof=1)) if len(self.score) > 1 else 0.0,
            "feature_columns": list(self.feature_columns),
        }
        validate_no_oracle_fields(record)
        return record


def fit_train_nuisance_score_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    x_columns: Sequence[str],
    w_columns: Sequence[str] = (),
    random_state: int = 20261017,
    clip_low: float = 0.05,
    clip_high: float = 0.95,
    max_iter: int = 150,
    min_samples_leaf: int = 20,
) -> TestDRResult:
    """Fit nuisance models on TRAIN and score untouched TEST only."""

    x_columns = tuple(x_columns)
    w_columns = tuple(w_columns)
    columns = tuple(dict.fromkeys((*x_columns, *w_columns)))
    required = (*columns, "T", "Y")
    _validate_analyst_frame(train, required=required)
    _validate_analyst_frame(test, required=required)
    if (
        set(train.columns).intersection(test.columns)
        and "unit_id" in train
        and "unit_id" in test
        and set(train["unit_id"]).intersection(test["unit_id"])
    ):
        raise ValueError("TRAIN and TEST contain overlapping unit IDs")
    if not (0.0 < clip_low < clip_high < 1.0):
        raise ValueError("invalid propensity clipping interval")
    x_train = train[list(columns)].to_numpy(dtype=float)
    x_test = test[list(columns)].to_numpy(dtype=float)
    t_train = train["T"].to_numpy(dtype=int)
    y_train = train["Y"].to_numpy(dtype=float)
    t_test = test["T"].to_numpy(dtype=int)
    y_test = test["Y"].to_numpy(dtype=float)
    if not np.isin(t_train, [0, 1]).all() or np.unique(t_train).size != 2:
        raise ValueError("TRAIN treatment must contain both binary arms")
    common = {
        "max_iter": max_iter,
        "min_samples_leaf": min_samples_leaf,
    }
    propensity_model = HistGradientBoostingClassifier(**common, random_state=random_state)
    outcome_control = HistGradientBoostingRegressor(**common, random_state=random_state + 1)
    outcome_treated = HistGradientBoostingRegressor(**common, random_state=random_state + 2)
    with threadpool_limits(limits=1):
        propensity_model.fit(x_train, t_train)
        outcome_control.fit(x_train[t_train == 0], y_train[t_train == 0])
        outcome_treated.fit(x_train[t_train == 1], y_train[t_train == 1])
    raw_propensity = propensity_model.predict_proba(x_test)[:, 1]
    mu0 = outcome_control.predict(x_test)
    mu1 = outcome_treated.predict(x_test)
    score, clip_fraction = doubly_robust_score(
        y_test,
        t_test,
        mu0,
        mu1,
        raw_propensity,
        clip_low=clip_low,
        clip_high=clip_high,
    )
    return TestDRResult(
        propensity=np.clip(raw_propensity, clip_low, clip_high),
        mu0=mu0,
        mu1=mu1,
        score=score,
        clip_fraction=clip_fraction,
        nuisance_train_rows=len(train),
        feature_columns=columns,
    )


def morphology_record(label: Any, *, candidate_moderator: str, moderator_construct: str) -> dict[str, Any]:
    """Serialize the prototype's ordered three-layer classification."""

    record = {
        "candidate_moderator": candidate_moderator,
        "moderator_construct": moderator_construct,
        "continuity": label.continuity,
        "monotonicity": label.monotonicity,
        "sign_relation": label.sign_relation,
        "shape": label.shape,
        "cell": int(label.cell),
        "requires_abduction": bool(label.requires_abduction),
        "refused": bool(label.refused),
        "refusal_reason": label.refusal_reason,
    }
    validate_no_oracle_fields(record)
    return record
