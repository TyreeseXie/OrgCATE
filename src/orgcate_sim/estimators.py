from __future__ import annotations

import numpy as np
import pandas as pd


def fit_causal_forest_dml(
    analyst: pd.DataFrame,
    *,
    x_cols: list[str],
    w_cols: list[str],
    random_state: int,
    n_estimators: int = 400,
    min_samples_leaf: int = 10,
):
    """Fit ONE multivariable CausalForestDML.

    Oracle columns must not be present in `analyst`. The wrapper imports EconML
    lazily so truth-generation tests can run even before EconML is installed.
    """
    forbidden = {"tau_true", "propensity_true", "Y0_true", "Y1_true", "mu_true"}
    leaked = forbidden.intersection(analyst.columns)
    if leaked:
        raise ValueError(f"Oracle leakage detected: {sorted(leaked)}")
    # EconML's default subforest_size is four; enforcing divisibility here
    # turns a deep library error into a reproducible wrapper-level contract.
    if n_estimators < 4 or n_estimators % 4:
        raise ValueError("n_estimators must be a positive multiple of 4 for CausalForestDML")
    if min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be positive")

    try:
        from econml.dml import CausalForestDML
    except ImportError as exc:
        raise ImportError("Install a compatible EconML version before fitting CausalForestDML") from exc

    est = CausalForestDML(
        cv=2,
        discrete_treatment=True,
        honest=True,
        inference=True,
        n_estimators=n_estimators,
        max_samples=0.45,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=1,
    )
    X = analyst[x_cols].to_numpy(dtype=float)
    W = analyst[w_cols].to_numpy(dtype=float) if w_cols else None
    est.fit(
        analyst["Y"].to_numpy(dtype=float),
        analyst["T"].to_numpy(dtype=int),
        X=X,
        W=W,
    )
    cate_hat = np.asarray(est.effect(X)).reshape(-1)
    return est, cate_hat
