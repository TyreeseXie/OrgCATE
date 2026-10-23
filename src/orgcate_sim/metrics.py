"""Pure oracle-evaluation metrics for completed simulation fits.

These functions are evaluator-side only.  They must not be used to tune the
practical causal-forest workflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def evaluate_cate_predictions(
    analyst: pd.DataFrame,
    oracle: pd.DataFrame,
    cate_hat: np.ndarray,
    *,
    top_fraction: float = 0.20,
) -> dict[str, float | None]:
    """Evaluate fixed predictions against hidden oracle truth after fitting."""
    if not (0 < top_fraction < 0.5):
        raise ValueError("top_fraction must be in (0, 0.5)")
    if len(cate_hat) != len(analyst) or len(oracle) != len(analyst):
        raise ValueError("prediction, analyst, and oracle lengths must match")
    if not np.array_equal(analyst["unit_id"].to_numpy(), oracle["unit_id"].to_numpy()):
        raise ValueError("analyst and oracle IDs must align before evaluation")

    predicted = np.asarray(cate_hat, dtype=float).reshape(-1)
    truth = oracle["tau_true"].to_numpy(dtype=float)
    if not np.isfinite(predicted).all():
        raise ValueError("CATE predictions contain non-finite values")
    rmse = float(np.sqrt(np.mean((predicted - truth) ** 2)))
    # CSV round-trips can turn an exactly constant oracle into tiny floating
    # point variation; treat near-constant arrays as undefined for ranking.
    if np.isclose(np.std(truth), 0.0) or np.isclose(np.std(predicted), 0.0):
        rank_correlation: float | None = None
    else:
        rank_correlation = float(spearmanr(predicted, truth).statistic)

    k = max(1, round(top_fraction * len(predicted)))
    ranking = np.argsort(predicted, kind="mergesort")
    true_top_gate = float(truth[ranking[-k:]].mean())
    true_bottom_gate = float(truth[ranking[:k]].mean())
    return {
        "cate_rmse": rmse,
        "cate_rank_correlation": rank_correlation,
        "cate_hat_mean": float(predicted.mean()),
        "oracle_ate": float(truth.mean()),
        "ate_error_from_cate_mean": float(predicted.mean() - truth.mean()),
        "predicted_cate_sd": float(predicted.std(ddof=0)),
        "true_top_minus_bottom_gate_from_predicted_ranking": float(true_top_gate - true_bottom_gate),
    }
