from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


def nonlinear_assignment_score(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """A controlled observational treatment-selection score."""
    terms = []
    if W.shape[1] > 0:
        terms.append(0.55 * W[:, 0])
    if W.shape[1] > 1:
        terms.append(-0.35 * W[:, 1])
    if W.shape[1] > 2:
        terms.append(0.20 * np.tanh(W[:, 2]))
    if X.shape[1] > 0:
        terms.append(0.25 * X[:, 0])
    if X.shape[1] > 2 and W.shape[1] > 0:
        terms.append(0.15 * X[:, 2] * W[:, 0])
    if not terms:
        return np.zeros(X.shape[0])
    score = np.sum(terms, axis=0)
    score = (score - score.mean()) / max(score.std(ddof=0), 1e-12)
    return score


def calibrate_intercept(score: np.ndarray, target_rate: float, strength: float) -> float:
    score = np.asarray(score, dtype=float)

    def moment(alpha: float) -> float:
        return float(expit(alpha + strength * score).mean() - target_rate)

    return float(brentq(moment, -20.0, 20.0))


def assign_treatment(
    X: np.ndarray,
    W: np.ndarray,
    *,
    target_rate: float,
    strength: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    score = nonlinear_assignment_score(X, W)
    alpha = calibrate_intercept(score, target_rate, strength)
    propensity = expit(alpha + strength * score)
    T = rng.binomial(1, propensity).astype(int)
    meta = {
        "target_treatment_rate": float(target_rate),
        "expected_treatment_rate": float(propensity.mean()),
        "realized_treatment_rate": float(T.mean()),
        "propensity_min": float(propensity.min()),
        "propensity_q01": float(np.quantile(propensity, 0.01)),
        "propensity_q50": float(np.quantile(propensity, 0.50)),
        "propensity_q99": float(np.quantile(propensity, 0.99)),
        "propensity_max": float(propensity.max()),
        "propensity_below_0_05": float((propensity < 0.05).mean()),
        "propensity_above_0_95": float((propensity > 0.95).mean()),
        "assignment_intercept": alpha,
    }
    return T, propensity, meta
