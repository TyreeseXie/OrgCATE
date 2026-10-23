from __future__ import annotations

import numpy as np


def baseline_mean(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Controlled nonlinear prognostic function, separate from tau(X)."""
    n = X.shape[0]
    mu = np.zeros(n, dtype=float)
    if W.shape[1] > 0:
        mu += 0.55 * W[:, 0]
    if W.shape[1] > 1:
        mu += 0.30 * (W[:, 1] ** 2 - np.mean(W[:, 1] ** 2))
    if X.shape[1] > 1:
        mu += 0.25 * np.sin(X[:, 1])
    if X.shape[1] > 2 and W.shape[1] > 0:
        mu += 0.20 * X[:, 2] * W[:, 0]
    mu -= mu.mean()
    return mu


def generate_outcomes(
    X: np.ndarray,
    W: np.ndarray,
    T: np.ndarray,
    tau: np.ndarray,
    *,
    noise_sd: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = baseline_mean(X, W)
    eps = rng.normal(0.0, noise_sd, size=len(T))
    y0 = mu + eps
    y1 = y0 + tau
    y = y0 + T * tau
    return mu, y0, y1, y
