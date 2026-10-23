from __future__ import annotations

import numpy as np


def percentile_rank(x: np.ndarray) -> np.ndarray:
    """Return deterministic mid-rank empirical percentiles in (0, 1)."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    return (ranks - 0.5) / len(x)


def center(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x - x.mean()


def standardize_centered(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = center(x)
    sd = x.std(ddof=0)
    if sd < eps:
        raise ValueError("cannot standardize near-constant shape")
    return x / sd
