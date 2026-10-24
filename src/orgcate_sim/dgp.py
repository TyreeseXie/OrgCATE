from __future__ import annotations

import numpy as np

from .transforms import percentile_rank, standardize_centered


def build_tau(
    X: np.ndarray,
    *,
    morphology: str,
    target_ate: float,
    heterogeneity_sd: float,
    active_x: tuple[int, ...] = (0,),
    cutoff_quantile: float = 0.35,
    steepness: float = 6.0,
    substantive_delta: float = 0.05,
) -> tuple[np.ndarray, dict]:
    """Construct oracle treatment effects from the full multivariable X matrix.

    This utility implements the first-stage worlds plus simple extensions.
    More elaborate moment calibration is outside its current scope.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if morphology == "constant":
        tau = np.full(n, target_ate, dtype=float)
        return tau, {"active_x": [], "true_morphology": morphology}

    j = active_x[0]
    u = percentile_rank(X[:, j])

    if morphology == "monotone":
        h = standardize_centered(u)
        if abs(target_ate) <= substantive_delta:
            raise ValueError("same-sign monotone world requires |target_ate| > substantive_delta")
        max_scale = np.inf
        if target_ate > 0 and h.min() < 0:
            max_scale = (target_ate - substantive_delta) / (-h.min())
        if target_ate < 0 and h.max() > 0:
            max_scale = (-target_ate - substantive_delta) / h.max()
        if max_scale <= 0:
            raise ValueError("requested monotone world cannot satisfy the same-sign threshold")
        scale = min(heterogeneity_sd, max_scale)
        tau = target_ate + scale * h
        return tau, {
            "active_x": [j],
            "true_morphology": morphology,
            "requested_heterogeneity_sd": float(heterogeneity_sd),
            "realized_heterogeneity_sd": float(tau.std(ddof=0)),
            "same_sign_threshold": float(substantive_delta),
        }

    if morphology in {"smooth_flip", "steep_continuous"}:
        k = steepness if morphology == "smooth_flip" else max(steepness, 20.0)
        raw = np.tanh(k * (u - cutoff_quantile))
        # Centering preserves target ATE but may move the zero crossing.
        raw_mean = float(raw.mean())
        raw_sd = float(raw.std(ddof=0))
        h = standardize_centered(raw)
        # A sign-flip world must actually contain both negative and positive effects.
        # Increase the requested heterogeneity scale only when needed to ensure this.
        min_scale_for_negative = 0.0
        min_scale_for_positive = 0.0
        if target_ate >= 0 and h.min() < 0:
            min_scale_for_negative = (target_ate + 0.02) / (-h.min())
        if target_ate <= 0 and h.max() > 0:
            min_scale_for_positive = (-target_ate + 0.02) / h.max()
        scale = max(heterogeneity_sd, min_scale_for_negative, min_scale_for_positive)
        tau = target_ate + scale * h
        root_argument = raw_mean - target_ate * raw_sd / scale
        realized_flip_u = None
        if -1.0 < root_argument < 1.0:
            realized_flip_u = float(cutoff_quantile + np.arctanh(root_argument) / k)
        return tau, {
            "active_x": [j],
            "true_morphology": morphology,
            "nominal_cutoff_quantile": cutoff_quantile,
            "realized_flip_u": realized_flip_u,
            "realized_heterogeneity_sd": float(tau.std(ddof=0)),
        }

    if morphology == "inverted_u":
        raw = -((u - 0.5) ** 2)
        h = standardize_centered(raw)
        tau = target_ate + heterogeneity_sd * h
        return tau, {"active_x": [j], "true_morphology": morphology}

    if morphology == "step":
        left = u < cutoff_quantile
        p = left.mean()
        # Teacher's discontinuity world is intended to have different signs.
        # Choose enough contrast to force tau_left < 0 < tau_right while preserving ATE.
        needed = 0.0
        if target_ate >= 0:
            needed = target_ate / max(1.0 - p, 1e-12) + 0.02
        else:
            needed = (-target_ate) / max(p, 1e-12) + 0.02
        contrast = max(2 * heterogeneity_sd, 0.20, needed)
        tau_l = target_ate - (1.0 - p) * contrast
        tau_r = target_ate + p * contrast
        tau = np.where(left, tau_l, tau_r)
        return tau, {
            "active_x": [j],
            "true_morphology": morphology,
            "cutoff_quantile": cutoff_quantile,
            "tau_left": float(tau_l),
            "tau_right": float(tau_r),
        }

    if morphology == "additive_multi":
        if len(active_x) < 2:
            raise ValueError("additive_multi requires at least two active_x indices")
        u1 = percentile_rank(X[:, active_x[0]])
        u2 = percentile_rank(X[:, active_x[1]])
        h1 = standardize_centered(u1)
        h2 = standardize_centered(-((u2 - 0.5) ** 2))
        raw = (h1 + h2) / np.sqrt(2.0)
        h = standardize_centered(raw)
        tau = target_ate + heterogeneity_sd * h
        return tau, {"active_x": list(active_x[:2]), "true_morphology": morphology}

    if morphology == "configuration":
        if len(active_x) < 2:
            raise ValueError("configuration requires at least two active_x indices")
        u1 = percentile_rank(X[:, active_x[0]])
        u2 = percentile_rank(X[:, active_x[1]])
        high = (u1 > 0.60) & (u2 > 0.60)
        h = standardize_centered(high.astype(float))
        tau = target_ate + heterogeneity_sd * h
        return tau, {"active_x": list(active_x[:2]), "true_morphology": morphology}

    if morphology == "complex_abstain":
        if len(active_x) < 3:
            raise ValueError("complex_abstain requires at least three active_x indices")
        u1 = percentile_rank(X[:, active_x[0]])
        u2 = percentile_rank(X[:, active_x[1]])
        u3 = percentile_rank(X[:, active_x[2]])
        raw = np.sin(4 * np.pi * u1) * (u2 - 0.5) + 0.5 * np.cos(3 * np.pi * u3)
        h = standardize_centered(raw)
        tau = target_ate + heterogeneity_sd * h
        return tau, {
            "active_x": list(active_x[:3]),
            "true_morphology": morphology,
            "expected_abstention": True,
        }

    raise ValueError(f"Unknown morphology: {morphology}")
