from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .audit import audit_truth
from .dgp import build_tau
from .outcomes import generate_outcomes
from .schemas import SimulationConfig, SimulationOutput
from .treatment import assign_treatment


def _correlated_normal(n: int, p: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    if p == 0:
        return np.empty((n, 0))
    cov = np.full((p, p), rho, dtype=float)
    np.fill_diagonal(cov, 1.0)
    return rng.multivariate_normal(np.zeros(p), cov, size=n)


def _split_labels(n: int, cfg: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(n)
    n_d = round(cfg.discovery_share * n)
    n_c = round(cfg.confirmation_share * n)
    labels = np.empty(n, dtype=object)
    labels[order[:n_d]] = "discovery"
    labels[order[n_d:n_d+n_c]] = "confirmation"
    labels[order[n_d+n_c:]] = "replication"
    return labels


def _oracle_feature_roles(cfg: SimulationConfig, *, active_x: set[int]) -> dict[str, list[str]]:
    """Return evaluator-only DGP role metadata; never append it to analyst data."""
    roles: dict[str, list[str]] = {}
    for j in range(cfg.n_x):
        name = f"X_{j}"
        if j in active_x:
            roles[name] = ["candidate_heterogeneity_axis", "effect_modifier", "treatment_predictor"]
        elif j == 0:
            roles[name] = ["candidate_heterogeneity_axis", "treatment_predictor_only"]
        elif j == 1:
            roles[name] = ["candidate_heterogeneity_axis", "prognostic_only"]
        elif j == 2:
            roles[name] = ["candidate_heterogeneity_axis", "confounder"]
        elif j == 3:
            roles[name] = ["candidate_heterogeneity_axis", "correlated_proxy"]
        else:
            roles[name] = ["candidate_heterogeneity_axis", "noise"]
    for j in range(cfg.n_w):
        name = f"W_{j}"
        if j in {0, 1}:
            roles[name] = ["adjustment_control", "confounder"]
        elif j == 2:
            roles[name] = ["adjustment_control", "treatment_predictor_only"]
        else:
            roles[name] = ["adjustment_control", "noise"]
    return roles


def simulate_one(cfg: SimulationConfig, *, seed: int) -> SimulationOutput:
    cfg.validate()
    root = np.random.SeedSequence(seed)
    seeds = root.spawn(5)
    rng_x = np.random.default_rng(seeds[0])
    rng_t = np.random.default_rng(seeds[1])
    rng_y = np.random.default_rng(seeds[2])
    rng_split = np.random.default_rng(seeds[3])

    X = _correlated_normal(cfg.n_units, cfg.n_x, cfg.rho, rng_x)
    W = _correlated_normal(cfg.n_units, cfg.n_w, cfg.rho, rng_x)

    tau, tau_meta = build_tau(
        X,
        morphology=cfg.morphology,
        target_ate=cfg.target_ate,
        heterogeneity_sd=cfg.heterogeneity_sd,
        active_x=cfg.active_x,
        cutoff_quantile=cfg.cutoff_quantile,
        steepness=cfg.steepness,
        substantive_delta=cfg.substantive_delta,
    )

    T, propensity, treatment_meta = assign_treatment(
        X,
        W,
        target_rate=cfg.target_treatment_rate,
        strength=cfg.overlap_strength,
        rng=rng_t,
    )
    mu, y0, y1, y = generate_outcomes(
        X,
        W,
        T,
        tau,
        noise_sd=cfg.outcome_noise_sd,
        rng=rng_y,
    )

    unit_id = np.arange(cfg.n_units)
    split_label = _split_labels(cfg.n_units, cfg, rng_split)

    analyst = pd.DataFrame({"unit_id": unit_id, "T": T, "Y": y, "split_label": split_label})
    for j in range(cfg.n_x):
        analyst[f"X_{j}"] = X[:, j]
    for j in range(cfg.n_w):
        analyst[f"W_{j}"] = W[:, j]

    oracle = pd.DataFrame({
        "unit_id": unit_id,
        "propensity_true": propensity,
        "mu_true": mu,
        "Y0_true": y0,
        "Y1_true": y1,
        "tau_true": tau,
    })
    oracle["true_morphology"] = cfg.morphology
    oracle["active_x_json"] = json.dumps(tau_meta.get("active_x", []))

    audit = audit_truth(
        analyst,
        oracle,
        target_ate=cfg.target_ate,
        target_treatment_rate=cfg.target_treatment_rate,
        substantive_delta=cfg.substantive_delta,
    )
    metadata = {
        **tau_meta,
        **treatment_meta,
        **audit,
        "seed": int(seed),
        "oracle_feature_roles": _oracle_feature_roles(cfg, active_x=set(tau_meta.get("active_x", []))),
    }
    return SimulationOutput(analyst=analyst, oracle=oracle, metadata=metadata)
