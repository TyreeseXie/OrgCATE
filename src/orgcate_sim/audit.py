from __future__ import annotations

import numpy as np
import pandas as pd


def audit_truth(
    analyst: pd.DataFrame,
    oracle: pd.DataFrame,
    *,
    target_ate: float,
    target_treatment_rate: float,
    substantive_delta: float,
    tol: float = 1e-8,
) -> dict:
    merged = oracle.sort_values("unit_id")
    if not np.allclose(merged["Y1_true"] - merged["Y0_true"], merged["tau_true"], atol=tol):
        raise AssertionError("Potential-outcome identity failed: Y1 - Y0 != tau_true")

    analyst_s = analyst.sort_values("unit_id")
    if not np.array_equal(analyst_s["unit_id"].to_numpy(), merged["unit_id"].to_numpy()):
        raise AssertionError("Analyst/oracle unit IDs do not align")

    expected_y = merged["Y0_true"].to_numpy() + analyst_s["T"].to_numpy() * merged["tau_true"].to_numpy()
    if not np.allclose(expected_y, analyst_s["Y"].to_numpy(), atol=tol):
        raise AssertionError("Observed outcome consistency failed")

    realized_ate = float(merged["tau_true"].mean())
    if abs(realized_ate - target_ate) > 1e-6:
        raise AssertionError(f"ATE mismatch: target={target_ate}, realized={realized_ate}")

    propensity = merged["propensity_true"].to_numpy(dtype=float)
    if not np.all((0.0 < propensity) & (propensity < 1.0)):
        raise AssertionError("Overlap identity failed: propensities are not strictly inside (0, 1)")
    if abs(float(propensity.mean()) - target_treatment_rate) > 1e-8:
        raise AssertionError("Propensity calibration failed")
    if not set(analyst["T"].unique()).issubset({0, 1}):
        raise AssertionError("Treatment is not binary")

    forbidden = {
        "tau_true",
        "propensity_true",
        "Y0_true",
        "Y1_true",
        "mu_true",
        "true_morphology",
        "active_x_json",
        "true_boundary_or_flip",
    }
    leaked = forbidden.intersection(analyst.columns)
    if leaked:
        raise AssertionError(f"Oracle leakage in analyst data: {sorted(leaked)}")

    split_sets = {
        s: set(analyst.loc[analyst["split_label"] == s, "unit_id"])
        for s in analyst["split_label"].unique()
    }
    labels = list(split_sets)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if split_sets[labels[i]].intersection(split_sets[labels[j]]):
                raise AssertionError("Unit leakage across splits")

    tau = merged["tau_true"].to_numpy(dtype=float)
    treatment = analyst_s["T"].to_numpy(dtype=int)
    sign_regions = {
        "negative": tau < 0.0,
        "zero": tau == 0.0,
        "positive": tau > 0.0,
    }
    substantive_effect_regions = {
        "negative": tau < -substantive_delta,
        "near_zero": np.abs(tau) <= substantive_delta,
        "positive": tau > substantive_delta,
    }

    def summarize_regions(regions: dict[str, np.ndarray]) -> dict[str, dict[str, int]]:
        return {
            name: {
                "n": int(mask.sum()),
                "treated": int((mask & (treatment == 1)).sum()),
                "control": int((mask & (treatment == 0)).sum()),
            }
            for name, mask in regions.items()
        }

    return {
        "realized_ate": realized_ate,
        "treatment_rate": float(analyst["T"].mean()),
        "expected_treatment_rate": float(propensity.mean()),
        "propensity_min": float(propensity.min()),
        "propensity_max": float(propensity.max()),
        "n": len(analyst),
        "sign_region_support": summarize_regions(sign_regions),
        "substantive_effect_region_support": summarize_regions(substantive_effect_regions),
        "truth_audit_passed": True,
    }
