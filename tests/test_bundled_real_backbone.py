from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BACKBONE = (
    ROOT
    / "execution"
    / "phase2_real_backbone_v1"
    / "headline_one_anchor_per_firm_L3.pkl.gz"
)
EXPECTED_SHA256 = "91c963e02cbdc17a19de85868ba44f5f4fd65791bec83906980f307a6cfd3174"
EXPECTED_FEATURES = {
    "at_rank_year_currency",
    "cash_at",
    "intangibles_at",
    "roa_ni",
    "debt_at",
    "ppe_at",
    "log_at_slope_L",
    "capx_at",
    "roa_ni_slope_L",
    "cash_at_slope_L",
    "debt_at_slope_L",
    "sale_rank_year_currency",
}
FORBIDDEN_ORACLE_COLUMNS = {
    "treatment",
    "outcome",
    "tau_true",
    "y0_true",
    "y1_true",
    "propensity_true",
    "true_morphology",
}


def test_bundled_backbone_integrity_and_scientific_contract() -> None:
    assert BACKBONE.is_file()
    assert hashlib.sha256(BACKBONE.read_bytes()).hexdigest() == EXPECTED_SHA256

    frame = pd.read_pickle(BACKBONE)
    assert frame.shape == (8648, 57)
    assert frame["gvkey"].nunique() == 8648
    assert frame["history_length_L"].eq(3).all()
    assert EXPECTED_FEATURES.issubset(frame.columns)
    assert FORBIDDEN_ORACLE_COLUMNS.isdisjoint(frame.columns)
