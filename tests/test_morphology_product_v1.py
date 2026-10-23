from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "execution" / "run_morphology_product_v1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("morphology_product_v1_tested", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_config_and_taxonomy_contract():
    runner = load_runner()
    config = json.loads((ROOT / "configs" / "morphology_product_v1.json").read_text())
    runner.validate_config(config)
    assert len(runner.HIGH_LEVEL_SHAPES) == 9
    assert len(runner.CONFIRMATION_DGPS) == 10
    assert runner.CONFIRMATION_DGPS[-1] == "multi-threshold-step"


def test_multi_step_truth_has_two_boundaries_but_one_high_level_label():
    runner = load_runner()
    config = json.loads((ROOT / "configs" / "morphology_product_v1.json").read_text())
    u = np.array([0.10, 0.29, 0.30, 0.50, 0.70, 0.90])
    tau = runner.multi_step_tau(u, config)
    assert np.allclose(tau, [0.2, 0.2, 1.2, 1.2, 2.2, 2.2])
    assert config["multi_step_truth"]["high_level_label"] == "threshold-step"


def test_piecewise_prediction_recovers_levels():
    runner = load_runner()
    u = np.linspace(0.01, 0.99, 200)
    truth = np.where(u < .30, .2, np.where(u < .70, 1.2, 2.2))
    t_residual = np.ones_like(u)
    model = runner.fit_piecewise_model(u, truth, t_residual, (.30, .70))
    prediction = runner.predict_model(model, np.array([.10, .50, .90]))
    assert np.allclose(prediction, [.2, 1.2, 2.2], atol=1e-10)


def test_analyst_frame_contains_no_oracle_or_forest_fields():
    runner = load_runner()
    frame, columns = runner.analyst_frame(np.ones((20, 12)), np.arange(20), np.tile([0, 1], 10))
    assert columns == [f"X_{index:02d}" for index in range(12)]
    assert not runner.fit_cross_fitted_aipw_ate.__module__.startswith("econml")
    assert not any(name in frame for name in ["tau_true", "cate_hat", "forest_prediction"])


def test_seed_grid_is_deterministic_and_unique():
    runner = load_runner()
    first = runner.seed_grid(2026102202, 10)
    second = runner.seed_grid(2026102202, 10)
    assert first == second
    assert len(set(first.values())) == 10
