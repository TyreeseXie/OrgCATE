from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "execution" / "run_kink_resolution_v2.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("kink_resolution_v2_tested", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_focused_scope_and_seed_contract():
    runner = load_runner()
    config = json.loads((ROOT / "configs" / "kink_resolution_v2.json").read_text())
    runner.validate_config(config)
    assert runner.TARGET_DGPS == ("kinked", "nonlinear-monotone", "steep-continuous")
    assert len(set(runner.seed_grid(2026102204, 10).values())) == 10


def test_ideal_kink_nested_evidence():
    runner = load_runner()
    config = json.loads((ROOT / "configs" / "kink_resolution_v2.json").read_text())
    rng = np.random.default_rng(8127)
    u = rng.uniform(0, 1, 3000)
    t_residual = rng.choice([-0.3, 0.7], size=len(u))
    tau = 1.5 + 2.0 * u + 7.0 * np.maximum(u - .5, 0)
    y_residual = t_residual * tau + rng.normal(0, .15, len(u))
    evidence = runner.nested_kink_evidence(u, y_residual, t_residual, config, 5541)
    assert abs(evidence["center_median"] - .5) <= .025
    assert evidence["loss_difference_upper"] < 0
    assert evidence["same_sign_rate"] >= .8
    assert evidence["median_slope_ratio"] >= 2


def test_smooth_linear_does_not_pass_combined_rule():
    runner = load_runner()
    config = json.loads((ROOT / "configs" / "kink_resolution_v2.json").read_text())
    rng = np.random.default_rng(9128)
    u = rng.uniform(0, 1, 3000)
    t_residual = rng.choice([-0.3, 0.7], size=len(u))
    tau = 1.5 + 3.0 * u
    y_residual = t_residual * tau + rng.normal(0, .15, len(u))
    items = [runner.nested_kink_evidence(u, y_residual, t_residual, config, 6000 + index) for index in range(3)]
    assert not runner.combine_evidence(items, config)["kink_supported_train"]


def test_no_oracle_or_test_input_in_evidence_signature():
    runner = load_runner()
    names = set(runner.nested_kink_evidence.__code__.co_varnames[:5])
    assert names == {"u", "y_residual", "t_residual", "config", "seed"}
