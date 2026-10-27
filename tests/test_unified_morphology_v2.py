from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "execution" / "run_unified_morphology_v2.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("unified_morphology_v2_tested", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_config() -> dict:
    return json.loads((ROOT / "configs" / "unified_morphology_v2.json").read_text())


def test_unified_contract_has_no_scenario_dispatch() -> None:
    runner = load_runner()
    config = load_config()
    runner.validate_config(config)

    selection_arguments = set(inspect.signature(runner.select_family).parameters)
    assert selection_arguments == {"libraries", "config", "grid", "ate_train"}
    assert config["selection_policy"]["dgp_label_used_for_shape_selection"] is False
    assert config["selection_policy"]["scenario_specific_dispatch"] is False


def test_pooled_tanh_parameters_are_shared_across_refits() -> None:
    runner = load_runner()
    records = []
    for offset in (0.0, 0.2, -0.1):
        records.append(
            {
                "records": [
                    {
                        "family": "tanh",
                        "params": {"k": 4.0, "c": 0.5},
                        "fold_losses": [1.0 + offset, 1.1 + offset],
                    },
                    {
                        "family": "tanh",
                        "params": {"k": 8.0, "c": 0.5},
                        "fold_losses": [0.4 + offset, 0.5 + offset],
                    },
                ]
            }
        )
    params, _ = runner.pooled_tanh_parameters(records)
    assert params == {"k": 8.0, "c": 0.5}


def test_nested_kink_rule_recovers_ideal_kink() -> None:
    runner = load_runner()
    config = load_config()
    rng = np.random.default_rng(8127)
    u = rng.uniform(0, 1, 3000)
    t_residual = rng.choice([-0.3, 0.7], size=len(u))
    tau = 1.5 + 2.0 * u + 7.0 * np.maximum(u - 0.5, 0)
    y_residual = t_residual * tau + rng.normal(0, 0.15, len(u))
    items = [
        runner.nested_kink_evidence(
            u, y_residual, t_residual, config, 6000 + refit
        )
        for refit in range(3)
    ]
    evidence = runner.combine_nested_kink_evidence(items, config)
    assert evidence["kink_supported_train"]
    assert abs(evidence["center_median"] - 0.5) <= 0.025


def test_nested_kink_rule_rejects_smooth_linear_control() -> None:
    runner = load_runner()
    config = load_config()
    rng = np.random.default_rng(9128)
    u = rng.uniform(0, 1, 3000)
    t_residual = rng.choice([-0.3, 0.7], size=len(u))
    tau = 1.5 + 3.0 * u
    y_residual = t_residual * tau + rng.normal(0, 0.15, len(u))
    items = [
        runner.nested_kink_evidence(
            u, y_residual, t_residual, config, 7000 + refit
        )
        for refit in range(3)
    ]
    assert not runner.combine_nested_kink_evidence(items, config)[
        "kink_supported_train"
    ]


def test_shape_parameter_readout_reports_effective_gradient_and_width() -> None:
    runner = load_runner()
    model = {
        "family": "tanh",
        "coefficient": np.array([0.2, 1.4]),
        "params": {"c": 0.5, "k": 10.0},
    }
    record = runner.parameter_record(
        "steep-continuous", [model, model, model], np.linspace(0.1, 0.9, 161)
    )
    details = json.loads(record["details_json"])
    assert all(np.isclose(item["effective_gradient"], 14.0) for item in details)
    assert all(item["transition_width_25_75"] > 0 for item in details)


def test_hinge_sign_reversal_does_not_require_confirmed_kink_center() -> None:
    runner = load_runner()
    source = inspect.getsource(runner.run_cell)
    assert 'selected_family == "kink" and "selected_center" in selection' in source


def test_append_rows_preserves_schema_for_failure_records(tmp_path: Path) -> None:
    runner = load_runner()
    path = tmp_path / "rows.csv"
    runner.append_rows(
        path,
        [{"status": "completed", "dgp": "linear-monotone", "replication": 0}],
    )
    runner.append_rows(
        path,
        [
            {
                "status": "failed",
                "dgp": "sign-reversal",
                "replication": 1,
                "error": "KeyError: example",
            }
        ],
    )
    frame = pd.read_csv(path)
    assert frame.loc[1, "replication"] == 1
    assert frame.loc[1, "error"] == "KeyError: example"
