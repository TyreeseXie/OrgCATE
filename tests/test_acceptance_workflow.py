from __future__ import annotations

import json

import pandas as pd

from orgcate_sim.public_demo import _prepare_output
from orgcate_sim.release import NINE_SITUATIONS, repository_root, sha256_file


def test_acceptance_config_matches_bundled_backbone() -> None:
    root = repository_root()
    config = json.loads(
        (root / "configs" / "data_to_results_acceptance_v1.json").read_text(
            encoding="utf-8"
        )
    )
    path = root / config["input"]["path"]
    frame = pd.read_pickle(path, compression="gzip")

    assert sha256_file(path) == config["input"]["sha256"]
    assert frame.shape == (
        config["input"]["expected_rows"],
        config["input"]["expected_columns"],
    )
    assert config["analysis"]["sample_size"] == 3000
    assert tuple(config["analysis"]["scenarios"]) == NINE_SITUATIONS
    assert config["output_policy"]["include_firm_level_rows"] is False


def test_acceptance_output_plan_excludes_row_level_tables(tmp_path) -> None:
    paths = _prepare_output(tmp_path, save_row_level=False)

    assert "analyst" not in paths
    assert "oracle" not in paths
    assert "practical" in paths
    assert "evaluation" in paths
