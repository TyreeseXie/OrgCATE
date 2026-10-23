from orgcate_sim import SimulationConfig, simulate_one


def test_oracle_columns_absent_from_analyst_view():
    out = simulate_one(SimulationConfig(n_units=200), seed=11)
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
    assert forbidden.isdisjoint(out.analyst.columns)


def test_split_ids_are_disjoint():
    out = simulate_one(SimulationConfig(n_units=200), seed=12)
    groups = {
        s: set(out.analyst.loc[out.analyst["split_label"] == s, "unit_id"])
        for s in out.analyst["split_label"].unique()
    }
    labels = list(groups)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            assert groups[labels[i]].isdisjoint(groups[labels[j]])
