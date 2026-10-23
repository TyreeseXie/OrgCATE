import numpy as np
import pandas as pd
import pytest

from orgcate_sim import SimulationConfig, simulate_one
from orgcate_sim.ate import (
    ATEAnchorConfig,
    fit_cross_fitted_aipw_ate,
    resolve_adjustment_columns,
)


def test_oof_and_adjustment_set():
    out = simulate_one(SimulationConfig(n_units=300, morphology="constant"), seed=111)
    r = fit_cross_fitted_aipw_ate(out.analyst, config=ATEAnchorConfig(n_splits=3, max_iter=40, min_samples_leaf=8, random_state=9))
    assert r.adjustment_columns == tuple(resolve_adjustment_columns(out.analyst)); assert np.array_equal(np.sort(np.unique(r.fold_assignments)), np.arange(3)); assert np.isfinite(r.dr_scores).all()

def test_rejects_oracle_and_forest_fields():
    a = pd.DataFrame({"Y":[0.,1.,.5,1.5],"T":[0,1,0,1],"X_0":[0.,1.,.2,.8],"tau_true":[.2]*4})
    with pytest.raises(ValueError, match="oracle or forest"): fit_cross_fitted_aipw_ate(a, config=ATEAnchorConfig(n_splits=2, max_iter=10, min_samples_leaf=1))

def test_constant_effect_anchor_is_reasonable():
    out = simulate_one(SimulationConfig(n_units=1000, morphology="constant", target_ate=.2), seed=113)
    r = fit_cross_fitted_aipw_ate(out.analyst, config=ATEAnchorConfig(n_splits=5, max_iter=80, min_samples_leaf=15, random_state=13))
    assert abs(r.ate - .2) < .15; assert r.ci_low < .2 < r.ci_high
