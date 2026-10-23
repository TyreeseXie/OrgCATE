"""OrgCATE methods for estimating treatment-effect heterogeneity."""

from .config import load_pilot_config
from .experiment import simulate_one
from .guarded_pipeline import (
    discover_moderator,
    fit_train_nuisance_score_test,
    make_external_split,
)
from .release import (
    NINE_SITUATIONS,
    audit_method_contract,
    load_integrated_evidence,
    verify_release,
)
from .schemas import SimulationConfig, SimulationOutput

__version__ = "0.1.0"

__all__ = [
    "NINE_SITUATIONS",
    "SimulationConfig",
    "SimulationOutput",
    "audit_method_contract",
    "discover_moderator",
    "fit_train_nuisance_score_test",
    "load_integrated_evidence",
    "load_pilot_config",
    "make_external_split",
    "simulate_one",
    "verify_release",
]
