"""Held-out theory-confirmation primitives.

This module consumes an already-created TEST table.  It never fits a forest,
searches for a moderator, chooses a region, or reads oracle columns.  All
scientific thresholds are explicit function arguments so a caller cannot
accidentally rely on hidden defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from .theory_contract import (
    DecisionState,
    Morphology,
    TheoryDecision,
    validate_no_oracle_fields,
)


@dataclass(frozen=True)
class ContrastPolicy:
    """Explicit policy supplied by an approved configuration."""

    propensity_clip_low: float
    propensity_clip_high: float
    minimum_treated: int
    minimum_control: int
    maximum_clip_fraction: float
    confidence_level: float
    practical_delta: float
    require_ci_excludes_zero: bool

    def validate(self) -> None:
        if not (0.0 < self.propensity_clip_low < self.propensity_clip_high < 1.0):
            raise ValueError("propensity clipping must satisfy 0 < low < high < 1")
        if self.minimum_treated < 1 or self.minimum_control < 1:
            raise ValueError("minimum treatment/control support must be positive")
        if not (0.0 <= self.maximum_clip_fraction <= 1.0):
            raise ValueError("maximum_clip_fraction must be in [0, 1]")
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError("confidence_level must be in (0, 1)")
        if self.practical_delta < 0.0:
            raise ValueError("practical_delta must be non-negative")


@dataclass(frozen=True)
class ContrastEstimate:
    left_name: str
    right_name: str
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    p_two_sided: float
    n_left: int
    n_right: int
    treated_left: int
    control_left: int
    treated_right: int
    control_right: int
    clip_fraction: float
    support_passed: bool
    clipping_passed: bool
    expected_sign: int
    practical_delta: float
    require_ci_excludes_zero: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "left_region": self.left_name,
            "right_region": self.right_name,
            "estimate_right_minus_left": self.estimate,
            "standard_error": self.standard_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_two_sided": self.p_two_sided,
            "n_left": self.n_left,
            "n_right": self.n_right,
            "treated_left": self.treated_left,
            "control_left": self.control_left,
            "treated_right": self.treated_right,
            "control_right": self.control_right,
            "propensity_clip_fraction": self.clip_fraction,
            "support_passed": self.support_passed,
            "clipping_passed": self.clipping_passed,
            "expected_sign": self.expected_sign,
            "practical_delta": self.practical_delta,
            "require_ci_excludes_zero": self.require_ci_excludes_zero,
        }


def _as_vector(name: str, values: Any, *, dtype: Any = float) -> np.ndarray:
    vector = np.asarray(values, dtype=dtype).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    return vector


def doubly_robust_score(
    outcome: Any,
    treatment: Any,
    mu0: Any,
    mu1: Any,
    propensity: Any,
    *,
    clip_low: float,
    clip_high: float,
) -> tuple[np.ndarray, float]:
    """Compute a TEST-only AIPW/DR score and return its clip fraction."""

    y = _as_vector("outcome", outcome)
    t = _as_vector("treatment", treatment, dtype=int)
    m0 = _as_vector("mu0", mu0)
    m1 = _as_vector("mu1", mu1)
    e_raw = _as_vector("propensity", propensity)
    if not (len(y) == len(t) == len(m0) == len(m1) == len(e_raw)):
        raise ValueError("outcome, treatment, nuisance predictions, and propensity must align")
    if not np.isin(t, [0, 1]).all():
        raise ValueError("treatment must be binary")
    if not (0.0 < clip_low < clip_high < 1.0):
        raise ValueError("invalid propensity clipping interval")
    e = np.clip(e_raw, clip_low, clip_high)
    gamma = m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
    return gamma, float(np.mean(e != e_raw))


def rank_interval_mask(rank: Any, lower: float, upper: float) -> np.ndarray:
    """Create a frozen rank/ECDF interval mask; bounds are not data-searched."""

    values = _as_vector("rank", rank)
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError("rank interval must satisfy 0 <= lower < upper <= 1")
    upper_mask = values <= upper if upper == 1.0 else values < upper
    return (values >= lower) & upper_mask


def _welch_contrast(left: np.ndarray, right: np.ndarray, confidence_level: float) -> tuple[float, float, float, float, float]:
    estimate = float(np.mean(right) - np.mean(left))
    left_var = float(np.var(left, ddof=1)) if len(left) > 1 else 0.0
    right_var = float(np.var(right, ddof=1)) if len(right) > 1 else 0.0
    variance = left_var / len(left) + right_var / len(right)
    se = float(np.sqrt(max(variance, 0.0)))
    if se == 0.0:
        p_value = 0.0 if estimate != 0.0 else 1.0
        critical = 0.0
    else:
        df_num = variance**2
        df_den = (left_var / len(left)) ** 2 / max(len(left) - 1, 1) + (right_var / len(right)) ** 2 / max(len(right) - 1, 1)
        degrees_of_freedom = max(df_num / df_den, 1.0) if df_den > 0 else 1.0
        p_value = float(2.0 * stats.t.sf(abs(estimate / se), degrees_of_freedom))
        critical = float(stats.t.ppf(0.5 + confidence_level / 2.0, degrees_of_freedom))
    return estimate, se, float(estimate - critical * se), float(estimate + critical * se), p_value


def evaluate_contrast(
    gamma: Any,
    treatment: Any,
    propensity_clip_fraction: float,
    left_mask: Any,
    right_mask: Any,
    *,
    left_name: str,
    right_name: str,
    expected_sign: int,
    policy: ContrastPolicy,
) -> ContrastEstimate:
    """Evaluate a pre-specified contrast on TEST DR scores."""

    policy.validate()
    g = _as_vector("gamma", gamma)
    t = _as_vector("treatment", treatment, dtype=int)
    left = np.asarray(left_mask, dtype=bool).reshape(-1)
    right = np.asarray(right_mask, dtype=bool).reshape(-1)
    if not (len(g) == len(t) == len(left) == len(right)):
        raise ValueError("contrast inputs must align")
    if np.any(left & right):
        raise ValueError("left and right regions must be disjoint")
    if expected_sign not in {-1, 1}:
        raise ValueError("expected_sign must be -1 or 1")
    if not (0.0 <= propensity_clip_fraction <= 1.0):
        raise ValueError("propensity_clip_fraction must be in [0, 1]")

    left_values = g[left]
    right_values = g[right]
    treated_left = int(np.sum(t[left] == 1))
    control_left = int(np.sum(t[left] == 0))
    treated_right = int(np.sum(t[right] == 1))
    control_right = int(np.sum(t[right] == 0))
    support_passed = (
        treated_left >= policy.minimum_treated
        and control_left >= policy.minimum_control
        and treated_right >= policy.minimum_treated
        and control_right >= policy.minimum_control
    )
    if len(left_values) < 2 or len(right_values) < 2:
        raise ValueError("each contrast region needs at least two observations")
    estimate, se, ci_low, ci_high, p_value = _welch_contrast(left_values, right_values, policy.confidence_level)
    return ContrastEstimate(
        left_name=left_name,
        right_name=right_name,
        estimate=estimate,
        standard_error=se,
        ci_low=ci_low,
        ci_high=ci_high,
        p_two_sided=p_value,
        n_left=len(left_values),
        n_right=len(right_values),
        treated_left=treated_left,
        control_left=control_left,
        treated_right=treated_right,
        control_right=control_right,
        clip_fraction=float(propensity_clip_fraction),
        support_passed=support_passed,
        clipping_passed=propensity_clip_fraction <= policy.maximum_clip_fraction,
        expected_sign=expected_sign,
        practical_delta=policy.practical_delta,
        require_ci_excludes_zero=policy.require_ci_excludes_zero,
    )


def decision_from_contrast(
    contrast: ContrastEstimate,
    *,
    world: str,
    sample_size: int,
    replication: int,
    morphology: Morphology,
    candidate_moderator: str,
    empirical_proposition: str,
    allowed_claim: str,
    reason_prefix: str = "",
) -> TheoryDecision:
    """Map one approved TEST contrast to a guarded, claim-hygienic state."""

    if not contrast.support_passed or not contrast.clipping_passed:
        state = DecisionState.INCONCLUSIVE
        reason = f"{reason_prefix}support or propensity clipping rule failed"
    else:
        signed_estimate = contrast.expected_sign * contrast.estimate
        signed_ci_low = contrast.ci_low if contrast.expected_sign > 0 else -contrast.ci_high
        signed_ci_high = contrast.ci_high if contrast.expected_sign > 0 else -contrast.ci_low
        ci_supports_direction = signed_ci_low >= 0.0
        practical = signed_estimate >= contrast.practical_delta
        precision_rule_passed = ci_supports_direction or not contrast.require_ci_excludes_zero
        if practical and precision_rule_passed:
            state = DecisionState.SUPPORTED_CANDIDATE
            reason = f"{reason_prefix}pre-specified TEST contrast met the supplied direction and practical-effect rules"
        elif signed_ci_high < contrast.practical_delta:
            state = DecisionState.NO_THEORY_CLAIM
            reason = f"{reason_prefix}the confidence interval excludes the supplied practical-effect threshold"
        else:
            state = DecisionState.INCONCLUSIVE
            reason = f"{reason_prefix}point direction is not contradicted, but precision or practical support is insufficient"
    practical_payload = contrast.to_record()
    validate_no_oracle_fields(practical_payload)
    return TheoryDecision(
        state=state,
        world=world,
        sample_size=sample_size,
        replication=replication,
        morphology=morphology,
        candidate_moderator=candidate_moderator,
        empirical_proposition=empirical_proposition,
        allowed_claim=allowed_claim if state is DecisionState.SUPPORTED_CANDIDATE else None,
        reason=reason,
        practical_payload=practical_payload,
    )
