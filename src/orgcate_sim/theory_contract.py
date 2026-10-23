"""Typed output contract for the interpretation layer.

The contract intentionally does *not* encode the unresolved scientific
thresholds (region definitions, practical delta, clipping, or multiplicity).
It only enforces separation and claim hygiene once those choices are approved
and supplied by a future runner.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DecisionState(str, Enum):
    SUPPORTED_CANDIDATE = "SUPPORTED_CANDIDATE"
    INCONCLUSIVE = "INCONCLUSIVE"
    NO_THEORY_CLAIM = "NO_THEORY_CLAIM"
    DESIGN_NOT_ELIGIBLE = "DESIGN_NOT_ELIGIBLE"


class Morphology(str, Enum):
    HOMOGENEOUS = "homogeneous"
    SAME_SIGN_DIFFERENT_INTENSITY = "same_sign_different_intensity"
    SMOOTH_SIGN_REVERSAL = "smooth_sign_reversal"
    INVERTED_U = "inverted_u"
    STEEP_CONTINUOUS = "steep_continuous"
    TRUE_JUMP = "true_jump"


# Names that may be present in evaluator-only tables, but never in a practical
# train/test input, discovery object, or theory decision record.
ORACLE_ONLY_FIELDS: frozenset[str] = frozenset(
    {
        "tau_true",
        "y0_true",
        "y1_true",
        "Y0_true",
        "Y1_true",
        "propensity_true",
        "true_morphology",
        "active_moderator",
        "active_moderator_labels",
        "true_boundary",
        "true_zero_crossing",
        "true_flip_point",
        "oracle_construct_diagnostic",
        "latent_confounder_truth",
    }
)


def _find_oracle_fields(value: Any, path: str = "$") -> list[str]:
    """Return paths at which evaluator-only fields occur in nested records."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text in ORACLE_ONLY_FIELDS:
                found.append(child_path)
            found.extend(_find_oracle_fields(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_find_oracle_fields(child, f"{path}[{index}]"))
    return found


def validate_no_oracle_fields(record: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if a practical record contains oracle fields."""

    leaked = _find_oracle_fields(record)
    if leaked:
        raise ValueError(f"oracle-only fields in practical record: {sorted(leaked)}")


@dataclass(frozen=True)
class TheoryDecision:
    """One recorded decision for one world × replication × route.

    ``practical_payload`` must contain only analyst-facing fields.  Oracle
    metrics belong in a separate evaluation object and are deliberately not a
    field on this dataclass.
    """

    state: DecisionState
    world: str
    sample_size: int
    replication: int
    morphology: Morphology | None
    candidate_moderator: str | None
    empirical_proposition: str | None
    allowed_claim: str | None
    reason: str
    practical_payload: Mapping[str, Any] = field(default_factory=dict)
    audit_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if self.replication < 0:
            raise ValueError("replication must be non-negative")
        if not self.world.strip():
            raise ValueError("world must be non-empty")
        if not self.reason.strip():
            raise ValueError("reason must be non-empty")
        validate_no_oracle_fields(self.practical_payload)
        if self.state is DecisionState.SUPPORTED_CANDIDATE and (
            not self.candidate_moderator or not self.empirical_proposition or not self.allowed_claim
        ):
            raise ValueError("SUPPORTED_CANDIDATE requires moderator, proposition, and allowed claim")
        if self.state is DecisionState.DESIGN_NOT_ELIGIBLE and self.allowed_claim:
            raise ValueError("DESIGN_NOT_ELIGIBLE cannot contain an allowed substantive claim")

    def to_record(self) -> dict[str, Any]:
        """Serialize without introducing oracle fields."""

        record = {
            "state": self.state.value,
            "world": self.world,
            "sample_size": self.sample_size,
            "replication": self.replication,
            "morphology": self.morphology.value if self.morphology else None,
            "candidate_moderator": self.candidate_moderator,
            "empirical_proposition": self.empirical_proposition,
            "allowed_claim": self.allowed_claim,
            "reason": self.reason,
            "practical_payload": dict(self.practical_payload),
            "audit_refs": list(self.audit_refs),
        }
        validate_no_oracle_fields(record)
        return record


def validate_confirmation_record(record: Mapping[str, Any]) -> None:
    """Validate a serialized practical decision before it is written."""

    validate_no_oracle_fields(record)
    required = {"state", "world", "sample_size", "replication", "reason"}
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"confirmation record missing required fields: {missing}")
    try:
        DecisionState(str(record["state"]))
    except ValueError as error:
        raise ValueError(f"unknown decision state: {record['state']!r}") from error
