"""Safe external-data plumbing for E008.

This module deliberately does not manufacture organizer QC labels. External
records carry task-specific auxiliary labels and provenance, while pseudo-QC
records require an explicit calibrated teacher decision.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Iterable

import numpy as np


class ExternalRole(str, Enum):
    DIRECT_QC = "direct_supervised_qc"
    ANATOMY = "anatomy_pretraining"
    GEOMETRY = "geometry_pretraining"
    SEGMENTATION = "segmentation_pretraining"
    LANDMARK = "landmark_spatial_representation"
    SSL = "self_supervised_learning"
    DOMAIN = "domain_adaptation"
    AUXILIARY = "auxiliary_supervised"
    HARD_NEGATIVE = "hard_negative_mining"
    REPRESENTATION = "representation_learning"
    PSEUDO_CANDIDATE = "pseudo_label_candidate"
    OOD = "ood_reference"
    UNUSED = "not_used"


@dataclass(frozen=True)
class ExternalRecord:
    record_id: str
    dataset: str
    path: str
    group_id: str
    modality: str
    anatomy: str
    roles: tuple[str, ...]
    label_provenance: str = "none"
    source_label_names: tuple[str, ...] = ()
    pixel_spacing_known: bool = False
    license_status: str = "verify_before_redistribution"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PseudoLabelDecision:
    decision: str
    probability: float
    calibrated_probability: float
    ensemble_disagreement: float
    fold_seed_agreement: float
    ood_score: float
    teacher_version: str
    confidence: float
    predicted_label: int
    reason: str
    provenance: str = "teacher_oof_calibrated_external_inference"


def group_key(dataset: str, path: str, study_uid: str | None = None) -> str:
    """Return a stable grouping key without treating anonymized PatientID as identity."""
    if study_uid:
        return f"{dataset}:study:{study_uid}"
    # Dataset-specific folder is a safer fallback than a filename stem.
    from pathlib import PurePath
    parent = PurePath(path).parent.as_posix()
    return f"{dataset}:folder:{parent}"


def decide_pseudo_label(
    probability: float,
    calibrated_probability: float,
    ensemble_disagreement: float,
    fold_seed_agreement: float,
    ood_score: float,
    *,
    reliability: dict[str, float],
    teacher_version: str,
) -> PseudoLabelDecision:
    """Fail closed; acceptance needs independently certified class/label policy.

    A policy is NOT authorized by simply filling probability thresholds. During
    an outer experiment certification must use outer-train groups only. Global
    OOF diagnostics may authorize future research, not feed the same outer run.
    """
    required = {
        "auto_accept_min_probability", "auto_accept_max_disagreement",
        "auto_accept_min_agreement", "auto_accept_max_ood", "certified",
        "image_scope_verified", "confirmed_modality", "teacher_version",
        "predicted_label", "provenance_hash", "risk_upper_bound", "risk_budget",
    }
    missing = required.difference(reliability)
    if missing:
        raise ValueError(f"Missing reliability thresholds: {sorted(missing)}")
    if not all(np.isfinite(x) for x in (
        probability, calibrated_probability, ensemble_disagreement,
        fold_seed_agreement, ood_score,
    )):
        raise ValueError("Nonfinite pseudo-label evidence")
    if not all(0 <= x <= 1 for x in (probability, calibrated_probability,
                                    ensemble_disagreement, fold_seed_agreement)):
        raise ValueError("Probability/agreement/disagreement must be in [0,1]")
    confidence = max(calibrated_probability, 1-calibrated_probability)
    predicted_label = int(calibrated_probability >= .5)
    accept = (
        confidence >= reliability["auto_accept_min_probability"]
        and ensemble_disagreement <= reliability["auto_accept_max_disagreement"]
        and fold_seed_agreement >= reliability["auto_accept_min_agreement"]
        and ood_score <= reliability["auto_accept_max_ood"]
        and reliability["certified"] is True
        and reliability["image_scope_verified"] is True
        and reliability["confirmed_modality"] == "DXA"
        and reliability["teacher_version"] == teacher_version
        and reliability["predicted_label"] == predicted_label
        and bool(reliability["provenance_hash"])
        and reliability["risk_upper_bound"] <= reliability["risk_budget"]
    )
    reject = reliability["confirmed_modality"] != "DXA" or ood_score > reliability["auto_accept_max_ood"]
    decision = "AUTO_ACCEPT" if accept else ("REJECT" if reject else "REVIEW")
    return PseudoLabelDecision(
        decision, float(probability), float(calibrated_probability),
        float(ensemble_disagreement), float(fold_seed_agreement),
        float(ood_score), teacher_version, float(confidence), predicted_label,
        "certified" if accept else "out_of_domain" if reject else "uncertified_or_uncertain",
    )


def validate_role_map(records: Iterable[ExternalRecord]) -> None:
    """Fail closed on unsafe role assignments."""
    for record in records:
        roles = set(record.roles)
        if ExternalRole.DIRECT_QC.value in roles and record.label_provenance != "organizer_expert":
            raise ValueError(f"Direct QC role without organizer labels: {record.record_id}")
        if ExternalRole.PSEUDO_CANDIDATE.value in roles and record.modality != "DXA":
            raise ValueError(f"Pseudo-QC candidate is not confirmed DXA: {record.record_id}")
        if ExternalRole.GEOMETRY.value in roles and not (
            record.label_provenance in {"landmark", "segmentation", "synthetic_geometry"}
        ):
            raise ValueError(f"Geometry role lacks physical/label provenance: {record.record_id}")
