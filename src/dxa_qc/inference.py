"""Cached-study inference, deliberately NOT an image-level clinical router."""
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from . import TARGETS
from .model import E007


def load_checkpoint(path, expected_signature, device="cpu"):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    provenance = checkpoint["provenance"]
    if provenance["target_order"] != list(TARGETS):
        raise ValueError("Checkpoint target order mismatch")
    if provenance["feature_signature"] != expected_signature:
        raise ValueError("Checkpoint and feature preprocessing/encoders differ")
    model = E007(provenance["ablation"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def infer_bag(model, batch, calibration, device="cpu"):
    device = torch.device(device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        result = model({k: v.to(device) for k, v in batch.items()})
    logits = result["logits"].float().cpu().numpy()
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite study logits")
    p = expit(logits / np.asarray(calibration["temperature"]))
    return {
        "scope": "source_study_QC_not_verified_image_level",
        "targets": {target: {"probability": float(p[i]),
                             "positive": bool(p[i] >= calibration["threshold"][i]),
                             "calibration_status": calibration["status"][i]}
                    for i, target in enumerate(TARGETS)},
        "class_specific_modality_gates": result["gates"].float().cpu().tolist(),
        "image_attention": {k: v.float().cpu().tolist() for k, v in result["attention"].items()},
        "review_policy": "not_validated_no_auto_label_eligibility",
    }
