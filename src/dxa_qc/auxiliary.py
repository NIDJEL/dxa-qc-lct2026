"""Frozen-DINO spatial adapter and annotation-safe letterboxing.

No external anatomy/segmentation target is called organizer QC ground truth.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from .features import letterbox
from .geometry import resize_transform


class SpatialAdapter(nn.Module):
    def __init__(self, outputs, width=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1024, width, 1),
                                     nn.GroupNorm(8, width), nn.GELU())
        self.head = nn.Conv2d(width, outputs, 1)

    def forward(self, dense):
        latent = self.encoder(dense)
        return self.head(latent), latent

    def descriptor(self, dense, valid):
        logits, latent = self(dense)
        mask = F.interpolate(valid.float(), latent.shape[-2:], mode="nearest")
        pooled = (latent*mask).sum((-2, -1))/mask.sum((-2, -1)).clamp_min(1)
        occupancy = (logits.sigmoid()*mask).sum((-2, -1))/mask.sum((-2, -1)).clamp_min(1)
        return torch.cat([pooled, occupancy], -1)


def read_gray(path):
    with Image.open(path) as im:
        pixels = np.array(im.convert("L"))
    if pixels.ndim != 2:
        raise ValueError("Expected 2-D grayscale")
    return pixels


def keypoint_targets(points, visible, shape, output_size=28, input_size=448):
    points = np.asarray(points, dtype=float)
    visible = np.asarray(visible, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 2 or len(visible) != len(points):
        raise ValueError("Expected [K,xy] and explicit visibility mask")
    h, w = shape
    if not np.isfinite(points[visible]).all():
        raise ValueError("Nonfinite visible keypoints")
    if ((points[visible] < 0).any() or (points[visible, 0] >= w).any()
            or (points[visible, 1] >= h).any()):
        raise ValueError("Visible point outside image")
    matrix, _ = resize_transform(h, w, input_size)
    # PIL/resize_transform uses pixel-centre coordinates; invertible mapping.
    safe = np.where(visible[:, None], points, 0)
    xy = np.c_[safe, np.ones(len(points))] @ matrix.T
    xy = (xy[:, :2]+.5)*(output_size/input_size)-.5
    yy, xx = np.mgrid[:output_size, :output_size]
    heatmaps = np.exp(-((xx[None]-xy[:, 0, None, None])**2+
                       (yy[None]-xy[:, 1, None, None])**2)/2)
    heatmaps[~visible] = 0
    return heatmaps.astype(np.float32), visible, matrix


def segmentation_target(path, shape):
    with Image.open(path) as im:
        raw = np.array(im)
    if raw.ndim != 2 or raw.shape != shape or not set(np.unique(raw)) <= {0, 255}:
        raise ValueError("Expected aligned binary 0/255 source mask")
    mask, valid, matrix = letterbox(raw, 448, Image.Resampling.NEAREST)
    # area downsampling preserves small structures; padding explicitly masked.
    return (F.interpolate(mask[None], (28, 28), mode="area")[0],
            F.interpolate(valid[None], (28, 28), mode="nearest")[0], matrix)


def masked_aux_loss(logits, target, mask, task):
    if not torch.isfinite(target).all():
        raise ValueError("Nonfinite auxiliary target")
    mask = mask.expand_as(logits).float()
    if mask.sum() == 0:
        raise ValueError("No observed auxiliary supervision")
    if task == "segmentation":
        loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    elif task == "keypoint":
        loss = (logits.float().sigmoid()-target.float()).square()
        # Positively weight narrow heatmaps, without altering anatomical labels.
        loss = loss*(1+20*target.float())
    else:
        raise ValueError(task)
    return (loss*mask).sum()/mask.sum()


def coco_keypoints(annotation_path, image_dir):
    data = json.loads(Path(annotation_path).read_text(encoding="utf8"))
    images = {x["id"]: x for x in data["images"]}
    records = []
    for annotation in data["annotations"]:
        if "keypoints" not in annotation:
            continue  # polygons/bboxes are NOT keypoints
        image = images[annotation["image_id"]]
        points = np.asarray(annotation["keypoints"], float).reshape(-1, 3)
        path = Path(image_dir) / image["file_name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as im:
            shape = (im.height, im.width)
        if shape != (image["height"], image["width"]):
            raise ValueError("COCO dimensions do not match source pixels")
        keypoint_targets(points[:, :2], points[:, 2] > 0, shape)
        records.append({"path": str(path), "points": points[:, :2].tolist(),
                        "visible": (points[:, 2] > 0).tolist(),
                        "provenance": "external_coco_keypoints"})
    return records
