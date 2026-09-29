"""Coordinates: x=column/right, y=row/down; pixel centres, angles from vertical.

Latent maps are QC representations, NOT anatomical landmarks or calibrated
uncertainty. Physical utilities do not imply anatomical measurement validity.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Calibration:
    row_mm: float | None
    col_mm: float | None
    provenance: str

    def __post_init__(self):
        for value in (self.row_mm, self.col_mm):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("Spacing must be positive finite mm/pixel")
        if (self.row_mm is None) != (self.col_mm is None):
            raise ValueError("Both axes required")

    @property
    def known(self):
        return self.row_mm is not None and self.col_mm is not None


def calibration_for(ds, allow_supplied_scanner=False):
    spacing = getattr(ds, "PixelSpacing", None)
    if spacing is not None:
        try:
            return Calibration(float(spacing[0]), float(spacing[1]), "DICOM.PixelSpacing")
        except (ValueError, IndexError, TypeError):
            return Calibration(None, None, "invalid_DICOM_spacing")
    scanner = (str(getattr(ds, "Manufacturer", "")),
               str(getattr(ds, "ManufacturerModelName", "")))
    if allow_supplied_scanner and scanner == ("GE Healthcare", "Lunar Prodigy Advance"):
        return Calibration(1.05, .60, "organizer_DOCX_supplied_scanner_only")
    return Calibration(None, None, "unknown")


def resize_transform(height, width, size):
    """Original pixel centres -> letterboxed pixel centres, actual rounded scales."""
    if min(height, width, size) <= 0:
        raise ValueError("Positive dimensions required")
    scale = size / max(height, width)
    rh, rw = max(1, round(height * scale)), max(1, round(width * scale))
    top, left = (size - rh) // 2, (size - rw) // 2
    sx, sy = rw / width, rh / height
    matrix = np.array([[sx, 0, left + (sx - 1) / 2],
                       [0, sy, top + (sy - 1) / 2], [0, 0, 1]], dtype=np.float64)
    return matrix, (rh, rw, top, left)


def physical_transform(original_to_model, calibration):
    if not calibration.known:
        return None
    return np.diag([calibration.col_mm, calibration.row_mm, 1.]) @ np.linalg.inv(original_to_model)


def tilt_from_vertical(points_xy, calibration):
    """Unsigned axial tilt [0,90]; NaN without calibration or a nonzero segment."""
    if not calibration.known:
        return float("nan")
    delta = np.asarray(points_xy[1], float) - points_xy[0]
    dx, dy = delta * [calibration.col_mm, calibration.row_mm]
    if dx == 0 and dy == 0:
        return float("nan")
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def spatial_moments(prob):
    """Differentiable normalized centre, covariance and entropy per latent slot."""
    n, k, h, w = prob.shape
    y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=prob.device),
                          torch.linspace(-1, 1, w, device=prob.device), indexing="ij")
    mx, my = (prob * x).sum((-2, -1)), (prob * y).sum((-2, -1))
    dx, dy = x - mx[..., None, None], y - my[..., None, None]
    xx, yy = (prob * dx.square()).sum((-2, -1)), (prob * dy.square()).sum((-2, -1))
    xy = (prob * dx * dy).sum((-2, -1))
    entropy = -(prob * prob.clamp_min(1e-9).log()).sum((-2, -1)) / math.log(max(2, h*w))
    return torch.stack((mx, my, xx, yy, xy, entropy), -1)


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         nn.GroupNorm(8, cout), nn.GELU())


class DenseGeometry(nn.Module):
    """Dense DINO + image pyramid -> 8 latent slots + axial/boundary profiles.

    Clinical criteria motivate retaining position/shape/coverage information, not
    hand-setting gates or claiming a latent slot identifies Th12/ROI/trochanter.
    """
    def __init__(self, width=64, slots=8, output=128):
        super().__init__()
        self.dense = nn.Sequential(nn.Conv2d(1024, width, 1), nn.GroupNorm(8, width), nn.GELU())
        self.image = nn.Sequential(block(1, 16, 2), block(16, 32, 2), block(32, width, 2))
        self.mix = nn.Sequential(block(width * 2 + 2, width), block(width, width))
        self.maps = nn.Conv2d(width, slots, 1)
        # slot content, moments, row/column profiles, 3x3 coverage grid
        self.out = nn.Sequential(nn.LayerNorm(slots * (width+6) + width * 17),
                                 nn.Linear(slots * (width+6) + width * 17, output),
                                 nn.GELU(), nn.Dropout(.2))

    def forward(self, dense, image, valid):
        raw = self.image(image)
        d = self.dense(dense)
        if d.shape[-2:] != raw.shape[-2:]:
            raise ValueError("Raw/dense grids must match the registered preprocessing")
        h, w = raw.shape[-2:]
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=raw.device),
                              torch.linspace(-1, 1, w, device=raw.device), indexing="ij")
        coords = torch.stack((x, y))[None].expand(raw.shape[0], -1, -1, -1)
        f = self.mix(torch.cat((raw, d, coords), 1))
        mask = F.interpolate(valid.float(), (h, w), mode="nearest") > .5
        if not mask.flatten(1).any(1).all():
            raise ValueError("Empty geometry support")
        logits = self.maps(f).float().masked_fill(~mask, -1e4)
        p = logits.flatten(2).softmax(-1).reshape_as(logits)
        moments = spatial_moments(p)
        content = torch.einsum("nkhw,nchw->nkc", p, f.float())
        def pooled(shape):
            # Explicit bin reductions avoid nondeterministic CUDA adaptive-pool
            # backward while preserving adaptive pooling's floor/ceil windows.
            bins = []
            for iy in range(shape[0]):
                for ix in range(shape[1]):
                    ys = slice(math.floor(iy*h/shape[0]), math.ceil((iy+1)*h/shape[0]))
                    xs = slice(math.floor(ix*w/shape[1]), math.ceil((ix+1)*w/shape[1]))
                    support = mask[:, :, ys, xs]
                    bins.append((f[:, :, ys, xs] * support).float().sum((-2, -1)) /
                                support.float().sum((-2, -1)).clamp_min(1))
            return torch.stack(bins, -1).flatten(1)
        features = torch.cat((content.flatten(1), moments.flatten(1),
                              pooled((4, 1)), pooled((1, 4)), pooled((3, 3))), 1)
        return self.out(features), {"latent_maps": p, "moments": moments}
