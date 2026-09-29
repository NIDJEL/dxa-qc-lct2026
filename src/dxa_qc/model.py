"""Small class-specific gated multiple-instance fusion over source-study bags."""
import torch
from torch import nn
from . import ABLATIONS, TARGETS
from .geometry import DenseGeometry, block


def pixel_block(cin, cout, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, padding=dilation, dilation=dilation,
                  bias=False),
        nn.GroupNorm(8, cout), nn.GELU())


class E007(nn.Module):
    def __init__(self, ablation="A6", hidden=128):
        super().__init__()
        self.modalities = ABLATIONS[ablation]
        self.geometry = DenseGeometry(output=hidden) if "geometry" in self.modalities else None
        self.project = nn.ModuleDict({
            m: nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, hidden), nn.GELU(), nn.Dropout(.2))
            for m in self.modalities if m != "geometry"})
        self.attention = nn.ModuleDict({m: nn.Sequential(nn.Linear(hidden, 32), nn.Tanh(),
                                                        nn.Linear(32, len(TARGETS)))
                                        for m in self.modalities})
        self.gates = nn.ModuleList([nn.Linear(hidden * len(self.modalities), len(self.modalities))
                                    for _ in TARGETS])
        self.heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(.2),
                                                  nn.Linear(hidden, 1)) for _ in TARGETS])

    def forward(self, batch):
        # One study, variable number of deduplicated images; no filename/size routing.
        representations = {m: self.project[m](batch[m]) for m in self.project}
        aux = {}
        if self.geometry is not None:
            representations["geometry"], aux = self.geometry(batch["dense"], batch["image"], batch["valid"])
        pooled, attention = [], {}
        for m in self.modalities:
            z = representations[m]
            a = self.attention[m](z).softmax(dim=0)
            attention[m] = a
            pooled.append(torch.einsum("nt,nh->th", a, z))
        stack = torch.stack(pooled, 1)  # target, modality, hidden
        gates = torch.stack([self.gates[t](stack[t].flatten()).softmax(0) for t in range(len(TARGETS))])
        fused = (stack * gates[..., None]).sum(1)
        logits = torch.cat([head(fused[t]) for t, head in enumerate(self.heads)])
        # Quality targets use their own attention, gates, and heads.
        return {"logits": logits, "gates": gates, "attention": attention, **aux}


class E010MeanMax(nn.Module):
    """Lower-variance frozen-feature MIL baseline with explicit mean/max pooling.

    This is a deliberately different inductive bias from E007's target-specific
    attention/gating: each encoder is pooled independently, then the shared
    study representation feeds one head per target.
    """
    def __init__(self, hidden=128):
        super().__init__()
        self.modalities = ("dino", "mi2")
        self.project = nn.ModuleDict({
            m: nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, hidden), nn.GELU(),
                            nn.Dropout(.1))
            for m in self.modalities
        })
        width = hidden * len(self.modalities) * 2
        self.heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Dropout(.1), nn.Linear(width, 1))
            for _ in TARGETS
        ])

    def forward(self, batch):
        pooled = []
        for m in self.modalities:
            z = self.project[m](batch[m])
            pooled.extend((z.mean(0), z.max(0).values))
        study = torch.cat(pooled, dim=-1)
        logits = torch.cat([head(study) for head in self.heads])
        return {"logits": logits}


class PixelKeypointBackbone(nn.Module):
    """Architecture-compatible copy of the externally pretrained E013 encoder."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            pixel_block(1, 16, 2), pixel_block(16, 32, 2),
            pixel_block(32, 64, 2), pixel_block(64, 64, dilation=2),
            pixel_block(64, 64, dilation=4), pixel_block(64, 64, dilation=2))
        self.head = nn.Conv2d(64, 8, 1)

    def forward(self, image):
        return self.encoder(image)


class E015PixelFusion(nn.Module):
    """DINO/MI2 plus intact pretrained pixel-spatial representation."""
    def __init__(self, hidden=128):
        super().__init__()
        self.modalities = ("dino", "mi2", "pixel")
        self.pixel = PixelKeypointBackbone()
        self.project = nn.ModuleDict({
            m: nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, hidden), nn.GELU(),
                            nn.Dropout(.2))
            for m in ("dino", "mi2")
        })
        self.pixel_project = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, hidden),
                                           nn.GELU(), nn.Dropout(.2))
        self.attention = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(hidden, 32), nn.Tanh(),
                             nn.Linear(32, len(TARGETS)))
            for m in self.modalities
        })
        self.gates = nn.ModuleList([
            nn.Linear(hidden * len(self.modalities), len(self.modalities))
            for _ in TARGETS
        ])
        self.heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(.2), nn.Linear(hidden, 1))
            for _ in TARGETS
        ])

    def forward(self, batch):
        representations = {
            m: self.project[m](batch[m]) for m in ("dino", "mi2")
        }
        latent = self.pixel(batch["image"])
        representations["pixel"] = self.pixel_project(torch.cat(
            (latent.mean((-2, -1)), latent.amax((-2, -1))), dim=-1))
        pooled = []
        attention = {}
        for m in self.modalities:
            z = representations[m]
            a = self.attention[m](z).softmax(dim=0)
            attention[m] = a
            pooled.append(torch.einsum("nt,nh->th", a, z))
        stack = torch.stack(pooled, 1)
        gates = torch.stack([
            self.gates[t](stack[t].flatten()).softmax(0)
            for t in range(len(TARGETS))
        ])
        fused = (stack * gates[..., None]).sum(1)
        logits = torch.cat([head(fused[t]) for t, head in enumerate(self.heads)])
        return {"logits": logits, "gates": gates, "attention": attention}


def initialize_geometry_from_spatial_adapter(model, checkpoint):
    """Transfer compatible frozen-DINO spatial weights from audited hip pretraining."""
    if model.geometry is None:
        raise ValueError("Spatial initialization requires a geometry ablation")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["state_dict"] if "state_dict" in payload else payload
    required = {"encoder.0.weight", "encoder.0.bias", "encoder.1.weight",
                "encoder.1.bias", "head.weight", "head.bias"}
    if not required.issubset(state):
        raise ValueError("Unexpected spatial adapter checkpoint")
    g = model.geometry
    with torch.no_grad():
        width = state["encoder.0.weight"].shape[0]
        if width > g.dense[0].out_channels or state["head.weight"].shape[1] != width:
            raise ValueError("Incompatible spatial adapter width")
        g.dense[0].weight[:width].copy_(state["encoder.0.weight"])
        g.dense[0].bias[:width].copy_(state["encoder.0.bias"])
        g.dense[1].weight[:width].copy_(state["encoder.1.weight"])
        g.dense[1].bias[:width].copy_(state["encoder.1.bias"])
        g.maps.weight[:, :width].copy_(state["head.weight"])
        g.maps.bias.copy_(state["head.bias"])
    return {"source": str(checkpoint), "transferred_width": int(width),
            "source_keys": sorted(required)}


def initialize_geometry_image_from_pixel_encoder(model, checkpoint):
    """Transfer the compatible raw-image stem from E013 pixel pretraining."""
    if model.geometry is None:
        raise ValueError("Pixel initialization requires a geometry ablation")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["state_dict"] if "state_dict" in payload else payload
    required = {
        "encoder.0.0.weight", "encoder.0.1.weight", "encoder.0.1.bias",
        "encoder.1.0.weight", "encoder.1.1.weight", "encoder.1.1.bias",
        "encoder.2.0.weight", "encoder.2.1.weight", "encoder.2.1.bias",
    }
    if not required.issubset(state):
        raise ValueError("Unexpected pixel encoder checkpoint")
    target = model.geometry.image
    with torch.no_grad():
        for source_i, target_i in enumerate((0, 1, 2)):
            dest = target[target_i]
            dest[0].weight.copy_(state[f"encoder.{source_i}.0.weight"])
            dest[1].weight.copy_(state[f"encoder.{source_i}.1.weight"])
            dest[1].bias.copy_(state[f"encoder.{source_i}.1.bias"])
    return {"source": str(checkpoint), "transferred_blocks": 3,
            "source_keys": sorted(required)}


def initialize_intact_pixel_encoder(model, checkpoint):
    """Load all E013 pixel encoder weights; the keypoint head is retained too."""
    if not hasattr(model, "pixel"):
        raise ValueError("Intact pixel initialization requires E015PixelFusion")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["state_dict"] if "state_dict" in payload else payload
    model_state = model.pixel.state_dict()
    for key in model_state:
        source_key = key
        if source_key not in state:
            raise ValueError(f"Missing pixel encoder key: {source_key}")
        if model_state[key].shape != state[source_key].shape:
            raise ValueError(f"Pixel encoder shape mismatch: {source_key}")
    model.pixel.load_state_dict({k: state[k] for k in model_state}, strict=True)
    return {"source": str(checkpoint), "intact_encoder": True,
            "transferred_keys": sorted(model_state)}


def masked_loss(logits, targets, pos_weight=None, mode="bce"):
    mask = torch.isfinite(targets)
    safe = torch.where(mask, targets, torch.zeros_like(targets))
    loss = nn.functional.binary_cross_entropy_with_logits(logits, safe, pos_weight=pos_weight, reduction="none")
    if mode == "asymmetric_focal":
        # Positive gamma=0, negative gamma=2; compute stably in float32.
        p = logits.float().sigmoid()
        loss = loss * torch.where(safe == 1, torch.ones_like(p), p.square())
    elif mode != "bce":
        raise ValueError(f"Unknown loss mode: {mode}")
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def positive_weights(targets, mode="sqrt_cap5"):
    """Only call with inner-training targets. Missing/one-class targets use 1."""
    pos = (targets == 1).sum(0)
    neg = (targets == 0).sum(0)
    raw = neg / pos.clamp_min(1)
    if mode == "linear_cap10":
        ratio = raw.clamp(1, 10)
    elif mode == "sqrt_cap5":
        ratio = raw.sqrt().clamp(1, 5)
    else:
        raise ValueError(f"Unknown positive-weight mode: {mode}")
    return torch.where((pos > 0) & (neg > 0), ratio, torch.ones_like(ratio))
