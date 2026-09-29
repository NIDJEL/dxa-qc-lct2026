"""Offline sequential frozen encoders and content-addressed, integrity-checked cache."""
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from safetensors.torch import load_file

from .geometry import calibration_for, resize_transform, physical_transform

PREPROCESS = {
    "version": "e007_native_norm_v2", "dino_size": 448, "mi2_size": 480,
    "image_size": 224, "interpolation": "PIL_bicubic", "pad": 0,
    "intensity": "uint8_div255_no_per_image_normalization",
    "dino_norm": [[.485, .456, .406], [.229, .224, .225]],
    "dino_resize": "letterbox_bilinear_448_registered_dense_geometry",
    "mi2_resize": "native_RGB_square_bicubic_480_no_crop",
    "mi2_norm": [[.485, .456, .406], [.229, .224, .225]],
    "dense": "final_normalized_patch_tokens_exclude_cls_and_4_registers",
    "calibration": "valid_DICOM_else_explicit_supplied_scanner_DOCX",
}


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pixel_hash(pixels):
    # Same decoded-pixel identity as E000.
    h = hashlib.sha256()
    h.update(f"{pixels.shape}|{pixels.dtype}|".encode())
    h.update(pixels.tobytes())
    return h.hexdigest()


def letterbox(pixels, size, resample=Image.Resampling.BICUBIC):
    matrix, (rh, rw, top, left) = resize_transform(*pixels.shape, size)
    resized = np.asarray(Image.fromarray(pixels).resize((rw, rh), resample))
    out, valid = np.zeros((size, size), np.uint8), np.zeros((size, size), np.float32)
    out[top:top+rh, left:left+rw] = resized
    valid[top:top+rh, left:left+rw] = 1
    return torch.from_numpy(out.copy()).float()[None] / 255, torch.from_numpy(valid)[None], matrix


def load_image(path):
    import pydicom
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array
    if pixels.ndim != 2 or pixels.dtype != np.uint8 or ds.PhotometricInterpretation != "MONOCHROME2":
        raise ValueError("E007 v1 supports audited uint8 MONOCHROME2 single-frame DICOM only")
    cal = calibration_for(ds, allow_supplied_scanner=True)
    return pixels, cal


def provenance(root):
    paths = {
        "dino": root / "models/dinov3-vitl16/model.safetensors",
        "mi2": root / "models/MedImageInsight/mlflow_model_folder/artifacts/checkpoints/vision_model/medimageinsigt-v1.0.0.pt",
    }
    vendor = root / "models/MedImageInsight/mlflow_model_folder"
    # Include all vendor Python dependencies, not just the entry point.
    code_hash = json_hash({str(p.relative_to(vendor)): digest_file(p)
                           for p in sorted((vendor / "code").rglob("*.py"))})
    return {
        "checkpoint_sha256": {k: digest_file(v) for k, v in paths.items()},
        "preprocessing_hash": json_hash(PREPROCESS), "preprocessing": PREPROCESS,
        "encoder_version": {n: importlib.metadata.version(n) for n in
                            ("torch", "transformers", "torchvision", "pillow", "safetensors")},
        "mi2_vendor_code_sha256": code_hash,
        "mi2_config_sha256": digest_file(vendor / "artifacts/checkpoints/config.yaml"),
        "implementation_sha256": digest_file(Path(__file__)),
        "geometry_implementation_sha256": digest_file(Path(__file__).with_name("geometry.py")),
        "dtype": "float32_weights_fp16_autocast_cuda",
    }


class FeatureStore:
    REQUIRED = {"dino": (1024,), "mi2": (1024,), "dense": (1024, 28, 28),
                "image": (1, 224, 224), "valid": (1, 224, 224)}

    def __init__(self, directory, signature):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.signature = signature

    def key(self, pixel, calibration):
        return json_hash({"pixel_hash": pixel, "signature": self.signature, "calibration": calibration})

    def write(self, key, values, metadata):
        self.validate(values)
        path = self.directory / f"{key}.npz"
        if path.exists():
            raise FileExistsError(path)
        temp = self.directory / f".{key}.{os.getpid()}.tmp"
        with temp.open("wb") as f:
            np.savez(f, **values)
        os.replace(temp, path)
        record = {"signature": self.signature, "metadata": metadata, "sha256": digest_file(path)}
        (self.directory / f"{key}.json").write_text(json.dumps(record, indent=2), encoding="utf8")

    def read(self, key):
        path = self.directory / f"{key}.npz"
        record = json.loads((self.directory / f"{key}.json").read_text(encoding="utf8"))
        if record["signature"] != self.signature or digest_file(path) != record["sha256"]:
            raise ValueError("Stale or corrupt feature cache")
        with np.load(path, allow_pickle=False) as z:
            values = {k: z[k].copy() for k in z.files}
        self.validate(values)
        return values

    @classmethod
    def validate(cls, values):
        for key, shape in cls.REQUIRED.items():
            if key not in values or values[key].shape != shape or not np.isfinite(values[key]).all():
                raise ValueError(f"Invalid feature {key}")


def load_encoder(root, name, device):
    if name == "dino":
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        model = DINOv3ViTModel(DINOv3ViTConfig(
            patch_size=16, hidden_size=1024, intermediate_size=4096,
            num_hidden_layers=24, num_attention_heads=16, num_register_tokens=4,
            image_size=224, num_channels=3, apply_layernorm=True, reshape_hidden_states=True))
        state = load_file(str(root / "models/dinov3-vitl16/model.safetensors"))
        model.load_state_dict({f"model.{k}" if k.startswith("layer.") else k: v for k, v in state.items()},
                              strict=True)
    else:
        import yaml
        vendor = root / "models/MedImageInsight/mlflow_model_folder"
        sys.path.insert(0, str(vendor / "code"))
        from MedImageInsight.ImageEncoder.davit_v1 import create_encoder
        config = yaml.safe_load((vendor / "artifacts/checkpoints/config.yaml").read_text())
        config["IMAGE_ENCODER"]["SPEC"]["ENABLE_CHECKPOINT"] = False
        model = create_encoder(config["IMAGE_ENCODER"])
        state = load_file(str(vendor / "artifacts/checkpoints/vision_model/medimageinsigt-v1.0.0.pt"))
        model.load_state_dict({k[len("image_encoder."):]: v for k, v in state.items()
                               if k.startswith("image_encoder.")}, strict=True)
        model.register_buffer("qc_projection", state["image_projection"])
    return model.to(device).eval().requires_grad_(False)


def encoder_input(name, pixels, device):
    size = PREPROCESS[f"{name}_size"]
    if name == "mi2":
        rgb = Image.fromarray(pixels).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        image = torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).float() / 255
    else:
        image, _, _ = letterbox(pixels, size, Image.Resampling.BILINEAR)
    mean, std = PREPROCESS[f"{name}_norm"]
    x = image.expand(3, -1, -1).unsqueeze(0).to(device)
    return (x - torch.tensor(mean, device=device)[None, :, None, None]) / torch.tensor(std, device=device)[None, :, None, None]


def encoder_forward(model, name, pixels, device):
    x = encoder_input(name, pixels, device)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        if name == "dino":
            out = model(pixel_values=x)
            patches = out.last_hidden_state[:, 5:]
            if patches.shape != (1, 784, 1024):
                raise ValueError(f"Unexpected DINO patch layout {patches.shape}")
            return {"dino": out.pooler_output[0].float().cpu().numpy(),
                    "dense": patches[0].T.reshape(1024, 28, 28).half().cpu().numpy()}
        out = model.forward_features(x)
        # Mirrors local vendor UniCL image-only math.
        out = out @ model.qc_projection
        return {"mi2": F.normalize(out.float(), dim=-1)[0].cpu().numpy()}


def extract(root, records, store, device):
    """records are E000 manifest rows; one encoder resident at a time."""
    prepared, failures, stats = {}, [], {}
    for row in records:
        try:
            pixels, cal = load_image(root / row["source_path"])
            if pixel_hash(pixels) != row["pixel_hash"]:
                raise ValueError("Decoded pixels do not match E000")
            calibration = vars(cal)
            key = store.key(row["pixel_hash"], calibration)
            if key in prepared:
                continue
            image, valid, matrix = letterbox(pixels, 224)
            physical = physical_transform(matrix, cal)
            matrix448, _ = resize_transform(*pixels.shape, 448)
            sy, sx = 480 / pixels.shape[0], 480 / pixels.shape[1]
            matrix480 = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]])
            prepared[key] = {
                "pixels": pixels, "values": {"image": image.numpy(), "valid": valid.numpy()},
                "metadata": {"pixel_hash": row["pixel_hash"], "calibration": calibration,
                             "original_to_224": matrix.tolist(),
                             "original_to_448": matrix448.tolist(),
                             "original_to_480": matrix480.tolist(),
                             "model224_to_mm": None if physical is None else physical.tolist()},
            }
        except Exception as exc:
            failures.append({"record_id": row["canonical_record_id"], "error": repr(exc)})
    for name in ("dino", "mi2"):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model = load_encoder(root, name, device)
        params = sum(p.numel() for p in model.parameters())
        buffers = model.qc_projection.numel() if name == "mi2" else 0
        timings, repeat_equal = [], None
        for key, item in prepared.items():
            if item.get("failed"):
                continue
            try:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                inference_start = time.perf_counter()
                first = encoder_forward(model, name, item["pixels"], device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - inference_start)
                if repeat_equal is None:
                    repeated = encoder_forward(model, name, item["pixels"], device)
                    repeat_equal = all(np.array_equal(first[k], repeated[k]) for k in first)
                    if not repeat_equal:
                        raise ValueError("Frozen encoder repeated output differs")
                if not all(np.isfinite(v).all() for v in first.values()):
                    raise FloatingPointError("Nonfinite encoder features")
                item["values"].update(first)
            except Exception as exc:
                item["failed"] = True
                failures.append({"pixel_hash": item["metadata"]["pixel_hash"],
                                 "encoder": name, "error": repr(exc)})
        if device.type == "cuda":
            torch.cuda.synchronize()
        stats[name] = {"parameters": params, "frozen_projection_values": buffers,
                       "all_parameters_frozen": all(not p.requires_grad for p in model.parameters()),
                       "repeat_first_image_exact": repeat_equal,
                       "per_image_inference_seconds_including_preprocess": timings,
                       "seconds_including_load": time.perf_counter()-start,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None}
        del model
    for key, item in prepared.items():
        if item.get("failed"):
            continue
        store.write(key, item["values"], item["metadata"])
    return {v["metadata"]["pixel_hash"]: k for k, v in prepared.items() if not v.get("failed")}, failures, stats
