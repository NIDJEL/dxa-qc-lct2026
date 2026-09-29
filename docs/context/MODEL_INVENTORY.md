# Model inventory

## DINOv3 ViT-L/16

- Path: `models/dinov3-vitl16/model.safetensors`
- Status: structurally checked and inference smoke-tested.
- SHA256: `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`
- Parameters: 303,129,600; 24 transformer blocks; hidden/embedding dimension 1024.
- Input in smoke test: 224 x 224, grayscale repeated to RGB, normalization
  `(x - 0.5) / 0.5`.
- Eval output finite and deterministic on the local NVIDIA GeForce RTX 3060.
- Details: `artifacts/model_inventory.json`.

## MedImageInsight

- Path: `models/MedImageInsight/mlflow_model_folder/`.
- Status: package/config/checkpoint structurally checked and image-only forward pass
  smoke-tested on the local GPU.
- Checkpoint: `vision_model/medimageinsigt-v1.0.0.pt`.
- SHA256: `5eeda63bf616a61664bc95b2c09d3b3d7125209e635678bd3f5f324e9bdb1414`
- Supplied `.pt` file is safetensors-format with 601 tensors; all tensors were
  finite and the image encoder loaded with `strict=True`.
- Config: DaViT v1, 480 x 480, ImageNet mean/std, 1024 projection dimension.
- Local vendor `DaViT` import passed. The image-only embedding path produced a
  finite deterministic 1024-D result.
- Details and dependency requirements: `artifacts/model_inventory.json` and the
  unmodified vendor `requirements.txt`.

## Frozen research checkpoints (2026-09-23)

- E019_ASYMMETRIC_FOCAL_V1/A3 and E007_FULL_V1/A3: all 15 best.pt per run
  inference/recalibration replay verified; all 30 last.pt finite/load checked.
- Audit: artifacts/E019_all_checkpoints_audit_v1/audit.json (60 SHA256 hashes).
- Preservation: artifacts/E019_preserved_state_v1/manifest.json.
- Geometry research checkpoints exist; no geometrically validated final
  Teacher or Student exists. No seed chosen; no new training authorized.
