"""Evaluation-only latent map diagnostics; maps have no anatomical semantics."""
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F


@torch.no_grad()
def inspect_outer(model, dataset, indices, device, directory):
    model.eval()
    rows, latencies = [], []
    directory = Path(directory)
    maps_dir = directory / "latent_maps"
    if getattr(model, "geometry", None) is not None:
        maps_dir.mkdir(exist_ok=True)
    for idx in indices:
        batch, _, group = dataset[int(idx)]
        batch = {k: v.to(device) for k, v in batch.items()}
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
        if "latent_maps" not in out:
            continue
        maps = out["latent_maps"].float()
        flat = maps.flatten(2)
        normalized = F.normalize(flat, dim=-1)
        similarity = normalized @ normalized.transpose(1, 2)
        off = ~torch.eye(flat.shape[1], dtype=torch.bool, device=device)
        centres = out["moments"][..., :2].float()
        distances = torch.cdist(centres, centres)
        for j in range(len(maps)):
            rows.append({
                "source_study_id": group, "image_index_in_bag": j,
                "mean_entropy": float(out["moments"][j, :, 5].mean()),
                "mean_pairwise_cosine": float(similarity[j][off].mean()),
                "fraction_pairs_cosine_gt_099": float((similarity[j][off] > .99).float().mean()),
                "mean_centroid_distance_normalized": float(distances[j][off].mean()),
            })
        # Local sensitive research arrays only; never uploaded or staged.
        np.savez_compressed(maps_dir / f"study_index_{int(idx):03d}.npz",
                            maps=maps.cpu().numpy().astype(np.float16),
                            images=batch["image"].cpu().numpy().astype(np.float16),
                            moments=out["moments"].float().cpu().numpy(),
                            gates=out["gates"].float().cpu().numpy(),
                            source_study_id=np.array(group))
    (directory / "cases_spatial.json").write_text(json.dumps(rows, indent=2), encoding="utf8")
    result = {"cached_head_inference_seconds_mean": float(np.mean(latencies)),
              "cached_head_inference_seconds_p95": float(np.percentile(latencies, 95)),
              "inference_scope": "warm_head_only_excludes_encoders_transfer_and_disk",
              "images_with_maps": len(rows),
              "collapse_threshold": "cosine_gt_099_is_descriptive_not_calibrated"}
    if rows:
        for key in ("mean_entropy", "mean_pairwise_cosine", "fraction_pairs_cosine_gt_099",
                    "mean_centroid_distance_normalized"):
            result[key] = float(np.mean([r[key] for r in rows]))
    return result
