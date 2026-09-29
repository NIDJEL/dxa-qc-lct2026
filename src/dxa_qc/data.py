"""Source-level targets only. No propagation to unverified image-level targets."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from . import TARGETS
from .features import FeatureStore, digest_file


def study_table(root):
    root = Path(root)
    manifest = pd.read_parquet(root / "artifacts/E000_data_audit/dataset_manifest.parquet")
    folds = pd.read_csv(root / "artifacts/E000_data_audit/frozen_folds_v1.csv")
    learn = manifest[manifest.source_dataset == "learn"].copy()
    columns = [f"{t}_label" for t in TARGETS]
    # Refuse inconsistent source targets rather than silently taking the first.
    if (learn.groupby("source_study_id")[columns].nunique(dropna=False) > 1).any().any():
        raise ValueError("Inconsistent source-study targets")
    for column in columns:
        if not learn[column].dropna().isin([0, 1]).all():
            raise ValueError(f"Nonbinary target {column}")
    if folds.source_study_id.duplicated().any() or set(folds.fold) != set(range(5)):
        raise ValueError("Invalid frozen folds")
    groups_per_pixel = learn.dropna(subset=["pixel_hash"]).groupby("pixel_hash").source_study_id.nunique()
    if (groups_per_pixel > 1).any():
        raise ValueError("Cross-study duplicates require new grouping ADR")
    table = learn.groupby("source_study_id")[columns].first().reset_index()
    table = table.merge(folds, on="source_study_id", how="left", validate="one_to_one")
    if table.fold.isna().any() or len(table) != len(folds):
        raise ValueError("Study/fold coverage mismatch")
    # Failed records retained in manifest and reported; training fails closed below.
    return learn, table


class StudyBags(Dataset):
    def __init__(self, root, cache_dir, index_path):
        self.root = Path(root)
        self.manifest, self.table = study_table(root)
        index = json.loads(Path(index_path).read_text(encoding="utf8"))
        if index.get("failures"):
            raise ValueError("Unresolved extraction failures: review index before training")
        folds_path = self.root / "artifacts/E000_data_audit/frozen_folds_v1.csv"
        if index["folds_sha256"] != digest_file(folds_path):
            raise ValueError("Frozen fold hash changed")
        manifest_path = self.root / "artifacts/E000_data_audit/dataset_manifest.parquet"
        if index["manifest_sha256"] != digest_file(manifest_path):
            raise ValueError("Manifest/labels changed since cache extraction")
        self.store = FeatureStore(cache_dir, index["signature"])
        self.index = index
        self.pixel_to_key = index["pixel_to_key"]
        # Fail closed on an index pointing to the wrong otherwise-valid feature.
        for pixel, key in self.pixel_to_key.items():
            sidecar = json.loads((Path(cache_dir) / f"{key}.json").read_text())
            metadata = sidecar["metadata"]
            if metadata["pixel_hash"] != pixel or self.store.key(pixel, metadata["calibration"]) != key:
                raise ValueError("Cache index identity/calibration mismatch")
        self.bags = []
        for sid in self.table.source_study_id:
            rows = self.manifest[self.manifest.source_study_id == sid]
            if rows.read_error.notna().any() or rows.pixel_hash.isna().any():
                raise ValueError(f"Unresolved failed record in {sid}")
            pixels = sorted(set(rows.pixel_hash))
            if any(p not in self.pixel_to_key for p in pixels):
                raise ValueError("Incomplete cache; smoke cache cannot train")
            self.bags.append([self.pixel_to_key[p] for p in pixels])
        self.targets = self.table[[f"{t}_label" for t in TARGETS]].to_numpy(dtype=np.float32)
        self.folds = self.table.fold.to_numpy(dtype=int)
        self.groups = self.table.source_study_id.to_numpy()
        # Validate each immutable feature once, then reuse CPU tensors across
        # epochs/ablations rather than hashing and reading gigabytes each epoch.
        self._records = {key: self.store.read(key) for key in set(sum(self.bags, []))}

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        records = [self._records[key] for key in self.bags[idx]]
        batch = {name: torch.from_numpy(np.stack([r[name] for r in records])).float()
                 for name in FeatureStore.REQUIRED}
        return batch, torch.from_numpy(self.targets[idx]), str(self.groups[idx])


def nested_indices(folds, outer):
    """Deterministic inner holdout: next existing frozen fold, NEVER outer fold."""
    folds = np.asarray(folds)
    inner = (outer + 1) % 5
    fit, tune, test = [np.where(m)[0] for m in
                       ((folds != outer) & (folds != inner), folds == inner, folds == outer)]
    if not all(len(i) for i in (fit, tune, test)):
        raise ValueError("Empty grouped partition")
    return fit, tune, test
