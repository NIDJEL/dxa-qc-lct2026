"""Approval-gated nested grouped training. Outer data never selects checkpoints.

V1 uses 60 fit / 20 inner / 20 outer studies. It intentionally does not refit on
the inner holdout, preserving the exact selected checkpoint/calibration pairing.
"""
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from scipy.special import expit
import torch
import yaml

from . import TARGETS
from .data import StudyBags, nested_indices
from .features import digest_file, json_hash, provenance as encoder_provenance
from .metrics import fit_calibration, summarize, grouped_bootstrap
from .model import (E007, E010MeanMax, E015PixelFusion,
                    initialize_geometry_image_from_pixel_encoder,
                    initialize_geometry_from_spatial_adapter,
                    initialize_intact_pixel_encoder, masked_loss, positive_weights)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")
    os.replace(temporary, path)


def save_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_all(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def predict(model, dataset, indices, device, pos_weight=None):
    model.eval()
    logits, targets, losses = [], [], []
    for idx in indices:
        batch, target, _ = dataset[int(idx)]
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(to_device(batch, device))["logits"]
            loss = masked_loss(out, target.to(device), pos_weight,
                               getattr(model, "_loss_mode", "bce"))
        logits.append(out.float().cpu().numpy())
        targets.append(target.numpy())
        losses.append(float(loss))
    return np.stack(logits), np.stack(targets), float(np.mean(losses))


def selection_score(summary):
    # Fixed before any training. Undefined F1 contributes zero to selection
    # only; metric reports retain null and explicit support.
    f1 = [summary["targets"][t]["f1"] or 0. for t in TARGETS]
    return float(np.mean(f1)), float(np.mean(f1[7:]))


def fit_one(dataset, config, outer, seed, ablation, directory, device, resume=False,
            evaluate_outer=True):
    directory.mkdir(parents=True, exist_ok=resume)
    seed_all(seed)
    fit, tune, test = nested_indices(dataset.folds, outer)
    splits = {name: dataset.groups[idx].tolist() for name, idx in
              (("fit", fit), ("inner", tune), ("outer", test))}
    write_json(directory / "split.json", splits)
    model_type = config.get("model_type", "e007")
    model_class = (E010MeanMax if model_type == "e010_meanmax"
                   else E015PixelFusion if model_type == "e015_pixel_fusion"
                   else E007)
    model = (model_class().to(device) if model_class is not E007
             else model_class(ablation).to(device))
    transfer = None
    if config.get("geometry_init"):
        transfer = initialize_geometry_from_spatial_adapter(model, config["geometry_init"])
    if config.get("geometry_image_init"):
        transfer = initialize_geometry_image_from_pixel_encoder(
            model, config["geometry_image_init"])
    if config.get("pixel_init"):
        transfer = initialize_intact_pixel_encoder(model, config["pixel_init"])
    if config.get("freeze_pixel", False):
        if not hasattr(model, "pixel"):
            raise ValueError("freeze_pixel requires a pixel backbone")
        model.pixel.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config["plateau_scheduler"]["factor"],
        patience=config["plateau_scheduler"]["patience"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    weight_mode = config.get("pos_weight_mode", "sqrt_cap5")
    loss_mode = config.get("loss_mode", "bce")
    model._loss_mode = loss_mode
    weights = positive_weights(torch.from_numpy(dataset.targets[fit]), weight_mode).to(device)
    best = (-float("inf"), -float("inf"))
    stale, history = 0, []
    optimizer_steps, overflow_steps = 0, 0
    ckpt_dir = directory / "checkpoints"
    ckpt_dir.mkdir(exist_ok=resume)
    best_path = ckpt_dir / "best.pt"
    provenance = {"ablation": ablation, "seed": seed, "outer_fold": outer,
                  "target_order": list(TARGETS), "feature_signature": dataset.index["signature"],
                  "folds_sha256": dataset.index["folds_sha256"], "config": config,
                  "config_hash": json_hash(config), "split_hash": json_hash(splits),
                  "selection": "inner F1 at fixed .5, undefined=0; quality F1 tiebreak",
                  "model_type": config.get("model_type", "e007"),
                  "pos_weight_mode": weight_mode,
                  "loss_mode": loss_mode,
                  "geometry_init": config.get("geometry_init"),
                  "geometry_image_init": config.get("geometry_image_init"),
                  "pixel_init": config.get("pixel_init"),
                  "geometry_transfer": transfer}
    provenance["implementation_sha256"] = {
        p.name: digest_file(p) for p in Path(__file__).parent.glob("*.py")}
    last_path = ckpt_dir / "last.pt"
    start_epoch = 0
    if resume and last_path.exists():
        last = torch.load(last_path, map_location=device, weights_only=True)
        if last["provenance"] != provenance:
            raise ValueError("Resume provenance mismatch")
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        scaler.load_state_dict(last["scaler"])
        best, stale, history = tuple(last["best"]), last["stale"], last["history"]
        start_epoch = last["epoch"] + 1
        optimizer_steps, overflow_steps = last["optimizer_steps"], last["overflow_steps"]
        torch.set_rng_state(last["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in last["cuda_rng"]])
        random.setstate(last["python_rng"])
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, config["max_epochs"]):
        if stale >= config["early_stopping"]["patience"]:
            break
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(fit)
        losses = []
        accumulation = config["gradient_accumulation"]
        for offset in range(0, len(order), accumulation):
            chunk = order[offset:offset+accumulation]
            optimizer.zero_grad(set_to_none=True)
            for idx in chunk:
                batch, target, _ = dataset[int(idx)]
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(to_device(batch, device))["logits"]
                    loss = masked_loss(logits, target.to(device), weights, loss_mode)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                scaler.scale(loss / len(chunk)).backward()
                losses.append(float(loss.detach()))
            scaler.unscale_(optimizer)
            finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            if finite:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer_steps += 1
            elif not scaler.is_enabled():
                raise FloatingPointError("Nonfinite unscaled gradients without AMP")
            else:
                # GradScaler skips the update and reduces its scale on overflow.
                # Do not clip infinities into NaNs or bypass scaler bookkeeping.
                overflow_steps += 1
            scaler.step(optimizer)
            scaler.update()
            if overflow_steps > 20 and optimizer_steps == 0:
                raise FloatingPointError("Persistent AMP overflow before any optimizer step")
        val_logits, val_y, val_loss = predict(model, dataset, tune, device, weights)
        metrics = summarize(val_y, expit(val_logits))
        score = selection_score(metrics)
        scheduler.step(score[0])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "inner_loss": val_loss,
               "selection_score": list(score), "inner_metrics": metrics,
               "lr": optimizer.param_groups[0]["lr"], "optimizer_steps": optimizer_steps,
               "amp_overflow_steps": overflow_steps, "amp_scale": scaler.get_scale()}
        history.append(row)
        delta = config["early_stopping"]["min_delta"]
        improves = score[0] > best[0] + delta or (abs(score[0]-best[0]) <= 1e-12 and score[1] > best[1]+delta)
        if improves:
            best, stale = score, 0
            temporary = ckpt_dir / "best.tmp"
            torch.save({"model": model.state_dict(), "epoch": epoch, "provenance": provenance,
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict()}, temporary)
            os.replace(temporary, best_path)
        else:
            stale += 1
        write_json(directory / "learning_curves.json", history)
        save_checkpoint(last_path, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "provenance": provenance, "best": list(best),
            "stale": stale, "history": history, "torch_rng": torch.get_rng_state(),
            "optimizer_steps": optimizer_steps, "overflow_steps": overflow_steps,
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "python_rng": random.getstate(),
            # Epoch shuffling uses stateless default_rng(seed + epoch).
            "numpy_rng_policy": "stateless_per_epoch_seed_plus_epoch"})
        print(f"{ablation} seed={seed} fold={outer} epoch={epoch} inner={score[0]:.4f}", flush=True)
        if stale >= config["early_stopping"]["patience"]:
            break
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    inner_logits, inner_y, _ = predict(model, dataset, tune, device)
    calibration = fit_calibration(inner_y, inner_logits)
    if not evaluate_outer:
        checkpoint["calibration"] = calibration
        save_checkpoint(best_path, checkpoint)
        report = {"status": "training_smoke_passed", "epochs": len(history),
                  "best_epoch": checkpoint["epoch"], "outer_evaluated": False,
                  "optimizer_steps": optimizer_steps, "amp_overflow_steps": overflow_steps,
                  "elapsed_seconds": time.perf_counter()-start}
        write_json(directory / "report.json", report)
        return report
    # Outer access occurs only AFTER selection and calibration.
    outer_logits, outer_y, _ = predict(model, dataset, test, device)
    probability = expit(outer_logits / np.asarray(calibration["temperature"]))
    from .diagnostics import inspect_outer
    diagnostics = inspect_outer(model, dataset, test, device, directory)
    report = {"seed": seed, "outer_fold": outer, "ablation": ablation,
              "best_epoch": checkpoint["epoch"], "calibration": calibration,
              "diagnostics": diagnostics, "epochs_run": len(history),
              "optimizer_steps": optimizer_steps, "amp_overflow_steps": overflow_steps,
              "outer_metrics": summarize(outer_y, probability, calibration["threshold"]),
              "elapsed_seconds": time.perf_counter()-start,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
              "checkpoint_sha256": digest_file(best_path)}
    checkpoint["calibration"] = calibration
    torch.save(checkpoint, best_path)
    report["checkpoint_sha256"] = digest_file(best_path)
    write_json(directory / "report.json", report)
    np.savez(directory / "predictions.npz", indices=test, y=outer_y, p=probability,
             logits=outer_logits, groups=dataset.groups[test].astype(str),
             thresholds=np.tile(calibration["threshold"], (len(test), 1)))
    return test, probability, calibration["threshold"], report


def run_experiment(root, config_path, cache_run, output, resume=False):
    torch.set_num_threads(2)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf8"))
    if config["max_epochs"] <= 0 or config["gradient_accumulation"] <= 0:
        raise ValueError("Positive epochs/accumulation required")
    if config["seeds"] != [17, 29, 43]:
        raise ValueError("All fixed seeds required")
    cache_run, output = Path(cache_run), Path(output)
    dataset = StudyBags(root, cache_run / "cache", cache_run / "index.json")
    if dataset.index.get("smoke_only"):
        raise ValueError("Smoke cache is never a training cache")
    if dataset.index["signature"] != encoder_provenance(Path(root)):
        raise ValueError("Encoder/code/preprocessing changed since cache extraction")
    output.mkdir(parents=True, exist_ok=resume)
    if (output / "completion.json").exists():
        raise ValueError("Finalized experiments cannot be overwritten")
    if resume and (output / "config.json").exists():
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Resume config mismatch")
    write_json(output / "config.json", config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries = {}
    for ablation in config["ablations"]:
        summaries[ablation] = {}
        for seed in config["seeds"]:
            p = np.full_like(dataset.targets, np.nan)
            thresholds = np.full_like(dataset.targets, np.nan)
            fold_reports = []
            for outer in range(5):
                directory = output / ablation / f"seed_{seed}" / f"fold_{outer}"
                if resume and (directory / "predictions.npz").exists():
                    checkpoint = torch.load(directory / "checkpoints/best.pt", map_location="cpu", weights_only=True)
                    prov = checkpoint["provenance"]
                    if prov["config_hash"] != json_hash(config) or prov["feature_signature"] != dataset.index["signature"]:
                        raise ValueError("Completed fold resume provenance mismatch")
                    with np.load(directory / "predictions.npz", allow_pickle=False) as z:
                        p[z["indices"]], thresholds[z["indices"]] = z["p"], z["thresholds"]
                    fold_reports.append(json.loads((directory / "report.json").read_text()))
                    continue
                indices, probs, th, report = fit_one(
                    dataset, config, outer, seed, ablation,
                    directory, device, resume=resume)
                p[indices], thresholds[indices] = probs, th
                fold_reports.append(report)
            report = summarize(dataset.targets, p, thresholds)
            report["ci95"] = grouped_bootstrap(dataset.targets, p, dataset.groups, thresholds,
                                               config["bootstrap_replicates"], seed)
            report["folds"] = fold_reports
            summaries[ablation][str(seed)] = report
            np.savez(output / ablation / f"predictions_seed_{seed}.npz",
                     y=dataset.targets, p=p, thresholds=thresholds, groups=dataset.groups.astype(str))
            write_json(output / "summary.json", summaries)
    # Do not select an ablation/seed using outer results. A6 is prespecified full model.
    from .analysis import analyze_runs, analyze_single_run
    if len(config["ablations"]) == 7:
        analyze_runs(output, config["seeds"])
    elif len(config["ablations"]) == 1:
        analyze_single_run(output, config["ablations"][0], config["seeds"])
    else:
        raise ValueError("Use all seven ablations or exactly one prespecified ablation")
    write_json(output / "completion.json", {"status": "complete",
                                           "full_model": config["ablations"][0] if len(config["ablations"]) == 1 else "A6",
                                           "seeds": config["seeds"], "no_next_experiments_started": True})
