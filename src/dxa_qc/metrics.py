"""Explicit undefined metrics, AP != trapezoidal PR-AUC, grouped uncertainty."""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize_scalar
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, auc, confusion_matrix)
from . import TARGETS

OFFICIAL_COLUMNS = {
    "spine_positioning": (0,), "spine_axis": (1,), "spine_foreign_object": (2,),
    "hip_positioning_rotation": (3, 5), "hip_roi": (4, 6),
}


def competition_summary(y, p, thresholds=None):
    """Pool labelled side observations, never OR hips or fabricate image labels.

    This is a competition-aligned SOURCE-STUDY proxy, not an official image
    leaderboard score. Bootstrap the original studies before pooling sides.
    """
    th = np.broadcast_to(.5 if thresholds is None else thresholds, y.shape)
    scores = {name: binary_metrics(y[:, cols].reshape(-1), p[:, cols].reshape(-1),
                                  th[:, cols].reshape(-1))
              for name, cols in OFFICIAL_COLUMNS.items()}
    values = [s["f1"] for s in scores.values()]
    quality = binary_metrics(y[:, 7:].reshape(-1), p[:, 7:].reshape(-1), th[:, 7:].reshape(-1))
    return {"scope": "source_study_proxy_not_official_image_score", "classes": scores,
            "violation_macro_f1": float(np.mean(values)) if all(v is not None for v in values) else None,
            "defined_violation_classes": sum(v is not None for v in values),
            "quality_pooled": quality}


def binary_metrics(y, p, threshold=.5):
    valid = np.isfinite(y)
    y, p = np.asarray(y)[valid], np.asarray(p)[valid]
    if np.ndim(threshold):
        threshold = np.asarray(threshold)[valid]
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite predictions")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities outside [0,1]")
    n = len(y)
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    sensitivity = float(tp/(tp+fn)) if tp+fn else None
    specificity = float(tn/(tn+fp)) if tn+fp else None
    f1 = float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else None
    precision, recall, _ = precision_recall_curve(y, p) if both else ([], [], [])
    ece = 0.
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        take = (p >= low) & ((p < high) if high < 1 else (p <= high))
        if take.any():
            ece += take.mean() * abs(p[take].mean() - y[take].mean())
    return {"n": n, "positive": int(y.sum()), "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "f1": f1, "sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": (sensitivity+specificity)/2 if both else None,
            "roc_auc": float(roc_auc_score(y, p)) if both else None,
            "average_precision": float(average_precision_score(y, p)) if both else None,
            "pr_auc_trapezoidal": float(auc(recall, precision)) if both else None,
            "brier": float(np.mean((p-y)**2)) if n else None, "ece_10_bins": float(ece) if n else None,
            "log_loss": float(-np.mean(y*np.log(p.clip(1e-7, 1-1e-7)) +
                                      (1-y)*np.log((1-p).clip(1e-7, 1-1e-7)))) if n else None,
            "undefined_reason": None if both else "missing_or_single_class"}


def summarize(y, p, thresholds=None):
    thresholds = np.full(len(TARGETS), .5) if thresholds is None else np.asarray(thresholds)
    if thresholds.ndim == 2 and thresholds.shape != y.shape:
        raise ValueError("Per-record thresholds must have shape [records, targets]")
    scores = {t: binary_metrics(y[:, i], p[:, i], thresholds[:, i] if thresholds.ndim == 2 else thresholds[i])
              for i, t in enumerate(TARGETS)}
    def mean_metric(names, key):
        values = [scores[t][key] for t in names if scores[t][key] is not None]
        return float(np.mean(values)) if values else None
    return {"targets": scores, "competition": competition_summary(y, p, thresholds),
            "macro_f1": mean_metric(TARGETS, "f1"),
            "violation_macro_f1": mean_metric(TARGETS[:7], "f1"),
            "quality_macro_f1": mean_metric(TARGETS[7:], "f1"),
            "defined_f1_targets": sum(scores[t]["f1"] is not None for t in TARGETS)}


def fit_calibration(y, logits):
    """Scalar temperature and per-target thresholds on inner holdout ONLY.

    Temperature needs >=5 examples of EACH class; rare targets explicitly keep
    identity calibration and threshold .5. No invented confident auto-labeling.
    """
    temperatures, thresholds, status = [], [], []
    for i in range(y.shape[1]):
        valid = np.isfinite(y[:, i])
        yi, zi = y[valid, i], logits[valid, i]
        sufficient = min(int((yi == 0).sum()), int((yi == 1).sum())) >= 5
        if not sufficient:
            temperatures.append(1.)
            thresholds.append(.5)
            status.append("insufficient_inner_support_identity")
            continue
        def nll(log_t):
            z = zi / np.exp(log_t)
            return np.mean(np.logaddexp(0, z) - yi*z)
        opt = minimize_scalar(nll, bounds=(-2, 2), method="bounded")
        temperature = float(np.exp(opt.x)) if opt.success else 1.
        p = expit(zi / temperature)
        candidates = np.linspace(.1, .9, 17)
        # Deterministic tie: closest to .5, then lower threshold.
        best = max(candidates, key=lambda t: (binary_metrics(yi, p, t)["f1"] or 0, -abs(t-.5), -t))
        temperatures.append(temperature)
        thresholds.append(float(best))
        status.append("inner_temperature_and_f1_threshold")
    return {"temperature": temperatures, "threshold": thresholds, "status": status}


def grouped_bootstrap(y, p, groups, thresholds=None, repeats=2000, seed=17):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    keys = ("f1", "sensitivity", "specificity", "balanced_accuracy", "roc_auc",
            "average_precision", "pr_auc_trapezoidal", "brier", "ece_10_bins", "log_loss")
    values = {f"{t}/{m}": [] for t in TARGETS for m in keys}
    values.update({k: [] for k in ("macro_f1", "quality_macro_f1")})
    values["competition/violation_macro_f1"] = []
    for t in (*OFFICIAL_COLUMNS, "quality_pooled"):
        for m in keys:
            values[f"competition/{t}/{m}"] = []
    for _ in range(repeats):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        th = thresholds
        # Thresholds can differ by outer fold: transform to an equivalent .5
        # decision without changing ranking/calibration metrics is NOT valid.
        # Instead feed a per-row threshold into each target's binary_metrics.
        if thresholds is not None and np.asarray(thresholds).ndim == 2:
            th = np.asarray(thresholds)[indices]
        s = summarize(y[indices], p[indices], th)
        comp = s["competition"]
        if comp["violation_macro_f1"] is not None:
            values["competition/violation_macro_f1"].append(comp["violation_macro_f1"])
        for t, result in {**comp["classes"], "quality_pooled": comp["quality_pooled"]}.items():
            for m in keys:
                if result[m] is not None:
                    values[f"competition/{t}/{m}"].append(result[m])
        for t in TARGETS:
            for m in keys:
                if s["targets"][t][m] is not None:
                    values[f"{t}/{m}"].append(s["targets"][t][m])
        for k in ("macro_f1", "quality_macro_f1"):
            if s[k] is not None:
                values[k].append(s[k])
    return {k: {"ci95": np.percentile(v, [2.5, 97.5]).tolist() if v else None,
                "defined_replicates": len(v), "total_replicates": repeats} for k, v in values.items()}
