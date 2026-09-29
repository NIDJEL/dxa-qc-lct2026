"""E008 low-variance experts. No external QC targets and no outer selection.

Candidate selection uses cross-fitted predictions on the 60% fit partition.
The 20% calibration partition and 20% outer partition are never passed to it.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer


@dataclass
class Constant:
    probability: float

    def predict_proba(self, x):
        p = np.full(len(x), self.probability)
        return np.column_stack([1-p, p])


def fit_binary(x, y, c, seed):
    valid = np.isfinite(y)
    x, y = np.asarray(x)[valid], np.asarray(y)[valid]
    if len(np.unique(y)) < 2:
        # Jeffreys prior: single-class or missing targets are explicit fallbacks.
        return Constant(float((y.sum()+.5)/(len(y)+1)) if len(y) else .5)
    model = LogisticRegression(C=c, solver="liblinear", max_iter=1000,
                               random_state=seed, class_weight=None)
    return model.fit(x, y)


def bag_features(batch):
    """Permutation invariant; no file size/name/side shortcut, no learned router."""
    result = {}
    for name in ("dino", "mi2"):
        x = batch[name].detach().cpu().numpy()
        # Each frozen embedding is normalized before pooling; no population fit.
        x = Normalizer().fit_transform(x)
        pooled = np.concatenate([x.mean(0), x.max(0)])
        result[name] = pooled / max(np.linalg.norm(pooled), 1e-8)
    return result


def logloss(y, p):
    mask = np.isfinite(y)
    if not mask.any():
        return float("inf")
    p = np.clip(p[mask], 1e-7, 1-1e-7)
    return float(-np.mean(y[mask]*np.log(p)+(1-y[mask])*np.log1p(-p)))


class Expert:
    def __init__(self, views, c, seed):
        self.views, self.c, self.seed = tuple(views), c, seed

    def fit(self, features, y):
        self.models = {v: fit_binary(features[v], y, self.c, self.seed)
                       for v in self.views}
        return self

    def predict(self, features):
        # Equal probability averaging, no high-dimensional learned fusion.
        return np.mean([m.predict_proba(features[v])[:, 1]
                        for v, m in self.models.items()], axis=0)


def select_expert(features, y, folds, seed, candidates):
    """Only accepts the fit partition. All preprocessing is per-sample or fitted."""
    folds = np.asarray(folds)
    if len(set(folds)) != 3:
        raise ValueError("E008 selection requires exactly three frozen fit folds")
    scores = []
    for candidate in candidates:
        predictions = np.full(len(y), np.nan)
        for fold in sorted(set(folds)):
            train, valid = folds != fold, folds == fold
            model = Expert(candidate["views"], candidate["C"], seed).fit(
                {k: v[train] for k, v in features.items()}, y[train])
            predictions[valid] = model.predict({k: v[valid] for k, v in features.items()})
        scores.append(logloss(y, predictions))
    chosen = int(np.argmin(scores))  # order breaks ties; no outer inputs.
    spec = candidates[chosen]
    model = Expert(spec["views"], spec["C"], seed).fit(features, y)
    return model, {"chosen": chosen, "candidate_nll": [
        s if np.isfinite(s) else None for s in scores],
        "fallback": None if np.isfinite(scores[chosen]) else "no_observed_fit_targets",
        "spec": spec}


def calibrate_inner(y, probabilities):
    """Reuse the prespecified v3 calibration; never treat missing labels as zero."""
    from .metrics import fit_calibration
    z = logit(np.clip(probabilities, 1e-7, 1-1e-7))
    return fit_calibration(y, z)


def calibrated(probabilities, temperatures):
    return expit(logit(np.clip(probabilities, 1e-7, 1-1e-7))/np.asarray(temperatures))
