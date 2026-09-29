"""Post-training diagnostics from paired outer predictions, never model selection."""
import csv
import json
from pathlib import Path

import numpy as np
from . import TARGETS
from .metrics import summarize


def analyze_single_run(output, ablation, seeds=(17, 29, 43)):
    """Post-training report for a deliberately single-ablation experiment."""
    output = Path(output)
    rows, aggregates = [], {}
    reference = None
    for seed in seeds:
        with np.load(output / ablation / f"predictions_seed_{seed}.npz",
                     allow_pickle=False) as z:
            run = {k: z[k] for k in z.files}
        if reference is None:
            reference = run
        elif (not np.array_equal(run["groups"], reference["groups"]) or
              not np.array_equal(run["y"], reference["y"], equal_nan=True)):
            raise ValueError("Seed alignment mismatch")
        report = summarize(run["y"], run["p"], run["thresholds"])
        rows.append({
            "ablation": ablation, "seed": seed,
            "competition_violation_macro_f1": report["competition"]["violation_macro_f1"],
            "quality_pooled_f1": report["competition"]["quality_pooled"]["f1"],
            "macro_f1": report["macro_f1"],
            "quality_macro_f1": report["quality_macro_f1"],
            "violation_macro_f1": report["violation_macro_f1"],
        })
    with (output / "ablation_table.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for key in rows[0]:
        if key in {"ablation", "seed"}:
            continue
        values = [r[key] for r in rows if r[key] is not None]
        aggregates[key] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "values_in_seed_order": [r[key] for r in rows],
        }
    (output / "seed_mean_std.json").write_text(
        json.dumps({ablation: aggregates}, indent=2, allow_nan=False),
        encoding="utf8")
    (output / "analysis_status.json").write_text(json.dumps({
        "status": "single_ablation_analysis_complete",
        "ablation": ablation, "seeds": list(seeds),
        "scope": "aggregation_only_does_not_audit_upstream_selection",
    }, indent=2), encoding="utf8")


def analyze_runs(output, seeds=(17, 29, 43)):
    output = Path(output)
    runs, rows, cases, contribution = {}, [], {}, {}
    for a in range(7):
        ablation = f"A{a}"
        runs[ablation] = {}
        for seed in seeds:
            with np.load(output / ablation / f"predictions_seed_{seed}.npz", allow_pickle=False) as z:
                run = {k: z[k] for k in z.files}
            runs[ablation][seed] = run
            reference = runs["A0"][seeds[0]]
            if (not np.array_equal(run["groups"], reference["groups"]) or
                    not np.array_equal(run["y"], reference["y"], equal_nan=True)):
                raise ValueError("Ablation alignment mismatch")
            report = summarize(run["y"], run["p"], run["thresholds"])
            rows.append({"ablation": ablation, "seed": seed,
                         "competition_violation_macro_f1": report["competition"]["violation_macro_f1"],
                         "quality_pooled_f1": report["competition"]["quality_pooled"]["f1"],
                         **{k: report[k] for k in ("macro_f1", "quality_macro_f1", "violation_macro_f1")}})
    for base, augmented in (("A0", "A4"), ("A1", "A5"), ("A3", "A6")):
        comparison = f"{augmented}_minus_{base}"
        contribution[comparison] = {}
        for seed in seeds:
            a, b = runs[base][seed], runs[augmented][seed]
            ma, mb = [summarize(r["y"], r["p"], r["thresholds"]) for r in (a, b)]
            contribution[comparison][str(seed)] = {
                t: {metric: (None if ma["targets"][t][metric] is None or mb["targets"][t][metric] is None
                              else mb["targets"][t][metric]-ma["targets"][t][metric])
                    for metric in ("f1", "sensitivity", "specificity", "roc_auc", "average_precision")}
                for t in TARGETS}
            contribution[comparison][str(seed)]["official_classes"] = {
                t: {metric: (None if ma["competition"]["classes"][t][metric] is None or
                             mb["competition"]["classes"][t][metric] is None else
                             mb["competition"]["classes"][t][metric] - ma["competition"]["classes"][t][metric])
                    for metric in ("f1", "roc_auc", "average_precision", "pr_auc_trapezoidal")}
                for t in ma["competition"]["classes"]}
            fixed = [summarize(r["y"], r["p"])["competition"] for r in (a, b)]
            contribution[comparison][str(seed)]["fixed_05_official_f1_delta"] = {
                t: None if fixed[0]["classes"][t]["f1"] is None or fixed[1]["classes"][t]["f1"] is None else
                fixed[1]["classes"][t]["f1"] - fixed[0]["classes"][t]["f1"]
                for t in fixed[0]["classes"]}
            pa, pb = a["p"] >= a["thresholds"], b["p"] >= b["thresholds"]
            ca, cb = pa == a["y"], pb == b["y"]
            valid = np.isfinite(a["y"])
            cases[f"{comparison}_seed_{seed}"] = {
                kind: [{"source_study_id": str(a["groups"][i]), "target": TARGETS[j]}
                       for i, j in np.argwhere(mask & valid)]
                for kind, mask in (("geometry_helps", ~ca & cb), ("geometry_hurts", ca & ~cb))}
    unstable, reliability = {}, {}
    full = [runs["A6"][s] for s in seeds]
    deviations = np.std(np.stack([r["p"] for r in full]), axis=0)
    for i, t in enumerate(TARGETS):
        valid = np.isfinite(full[0]["y"][:, i])
        unstable[t] = float(deviations[valid, i].mean()) if valid.any() else None
    for seed in seeds:
        r = runs["A6"][seed]
        pred = r["p"] >= r["thresholds"]
        valid = np.isfinite(r["y"])
        cases[f"full_errors_seed_{seed}"] = {
            kind: [{"source_study_id": str(r["groups"][i]), "target": TARGETS[j],
                    "probability": float(r["p"][i, j]), "threshold": float(r["thresholds"][i, j])}
                   for i, j in np.argwhere(mask & valid)]
            for kind, mask in (("false_positive", pred & (r["y"] == 0)),
                               ("false_negative", ~pred & (r["y"] == 1)))}
        d, m = runs["A0"][seed], runs["A1"][seed]
        cases[f"dino_mi2_disagreement_seed_{seed}"] = [
            {"source_study_id": str(d["groups"][i]), "target": TARGETS[j],
             "dino_p": float(d["p"][i, j]), "mi2_p": float(m["p"][i, j])}
            for i, j in np.argwhere(((d["p"] >= d["thresholds"]) != (m["p"] >= m["thresholds"])) & valid)]
        reliability[str(seed)] = {}
        for j, t in enumerate(TARGETS):
            bins = []
            for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
                mask = valid[:, j] & (r["p"][:, j] >= lo) & (
                    (r["p"][:, j] < hi) if hi < 1 else (r["p"][:, j] <= hi))
                bins.append({"lower": float(lo), "upper": float(hi), "n": int(mask.sum()),
                             "mean_p": float(r["p"][mask, j].mean()) if mask.any() else None,
                             "positive_rate": float(r["y"][mask, j].mean()) if mask.any() else None})
            reliability[str(seed)][t] = bins
    with (output / "ablation_table.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates = {}
    for ablation, seed_runs in runs.items():
        entries = {}
        for seed, r in seed_runs.items():
            s = summarize(r["y"], r["p"], r["thresholds"])
            values = {"competition_violation_macro_f1": s["competition"]["violation_macro_f1"],
                      "quality_macro_f1": s["quality_macro_f1"]}
            for scope, targets in (("official", s["competition"]["classes"]),
                                   ("diagnostic", s["targets"]),
                                   ("quality", {"pooled": s["competition"]["quality_pooled"]})):
                for t, metrics in targets.items():
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)) or v is None:
                            values[f"{scope}/{t}/{k}"] = v
            for k, v in values.items():
                entries.setdefault(k, []).append(v)
        aggregates[ablation] = {
            k: {"mean": float(np.mean([v for v in vals if v is not None])) if any(v is not None for v in vals) else None,
                "std": float(np.std([v for v in vals if v is not None], ddof=1)) if sum(v is not None for v in vals) > 1 else None,
                "defined_seeds": sum(v is not None for v in vals), "values_in_seed_order": vals}
            for k, vals in entries.items()}
    for filename, value in (("geometry_contribution.json", contribution), ("cases_analysis.json", cases),
                            ("class_instability.json", unstable), ("reliability.json", reliability),
                            ("seed_mean_std.json", aggregates)):
        (output / filename).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")
