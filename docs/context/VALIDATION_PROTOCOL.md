# Validation protocol v3 — approved E007 reporting (D007)

v1 rules below are retained; D006 adds operational nested selection for E007.
No fold assignments changed. User authorized the full E007 series on 2026-09-22.

## Groups and exclusions

- Primary group: source study folder ID.
- Decoded-pixel duplicate descendants remain in the same source-study group.
- `Dataset_test/` is loader-only and never used for training, tuning or model
  selection.
- All 100 learn source studies have workbook rows; missing target values remain
  nullable with a loss mask.
- The anonymized literal PatientID cannot provide patient-level grouping.

## Folds and seeds

- Frozen assignment requested for E007: `artifacts/E000_data_audit/frozen_folds_v1.csv`.
- Five fixed grouped folds, 20 source studies per fold.
- Planned fixed seeds: 17, 29 and 43.
- Every ablation uses identical folds, labels, exclusions, preprocessing and
  seeds. No architecture-specific split.

## Thresholds and tuning

- Outer-fold predictions are evaluation-only.
- Thresholds, calibration, early stopping, class weights and hyperparameters are
  selected inside the outer-training partition using an inner grouped split.
- Compare clinical thresholds to calibrated learned thresholds without choosing
  on an outer validation fold.

## Metrics

Per target: F1, sensitivity, specificity, balanced accuracy, ROC-AUC, Average
Precision, and PR-AUC when defined. AP and PR-AUC are stored separately.
Quality heads additionally report F1, ROC-AUC, AP and sensitivity/specificity.
Geometry reports landmark/normalized/mm error, angle MAE/correlation and ROI
error only when reference geometry exists.

Use study-level bootstrap 95% confidence intervals. Report undefined metrics,
sample counts and masks rather than silently dropping cases.

## E007 operational additions (D006)

- Unit of supervision/evaluation is the source-study bag until image-level
  assignments are verified. Deduplicate pixels within each bag; no size/name router.
- For outer fold f, use existing fold (f+1)%5 as inner holdout and the other
  three folds to fit. Do not refit on inner holdout after checkpoint selection.
- Checkpoint criterion: mean of ten inner target F1s at fixed .5; undefined
  contributes zero to this criterion only. Quality mean F1 tie-break.
- Early stopping min_delta .001, patience 25, max 200; all predetermined.
- Inner temperature/F1 threshold fitting requires >=5 positives AND negatives;
  otherwise temperature 1 / threshold .5 with insufficient-support status.
- A6 full model is prespecified. Outer metrics cannot pick a seed/ablation.
- 2,000 study-bootstrap replicates per seed; do not pool seed predictions as
  independent studies. Report per-class/macro intervals and undefined counts.
- A0–A6 use identical data/preprocessing/splits/seeds. A2 includes frozen dense
  DINO features; A6−A3 isolates added Geometry beyond the two global branches.

## v3 reporting extension (D007)

- Keep all v2 selection/training rules unchanged.
- Five-class competition-aligned SOURCE-STUDY proxy: three spine targets,
  positioning/rotation pooled over observed right/left hip labels, and ROI pooled
  over observed right/left hip labels. Do not OR sides or invent image labels.
- Keep side-specific diagnostics and independent regional/pooled quality.
- Five-class macro is null if any class F1 is undefined; report defined count.
- Study bootstrap samples whole rows before pooling hips; include Brier, ECE,
  log-loss and five-class/quality intervals as well as discrimination metrics.
- Not an official image-level competition score: verified image routing is absent.

## v4 E020 controlled transfer pilot (D020)
- v3 historical results unchanged. E020 protocol: docs/E020_PREREGISTRATION.md.
- Outer 0 training partition only for initial pilot, all seeds, fixed residual probe.
- No outer predictions used for stop/go or model/threshold selection.
- Full protocol conditional on prespecified combined transfer go criterion.
- Image-level mapping is provisional anatomy only; no expert QC labels fabricated.
