# Official task specification

This file records facts from `about project.pdf` only. Source page numbers refer
to the supplied 12-page PDF (the first page is visually blank in text extraction).

## Scope

- Purpose: automated quality assessment of densitometric studies and anatomical
  markup, including the lumbar spine and proximal femur (pp. 2-3).
- Input: DICOM densitometry study, potentially one to three images/series (p. 7).
- Output: CSV or XLSX, one row per image/anatomical region (p. 8).

## Required violations

- Spine positioning: iliac-crest upper edges visible at the lower boundary and
  half of Th12 at the upper boundary (p. 3).
- Spine axis: acceptable tilt up to 5 degrees (p. 3).
- Spine foreign objects/marked artifacts/overlays (p. 4).
- Femur positioning: greater trochanter, femoral neck and ischium visible (p. 5).
- Femur rotation: evaluate the lesser trochanter; both over- and under-rotation
  are unacceptable (p. 6).
- Femur ROI: at least 3 cm above and below the ROI and 2 cm from the relevant
  side edge (p. 7).

## Output and runtime

Required fields include `path_to_study`, `study_uid`, `image_uid`,
`anatomical_region`, `quality_class`, `violation_type`, `processing_status`, and
`time_of_processing` (p. 8). Processing target is no more than three minutes per
study, with error reporting, reproducibility, batch archive support, and no
unhandled exceptions (p. 8).

## Delivery constraints

The final submission requires a trained model and weights, a containerized local
service, Linux/Unix build and run scripts, batch API, documentation, and support
for all required regions/classes (pp. 9-10). No external closed service may be
needed and medical images must not leave the local service (p. 9).

## Evaluation

Recommended metrics include sensitivity, specificity, balanced accuracy, F1,
ROC-AUC and/or PR-AUC, macro-F1, localization/segmentation metrics where
applicable, processing time, and successful-file rate. F1 and ROC-AUC are
priority clinical characteristics; 95% confidence intervals are recommended
(p. 12).
