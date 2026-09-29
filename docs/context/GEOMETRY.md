# Geometry design

## Implemented E007 update (supersedes the selected proposal below)

See `docs/E007_FULL_V1_DESIGN.md` and D005. Implemented dense DINO + small raw
image pyramid, eight unnamed latent maps, spatial moments, axial/coverage
profiles, 128-D embedding. Official source-study QC provides weak supervision
through class-specific MIL. Maps are NOT expert landmarks, masks or clinical
measurements. Geometry math/provenance is tested; physical metrics are not
validated or fed as anatomical measurements. GPU backward smoke passed; no
optimizer steps or training. The following is retained as historical alternatives.

## Physical coordinate contract

Use DICOM PixelSpacing when present. In this supplied set it is absent, so the
organizer's external scanner calibration is 0.60 mm/column and 1.05 mm/row.
Every resize, crop and pad must retain an explicit pixel-to-physical transform.
Unknown calibration must mask or downgrade physical measurements.

## Alternatives considered

### A: fixed three-point heuristic

Cheap and interpretable, but fragile to missing anatomy, cropping, artifacts and
hip side. Reject as the final geometry representation.

### B: centerline plus spline/landmark regression

Predict a spine centerline, vertebral centers and hip landmarks; derive robust
line angle, curvature, coverage and anatomical distances. Strong physical
interpretability, but needs stable targets and can fail when landmarks are
occluded.

### C: multi-resolution segmentation plus heatmap landmarks

Predict anatomy/ROI masks and heatmaps for spine/vertebral/hip landmarks, then
derive differentiable physical features. It provides dense evidence, supports
uncertainty and is more tolerant of partial views. It is selected as the
implementation direction, with B-style measurements on top.

## Selected design

1. Low-resolution global view for anatomy and coverage.
2. Region-specific high-resolution crops.
3. Heatmaps for multiple vertebral centers, spine endpoints, iliac crest/Th12,
   greater trochanter, femoral neck, shaft, ischium and lesser trochanter.
4. Optional soft masks for spine, hips and ROI boundaries.
5. Robust line/spline fit in physical coordinates; report angle, curvature,
   lateral displacement and confidence.
6. Explicit missing/uncertain landmark masks; no fabricated geometry.
7. Geometry embedding plus measurements supplied only to relevant class heads.

## Validation needed

The official workbook has class labels but not landmark coordinates. Before
training a geometry branch, create a small separate expert geometric annotation
protocol or use weak targets only as hypotheses. Report landmark error, mm error,
angle MAE and ROI error only when reference geometry exists.
