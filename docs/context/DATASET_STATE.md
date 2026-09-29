# Dataset state

## E000 verified facts

- Learn: 499 DICOM files, 100 source studies, 100 workbook rows.
- Format-test: 3 DICOM files in 2 DICOM studies.
- All DICOM files read successfully; modality is CR.
- Manufacturer/model: GE Healthcare / Lunar Prodigy Advance.
- Single-channel MONOCHROME2, unsigned 8-bit, no DICOM PixelSpacing.
- Widths are mainly 300 (spine hint) and 280 (hip hint); two images are 248-wide
  and remain unknown. Side is not a reliable DICOM field.
- 252 unique decoded pixel arrays among 499 learn files; 127 duplicate groups,
  all within source studies; raw-file duplicate groups: 0.
- Workbook labels are study-level. Hip positioning and rotation are combined.
- Spine aggregate quality disagrees with component OR in three studies.
- All 100 learn studies have workbook rows, NOT complete target coverage.
  E000 manifest/distribution retain missing spine targets for 1 study, right-hip
  targets for 28, left-hip targets for 22. These remain nullable/masked.
  This corrects the earlier summary; no original labels or E000 files changed.
  Test labels are intentionally not available for training.
- External scanner calibration from DOCX: 1.05 mm/row and 0.60 mm/column.
- Candidate grouped folds: `artifacts/E000_data_audit/frozen_folds_v1.csv`;
  5 folds, 20 source studies each.

Detailed manifest, metadata, distributions, duplicates and conflicts:
`artifacts/E000_data_audit/`.
