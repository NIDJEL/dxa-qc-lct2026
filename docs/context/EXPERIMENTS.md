# Experiment registry

## 2026-09-22 E007 preflight and run

- `E007_features_v2`: 252 unique learn images, zero extraction failures.
- `E007_training_smoke_v1`: retained failed AMP overflow run; no outer use.
- `E007_training_smoke_v2`: passed 3 real A6 epochs/fold0/seed17 and 23 steps.
- `E007_FULL_V1`: running A0-A6, five folds, seeds 17/29/43.
- Preflight regression suite: 30 tests passed; final metrics pending.

## E000 data audit

- Status: completed.
- Purpose: audit data, labels, DICOM metadata, duplicates and grouped folds.
- Artifacts: `artifacts/E000_data_audit/`.
- Result: 499 learn DICOMs, 100 learn studies, 252 unique pixel arrays,
  127 within-study pixel-duplicate groups, 3 label conflicts, no DICOM read
  errors, 5 candidate folds.
- Decision: preserve official quality labels independently; no training started.

## E000 model verification

- Status: completed for DINOv3 and MedImageInsight image-only structural/inference
  smoke tests.
- Artifacts: `artifacts/model_inventory.json`.
- Result: both encoders loaded locally, emitted finite deterministic 1024-D
  embeddings, and used no external image service.

## E007_FULL_V1 implementation and E007 smoke v1/v2/v3

- Status: implemented, unit/forward/backward/cache checked; NOT trained.
- Architecture: `docs/E007_FULL_V1_DESIGN.md`; D005 and validation-v2 D006.
- Reports: `artifacts/E007_smoke_v1/report.json`,
  `artifacts/E007_smoke_v2/report.json`, `artifacts/E007_smoke_v3/report.json`. New IDs preserve earlier evidence.
- 3 learn images, one study, no optimizer step or learned checkpoint.
- 711,294 trainable / 665,863,168 frozen values; MI2 extraction ~1.58 GiB
  allocated peak; final deterministic head backward ~103 MiB. Full feature extraction not run.
- No clinical metric/geometry gain is available. User approval required.

## Historical plan (E007 now groups A0–A6 in one infrastructure)

- E001 DINOv3 frozen baseline.
- E002 MedImageInsight frozen baseline.
- E003 geometry baseline.
- E004-E007 fixed ablation matrix.
- E008 calibration/ensemble.
- E009 Teacher confidence validation.
- E010 Teacher release.
- E011 pseudo-label generation.
- E012 Student training.

Every completed experiment must have immutable config, metrics, predictions,
per-fold results, report, input/checkpoint hashes and Git commit.

## E008_B0 and targeted external audit

- `E008_B0`: completed on the unchanged frozen organizer grouped protocol,
  seeds 17/29/43, with inner-only selection/calibration and 2,000 grouped
  bootstrap replicates per seed. It is a cached-feature source-study proxy,
  not official image-level scoring.
- Results: violation macro-F1 0.0484 ± 0.0000, pooled quality F1 0.3230
  ± 0.0000, quality ROC-AUC 0.5189 ± 0.0000 across fixed seeds. Full
  per-class F1/sensitivity/specificity/AP and CIs are in
  `artifacts/E008_B0/FINAL_REPORT.md` and seed JSON artifacts.
- `scripts/e008_external_audit.py` audited only Spinal-AI2024 and MTDDH
  Dataset1. Project participant permission attestation is recorded; missing
  embedded LICENSE files are not treated as exclusion. Both have zero decoded
  pixel overlap with official data; MTDDH contains 44 aliases.
- `E008_aux_hip_v1`: completed real frozen-DINO SpatialAdapter pretraining
  on MTDDH's eight explicit hip keypoints (780 train / 208 validation),
  AMP, gradient accumulation 8, 10 epochs. Organizer ablation used the same
  nested frozen protocol and all fixed seeds/folds. Quality F1 improved
  0.3230→0.3905; quality ROC-AUC 0.5189→0.5164; violation macro-F1 stayed
  0.0484. Full per-class metrics, learning curve and provenance are in
  `artifacts/E008_aux_hip_v1/`.

## E009 threshold diagnosis v1 (2026-09-22)

- Status: analysis-only; no model, threshold, fold or label was changed.
- Reused frozen E007 A3 outer predictions for seeds 17/29/43.
- Operational violation macro-F1 mean: 0.2534.
- Per-class threshold sweep using outer labels: 0.3771 mean, an explicitly
  optimistic upper bound rather than valid model selection.
- The 0.1237 gap supports threshold suppression as a material bottleneck, but
  seed/class threshold variation means the sweep must not be deployed.
- Evidence: `artifacts/E009_threshold_diagnosis_v1/FINAL_REPORT.md`,
  `report.json`, and `scripts/e009_threshold_diagnosis.py`.
- Next valid hypothesis: pre-register an inner-only positive-aware rare-class
  policy, then rerun all fixed folds and seeds; do not pseudo-label yet.

## E010 MEANMAX V3 (2026-09-22)

- Full alternative MIL architecture completed: DINO/MI2 mean+max pooling,
  independent ten-target heads, all 5 grouped folds and seeds 17/29/43.
- Inner-only checkpoint selection, calibration and thresholds were retained.
- Five-class source-study proxy violation macro-F1: 0.2240 mean; quality pooled
  F1: 0.4387. This did not improve E007 A3.
- Replay verification reproduced inner calibration and all 15 outer probability
  arrays exactly; no fold overlap was found. Evidence:
  `artifacts/E010_operational_replay_v1/verification.json`.
- The run failed only in the old A0-A6 post-training report helper because this
  was a single-ablation run; `E010_REPORT.json` and `FINAL_REPORT.md` are the
  corrected report. No results were discarded or overwritten.
- Decision: reject mean+max as a primary replacement. Continue with validated
  external spatial pretraining/domain-adaptation hypotheses rather than
  threshold-only tuning or pseudo-labeling.

## E011 HIP transfer V2 (2026-09-23)

- Full 15-fit grouped run completed for A6 with compatible initialization from
  the corrected, duplicate-group-safe MTDDH spatial adapter.
- Inner-only checkpoint selection, calibration and thresholds were preserved;
  external keypoints were not used as official QC labels.
- Five-class source-study proxy violation macro-F1: 0.1908 mean, below E007
  A3 and E010. Pooled quality F1: 0.4897, above E010 but not sufficient to
  establish a primary QC improvement.
- Replay verification reproduced calibration and outer inference for all 15
  folds/seeds: `artifacts/E011_operational_replay_v2/verification.json`.
- The old A0-A6 analyzer again failed only because this is a single-ablation
  run; corrected metrics and grouped 2,000-bootstrap CIs are in
  `artifacts/E011_HIP_TRANSFER_V2/E011_REPORT.json`.
- Decision: reject this transfer as a violation-model improvement. Retain the
  checkpoint as an auxiliary representation artifact; do not pseudo-label.

## E012 linear positive weighting (2026-09-23)

- Full A3 grouped run completed for seeds 17/29/43 with inner-fit positive
  weights changed from sqrt(cap5) to linear(cap10); all other protocol elements
  unchanged.
- Five-class source-study proxy violation macro-F1: 0.2470 mean; quality pooled
  F1: 0.4648. This does not improve the E007 A3 competition proxy.
- The run again only hit the legacy A0-A6 analyzer after successful single-
  ablation training; corrected metrics and 2,000 grouped bootstrap CIs are in
  `artifacts/E012_LINEAR_POSWEIGHT_V1/E012_REPORT.json`.
- Decision: reject as a primary improvement; preserve as evidence that stronger
  class weighting alone does not solve the representation/support bottleneck.

## E013 pixel hip pretraining (2026-09-23)

- Trained a full small pixel encoder directly on 944 unique MTDDH images and
  eight explicit keypoints; no DINO features or organizer QC labels were used.
- Corrected duplicate-safe split: 736 train / 208 validation, decoded-pixel
  overlap with organizer data 0, failures 0.
- Across seeds 17/29/43 validation normalized coordinate error fell from about
  0.21--0.23 initially to 0.0125--0.0140, and NLL from about 6.6--6.8 to
  2.95--2.96. This validates the external spatial learning pipeline, not QC
  improvement.
- Evidence: `artifacts/E013_PIXEL_HIP_V1/`; next pre-registered step is
  transfer of compatible pixel image blocks into the QC Geometry branch.

## E014 pixel stem transfer (2026-09-23)

- Full A6 grouped run completed with the first three raw-image blocks
  initialized from E013 seed17 pixel-to-keypoint pretraining.
- Five-class source-study proxy violation macro-F1: 0.1755 mean; pooled quality
  F1: 0.4223. This is below E007 A3 and E011, so partial stem transfer is
  rejected as a QC improvement.
- Replay verification passed all 15 checkpoints in
  `artifacts/E014_operational_replay_v1/verification.json`.
- The active process loaded the old seven-ablation finalizer and failed only
  after successful training. Saved predictions were aggregated afterward with
  the current single-ablation analyzer; no retraining or outer selection was
  performed.
- Decision: reject partial transfer. If continuing, test an intact external
  pixel representation branch rather than another partial weight transplant.

## E015/E016 controlled intact-pixel comparison (2026-09-23)
- E015 currently running; external E013 seed17 six-block encoder is fine-tuned
  with cached frozen DINO/MI2 fusion, fixed folds/seeds and inner-only selection.
- E016 matched scratch architecture registered before E015 pooled inspection;
  only external pixel initialization removed. Both branches must be reported.
- tests/test_pixel_transfer_integrity.py proves exact latent equivalence after
  full loading and finite nonzero QC gradients through all encoder parameters.
- Loaded keypoint output head is unused, not learned landmarks on organizer DXA.
- Full scoped tests: 45 passed. No QC improvement claimed yet.

## E015 complete, E016 active (2026-09-23)
- E015 completed 15 fits and exited successfully; replay passed all checkpoints.
- Verified report: artifacts/E015_verified_report_v1/FINAL_REPORT.md.
- Five-class proxy F1 0.1321, quality F1 0.4728; macro AP 0.2394, macro AUC
  0.6485. No improvement over E007 A3 established. No seed was selected.
- Export validates hashes/coverage and reuses existing 2,000 study bootstrap
  intervals; does not silently treat float64-to-float32 storage as corruption.
- Preregistered E016 capacity-matched scratch control launched, session 28889.
- Attribution to external pretraining awaits full E015/E016 comparison.

## E016 completed and E017/E018 frozen pair (2026-09-23)
- E016 all 15 fits/replay verified: five-class F1 0.1708, quality 0.4533.
- Paired E015 minus E016 F1 -0.03875, 95% study CI [-0.07747,-0.00406];
  AP, AUC and quality deltas include zero. No external pretraining gain proven.
- Report: artifacts/E015_E016_paired_v1/FINAL_REPORT.md. 2000 paired study draws
  shared across models/seeds; undefined replicates reported, no seed pooling.
- Fit/inner loss gap is severe in both arms; zero AMP overflow in all 30 fits.
- E017 frozen pretrained / E018 frozen random pair registered before either
  run; unchanged architecture, grouped folds/seeds, inner selection/thresholds.
- Both queued sequentially in session 71309. Tests: 48 passed.

## E017 frozen pretrained complete (2026-09-23)
- All 15 fits complete, operational replay passed, exact frozen weights verified
  against external checkpoint for all 15 models.
- Proxy F1 0.2093; quality F1 0.4699; macro AP 0.2941, macro AUC 0.6502.
- Evidence: artifacts/E017_verified_report_v1/ and E017_frozen_audit_v1/.
- E018 frozen-random control now running in the original session 71309; no
  pretraining benefit or primary improvement claimed before paired comparison.

## E017/E018 frozen pair complete (2026-09-23)
- E017 pretrained frozen pixel: proxy F1 0.2093, quality F1 0.4699.
- E018 random frozen control: proxy F1 0.2345, quality F1 0.4749.
- Paired pretrained-minus-random F1 delta -0.0252, CI [-0.0543, 0.0045];
  quality F1 delta -0.0050, CI [-0.0321, 0.0211]. Macro AP/AUC deltas include zero.
- All 30 frozen checkpoints exactly preserved initialization; all replays passed.
- Decision: no reproducible external pixel-pretraining benefit. Stop repeating
  pixel transfer variants; representation/support bottleneck remains unresolved.

## E019 asymmetric negative focusing (2026-09-23)
- Registered A3 hypothesis: positive BCE unchanged, negative BCE multiplied by
  sigmoid(logit)^2; existing sqrt-cap5 weights/missing masks unchanged.
- Same nested folds/seeds, checkpoint criterion, inner calibration/thresholds.
- Not a published ASL reproduction: no probability clipping or detached weights.
- Full scoped tests: 50 passed. Running session 47715; no outer metrics yet.

## E019 status reconciliation for work report (2026-09-23)
- Completion artifact and all three seed aggregates exist (15 fits complete).
- Saved five-class proxy F1 0.2684301; pooled quality F1 0.4594612.
- Operational replay and paired baseline comparison pending; no validated gain.
- Earlier running notes above are historical. No new training launched here.
- Work report: docs/WORK_REPORT_2026-09-23.md; fresh tests: 50 passed in 49.44s.

## E019 verification-only closure — 2026-09-23
- No training; all seeds 17/29/43, all five folds compared with E007 A3.
- Fresh replay passed all 30 operational checkpoints; all 60 best/last audited.
- F1 0.2684301 vs 0.2469687; delta CI [-0.0155834,0.0685233].
- AP 0.2664548 vs 0.2835845; ROC-AUC 0.6725722 vs 0.6424619.
- Quality F1 0.4594612 vs 0.4743595. Calibration errors worsened.
- Evidence: artifacts/E019_A3_recheck_v1/comparison.json and all_metrics.json.
- Report: docs/E019_RECHECK_REPORT_2026-09-23.md. No robust improvement claim.
- All-seed state frozen locally; experiments stopped pending user permission.

## 2026-09-23 E020 external pretraining and transfer

- Verified rtifacts/E020_EXTERNAL_PRETRAINING/ by completion markers, cache/checkpoint hashes, optimizer counters, split isolation, nullable labels, and calibration replay.
- Spine, hip, VinDr: 35 epochs, seeds 17/29/43; 36,015 external optimizer steps; 3.155 conservative GPU-hours.
- Preregistered QC pilot completed using frozen E007 A3 and matched random controls. Combined pretrained descriptor failed: AP delta vs A3 -0.000253, ROC-AUC delta -0.002185; full grouped evaluation was correctly not run.
- Final factual report: docs/E020_FINAL_REPORT.md; no E021 or new external training started.


## 2026-09-24 E007 A3 technical replay (not a new ML experiment)
- Fresh 252-image encoder extraction and all 15 matching fold/seed checkpoints reproduce embeddings, attention/gates, logits and calibrated probabilities exactly (max/mean abs 0/0).
- Service DICOM/bag/CSV integration also passed; no fitting, checkpoint replacement or outer selection.
- Evidence: artifacts/E007_service_end_to_end_v1/verification.json and artifacts/E007_service_integration_v1/verification.json.
- Official contract and Linux readiness are tracked separately in SUBMISSION_READINESS_REPORT.md.
## E023 — frozen-feature anatomy router, diagnostic (2026-09-24)
- Authorized single pilot: `scripts/pilot_anatomy_router.py`; local evidence
  `artifacts/E023_ANATOMY_ROUTER_PILOT_V1_FINAL/results.json`.
- E020 nonexpert visual anatomy candidates, 251 scored high-visual, one
  moderate withheld from fit/metrics. Frozen DINO+MI2 global embeddings,
  regularized linear model, grouped frozen folds, inner-only C/margin.
- Seeds 17/29/43: each coverage 100%, balanced accuracy 1.0, spine→hip 0,
  hip→spine 0, abstain 0; C=.01/margin 0. Diagnostic GO, not independent:
  candidate visual mapping used DINO clustering and prior inspection.
- Final router integrated separately; E007 A3 remains frozen. Windows CUDA
  smoke 3/3 format-test and 29/29 real two-series DICOM successful.
  Current Docker image unbuilt; no Linux container claim.

## E024 — independent router and final container check (2026-09-24)
- Status: completed diagnostic verification; no training or tuning.
- Independent anatomy sources not used by E020: Spinal-AI2024 validation (1,000 spine) and MTDDH Dataset1 validation (208 hip); 1,208 exact-deduplicated source records.
- Coverage 1.000, abstention 0, spine sensitivity 1.000, hip sensitivity 0.798, balanced accuracy 0.899, 42 cross-region errors. Confirmed anatomy only; not official DXA QC.
- VinDr SpineXR test stress: 120/120 spine, radiographic domain shift.
- Docker image rebuilt and offline CUDA verified on Dataset_test (3/3) and the 29-DICOM/two-series study (29/29); CSV/JSON, provenance, time/RAM/VRAM checked.
- Evidence: `artifacts/E024_ROUTER_INDEPENDENT_CHECK_20260924/`; readiness report remains blocked.
## E025 - read-only inference discrepancy audit (2026-09-24)
- Three Dataset_test DICOMs: one spine bag, two-image hip bag. No training,
  threshold/model selection, label correction or checkpoint change.
- Direct CUDA inference checked DICOM decode, preprocessing, router, logits,
  temperatures and ten-head indexing. The two hip rows share the same study
  vector by MIL design; no broadcast/serialization error was demonstrated.
- Saved grouped outer validation replay matched all 15 seed/fold predictions
  exactly. Diagnostic singleton-bag hip-positioning decisions differed from
  study-bag decisions for 84/230, 61/230, 61/230 cached images of mixed regions (seeds 17/29/43).
- These are differences in predictions, NOT image-level accuracy estimates.
  Details: `docs/E025_DIAGNOSTIC_REPORT_2026-09-24.md`; sensitive evidence
  remains local in `artifacts/E025_DIAGNOSTIC_AUDIT_20260924/`.
