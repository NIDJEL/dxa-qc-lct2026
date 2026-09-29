#!/usr/bin/env bash
set -euo pipefail
# INPUT_DIR may contain several studies. Ambiguous boundaries require STUDY_GROUPS.
: "${INPUT_DIR:?set INPUT_DIR to a study directory or batch archive}"
: "${ASSET_ROOT:?set ASSET_ROOT to repository containing models and E007 checkpoints}"
IMAGE=${IMAGE:-dxa-qc:submission}
DEVICE=${DEVICE:-cuda}
OUTPUT_DIR=${OUTPUT_DIR:-"$PWD/submission-output"}
mkdir -p "$OUTPUT_DIR"
args=()
if [[ "$DEVICE" == cuda ]]; then args+=(--gpus all); fi
group_mount=()
group_arg=()
if [[ -n "${STUDY_GROUPS:-}" ]]; then
  group_mount=(--mount "type=bind,source=$(realpath "$STUDY_GROUPS"),target=/input-study-groups.json,readonly")
  group_arg=(--study-groups /input-study-groups.json)
fi
docker run --rm --network none "${args[@]}" \
  --mount "type=bind,source=$(realpath "$INPUT_DIR"),target=/input,readonly" \
  "${group_mount[@]}" \
  --mount "type=bind,source=$(realpath "$ASSET_ROOT/models"),target=/app/models,readonly" \
  --mount "type=bind,source=$(realpath "$ASSET_ROOT/artifacts/E007_FULL_V1/A3"),target=/app/artifacts/E007_FULL_V1/A3,readonly" \
  --mount "type=bind,source=$(realpath "$ASSET_ROOT/artifacts/E023_ANATOMY_ROUTER_PILOT_V1_FINAL/router_production.json"),target=/app/artifacts/E023_ANATOMY_ROUTER_PILOT_V1_FINAL/router_production.json,readonly" \
  --mount "type=bind,source=$(realpath "$OUTPUT_DIR"),target=/output" \
  --entrypoint python "$IMAGE" scripts/run_dxa_qc_service.py /input "${group_arg[@]}" --device "$DEVICE" --json /output/result.json --csv /output/result.csv
