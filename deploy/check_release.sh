#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test -f Dockerfile
test -f requirements-inference.txt
test -f src/dxa_qc/web.py
test -n "${DXA_ASSETS_DIR:-}"
test -f "$DXA_ASSETS_DIR/models/dinov3-vitl16/model.safetensors"
test -f "$DXA_ASSETS_DIR/artifacts/E007_FULL_V1/A3/seed_17/fold_0/checkpoints/best.pt"
test -f "$DXA_ASSETS_DIR/artifacts/E023_ANATOMY_ROUTER_PILOT_V1_FINAL/router_production.json"
docker compose -f deploy/docker-compose.yml config --quiet
docker compose -f deploy/docker-compose.yml build
docker compose -f deploy/docker-compose.yml up -d
curl --fail --retry 12 --retry-delay 5 http://127.0.0.1:8000/api/health
docker image inspect dxa-qc:stopcode-20260929 --format '{{.Id}}'
