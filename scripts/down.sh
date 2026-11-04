#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
unset HF_TOKEN || true
unset HUGGING_FACE_HUB_TOKEN || true
docker-compose down

