#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
# Ensure shell env doesn't override .env tokens.
unset HF_TOKEN || true
unset HUGGING_FACE_HUB_TOKEN || true
docker-compose up -d

