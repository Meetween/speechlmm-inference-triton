#!/usr/bin/env bash
set -euo pipefail

TRITON_URL="${TRITON_URL:-http://localhost:${TRITON_HTTP_PORT:-9000}}"

curl -sS "${TRITON_URL%/}/v2/repository/index" \
  -H "Content-Type: application/json" \
  -d '{}' \
  | python3 -m json.tool

