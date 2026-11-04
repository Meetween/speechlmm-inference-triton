#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-}"
if [[ -z "${MODEL_NAME}" ]]; then
  echo "Usage: $0 <model_name>"
  exit 2
fi

TRITON_URL="${TRITON_URL:-http://localhost:${TRITON_HTTP_PORT:-9000}}"

curl -sS -X POST "${TRITON_URL%/}/v2/repository/models/${MODEL_NAME}/unload" \
  -H "Content-Type: application/json" \
  -d '{}' \
  | sed -e 's/^/RESPONSE: /'

echo "OK: requested unload for ${MODEL_NAME}"

