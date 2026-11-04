#!/bin/bash
# Build the self-contained SpeechLMM v2 serving image and convert it to a .sif
# via the local Docker registry. Same flow as speechlmm-v2 scripts/build/.
#
# The .sif carries everything needed to serve — tritonserver, torch,
# transformers, torchcodec, the OpenAI gateway and the smoke clients — so
# serving needs no host venv prefix, exactly like the training SIF.
#
# Run on the Docker+Singularity build host (local disk). Do not run on a
# parallel filesystem such as Lustre. Expect ~18 GB for the image and the same again for the .sif.
#
# Required:
#   TRITON_SIF_PATH       local-disk path for the .sif
# Optional:
#   TRITON_SOURCE_IMAGE   default nvcr.io/nvidia/tritonserver:25.06-py3
#   TRITON_IMAGE_NAME     default meetween/lara-mtw-triton_25-06
#   TRITON_IMAGE_VERSION  default 0.1.1
#   TORCH_INDEX           default https://download.pytorch.org/whl/cu128
#   SINGULARITY_BIN       default /usr/bin/singularity

set -euo pipefail

if [ -z "${TRITON_SIF_PATH+x}" ] || [ -z "${TRITON_SIF_PATH}" ]; then
  echo "TRITON_SIF_PATH is required (local-disk path for the .sif)" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRITON_SOURCE_IMAGE="${TRITON_SOURCE_IMAGE:-nvcr.io/nvidia/tritonserver:25.06-py3}"
TRITON_IMAGE_NAME="${TRITON_IMAGE_NAME:-meetween/lara-mtw-triton_25-06}"
TRITON_IMAGE_VERSION="${TRITON_IMAGE_VERSION:-0.1.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
SINGULARITY_BIN="${SINGULARITY_BIN:-/usr/bin/singularity}"
IMAGE_REF="${TRITON_IMAGE_NAME}:${TRITON_IMAGE_VERSION}"
LOCAL_REF="localhost:5000/${TRITON_IMAGE_NAME}:${TRITON_IMAGE_VERSION}"

mkdir -p "$(dirname "${TRITON_SIF_PATH}")"

docker build "${REPO_ROOT}" \
  -f "${REPO_ROOT}/docker/serve.Dockerfile" \
  -t "${IMAGE_REF}" \
  --build-arg TRITON_SOURCE_IMAGE="${TRITON_SOURCE_IMAGE}" \
  --build-arg TORCH_INDEX="${TORCH_INDEX}"

docker tag "${IMAGE_REF}" "${LOCAL_REF}"
docker push "${LOCAL_REF}"

"${SINGULARITY_BIN}" --version
"${SINGULARITY_BIN}" pull --no-https --name "${TRITON_SIF_PATH}" "docker://${LOCAL_REF}"

"${SINGULARITY_BIN}" exec "${TRITON_SIF_PATH}" python3.12 --version
"${SINGULARITY_BIN}" exec "${TRITON_SIF_PATH}" sh -c "command -v tritonserver"
"${SINGULARITY_BIN}" exec "${TRITON_SIF_PATH}" python3.12 -c "\
import torch, transformers, torchcodec, openai, flash_attn; \
print('torch', torch.__version__, '| transformers', transformers.__version__); \
print('torchcodec', torchcodec.__version__, '| openai', openai.__version__); \
print('flash_attn', flash_attn.__version__)"
echo "SIF ready: ${TRITON_SIF_PATH}"
