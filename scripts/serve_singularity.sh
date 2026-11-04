#!/usr/bin/env bash
# Launch tritonserver + the OpenAI-compatible gateway under Singularity.
# Cluster-agnostic: everything comes from the environment. Wrap it in your
# scheduler's job script for scheduler headers and site paths (see README).
#
# Required:
#   SIF             Triton serving image (.sif or sandbox), self-contained:
#                   scripts/build/build_sif.sh bakes torch, transformers,
#                   torchcodec and the gateway deps in, so there is no venv
#   LOAD_MODEL      space-separated Triton model names to load
#                   (speechlmm_v2_xl, speechlmm_v2_l, qwen3_omni, qwen25_omni)
#   <MODEL_NAME>_MODEL_PATH  one per name in LOAD_MODEL, upper-cased: the checkpoint
#                   dir as seen INSIDE the container, or a Hub id. This is the
#                   convention every backend already reads, so a new model under
#                   models/ needs no change here.
#                   e.g. SPEECHLMM_V2_XL_MODEL_PATH, QWEN3_OMNI_MODEL_PATH
# Optional:
#   <MODEL_NAME>_ATTN_IMPL   per-model attn_implementation for SpeechLMM v2: sdpa,
#                   flash_attention_2 or eager. Unset keeps the transformers default.
#   <MODEL_NAME>_DEVICE_MAP  per-model device_map; falls back to DEVICE_MAP, and
#                   to the config.pbtxt parameter when neither is set
#   BINDS           space-separated extra host:container binds that expose the
#                   checkpoints and any file:// media the gateway must read
#   REPO_DIR        this repo on the host (default: parent of scripts/)
#   CONTAINER_REPO  mount point of REPO_DIR (default /workspace/speechlmm-inference-triton)
#   HF_HOME         container path of the HF cache (offline)
#   TRITON_HTTP_PORT 8100  TRITON_GRPC_PORT 8101  TRITON_METRICS_PORT 8102
#   TRITON_OPENAI_PORT 8000
#   DEVICE_MAP      device_map for every loaded model
#   TRITON_HTTP_MAX_INPUT_SIZE 268435456 bytes; raise for larger media requests
#   SPEECHLMM_VIDEO_FPS        frame sampling rate for the general vision tower,
#                   default 2 (matches vision_config.tokens_per_second). 0 keeps
#                   every decoded frame, which blows past the context on long
#                   clips. Lipread video is unaffected.
#   SPEECHLMM_VIDEO_NUM_FRAMES fixed frame budget per clip; overrides FPS when set
#   MAX_NEW_TOKENS TEMPERATURE DO_SAMPLE TRITON_TIMEOUT TMPDIR
#   SPEECHLMM_DISABLE_TALKER / SPEECHLMM_DISABLE_CODE2WAV
#                   default 1 (speech off). Set both to 0 to enable speech output.
#   <MODEL>_DISABLE_TALKER / <MODEL>_DISABLE_CODE2WAV
#                   per-model override of the globals above
#   SPEECHLMM_SPEAKER          default ethan
#   TALKER_MAX_NEW_TOKENS      default 4096
#   TALKER_DO_SAMPLE           default 1 (greedy Talker can loop to the cap)
#   TELEMETRY                  default 0. Set to 1 for per-request efficiency metrics.
#   <MODEL>_TELEMETRY          per-model override of TELEMETRY
#   SPEECHLMM_DEBUG_MEM        default 0. Set to 1 to log per-request input shapes
#                   and per-GPU allocator usage around generate, plus a full
#                   memory_summary on CUDA OOM.
#   <MODEL>_SITE_PACKAGES      container path of a pip --target sidecar for that model
#   <MODEL>_SPEAKER            default voice of a TTS/S2ST-only model
#   <MODEL>_INSTRUCT           voice-design instruction of a TTS model that has no preset voices
#   <MODEL>_SEED               per-request torch seed of a TTS/S2ST-only model (default 0)
#   CONTEXT_OVERFLOW           default warn. A request whose prompt plus max_new_tokens
#                              exceeds the model context is logged and still served;
#                              reject answers it with a 400 instead.
#   <MODEL>_CONTEXT_OVERFLOW   per-model override of CONTEXT_OVERFLOW

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONTAINER_REPO="${CONTAINER_REPO:-/workspace/speechlmm-inference-triton}"
TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-8100}"
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-8101}"
TRITON_METRICS_PORT="${TRITON_METRICS_PORT:-8102}"
TRITON_OPENAI_PORT="${TRITON_OPENAI_PORT:-8000}"
FFMPEG_LIB_DIR="${FFMPEG_LIB_DIR:-/opt/ffmpeg/lib}"

: "${SIF:?Set SIF to the Triton serving image}"
: "${LOAD_MODEL:?Set LOAD_MODEL to the Triton model names to load}"
if [[ ! -e "${SIF}" ]]; then
  echo "Missing Triton SIF: ${SIF}" >&2
  exit 1
fi
BIND_ARGS=(-B "${REPO_DIR}:${CONTAINER_REPO}")
for _b in ${BINDS:-}; do
  BIND_ARGS+=(-B "${_b}")
done

SING_COMMON=(
  singularity exec --nv
  --pwd "${CONTAINER_REPO}"
  "${BIND_ARGS[@]}"
  --env LD_LIBRARY_PATH="${FFMPEG_LIB_DIR}"
  --env SPEECHLMM_DEVICE_MAP="${DEVICE_MAP:-auto}"
  --env SPEECHLMM_DISABLE_TALKER="${SPEECHLMM_DISABLE_TALKER:-1}"
  --env SPEECHLMM_DISABLE_CODE2WAV="${SPEECHLMM_DISABLE_CODE2WAV:-1}"
  --env SPEECHLMM_SPEAKER="${SPEECHLMM_SPEAKER:-ethan}"
  --env TALKER_MAX_NEW_TOKENS="${TALKER_MAX_NEW_TOKENS:-4096}"
  --env TALKER_DO_SAMPLE="${TALKER_DO_SAMPLE:-1}"
  --env HF_HUB_OFFLINE=1
  --env TRANSFORMERS_OFFLINE=1
  --env TMPDIR="${TMPDIR:-/tmp}"
  --env SPEECHLMM_VIDEO_FPS="${SPEECHLMM_VIDEO_FPS:-2}"
  --env SPEECHLMM_VIDEO_NUM_FRAMES="${SPEECHLMM_VIDEO_NUM_FRAMES:-}"
  --env MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
  --env TEMPERATURE="${TEMPERATURE:-0.0}"
  --env DO_SAMPLE="${DO_SAMPLE:-0}"
  --env TELEMETRY="${TELEMETRY:-0}"
  --env CONTEXT_OVERFLOW="${CONTEXT_OVERFLOW:-warn}"
)
if [[ -n "${HF_HOME:-}" ]]; then
  SING_COMMON+=(--env HF_HOME="${HF_HOME}" --env TRANSFORMERS_CACHE="${HF_HOME}/hub")
fi

# One pass over LOAD_MODEL: every model contributes its own checkpoint variable,
# its optional device_map, a container-path check and a --load-model flag.
MODEL_ENV=()
CHECK_PATHS=()
LOAD_ARGS=()
RESOLVED=()
for _m in ${LOAD_MODEL}; do
  _path_var="${_m^^}_MODEL_PATH"
  _map_var="${_m^^}_DEVICE_MAP"
  if [[ -z "${!_path_var:-}" ]]; then
    echo "Set ${_path_var} (container path or Hub id) to load ${_m}." >&2
    exit 1
  fi
  MODEL_ENV+=(--env "${_path_var}=${!_path_var}")
  _map="${!_map_var:-${DEVICE_MAP:-}}"
  if [[ -n "${_map}" ]]; then
    MODEL_ENV+=(--env "${_map_var}=${_map}")
  fi
  for _suffix in DISABLE_TALKER DISABLE_CODE2WAV TELEMETRY SITE_PACKAGES SPEAKER SEED INSTRUCT CONTEXT_OVERFLOW ATTN_IMPL; do
    _flag_var="${_m^^}_${_suffix}"
    if [[ -n "${!_flag_var:-}" ]]; then
      MODEL_ENV+=(--env "${_flag_var}=${!_flag_var}")
    fi
  done
  # A Hub id has no directory inside the container, so only probe real paths.
  if [[ "${!_path_var}" == /* ]]; then
    CHECK_PATHS+=("${!_path_var}")
  fi
  LOAD_ARGS+=(--load-model "${_m}")
  RESOLVED+=("${_path_var}=${!_path_var}")
done
# Container paths, so check them through the binds.
for _p in "${CHECK_PATHS[@]}"; do
  if ! "${SING_COMMON[@]}" "${SIF}" test -d "${_p}"; then
    echo "Checkpoint ${_p} is not visible inside the container; add a bind via BINDS." >&2
    exit 1
  fi
done

echo "NODE=$(hostname)"
echo "TRITON_HTTP=http://$(hostname):${TRITON_HTTP_PORT}"
echo "OPENAI=http://$(hostname):${TRITON_OPENAI_PORT}"
printf '%s\n' "${RESOLVED[@]}"
echo "LOAD_MODEL=${LOAD_MODEL}"
echo "Starting tritonserver..."

"${SING_COMMON[@]}" \
  "${MODEL_ENV[@]}" \
  "${SIF}" \
  tritonserver \
    --model-repository="${CONTAINER_REPO}/models" \
    --model-control-mode=explicit \
    --strict-readiness=false \
    --exit-on-error=true \
    --http-max-input-size "${TRITON_HTTP_MAX_INPUT_SIZE:-268435456}" \
    --http-port "${TRITON_HTTP_PORT}" \
    --grpc-port "${TRITON_GRPC_PORT}" \
    --metrics-port "${TRITON_METRICS_PORT}" \
    --log-info=true \
    --log-warning=true \
    --log-error=true \
    "${LOAD_ARGS[@]}" &
TRITON_PID=$!

READY_URL="http://127.0.0.1:${TRITON_HTTP_PORT}/v2/health/ready"
echo "Waiting for ${READY_URL}"
for i in $(seq 1 180); do
  if curl -sf "${READY_URL}" >/dev/null; then
    echo "tritonserver ready after ${i} checks"
    break
  fi
  if ! kill -0 "${TRITON_PID}" 2>/dev/null; then
    echo "tritonserver exited before ready" >&2
    wait "${TRITON_PID}" || true
    exit 1
  fi
  sleep 10
done
if ! curl -sf "${READY_URL}" >/dev/null; then
  echo "tritonserver did not become ready in time" >&2
  kill "${TRITON_PID}" 2>/dev/null || true
  exit 1
fi

echo "Starting OpenAI gateway on :${TRITON_OPENAI_PORT}"
"${SING_COMMON[@]}" \
  --env "TRITON_URL=http://127.0.0.1:${TRITON_HTTP_PORT}" \
  --env "TRITON_TIMEOUT=${TRITON_TIMEOUT:-1800}" \
  "${SIF}" \
  python3.12 -m uvicorn gateway.openai_server:app \
    --host 0.0.0.0 \
    --port "${TRITON_OPENAI_PORT}" \
    --app-dir "${CONTAINER_REPO}" &
GW_PID=$!

cleanup() {
  kill "${GW_PID}" "${TRITON_PID}" 2>/dev/null || true
}
trap cleanup EXIT

wait -n "${TRITON_PID}" "${GW_PID}"
status=$?
cleanup
trap - EXIT
wait "${TRITON_PID}" 2>/dev/null || true
wait "${GW_PID}" 2>/dev/null || true
exit "${status}"
