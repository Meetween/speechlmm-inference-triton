#!/usr/bin/env bash
# Provision host-side artifacts for Triton-Unified:
#   - Python packages in VENV_PATH (shared /venv mount)
#   - Python packages in VOXTRAL_DEPS/voxtral_site (/deps/voxtral_site mount)
#   - Hugging Face model snapshots into repo-local huggingface_cache/
#   - Optional: pre-download Silero VAD weights (torch.hub) into the default torch hub cache
#
# Usage:
#   ./scripts/setup_host_dependencies.sh [options]
#
# Options:
#   --venv PATH          Python venv root (default: VENV_PATH from .env)
#   --voxtral-deps PATH  Directory that will contain voxtral_site/ (default: VOXTRAL_DEPS from .env)
#   --hf-cache PATH      Hugging Face cache dir (default: <repo>/huggingface_cache)
#   --torch-index URL    PyTorch wheel index (default: cu124)
#   --skip-pip           Do not run pip steps
#   --skip-voxtral-site  Do not populate VOXTRAL_DEPS/voxtral_site
#   --skip-hf            Do not snapshot Hub models
#   --skip-silero        Do not prewarm Silero VAD via torch.hub
#   -h, --help           Show this help
#
# Environment:
#   Reads .env from the repository root if present (same keys as docker-compose / .env.example).
#   QWEN25_OMNI_MODEL, VOXTRAL_MODEL, HF_TOKEN, HUGGING_FACE_HUB_TOKEN are respected for Hub downloads.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_PATH_ARG=""
VOXTRAL_DEPS_ARG=""
HF_CACHE_ARG=""
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
SKIP_PIP=0
SKIP_VOXTRAL_SITE=0
SKIP_HF=0
SKIP_SILERO=0

usage() {
  cat <<'EOF'
Provision host-side dependencies for Triton-Unified (see script header comments).

Options:
  --venv PATH            Python venv (default: VENV_PATH from .env)
  --voxtral-deps PATH    Parent of voxtral_site/ (default: VOXTRAL_DEPS from .env)
  --hf-cache PATH        HF cache directory (default: <repo>/huggingface_cache)
  --torch-index URL      PyTorch wheel index (default: https://download.pytorch.org/whl/cu124)
  --skip-pip             Skip pip installs
  --skip-voxtral-site    Skip VOXTRAL_DEPS/voxtral_site population
  --skip-hf              Skip Hugging Face snapshot_download
  --skip-silero          Skip torch.hub Silero prewarm
  -h, --help             This message
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV_PATH_ARG="${2:-}"; shift 2 ;;
    --voxtral-deps) VOXTRAL_DEPS_ARG="${2:-}"; shift 2 ;;
    --hf-cache) HF_CACHE_ARG="${2:-}"; shift 2 ;;
    --torch-index) TORCH_INDEX="${2:-}"; shift 2 ;;
    --skip-pip) SKIP_PIP=1; shift ;;
    --skip-voxtral-site) SKIP_VOXTRAL_SITE=1; shift ;;
    --skip-hf) SKIP_HF=1; shift ;;
    --skip-silero) SKIP_SILERO=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
done

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

VENV_PATH="${VENV_PATH_ARG:-${VENV_PATH:-}}"
VOXTRAL_DEPS="${VOXTRAL_DEPS_ARG:-${VOXTRAL_DEPS:-}}"
HF_CACHE="${HF_CACHE_ARG:-$REPO_ROOT/huggingface_cache}"
QWEN25_OMNI_MODEL="${QWEN25_OMNI_MODEL:-Qwen/Qwen2.5-Omni-7B}"
VOXTRAL_MODEL="${VOXTRAL_MODEL:-mistralai/Voxtral-Mini-3B-2507}"

mkdir -p "$HF_CACHE" "$REPO_ROOT/logs" "$REPO_ROOT/tmp"

PY=""
if [[ -n "${VENV_PATH:-}" && -x "$VENV_PATH/bin/python" ]]; then
  PY="$VENV_PATH/bin/python"
fi

if [[ "$SKIP_PIP" == 0 ]]; then
  if [[ -z "$PY" ]]; then
    echo "ERROR: VENV_PATH must point to an existing Python venv with bin/python (got: ${VENV_PATH:-empty})" >&2
    echo "Create one with: python3.10 -m venv /path/to/venv" >&2
    exit 2
  fi

  echo "==> Upgrading pip tooling ($VENV_PATH)"
  "$PY" -m pip install -U pip setuptools wheel

  echo "==> Installing PyTorch stack from: $TORCH_INDEX"
  "$PY" -m pip install --upgrade \
    torch torchvision torchaudio \
    --index-url "$TORCH_INDEX"

  echo "==> Installing shared venv requirements (Qwen + common runtime)"
  "$PY" -m pip install -r "$REPO_ROOT/scripts/requirements-host-venv.txt"
fi

if [[ "$SKIP_VOXTRAL_SITE" == 0 ]]; then
  if [[ -z "$VOXTRAL_DEPS" ]]; then
    echo "ERROR: VOXTRAL_DEPS is not set (required to build voxtral_site unless --skip-voxtral-site)" >&2
    exit 2
  fi
  if [[ -z "$PY" ]]; then
    echo "ERROR: voxtral_site install needs a venv python (set VENV_PATH or use --venv)" >&2
    exit 2
  fi
  SITE_TARGET="$VOXTRAL_DEPS/voxtral_site"
  mkdir -p "$SITE_TARGET"
  echo "==> Installing Voxtral-side packages into: $SITE_TARGET"
  "$PY" -m pip install --upgrade -r "$REPO_ROOT/scripts/requirements-voxtral-site.txt" --target "$SITE_TARGET"
fi

if [[ "$SKIP_HF" == 0 ]]; then
  if [[ -z "$PY" ]]; then
    echo "ERROR: Hugging Face snapshot step needs VENV_PATH with huggingface_hub installed" >&2
    exit 2
  fi
  export HF_HOME="$HF_CACHE"
  export TRANSFORMERS_CACHE="$HF_CACHE"
  export HF_HUB_DISABLE_TELEMETRY=1

  echo "==> Downloading Hub snapshot: $QWEN25_OMNI_MODEL"
  HF_CACHE="$HF_CACHE" QWEN25_OMNI_MODEL="$QWEN25_OMNI_MODEL" "$PY" -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id=os.environ['QWEN25_OMNI_MODEL'],
    cache_dir=os.environ['HF_CACHE'],
    local_files_only=False,
)
print('OK:', os.environ['QWEN25_OMNI_MODEL'])
"

  echo "==> Downloading Hub snapshot: $VOXTRAL_MODEL"
  HF_CACHE="$HF_CACHE" VOXTRAL_MODEL="$VOXTRAL_MODEL" "$PY" -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id=os.environ['VOXTRAL_MODEL'],
    cache_dir=os.environ['HF_CACHE'],
    local_files_only=False,
)
print('OK:', os.environ['VOXTRAL_MODEL'])
"
fi

if [[ "$SKIP_SILERO" == 0 ]]; then
  if [[ -z "$PY" ]]; then
    echo "WARN: skipping Silero prewarm (no venv python)" >&2
  else
    echo "==> Prewarming Silero VAD (torch.hub; uses TORCH_HOME/HUB cache)"
    "$PY" <<'PY'
import torch
torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
    verbose=False,
    trust_repo=True,
)
print("OK: silero_vad")
PY
  fi
fi

echo "Done. Next: set VENV_PATH and VOXTRAL_DEPS in .env if not already, then ./scripts/up.sh"
