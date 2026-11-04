#!/bin/bash
# Build a per-model Python sidecar for a model whose stack conflicts with the SIF
# (for example a different transformers). The sidecar is a venv made with the
# SIF's python3.12 and --system-site-packages, so pip reuses the SIF's torch and
# installs only what is missing or pinned differently. numpy and the torch family
# are constrained to the SIF's versions (the Triton Python stub needs its numpy<2).
# Point the model at it with
#   <MODEL>_SITE_PACKAGES=<dest>/lib/python3.12/site-packages
# and the backend puts it first on sys.path for that model only.
#
# Needs internet (on a cluster: a login node or a node with outbound access).
#
# Lines "#nodeps <requirement>" in the requirements file are installed afterwards
# with --no-deps (a package whose declared deps include an unneeded demo stack).
#
# Usage: SIF=<triton.sif> bash scripts/build/build_model_site.sh <requirements.txt> <dest>
# Optional: BINDS (space-separated singularity binds), PIP_CACHE_DIR

set -euo pipefail

: "${SIF:?Set SIF to the Triton serving image}"
REQUIREMENTS="${1:?requirements file}"
DEST="${2:?destination directory}"
if [[ -e "${DEST}" ]]; then
  echo "${DEST} exists; remove it or pick another destination" >&2
  exit 1
fi
REQ_DIR="$(cd "$(dirname "${REQUIREMENTS}")" && pwd)"
BIND_ARGS=(-B "${REQ_DIR}" -B "$(dirname "${DEST}")")
for _b in ${BINDS:-}; do
  BIND_ARGS+=(-B "${_b}")
done

singularity exec --no-home --cleanenv "${BIND_ARGS[@]}" \
  --env PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DEST}.pip-cache}" \
  "${SIF}" bash -c "
    set -euo pipefail
    # --without-pip: the SIF has no ensurepip; the venv runs the SIF's pip instead.
    python3.12 -m venv --system-site-packages --without-pip '${DEST}'
    python3.12 -m pip freeze 2>/dev/null | grep -iE '^(numpy|torch|torchaudio|torchvision)==' > '${DEST}/sif-constraints.txt'
    '${DEST}/bin/python' -m pip install --no-input --upgrade-strategy only-if-needed \
      -c '${DEST}/sif-constraints.txt' -r '${REQ_DIR}/$(basename "${REQUIREMENTS}")'
    NODEPS=\$(sed -n 's/^#nodeps[[:space:]]\+//p' '${REQ_DIR}/$(basename "${REQUIREMENTS}")')
    if [[ -n \"\${NODEPS}\" ]]; then
      '${DEST}/bin/python' -m pip install --no-input --no-deps -c '${DEST}/sif-constraints.txt' \${NODEPS}
    fi
    '${DEST}/bin/python' -m pip check || true
  "
rm -rf "${DEST}.pip-cache"
echo "SITE_PACKAGES=${DEST}/lib/python3.12/site-packages"
