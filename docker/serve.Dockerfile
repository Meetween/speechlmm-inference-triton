ARG TRITON_SOURCE_IMAGE=nvcr.io/nvidia/tritonserver:25.06-py3
FROM ${TRITON_SOURCE_IMAGE}

# Self-contained SpeechLMM v2 serving image: tritonserver, the model stack, the
# OpenAI gateway and the smoke clients all live here, so serving needs no host
# venv prefix. Everything is installed with the image's own python3.12 so the
# Triton python-backend stub ABI matches the interpreter that loads the model.
#
# Pins match the SpeechLMM-v2 training SIF (transformers 5.2.0, torch 2.10.0).
# pydantic stays at the version tritonfrontend expects; requirements-speechlmm-v2.txt
# pins it, so pip must not drift it.

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

COPY scripts/requirements-speechlmm-v2.txt scripts/requirements-clients.txt /tmp/requirements/

RUN python3.12 -m pip install --no-cache-dir \
        torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
        --index-url "${TORCH_INDEX}" \
 && python3.12 -m pip install --no-cache-dir \
        -r /tmp/requirements/requirements-speechlmm-v2.txt \
        -r /tmp/requirements/requirements-clients.txt \
        --extra-index-url "${TORCH_INDEX}" \
 && rm -rf /tmp/requirements

# FlashAttention-2 for <MODEL>_ATTN_IMPL=flash_attention_2. No upstream wheel
# exists for torch 2.10, so it is compiled against the installed torch with the
# image's own nvcc. Native sm_80 (A100; its cubins also run on sm_86) and sm_90
# (H100), same version as the speechlmm-v2 training image. MAX_JOBS
# caps parallel nvcc jobs to keep build-host RAM in check.
ARG FLASH_ATTN_VERSION=2.8.3.post1
ARG FLASH_ATTN_CUDA_ARCHS="80;90"
ARG MAX_JOBS=16
RUN python3.12 -m pip install --no-cache-dir ninja packaging psutil \
 && FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS}" MAX_JOBS="${MAX_JOBS}" NVCC_THREADS=2 \
    python3.12 -m pip install --no-cache-dir --no-build-isolation \
        "flash-attn==${FLASH_ATTN_VERSION}"

# torchcodec dlopens unhashed FFmpeg sonames; PyAV's wheel vendors FFmpeg 8
# under av.libs/ with hashed names. A missing soname must fail the build, not
# turn into a lipread 500 at serving time.
ENV FFMPEG_LIB_DIR=/opt/ffmpeg/lib
RUN set -eu; \
    site="$(python3.12 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    mkdir -p "${FFMPEG_LIB_DIR}"; \
    ln -sf "${site}/av.libs/"* "${FFMPEG_LIB_DIR}/"; \
    for spec in libavutil:60 libavcodec:62 libavformat:62 libavdevice:62 \
                libavfilter:11 libswscale:9 libswresample:6; do \
        prefix="${spec%:*}"; sover="${spec#*:}"; \
        src="$(ls "${FFMPEG_LIB_DIR}/${prefix}-"*.so.${sover}* 2>/dev/null | head -n 1)"; \
        if [ -z "${src}" ]; then \
            echo "no vendored FFmpeg lib for ${prefix}.so.${sover}" >&2; exit 1; \
        fi; \
        ln -sf "$(basename "${src}")" "${FFMPEG_LIB_DIR}/${prefix}.so.${sover}"; \
        echo "  ${prefix}.so.${sover} -> $(basename "${src}")"; \
    done
ENV LD_LIBRARY_PATH=${FFMPEG_LIB_DIR}:${LD_LIBRARY_PATH}

# Fail here rather than on the first inference request.
RUN python3.12 -c "\
import ctypes, torch, transformers, flash_attn, torchcodec, av, soundfile, fastapi, uvicorn, openai, pydantic; \
[ctypes.CDLL(n) for n in ('libavutil.so.60', 'libavcodec.so.62', 'libavformat.so.62', \
                          'libavdevice.so.62', 'libavfilter.so.11', 'libswscale.so.9', \
                          'libswresample.so.6')]; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
print('transformers', transformers.__version__, '| torchcodec', torchcodec.__version__); \
print('pydantic', pydantic.VERSION, '| openai', openai.__version__); \
print('flash_attn', flash_attn.__version__)"

CMD ["bash"]
