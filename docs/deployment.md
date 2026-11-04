# Deployment details

## Full-length audio and multiple GPUs

These apply to both deployment paths.

SpeechLMM and Qwen-Omni preprocessing explicitly disables audio truncation. In
particular, Qwen2.5-derived checkpoints no longer silently discard everything
beyond their configured 300-second feature-extractor window. Text padding is
configured separately so multiple audio clips can still be padded together.

For Qwen2.5-Omni and SpeechLMM-L, the backend evaluates the audio encoder's existing
independent attention windows in bounded batches, avoiding a global quadratic
block-diagonal mask. Every audio token still reaches the language model in one
request. During generation only, it also projects the final hidden-state position
through the vocabulary head. The hook is removed even if generation fails; normal
forwards and checkpoint files are unchanged. These two memory optimizations are
restricted to the recognized Qwen2.5 audio encoder, not Qwen3/XL.

To shard one model across two allocated, visible GPUs, set either
`SPEECHLMM_V2_L_DEVICE_MAP=balanced` or `QWEN25_OMNI_DEVICE_MAP=balanced`. Request the
GPUs in your cluster launcher. Both model configs anchor **one Triton instance**
on GPU 0; its layer placement may span both GPUs. Removing `gpus: [0]` from a
`KIND_GPU` instance group creates a replica on every visible GPU, which defeats
this configuration. Deliberate replica deployments must configure placement
separately.

Use one concurrent request for long recordings. There is no duration cap added
here: available GPU memory, input length, generated length and concurrency decide
whether a request fits. The checkpoint's declared context length still matters for
quality; successfully allocating a longer sequence does not establish accuracy.
`TRITON_HTTP_MAX_INPUT_SIZE` controls the Singularity launcher's HTTP body limit
(default 268435456 bytes = 256 MiB); set 1073741824 for a 1 GiB body limit when
needed. This transport limit is independent of model context and GPU capacity.
Other launchers must set Triton's `--http-max-input-size` themselves.

Before generating, the SpeechLMM and Qwen-Omni backends compare each request's prompt
length plus `max_new_tokens` with the language model's `max_position_embeddings`
(32,768 for Qwen2.5-Omni and SpeechLMM-L/M, 65,536 for Qwen3-Omni and SpeechLMM-XL).
`CONTEXT_OVERFLOW` (per-model `<MODEL>_CONTEXT_OVERFLOW`) decides what happens past it:

- `warn` (default): the Triton log gets a warning with both counts, and the request is
  served as before. With telemetry on, its metrics carry `context_overflow: true`.
- `reject`: the request fails with a **400** naming both counts, and nothing is generated.

The model load log reports the limit and the policy. The check measures only the
prompt and the output budget; it says nothing about memory, which still decides
whether a long request fits.

## Run on a Docker host

From the repository root.

1. **Directories** — `mkdir -p huggingface_cache logs tmp`
2. **Environment** — create `.env` next to `docker-compose.yml` with the variables [below](#env); at least set `VENV_PATH` (a Python **3.10** venv, matching the `24.01-py3` image) and `VOXTRAL_DEPS` to real host directories.
3. **Host dependencies** — `./scripts/setup_host_dependencies.sh` (flags in the script header: `--skip-pip`, `--skip-hf`, `--torch-index`, …).
4. **Start** — `./scripts/up.sh`, which unsets shell `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` so Compose reads secrets from `.env` only.
5. **Health** — `curl -sSf "http://127.0.0.1:${TRITON_HTTP_PORT:-9000}/v2/health/live"`
6. **Load a model** — `./scripts/model_index.sh` then `./scripts/model_load.sh voxtral_mini_3b_2507`
7. **Stop** — `./scripts/down.sh`

Set `TRITON_URL` when calling `model_load.sh` / `model_unload.sh` / `model_index.sh` to target a non-default endpoint (default `http://127.0.0.1:${TRITON_HTTP_PORT:-9000}`).

### `.env`

Compose loads `.env` next to `docker-compose.yml`; there is no committed `.env`.

| Variable | Meaning |
|----------|---------|
| `TRITON_HTTP_PORT` | Host → container **8000** (Triton HTTP) |
| `TRITON_GRPC_PORT` | Host → **8001** (gRPC) |
| `TRITON_METRICS_PORT` | Host → **8002** (metrics) |
| `TRITON_IMAGE` | `tritonserver` image (pin for reproducibility) |
| `NVIDIA_VISIBLE_DEVICES` | GPUs visible in container (`all`, `0`, …); compose defaults to `all` if unset |
| `VENV_PATH` | Host dir → `/venv` (read-only); must contain `lib/python3.10/site-packages` |
| `VOXTRAL_DEPS` | Host dir → `/deps` (read-only); must include `voxtral_site/` for Voxtral extras |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | Hugging Face auth for gated/private Hub IDs |
| `VOXTRAL_MODEL`, `QWEN25_OMNI_MODEL` | Hub model IDs |
| `MAX_NEW_TOKENS`, `TEMPERATURE`, `DO_SAMPLE` | Defaults when infer omits those inputs (`DO_SAMPLE`: `0`/`1` or `true`/`false`) |
| `VAD_ENABLED`, `VAD_THRESHOLD`, `VAD_ENERGY_FALLBACK` | VAD behavior |
| `PYTORCH_CUDA_ALLOC_CONF` | Passed into the container for PyTorch allocator |

Set only in `docker-compose.yml`: `NVIDIA_DRIVER_CAPABILITIES` (`compute,utility`), `TMPDIR` (`/tmp/triton`), `VENV_SITE_PACKAGES` (`/venv/lib/python3.10/site-packages`), `VOXTRAL_SITE_PACKAGES` (`/deps/voxtral_site`), `VOXTRAL_OVERLAY_SITE_PACKAGES` (`/unified_deps/voxtral_overlay`), `HF_HOME` / `TRANSFORMERS_CACHE` (under `./huggingface_cache`), `HF_HUB_DISABLE_TELEMETRY`, `OMP_NUM_THREADS` / `MKL_NUM_THREADS`.

Compose passes `--model-control-mode=explicit` (no auto-load), `--strict-readiness=false` (one bad model config does not block bring-up) and `--exit-on-error=false` (server survives a failed load). Adjust only if you know your ops constraints.

## Voice activity detection (VAD)

VAD applies to the **Voxtral and Qwen backends only**. When it is enabled and working, **silent** clips may return empty `GENERATED_TEXT`. Tune `VAD_*` and restart. If Silero or the thresholds fail, those backends log and continue without VAD (fail-open).

`backends/speechlmm_backend.py` has no VAD: it needs a `torch.hub` download, which a compute node without internet cannot do, so it silently disabled itself exactly where SpeechLMM runs. Non-speech audio now reaches the model and you see what it produces.
