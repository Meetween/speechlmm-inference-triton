# SpeechLMM-v2 Triton Inference

[![Triton](https://img.shields.io/badge/NVIDIA-Triton%20Inference%20Server-76B900)](https://github.com/triton-inference-server/server)
[![OpenAI-compatible](https://img.shields.io/badge/API-OpenAI--compatible-412991)](#openai-compatible-gateway)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Serving for [SpeechLMM-v2](https://github.com/Meetween/speechlmm-v2) and related
speech/audio-visual models on [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server),
with an OpenAI-compatible gateway in front.

One `tritonserver` process serves a shared model repository under `models/`. Every backend speaks the **same infer contract** (`AUDIO_BYTES`, `VIDEO_BYTES`, `TEXT_PROMPT`, generation scalars → `GENERATED_TEXT`), so clients switch models by name without changing payload shape. Triton runs with `--model-control-mode=explicit`: nothing occupies GPU memory until you load a model.

Backends subclass `backends/base_backend.py` (`BaseBackend`), which owns request decoding, grouping, errors and response tensors; a subclass implements `generate`. See [backends/README.md](./backends/README.md) to add a model.

The repository is cluster-agnostic: launchers are driven by environment variables and contain no scheduler headers, accounts or site paths.

## Models

| Triton model | Backend | Weights | Placement |
|---|---|---|---|
| `speechlmm_v2_xl` | `backends/speechlmm_backend.py` | SpeechLMM-v2 XL (Qwen3-Omni MoE 30B) | `KIND_MODEL`, `device_map=auto` over all visible GPUs |
| `speechlmm_v2_l` | `backends/speechlmm_backend.py` | SpeechLMM-v2 L (Qwen2.5-Omni 7B) | one `KIND_GPU` instance on GPU 0; configurable layer sharding |
| `voxtral_mini_3b_2507` | `models/voxtral_mini_3b_2507/1/model.py` | Voxtral Mini 3B | single GPU |
| `qwen25_omni` | `backends/qwen_omni_backend.py` | Qwen2.5-Omni | one instance on GPU 0; configurable layer sharding |
| `qwen3_omni` | `backends/qwen_omni_backend.py` | Qwen3-Omni MoE 30B (thinker-only unless speech is enabled) | `KIND_MODEL`, `device_map=auto` over all visible GPUs |
| `seamless_m4t_v2_large` | `backends/seamless_backend.py` | SeamlessM4T v2 large, `s2st` only | one `KIND_GPU` instance on GPU 0 |
| `omnivoice` | `backends/omnivoice_backend.py` | OmniVoice, `tts` only, 600+ languages (needs the `scripts/requirements-omnivoice-site.txt` sidecar) | one `KIND_GPU` instance on GPU 0 |
| `qwen3_tts` | `backends/qwen3_tts_backend.py` | Qwen3-TTS 12Hz CustomVoice, `tts` only (needs the `scripts/requirements-qwen3-tts-site.txt` sidecar) | one `KIND_GPU` instance on GPU 0 |

The two SpeechLMM models share one backend file and add optional `VIDEO_BYTES` plus dynamic batching (`max_batch_size: 8`); Voxtral and Qwen stay at `max_batch_size: 1`.

## Two ways to run

| | [HPC (Singularity / Apptainer)](#run-on-hpc-singularity--apptainer) | [Docker Compose](docs/deployment.md#run-on-a-docker-host) |
|---|---|---|
| Use for | SpeechLMM-v2 on a GPU cluster | Voxtral / Qwen on a single machine |
| Triton | `25.06-py3` image | `24.01-py3` image |
| Python deps | baked into the image, `scripts/build/build_sif.sh` | 3.10 venv, `scripts/setup_host_dependencies.sh` |
| Launcher | `scripts/serve_singularity.sh` | `docker-compose.yml` + `scripts/up.sh` |
| Gateway | yes, port 8000 | not wired up |

## Run on HPC (Singularity / Apptainer)

Serving SpeechLMM-v2 is two processes: `tritonserver` on 8100, and the OpenAI-compatible gateway on 8000 once Triton is ready.

### 1. Build the image (once, on a machine with Docker and Singularity)

```bash
TRITON_SIF_PATH=<local-disk>/speechlmm-triton.sif scripts/build/build_sif.sh
```

This builds `docker/serve.Dockerfile`, converts it through a local Docker registry (`localhost:5000`) and writes the `.sif`; copy it to the cluster afterwards. Build on local disk, not on a parallel filesystem. The image contains `tritonserver`, `torch`, `transformers`, `torchcodec`, FFmpeg and the gateway, so serving needs no venv on the cluster. Expect ~18 GB.

### 2. Serve

```bash
SIF=<path>/speechlmm-triton.sif \
LOAD_MODEL="speechlmm_v2_xl" \
SPEECHLMM_V2_XL_MODEL_PATH=/models/SpeechLMM-v2.0-XL-30B \
BINDS="/host/path/to/models:/models" \
scripts/serve_singularity.sh
```

Every name in `LOAD_MODEL` (space-separated) reads its own `<MODEL_NAME>_MODEL_PATH`: a path **inside** the container, or a Hub id already cached in `HF_HOME` (the launcher runs with `HF_HUB_OFFLINE=1`, since compute nodes often have no internet). `BINDS` must expose those paths and any `file://` media the gateway reads. Released SpeechLMM-v2 checkpoints carry `auto_map`, so `trust_remote_code=True` resolves the model class directly; `scripts/export_checkpoint.py` in speechlmm-v2 produces the same layout from a training checkpoint. The script header documents every variable (device maps, attention kernel, generation defaults, speech output, ports).

Ports: gateway **8000**, Triton HTTP **8100**, gRPC 8101, metrics 8102.

### 3. Run it as a batch job

`serve_singularity.sh` runs in the foreground, so any scheduler can wrap it. A SLURM example:

```bash
#!/bin/bash
#SBATCH --job-name=speechlmm-serve
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --account=<your-account>
#SBATCH --partition=<gpu-partition>

module load singularity 2>/dev/null || true   # site-specific

export SIF=/path/to/speechlmm-triton.sif
export LOAD_MODEL="speechlmm_v2_l"
export SPEECHLMM_V2_L_MODEL_PATH=/models/SpeechLMM-v2.0-L-7B
export BINDS="/path/to/models:/models /path/to/data:/data"

bash /path/to/speechlmm-inference-triton/scripts/serve_singularity.sh
```

Reach the gateway from the login node through the compute node's hostname (`squeue -j <jobid> -o %N`) or an SSH tunnel, e.g. `ssh -L 8000:<node>:8000 <login-node>`. For long recordings and sharding one model across GPUs, see [Full-length audio and multiple GPUs](docs/deployment.md#full-length-audio-and-multiple-gpus).

## OpenAI-compatible gateway

`gateway/openai_server.py` exposes `POST /v1/chat/completions`, `GET /v1/models` and `GET /health` on port 8000. It accepts the vLLM-Omni content-array shape, `input_audio`, `audio_url` / `video_url` parts, and `file://` or `data:` URLs.

```bash
curl -sS "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "speechlmm_v2_l",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "Transcribe the following speech."},
      {"type": "audio_url", "audio_url": {"url": "file:///data/clip.wav"}}
    ]}],
    "max_tokens": 256
  }'
```

The client owns all instructions: the server adds no system or task prompt. Speech output (`tts`, `s2st`), the `speechlmm_*` request fields, `finish_reason`, failure modes and the raw Triton tensor contract are in the [API reference](docs/api.md).

## Clients

| Path | Role |
|------|------|
| `examples/client/hello_speechlmm.py` | Read-first example: text, audio and video requests through the official `openai` SDK |
| `clients/infer_speechlmm_v2.py` | Debug probe: raw Triton tensors on 8100, bypassing the gateway |
| `clients/verify_speechlmm_v2.py` | Verification ladder against a live gateway: health, model list, chat, ASR, lipread, avspeakd, multi-clip, the 400s, batch invariance, throughput |

This repo serves models; it does not run evaluations. Batch inference over a
prepared dataset lives in [speechlmm-evaluation](https://github.com/Meetween/speechlmm-evaluation),
which knows the `inputs.jsonl` / `prompts.jsonl` schema and the task registry.
Point it at this gateway with `SPEECHLMM_TRITON_URL` and call
`speechlmm-run-inference`.

### Tests

```bash
python3 -m unittest discover -s tests            # wire-format tests, any interpreter
singularity exec "${SIF}" \
  python3.12 -m unittest discover -s tests       # adds the gateway parsing tests
```

`tests/test_gateway.py` imports `fastapi`, which lives in the SIF, so it skips
outside the container. No server or GPU is needed either way.

## Documentation

- [API reference](docs/api.md): Triton infer contract, `TEXT_PROMPT` modes, gateway request fields, speech output, failure modes
- [Deployment details](docs/deployment.md): full-length audio, multi-GPU placement, context-overflow policy, Docker Compose setup, VAD
- [Telemetry and metrics](docs/telemetry.md): per-request efficiency metrics and Prometheus endpoint
- [Adding a model](backends/README.md)

## Repository layout

```
models/<name>/config.pbtxt      Triton model config: batching, instance group, device map
models/<name>/1/model.py        Shim: instrument(TritonPythonModel) for telemetry
backends/README.md              How to add a model (BaseBackend subclass, shim, config)
docs/                           API reference, deployment details, telemetry
backends/base_backend.py        Shared execute pipeline (Sample, Result, BaseBackend)
backends/backend_common.py      Helpers shared by the SpeechLMM and Qwen backends
backends/telemetry.py           Opt-in efficiency metrics wrapper
models/speechlmm_v2_*/1/        Thin shim onto backends/speechlmm_backend.py
models/qwen25_omni/1/           Thin shim onto backends/qwen_omni_backend.py
models/qwen3_omni/1/            Thin shim onto backends/qwen_omni_backend.py
models/voxtral_mini_3b_2507/1/  Self-contained backend
backends/speechlmm_backend.py   Shared SpeechLMM-v2 backend: audio/video decode, batching, generation
backends/qwen_omni_backend.py   Shared stock Qwen-Omni backend
gateway/openai_server.py        The product: OpenAI-compatible server in front of Triton
triton_tensors.py               Triton tensor builders + infer envelope, shared by the gateway and the probe
clients/
  infer_speechlmm_v2.py         Diagnostic probe: raw tensors on 8100, bypassing the gateway
  verify_speechlmm_v2.py        Acceptance ladder against a live gateway
examples/client/
  hello_speechlmm.py            The read-first sample
tests/
  test_gateway.py               Request parsing, offline (needs the SIF for fastapi)
  test_triton_tensors.py        Wire format, stdlib only
deps/voxtral_overlay/           Extra wheels on sys.path via VOXTRAL_OVERLAY_SITE_PACKAGES
docker/serve.Dockerfile         The self-contained Singularity/HPC serving image
docker-compose.yml              Compose service, ports, volumes, env, tritonserver command
```

`scripts/` splits by deployment path — the two halves are independent:

```
scripts/
  # Singularity / HPC: SpeechLMM-v2
  build/build_sif.sh                Build the self-contained serving SIF off-cluster
  serve_singularity.sh              Start tritonserver + gateway under Singularity (env-driven)
  requirements-speechlmm-v2.txt     Server deps baked in by docker/serve.Dockerfile
  requirements-clients.txt          Client deps (the openai SDK), baked in by the same file

  # Docker Compose: Voxtral / Qwen
  setup_host_dependencies.sh        Host venv + HF cache + optional Silero prewarm
  up.sh, down.sh                    Compose up/down
  model_load.sh, model_unload.sh, model_index.sh    Repository HTTP API wrappers; default port
                                    9000, so set TRITON_URL to reach Triton on 8100 instead
  requirements-host-venv.txt        Deps for the 3.10 host venv
  requirements-voxtral-site.txt     Deps for the Voxtral sys.path overlay
```

Local-only (see `.gitignore`): `.env`, `huggingface_cache/`, `logs/`, `tmp/`.

## Troubleshooting

| Symptom | What to do |
|---------|------------|
| Infer **404** | Model not loaded — `model_load.sh <name>`, or add it to `LOAD_MODEL`. |
| CUDA OOM | Unload other models; reduce visible GPUs; do not load two large models on one GPU. |
| Empty text, audio present (Voxtral / Qwen) | VAD may be dropping "non-speech" — try `VAD_ENABLED=false` or non-silent audio. |
| `mode '…' requires audio` | Send non-empty `AUDIO_BYTES` for `transcribe` / `caption` / `summary`. |
| Gateway 400 `part carries no usable url` | Send `{"audio_url": {"url": "file:///…"}}`; the gateway will not fall back to text-only chat. |
| Gateway 400 `role '…' is not supported` | Only `system` and `user` are accepted; there is no conversation history. |
| Cannot resolve `SpeechLMMForConditionalGeneration` | The checkpoint's `config.json` has no `auto_map` — re-export it with `scripts/export_checkpoint.py` in **speechlmm-v2**. |
| `tritonserver: executable file not found` | Wrong SIF — `SIF` must be the Triton image, not a training image. |
| `No module named 'torch'` / `'fastapi'` | Old SIF — one built before the deps were baked in, which needed a venv prefix. Rebuild with `scripts/build/build_sif.sh`. |
| `libavutil.so.60` not found, or lipread 500s | FFmpeg sonames missing or dangling — rebuild the SIF; `docker/serve.Dockerfile` fails the build if PyAV stops vendoring one. |
| One instance per GPU when you wanted one | `KIND_GPU` creates an instance per visible GPU; pin with `gpus: [ 0 ]`. |
| Compose volume error | Ensure `huggingface_cache`, `logs`, `tmp`, `VENV_PATH` and `VOXTRAL_DEPS` exist. |
| Hub **401** / auth | Set `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` in `.env` and restart. |

## Adding a model

See [backends/README.md](./backends/README.md) for subclassing `BaseBackend`, keeping the unified I/O contract, GPU id in `initialize`, telemetry opt-in, dependencies and smoke tests.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Model weights served by this repository are distributed under their own licenses.

## Acknowledgements

This work is part of [Meetween](https://meetween.eu) — *My Personal AI Mediator
for Virtual Meetings* — funded by the European Union under the Horizon Europe
programme, grant agreement No. 101135798 (HORIZON-CL4-2023-HUMAN-01-CNECT).

Views and opinions expressed are those of the author(s) only and do not
necessarily reflect those of the European Union or the European Commission.
Neither the European Union nor the granting authority can be held responsible
for them.
