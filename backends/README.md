# Adding a new model to Triton-Unified

Add a new subdirectory under `models/` instead of creating a separate top-level repository for each model. New backends must expose the **same** input and output tensor names as `qwen25_omni` and `voxtral_mini_3b_2507` unless every client is updated in the same change.

## Subclass `BaseBackend`

`backends/base_backend.py` is the pipeline. A new model is a subclass, not a copy of `execute`.

```python
from base_backend import BaseBackend, Result, Sample

class TritonPythonModel(BaseBackend):
    supported_modes = frozenset({"chat", "tts"})

    def load_model(self, args, model_config) -> None:
        ...  # or keep the work in initialize(), which the base calls load_model from

    def decode_request(self, index, request) -> Sample:
        ...

    def generate(self, samples) -> list[Result]:
        ...  # text, finish_reason, optional audio_b64, optional usage counts
```

`tests/test_base_backend.py` (`EchoBackend`) is the worked example: text in, text out, unknown modes rejected, telemetry left to the shim. OmniVoice (text in, audio out) and SeamlessM4T follow the same shape: speech-only models subclass `speech_model_backend.SpeechModelBackend`, which owns decoding, speech validation, voices and the per-model `<MODEL>_SITE_PACKAGES` sidecar, and implement `load_weights` and `synthesize` (see `seamless_backend.py`). Put exact token counts on `Result.usage` (`input_tokens`, `output_tokens`, `generate_time_s`, `skipped`).

The Triton shim is still two lines plus `instrument` (see section 7b). `config.pbtxt` still declares the shared tensor names, including `METRICS` when telemetry is on.

Use a **separate** Triton deployment (another compose project or host) when the model requires a different Python or CUDA stack, a different Triton server version, or strict memory isolation from the models in this stack.

---

## 1. Copy an existing model as a template

Fastest path:

```bash
cp -r models/qwen25_omni models/<your_model_name>
# or voxtral_mini_3b_2507 if closer to your stack
```

Then:

1. Rename `name: "qwen25_omni"` inside `config.pbtxt` to **`your_model_name`** (must match the directory name).
2. Replace `1/model.py` with your implementation, **keeping I/O names** unless you intentionally fork the contract (then update **all** clients).

### Sharing one backend between models

When two models differ only in weights and GPU placement, do not copy the backend. `speechlmm_v2_xl` and `speechlmm_v2_l` are the reference: each `models/speechlmm_v2_*/1/model.py` is a shim that imports `TritonPythonModel` from `backends/speechlmm_backend.py`, and everything model-specific comes from outside the code. `qwen25_omni` and `qwen3_omni` use the same pattern via `backends/qwen_omni_backend.py`.

- `config.pbtxt` carries batching, `instance_group` and a `device_map` parameter.
- `initialize` reads `<MODEL_NAME>_MODEL_PATH` and `<MODEL_NAME>_DEVICE_MAP` (see `_model_env`), so `SPEECHLMM_V2_L_MODEL_PATH` configures `speechlmm_v2_l` without touching the shared file.

A fix then lands for both models at once. Only fork the backend when the generation path itself differs.

---

## 2. Keep the unified HTTP contract

### `config.pbtxt`

- **`backend: "python"`**
- **`max_batch_size: 1`** (matches typical HTTP JSON infer payloads using shape `[1, 1]` per input).
- **`instance_group`**: use `{ count: 1 kind: KIND_GPU }` and select GPUs with **`NVIDIA_VISIBLE_DEVICES`** at deploy time. Only add `gpus: [ n ]` if you truly need a fixed mapping inside Triton.

### Inputs / output (names and types)

Align with existing models:

| Name | `data_type` | `dims` | `optional` |
|------|---------------|--------|------------|
| `AUDIO_BYTES` | `TYPE_STRING` | `[ 1 ]` | `true` if text-only is supported |
| `VIDEO_BYTES` | `TYPE_STRING` | `[ 1 ]` | `true` (SpeechLMM-v2 only; mouth-ROI MP4) |
| `TEXT_PROMPT` | `TYPE_STRING` | `[ 1 ]` | no |
| `MAX_NEW_TOKENS` | `TYPE_INT32` | `[ 1 ]` | yes |
| `DO_SAMPLE` | `TYPE_BOOL` | `[ 1 ]` | yes |
| `REPETITION_PENALTY` | `TYPE_FP32` | `[ 1 ]` | yes |
| `TEMPERATURE` | `TYPE_FP32` | `[ 1 ]` | yes |
| **`GENERATED_TEXT`** (output) | `TYPE_STRING` | `[ 1 ]` | — |

**HTTP clients:** send each input as shape **`[1, 1]`** (nested list of one batch row, one element).

**`AUDIO_BYTES`:** base64-encoded bytes (WAV is the most tested format in this repo).

**`TEXT_PROMPT`:** either:

- Plain string (treated as chat-style / legacy), or  
- A **JSON string** with at least `mode` and `prompt`, e.g.  
  `{"mode":"transcribe","prompt":"...","system":"...","format":"verbatim","language":"en","caption_chars":64}`  

See `backends/qwen_omni_backend.py` → `_parse_text_prompt` for the canonical parser.

`prompt` and the optional `system` are the client's, and `backends/speechlmm_backend.py`
passes both to the model verbatim. A `mode` therefore selects mechanics (media
requirements, lipread placeholders, caption wrapping) and **defaults**, not mandatory
wording: the mode's own instruction is used only for a turn the client omitted. Follow the
same rule in a new backend, so a caller can reproduce a prescribed prompt exactly.

### Mode vs `AUDIO_BYTES` (must match `model.py`)

Backends enforce: **`transcribe`**, **`caption`**, **`summary`** and **`st`** require non-empty **`AUDIO_BYTES`**; **`chat`**, **`mt`** and **`text_summary`** are text-in. If you add modes, document and implement the same matrix in **`README.md`**.

| `mode` | `AUDIO_BYTES` | `VIDEO_BYTES` |
|--------|----------------|---------------|
| `chat` | Optional (text-only or multimodal) | Optional |
| `transcribe`, `caption`, `summary`, `st` | **Required** | — |
| `mt`, `text_summary` | — | — |
| `lipread` (`speechlmm_v2_xl` / `_l`) | — | **Required** (mouth-ROI MP4) |
| `avspeakd` (`speechlmm_v2_xl` / `_l`) | Optional | **Required** |

An unknown `mode` string is treated as chat by `_parse_text_prompt`, so normalize task ids in HTTP clients. A `TEXT_PROMPT` that *looks* like JSON but does not parse is an error rather than a chat fallback — do the same in a new backend, since a silent downgrade to chat produces a fluent answer to the wrong question.

---

## 3. GPU placement in `model.py` (required)

In `TritonPythonModel.initialize(self, args)`:

```python
instance_device_id = int(args.get("model_instance_device_id", 0))
# Load HF / torch model on cuda:instance_device_id — do not hardcode cuda:0
```

Triton sets `model_instance_device_id` from the visible GPU set.

---

## 4. Dependencies

### Shared venv (`VENV_PATH` → `/venv`)

Put **common** packages here (torch, transformers base, etc.).

### Voxtral-style “sidecar” (`VOXTRAL_DEPS` → `/deps/voxtral_site`)

Use when a model needs **extra** wheels that should not pollute every backend.

### Overlay inside this repo (`./deps` → `/unified_deps`)

**`deps/voxtral_overlay/`** is mounted at `VOXTRAL_OVERLAY_SITE_PACKAGES` for small pins (e.g. specific `accelerate` versions). Prepend that path in `sys.path` **before** `VENV_SITE_PACKAGES` in `initialize()` if you need it.

If two models need **mutually exclusive** stacks, do **not** force both into one container — split deployments.

---

## 5. Optional: environment-driven model IDs

Follow the pattern in existing backends:

- Read `os.environ.get("YOUR_MODEL_HUB_ID", "org/name")` so operators can override in `.env` without editing code.
- Document new variables in **`README.md`** and **`.env.example`**.

---

## 6. Optional: VAD (silence skipping)

Only `qwen25_omni` and `voxtral_mini_3b_2507` do this. If your backend supports it, reuse their env vars:

- `VAD_ENABLED` (`true` / `1` / `yes`)
- `VAD_THRESHOLD`
- `VAD_ENERGY_FALLBACK`

Implement **fail-open**: if Silero or thresholds fail to load, log a warning and run without VAD.

Think twice on an air-gapped cluster: Silero comes from `torch.hub`, so a compute node with no internet fails the load and runs unfiltered anyway. `backends/speechlmm_backend.py` dropped VAD for exactly that reason — a guarantee it could not keep is worse than none.

---

## 7. Load without restarting Triton

```bash
./scripts/model_load.sh <your_model_name>
./scripts/model_unload.sh <your_model_name>
./scripts/model_index.sh
```

---

## 7b. Telemetry (optional, two lines)

Efficiency metrics are generic. They do not need backend code.

1. In `models/<name>/1/model.py`: `from telemetry import instrument` and `TritonPythonModel = instrument(YourBackend)`.
2. In `config.pbtxt`, add `{ name: "METRICS" data_type: TYPE_STRING dims: [ 1 ] }` to `output`.
3. If the model emits speech from a codec, add `parameters { key: "codec_frame_rate" value: { string_value: "12.5" } }` with the real frame rate (Qwen3 / `speechlmm_v2_xl`: `12.5`; Qwen2.5 Talker / `speechlmm_v2_l`: `50`). Omit it if you do not know the rate; `codec_*` fields are then left out.

`TELEMETRY=1` (or `<MODEL>_TELEMETRY=1`) turns measurement on. The wrapper times `execute`, reads audio and video durations from the bytes, and re-tokenizes text with `self.tokenizer` or `self.processor.tokenizer`.

For exact prompt lengths that include media tokens, set `self.telemetry_usage[request_index]` during `execute` (the wrapper creates the dict in `initialize`). Allowed keys: `input_tokens`, `output_tokens`, `generate_time_s`, `skipped`. Anything else is ignored. See `backends/speechlmm_backend.py` (`record_usage`).

---

## 8. Smoke test checklist

- [ ] `./scripts/model_index.sh` lists your model after load.
- [ ] **Chat**, text-only (no `AUDIO_BYTES`): JSON `{"mode":"chat",...}` returns non-empty `GENERATED_TEXT` for a tiny prompt.
- [ ] **Transcribe** or **caption** with a short WAV (and matching JSON `mode`) returns plausible output (or empty if the backend has VAD and treats the clip as silence).
- [ ] `nvidia-smi` shows memory on the expected GPU when loaded.

---

## 9. Documentation you should update in the same PR

- **`README.md`** — env table, model list, troubleshooting if new failure modes exist.
- **`.env.example`** — any new `YOUR_MODEL_*` variables.
- **This file** — if you introduce a new I/O name or a second contract (strongly discouraged).

---

## 10. Naming

Use a **stable** Triton model name (lowercase, digits, underscores) — it appears in infer URLs and in any client configuration that selects `model_name`.
