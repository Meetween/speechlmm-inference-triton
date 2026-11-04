# API reference

The Triton HTTP contract every backend implements, and the OpenAI-compatible gateway in front of it.

## Model control lifecycle

```text
server started (compose up / serve_singularity.sh)
        │
        ▼
GET /v2/health/ready
        │
        ▼
POST /v2/repository/index                 (scripts/model_index.sh)
        │
        ▼
POST /v2/repository/models/<name>/load    (scripts/model_load.sh)
        │
        ▼
POST /v2/models/<name>/infer
        │
        ▼
POST /v2/repository/models/<name>/unload  (scripts/model_unload.sh)
```

Until **load** succeeds, infer returns **404** for that model name. `serve_singularity.sh` passes `--load-model` for each name in `LOAD_MODEL`, so it is ready without a manual load step.

## HTTP inference contract

**Endpoint:** `POST /v2/models/<model_name>/infer`, where `<model_name>` is a directory under `models/`.

### Tensors

| Tensor | Datatype | Required | Description |
|--------|----------|----------|-------------|
| `AUDIO_BYTES` | BYTES | See [modes](#text_prompt-modes) | Base64-encoded audio (WAV is what we test most). One request may carry N clips as shape `[1, N]`. |
| `VIDEO_BYTES` | BYTES | See [modes](#text_prompt-modes) | Base64 MP4; N clips as shape `[1, N]`. `lipread` consumes exactly one as a mouth-ROI crop and needs `speechlmm_v2_*`; other modes send every clip to the vision tower. |
| `TEXT_PROMPT` | BYTES | Yes | Plain UTF-8 string **or** JSON string with `mode` + `prompt`. |
| `MAX_NEW_TOKENS` | INT32 | No | Falls back to env defaults when omitted. |
| `DO_SAMPLE` | BOOL | No | |
| `REPETITION_PENALTY` | FP32 | No | |
| `TEMPERATURE` | FP32 | No | |
| `GENERATED_TEXT` | BYTES | Output | Generated text. |
| `FINISH_REASON` | BYTES | Output (SpeechLMM-v2 L/XL) | `stop` when EOS is emitted; `length` when the generation token budget is exhausted without EOS. |

HTTP JSON infer uses shape `[1, 1]` for the scalar tensors and `[1, N]` for
`AUDIO_BYTES` / `VIDEO_BYTES`. Triton's dynamic batcher only groups requests whose
non-batch shapes match, so a 1-clip request and a 3-clip request form separate
batches; today's single-clip traffic batches as before.

### `TEXT_PROMPT` modes

| `mode` (JSON) | `AUDIO_BYTES` | `VIDEO_BYTES` | Behavior |
|---------------|---------------|---------------|----------|
| `chat` | Optional | Optional | Text-only if both missing; multimodal if present. |
| `transcribe` | **Required** | — | ASR-style; error if no audio. |
| `caption` | **Required** | — | Short-line captions; error if no audio. |
| `summary` | **Required** | — | Spoken-audio summary; error if no audio. |
| `text_summary` | — | — | Summary of text carried in `prompt`; error if the prompt is empty. |
| `lipread` | — | **Required** | VSR from a mouth-ROI crop; `speechlmm_v2_*` only, a **400** elsewhere. |
| `avspeakd` | Optional | **Required** | Audiovisual speaker diarization. |
| `st` | **Required** | — | Speech translation (text out). |
| `mt` | Optional | — | Text translation. |
| `tts` | — | — | Speech from the text in `prompt`; needs `return_audio`. |
| `s2st` | **Required** | — | Speech translation (speech out); needs `return_audio`. |

Optional JSON keys: `format`, `language`, `caption_chars`, and `parts` (the user
content in order). A text entry is `{"type": "text", "text": "..."}`; a media
entry is `{"type": "audio"}` or `{"type": "video"}` and carries no bytes. The
Nth `audio` is `AUDIO_BYTES[N]`, and the Nth `video` is `VIDEO_BYTES[N]`.
`parts` is sent whenever the request has media, so text can sit before,
between, or after clips. Without `parts`, clips are audio then video and the
prompt is last. A plain-string `TEXT_PROMPT` is treated as legacy chat with no structured mode. Clients sending other `mode` values (e.g. uppercase task ids) must map them to the table above, or extend the backend and this table in the same change.

Parser reference: `_parse_text_prompt` and `_build_mode_instruction` in `backends/speechlmm_backend.py` and in each `models/*/1/model.py`.

### Minimal text-only chat

```bash
curl -sS "http://127.0.0.1:${TRITON_HTTP_PORT:-9000}/v2/models/voxtral_mini_3b_2507/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {"name":"TEXT_PROMPT","shape":[1,1],"datatype":"BYTES","data":[["{\"mode\":\"chat\",\"prompt\":\"Say hi in one word.\"}"]]},
      {"name":"MAX_NEW_TOKENS","shape":[1,1],"datatype":"INT32","data":[[128]]},
      {"name":"DO_SAMPLE","shape":[1,1],"datatype":"BOOL","data":[[false]]},
      {"name":"REPETITION_PENALTY","shape":[1,1],"datatype":"FP32","data":[[1.0]]},
      {"name":"TEMPERATURE","shape":[1,1],"datatype":"FP32","data":[[1.0]]}
    ],
    "outputs":[{"name":"GENERATED_TEXT"}]
  }'
```

## OpenAI-compatible gateway

`gateway/openai_server.py` fronts Triton for the SpeechLMM models: `POST /v1/chat/completions`, `GET /v1/models`, `GET /health` on port 8000 (Triton 8100 stays internal). It accepts the vLLM-Omni content-array shape, both `input_audio` and `audio_url` / `video_url` parts, and `file://` or `data:` URLs. `stream: true` returns 400 — real streaming needs Triton decoupled mode.

```bash
curl -sS "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "speechlmm_v2_l",
    "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
    "max_tokens": 32,
    "speechlmm_mode": "chat"
  }'
```

Speech output uses the OpenAI audio shape. It is off unless `SPEECHLMM_DISABLE_TALKER=0` and `SPEECHLMM_DISABLE_CODE2WAV=0` (or the per-model `<MODEL>_DISABLE_TALKER` / `<MODEL>_DISABLE_CODE2WAV` pair). `modalities` must include `audio`, and `audio.format` is `wav` only. The waveform is 24 kHz PCM16 in `message.audio.data`; `message.content` stays the text transcript. Video and lipread input cannot be combined with audio output. `tts` is text in, speech out; `s2st` is speech in, speech out. Qwen2.5 speech (`qwen25_omni`, `speechlmm_v2_l`) uses the official Qwen system prompt when the request does not include one. A client system message replaces it, but the Qwen2.5 talker was trained with that prompt and may produce unintelligible audio under another one; the backend logs a warning the first time this happens. Qwen3 (`qwen3_omni`, `speechlmm_v2_xl`) has no such default.

Speech-only models (`backends/speech_model_backend.py`) have no talker switch: every request must ask for audio. They speak the prompt as is in `tts`, so the client sends the text to read, not an instruction. In `s2st` they ignore the prompt and translate into `speechlmm_language`. Their WAV is at the model's native rate (24 kHz for `qwen3_tts` and `omnivoice`, 16 kHz for `seamless_m4t_v2_large`). `qwen3_tts` speaks de, en, es, fr, it, pt, ja, ko, ru and zh with preset voices (default `ryan`) and samples as its checkpoint config says, reseeded with `<MODEL>_SEED` before every request. `omnivoice` has no preset voices: it ignores `audio.voice` and picks a voice itself (fixed by the seed), or designs one from `<MODEL>_INSTRUCT`. `seamless_m4t_v2_large` has speech output for 36 target languages (not Hungarian, Greek or Bulgarian, for example); other targets, like other `qwen3_tts` languages, return an error. `audio.voice` must be one of the model's voices, except the gateway default `ethan`, which falls back to `<MODEL>_SPEAKER` or the model's own default. `<MODEL>_SITE_PACKAGES` points at a sidecar placed first on `sys.path` for that model only, for a model that needs a different `transformers` than the SIF; build it with `SIF=<triton.sif> bash scripts/build/build_model_site.sh scripts/requirements-<model>-site.txt <dest>` (a `--system-site-packages` venv, so torch comes from the SIF) and set the printed `site-packages` path.

```bash
curl -sS "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "speechlmm_v2_l",
    "messages": [{"role": "user", "content": "The weather is clear."}],
    "speechlmm_mode": "tts",
    "modalities": ["text", "audio"],
    "audio": {"voice": "ethan", "format": "wav"}
  }'
```

```bash
curl -sS "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "speechlmm_v2_l",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "Translate the speech into Italian."},
      {"type": "audio_url", "audio_url": {"url": "file:///scratch/clip.wav"}}
    ]}],
    "speechlmm_mode": "s2st",
    "modalities": ["text", "audio"],
    "audio": {"voice": "ethan", "format": "wav"}
  }'
```

SpeechLMM-v2 responses report `choices[0].finish_reason` per sample: `stop` for
an emitted EOS token (including EOS on the final allowed token), or `length`
for token-budget exhaustion without EOS. Partial text is returned unchanged;
this metadata does not trigger retries. Padding after EOS in a batch does not
change a completed sample's reason. The reason reflects the effective generation
budget, including the existing OOM fallback's reduced budget.

Deploy the backend, model configs, and gateway together: SpeechLMM-v2 responses
missing `FINISH_REASON` are rejected instead of guessed. Older non-SpeechLMM
backends retain their existing `stop` fallback. The raw Python `infer()` helper
still returns text; `infer_result()` returns `(text, finish_reason)` and
`infer_output()` also returns the base64 WAV when the model emitted one.

Media are referenced by URL, so a gateway sharing a filesystem with the data skips base64:

```json
{"role": "user", "content": [
  {"type": "text", "text": "Transcribe the following speech."},
  {"type": "audio_url", "audio_url": {"url": "file:///scratch/clip.wav"}}
]}
```

### Request fields

| Field | Notes |
|---|---|
| `messages` | **`system` and `user` roles.** Client instructions are used without server-generated defaults; omitted or blank system messages produce no system turn. An `assistant` message is a 400: this endpoint takes no conversation history. A `system` message carries text only. |
| `max_tokens`, `max_completion_tokens` | Either is accepted; `max_completion_tokens` wins. Default 128. |
| `temperature` | `> 0` enables sampling; `0` (the default) is greedy. |
| `stream` | `true` returns 400 — real streaming needs Triton decoupled mode. |
| `speechlmm_mode` | Overrides mode derivation; otherwise audio → `transcribe`, video → `lipread`, both → `avspeakd`, neither → `chat`. |
| `speechlmm_language` | Accepted for compatibility; does not add instructions. State the target/content language in your prompt. |
| `speechlmm_format`, `speechlmm_caption_chars` | `format` is accepted for compatibility but does not add instructions. `caption_chars` still controls output line wrapping in `caption` mode. |
| `modalities`, `audio` | `modalities` is `text` and/or `audio`. `audio.voice` selects the speaker (`ethan` by default) and `audio.format` must be `wav`. Audio output is a PCM16 WAV in `message.audio.data`, at 24 kHz for the talkers and at the native rate for speech-only models. |

With the official OpenAI SDK the four `speechlmm_*` fields go in `extra_body`.

`usage` is present only when telemetry is on. SpeechLMM and Qwen backends then report exact prompt and completion counts (audio and video tokens included). With telemetry off the response has no `usage` field.

### Who owns the prompts

The client owns all natural-language instructions. No mode supplies automatic
system or user prompts. Supplied nonempty system messages are used; absent,
empty, or whitespace-only system messages produce no system turn. Missing user
text stays empty, even when audio/video is present. Existing surrounding-whitespace
trimming and newline joining of text parts remain unchanged.

Modes still control media requirements, lipreading encoder routing and placeholder
expansion, batching groups, and caption output wrapping. Chat-template delimiters
and media tokens remain required model input structure. `text_summary` still
requires nonempty source text; modes requiring audio/video still reject missing
media. Caption wrapping still defaults to 64 characters (clamped to 32–120).

Clients that previously relied on defaults must supply instructions explicitly:
for example, `Transcribe the audio verbatim.` or `Translate the speech into Italian.`
`speechlmm_language` and `speechlmm_format` remain accepted fields but no longer
influence wording or enforce a target language/format. Use your prompt for those
choices. Selecting `mt` or `st` alone does not specify a translation target.

This prevents evaluation instructions from being overridden by hidden server
policy. In particular, ACHAP can continue using `transcribe` for audio processing
while explicitly requesting Markdown chapter titles and transcripts; the server
no longer adds the conflicting instruction to output only spoken words.

### Failure modes

The gateway never downgrades a request it could not parse. An audio or video part
that yields no bytes — an `audio_url` carrying no URL, an empty URL, a `file://`
path that does not exist, `input_audio` with no `data`, undecodable base64 — is a
**400**, not a text-only chat completion with a plausible answer. Several
`audio_url` / `video_url` parts in one user message are accepted, and text keeps
its position among them: `[text, audio]` is not rewritten as `[audio, text]`.
That order is the `parts` list on `TEXT_PROMPT`. More than
`GATEWAY_MAX_AUDIO_PARTS` or `GATEWAY_MAX_VIDEO_PARTS` parts (both default 8) is a
**400**. `lipread` still takes exactly one video clip. Likewise an unknown
content-part type, an `assistant` role, a media part inside a `system` message, and
an unrecognised `speechlmm_*` field, so a typo cannot quietly fall back to mode
derivation. If Triton is unreachable, `/v1/models` answers **503** rather than an
empty model list.

Both media shapes are accepted: `{"audio_url": {"url": "file:///clip.wav"}}` and the
shorthand `{"audio_url": "file:///clip.wav"}`.

`http(s)` media URLs are refused unless `GATEWAY_ALLOW_REMOTE_MEDIA=1`, and a fetch
larger than `GATEWAY_MAX_MEDIA_BYTES` (256 MiB) is a 413. `file://` and `data:` URLs
always work.

`/v1/chat/completions` is synchronous and does blocking I/O, so concurrency is
capped by the anyio worker threadpool (40 by default). That is well above the
concurrency the verification client drives, but one wedged model can occupy the
gateway.

## Integrating with Meetween FastAPI

The companion HTTP API lives in [Meetween/fastapi-server](https://github.com/Meetween/fastapi-server/tree/develop) (`develop` branch).

- Configure `model_profiles.yml` with this stack's `TRITON_URL` and `model_name`.
- Use `prompt_protocol: general_json` so `TEXT_PROMPT` is the JSON envelope FastAPI builds (see `butler/triton_client.py` there).

There is no automatic discovery: you choose URL and model name in the profile.
