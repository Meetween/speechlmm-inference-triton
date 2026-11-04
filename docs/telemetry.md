# Telemetry and metrics

Per-request efficiency metrics are computed inside Triton, on the `execute` wall time, and are off unless you turn them on. The client only sees network latency, so real-time factor and tokens per second are not measured there.

```bash
TELEMETRY=1                    # every opted-in model
SPEECHLMM_V2_XL_TELEMETRY=0    # per-model override; a non-empty value wins
```

`scripts/serve_singularity.sh` forwards both. A model opts in from its shim (`TritonPythonModel = instrument(...)` in `models/<name>/1/model.py`) and a `METRICS` output in `config.pbtxt`. Voxtral is not opted in.

With telemetry on, the gateway adds `metrics` and, when both token counts exist, OpenAI `usage`. The Triton log gets one line per successful request, prefixed `telemetry `, so `grep -o 'telemetry {.*'` is JSONL. With telemetry off the `METRICS` tensor is an empty string and the gateway response is unchanged.

Ratios use the whole `execute` time (`processing_time_s`), which includes media decode and generate. A backend may also report `generate_time_s`. Set the `telemetry_time_basis` config parameter to `generate` to compute the ratios from that time instead. A ratio whose denominator is missing or zero is omitted. `batch_size` is the number of requests in that `execute` call; they share one `processing_time_s`.

| Field | Definition |
|---|---|
| `rtf` | `processing_time_s / input_audio_s` |
| `video_rtf` | `processing_time_s / input_video_s` |
| `output_rtf` | `processing_time_s / output_audio_s` (TTS, S2ST) |
| `input_tokens_per_s` | `input_tokens / processing_time_s` |
| `output_tokens_per_s` | `output_tokens / processing_time_s` |
| `codec_tokens_per_s` | estimated codec tokens / `processing_time_s` |
| `compression_factor` | `input_tokens / output_tokens`, only for `summary` and `text_summary` |

`input_tokens_source` is `exact` when the backend reported its prompt length (media placeholder tokens included) and `text_only` when the wrapper re-tokenized the text prompt. `output_tokens_source` is `exact` or `retokenized` the same way. Codec tokens are `output_audio_s * codec_frame_rate`, and only when `config.pbtxt` sets `codec_frame_rate`. Qwen3-Omni and `speechlmm_v2_xl` use 12.5 Hz (24 kHz / 1920). Qwen2.5-Omni and `speechlmm_v2_l` use 50 Hz (24 kHz / 480: DiT repeats=2 × BigVGAN upsample product 240).

Task names in the evaluation plan map onto backend modes:

| Task | Mode | Metrics this server reports |
|---|---|---|
| ASR | `transcribe` | `rtf`, `input_tokens_per_s` |
| LREAD | `lipread` | `output_tokens_per_s`, `video_rtf` |
| SLU | `chat` + audio | `output_tokens_per_s` |
| ST | `st` | `rtf`, `output_tokens_per_s` |
| SSUM | `summary` | `output_tokens_per_s` |
| TSUM | `text_summary` | `input_tokens_per_s`, `output_tokens_per_s`, `compression_factor` |
| MT | `mt` | `input_tokens_per_s`, `output_tokens_per_s` |
| SQA | `chat` + audio | `rtf`, `output_tokens_per_s` |
| TQA | `chat`, text only | `input_tokens_per_s`, `output_tokens_per_s` |
| ACHAP | `chat` + audio | `rtf`, `output_tokens_per_s` |
| AVSPEAKD | `avspeakd` | `input_tokens_per_s`, `video_rtf` |
| TTS | `tts` | `output_rtf`, `input_tokens_per_s`, `codec_tokens_per_s` |
| S2ST | `s2st` | `rtf`, `codec_tokens_per_s` |

Not reported here: TQA has no input duration, so it has no real-time factor. ACHAP's forced-alignment time is computed outside this server. There is no streaming, so there is no time-to-first-token. A qwen silence skip is `skipped: "silence"` rather than dropped. A request whose prompt plus `max_new_tokens` exceeds the model context carries `context_overflow: true` (see [Full-length audio and multiple GPUs](deployment.md#full-length-audio-and-multiple-gpus)).

## Metrics

Triton exposes Prometheus metrics on container port 8002 — `http://127.0.0.1:${TRITON_METRICS_PORT:-9002}/metrics` under Compose, port 8102 under `serve_singularity.sh`.
