"""Opt-in per-request efficiency metrics for Triton Python backends.

A model opts in by wrapping its backend class in the Triton shim::

    TritonPythonModel = instrument(TritonPythonModel)

and by declaring a ``METRICS`` string output in ``config.pbtxt``. Measurement
stays off unless ``TELEMETRY`` or ``<MODEL>_TELEMETRY`` is set.
``instrument`` returns a subclass, so the unwrapped backend class (what the
unit tests call) is unchanged.

The wrapper times ``execute``, reads durations from the input and output
bytes, and counts tokens with the backend tokenizer when the backend did not
report exact counts through ``self.telemetry_usage[index]``.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import time
from typing import Any, Callable, Optional

import numpy as np

_HOOK_KEYS = {"input_tokens", "output_tokens", "generate_time_s", "skipped", "context_overflow"}
_SUMMARY_MODES = {"summary", "text_summary"}
_WARN_AT: dict[str, float] = {}
_VIDEO_WARNED = False


def _env_on(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def telemetry_enabled(model_name: str) -> bool:
    """Per-model ``<MODEL>_TELEMETRY`` wins over ``TELEMETRY`` (default off)."""
    if model_name:
        per = os.environ.get(f"{model_name.upper()}_TELEMETRY")
        if per is not None and per != "":
            return _env_on(per)
    return _env_on(os.environ.get("TELEMETRY"))


def _pb():
    import triton_python_backend_utils as pb_utils

    return pb_utils


def _warn_limited(model: str, message: str) -> None:
    now = time.monotonic()
    if now - _WARN_AT.get(model, 0.0) < 60.0:
        return
    _WARN_AT[model] = now
    logger = getattr(_pb(), "Logger", None)
    if logger is not None:
        logger.log_warn(message)


def _config_value(args: Optional[dict], key: str) -> str:
    if not args:
        return ""
    raw = args.get("model_config") or "{}"
    try:
        config = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return ""
    params = (config or {}).get("parameters") or {}
    entry = params.get(key) or {}
    if isinstance(entry, dict):
        return str(entry.get("string_value") or "").strip()
    return str(entry or "").strip()


def wav_duration_s(wav_bytes: bytes) -> Optional[float]:
    """Duration from a WAV header. ``None`` when the bytes are not a WAV."""
    try:
        import soundfile as sf

        info = sf.info(io.BytesIO(wav_bytes))
        if info.samplerate <= 0:
            return None
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def video_duration_s(video_bytes: bytes) -> Optional[float]:
    """Container duration via torchcodec. ``None`` when it cannot be read."""
    global _VIDEO_WARNED
    try:
        from torchcodec.decoders import VideoDecoder
    except Exception:
        if not _VIDEO_WARNED:
            _VIDEO_WARNED = True
            _warn_limited("video", "torchcodec is not available; input_video_s omitted")
        return None
    path = None
    try:
        import tempfile

        handle, path = tempfile.mkstemp(suffix=".mp4")
        os.write(handle, video_bytes)
        os.close(handle)
        meta = VideoDecoder(path).metadata
        duration = getattr(meta, "duration_seconds", None)
        if not duration:
            frames = getattr(meta, "num_frames", None)
            fps = getattr(meta, "average_fps", None)
            if frames and fps:
                duration = float(frames) / float(fps)
        return float(duration) if duration else None
    except Exception:
        return None
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def count_tokens(owner: Any, text: str) -> Optional[int]:
    """Re-tokenize ``text`` with the backend tokenizer. ``None`` if there is none."""
    tokenizer = getattr(owner, "tokenizer", None)
    if tokenizer is None:
        processor = getattr(owner, "processor", None)
        tokenizer = getattr(processor, "tokenizer", None) if processor is not None else None
    if tokenizer is None:
        return None
    if not text:
        return 0
    try:
        encoded = tokenizer(text, add_special_tokens=False)
    except Exception:
        return None
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
    if hasattr(ids, "shape"):
        return int(ids.shape[-1])
    return len(ids)


def output_token_count(tokens: list, eos_ids: set) -> int:
    """Generated tokens up to and including the first EOS, else the whole suffix."""
    for index, token in enumerate(tokens):
        if token in eos_ids:
            return index + 1
    return len(tokens)


def stash_generate_usage(
    owner: Any,
    model_inputs: Any,
    prompt_len: int,
    generate_time_s: float,
) -> list[dict[str, Any]]:
    """Build per-row exact counts from ``_last_output_counts`` and the attention mask."""
    counts = [int(value) for value in getattr(owner, "_last_output_counts", []) or []]
    mask = model_inputs.get("attention_mask") if hasattr(model_inputs, "get") else None
    inputs: list[int]
    if mask is not None and hasattr(mask, "sum"):
        try:
            summed = mask.sum(-1).reshape(-1)
            inputs = [int(value) for value in summed.tolist()]
        except Exception:
            inputs = []
    else:
        inputs = []
    if len(inputs) != len(counts):
        inputs = [int(prompt_len)] * len(counts)
    usage = [
        {
            "input_tokens": int(n_in),
            "output_tokens": int(n_out),
            "generate_time_s": float(generate_time_s),
        }
        for n_in, n_out in zip(inputs, counts)
    ]
    owner._last_usage = usage
    return usage


def record_usage(owner: Any, index: int, usage: Optional[dict]) -> None:
    """Store hook fields when the wrapped model created ``telemetry_usage``."""
    bucket = getattr(owner, "telemetry_usage", None)
    if not isinstance(bucket, dict) or not usage:
        return
    bucket[index] = {key: usage[key] for key in _HOOK_KEYS if key in usage}


def build_metrics(raw: dict) -> dict:
    """Add derived ratios. A ratio whose denominator is missing or zero is omitted."""
    record: dict[str, Any] = {}
    passthrough = (
        "model",
        "mode",
        "batch_size",
        "processing_time_s",
        "generate_time_s",
        "input_audio_s",
        "input_audio_clips",
        "input_video_s",
        "input_video_clips",
        "input_text_chars",
        "input_tokens",
        "input_tokens_source",
        "output_tokens",
        "output_tokens_source",
        "output_text_chars",
        "output_audio_s",
        "finish_reason",
        "skipped",
        "context_overflow",
    )
    for key in passthrough:
        if raw.get(key) is not None:
            record[key] = raw[key]

    if raw.get("telemetry_time_basis") == "generate" and raw.get("generate_time_s"):
        elapsed = raw.get("generate_time_s")
    else:
        elapsed = raw.get("processing_time_s")

    def ratio(name: str, numerator: Any, denominator: Any) -> None:
        if numerator is None or denominator is None or denominator == 0:
            return
        record[name] = round(float(numerator) / float(denominator), 4)

    ratio("rtf", elapsed, raw.get("input_audio_s"))
    ratio("video_rtf", elapsed, raw.get("input_video_s"))
    ratio("output_rtf", elapsed, raw.get("output_audio_s"))
    ratio("input_tokens_per_s", raw.get("input_tokens"), elapsed)
    ratio("output_tokens_per_s", raw.get("output_tokens"), elapsed)
    frame_rate = raw.get("codec_frame_rate")
    output_audio = raw.get("output_audio_s")
    if frame_rate and output_audio is not None:
        record["codec_tokens_est"] = int(round(float(output_audio) * float(frame_rate)))
        ratio("codec_tokens_per_s", record["codec_tokens_est"], elapsed)
    if raw.get("mode") in _SUMMARY_MODES:
        ratio("compression_factor", raw.get("input_tokens"), raw.get("output_tokens"))
    return record


def _tensor_strings(tensor: Any) -> list[str]:
    if tensor is None:
        return []
    array = tensor.as_numpy().reshape(-1)
    values = []
    for item in array.tolist():
        if isinstance(item, bytes):
            values.append(item.decode("utf-8", errors="replace"))
        elif item is None:
            continue
        else:
            values.append(str(item))
    return values


def _prompt_mode_and_text(raw: str) -> tuple[str, str]:
    text = (raw or "").strip()
    if not text.startswith("{"):
        return "chat", text
    try:
        spec = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return "chat", text
    if not isinstance(spec, dict):
        return "chat", text
    mode = str(spec.get("mode") or "chat").strip().lower() or "chat"
    parts = spec.get("parts")
    if isinstance(parts, list):
        chunks = [
            str(part.get("text") or "")
            for part in parts
            if isinstance(part, dict) and str(part.get("type") or "") == "text"
        ]
        if chunks:
            return mode, "".join(chunks)
    prompt = spec.get("prompt", spec.get("text", ""))
    return mode, "" if prompt is None else str(prompt)


def _decode_b64_list(values: list[str]) -> list[bytes]:
    blobs = []
    for value in values:
        try:
            blobs.append(base64.b64decode(value, validate=False))
        except (binascii.Error, ValueError):
            continue
    return blobs


def _sum_durations(blobs: list[bytes], reader: Callable[[bytes], Optional[float]]) -> Optional[float]:
    total = 0.0
    found = False
    for blob in blobs:
        duration = reader(blob)
        if duration is None:
            continue
        total += duration
        found = True
    return total if found else None


def _input_tensor(request: Any, name: str) -> Any:
    getter = getattr(_pb(), "get_input_tensor_by_name", None)
    if getter is None:
        return None
    try:
        return getter(request, name)
    except Exception:
        return None


def _measure_request(request: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    try:
        prompt_values = _tensor_strings(_input_tensor(request, "TEXT_PROMPT"))
        mode, text = _prompt_mode_and_text(prompt_values[0] if prompt_values else "")
        raw["mode"] = mode
        raw["input_text_chars"] = len(text)
        raw["_prompt_text"] = text
        audio_blobs = _decode_b64_list(_tensor_strings(_input_tensor(request, "AUDIO_BYTES")))
        video_blobs = _decode_b64_list(_tensor_strings(_input_tensor(request, "VIDEO_BYTES")))
        raw["input_audio_clips"] = len(audio_blobs)
        raw["input_video_clips"] = len(video_blobs)
        audio_s = _sum_durations(audio_blobs, wav_duration_s)
        video_s = _sum_durations(video_blobs, video_duration_s)
        if audio_s is not None:
            raw["input_audio_s"] = audio_s
        if video_s is not None:
            raw["input_video_s"] = video_s
    except Exception:
        raw.setdefault("mode", "chat")
    return raw


def _output_string(response: Any, name: str) -> Optional[str]:
    tensor = None
    getter = getattr(_pb(), "get_output_tensor_by_name", None)
    if getter is not None:
        try:
            tensor = getter(response, name)
        except Exception:
            tensor = None
    if tensor is None and hasattr(response, "output_tensors"):
        tensors = response.output_tensors()
        for item in tensors:
            if getattr(item, "name", None) == name or (isinstance(item, tuple) and item and item[0] == name):
                tensor = item
                break
    if tensor is None:
        return None
    if isinstance(tensor, tuple):
        data = tensor[1]
        if isinstance(data, list) and data:
            value = data[0]
            return value.decode() if isinstance(value, bytes) else str(value)
        return None
    values = _tensor_strings(tensor)
    return values[0] if values else None


def _has_error(response: Any) -> bool:
    checker = getattr(response, "has_error", None)
    if callable(checker):
        return bool(checker())
    return getattr(response, "error", None) is not None


def _attach_metrics(response: Any, payload: str) -> Any:
    if _has_error(response):
        return response
    pb = _pb()
    tensor = pb.Tensor("METRICS", np.array([payload], dtype=object))
    tensors = list(response.output_tensors())
    return pb.InferenceResponse(output_tensors=tensors + [tensor])


def _arm(owner: Any, args: Optional[dict]) -> None:
    model_name = str((args or {}).get("model_name") or "")
    owner._telemetry_on = telemetry_enabled(model_name)
    owner._telemetry_model = model_name
    rate = _config_value(args, "codec_frame_rate")
    try:
        owner._codec_frame_rate = float(rate) if rate else None
    except ValueError:
        owner._codec_frame_rate = None
    owner._telemetry_time_basis = _config_value(args, "telemetry_time_basis")
    owner.telemetry_usage = {}
    logger = getattr(_pb(), "Logger", None)
    if logger is not None:
        state = "enabled" if owner._telemetry_on else "disabled"
        logger.log_info(f"telemetry {state} for {model_name or 'model'}")


def _fill_record(owner: Any, measured: dict, response: Any, elapsed: float, batch_size: int, index: int) -> dict:
    raw = {
        "model": getattr(owner, "_telemetry_model", ""),
        "mode": measured.get("mode", "chat"),
        "batch_size": batch_size,
        "processing_time_s": round(float(elapsed), 6),
        "input_audio_clips": measured.get("input_audio_clips", 0),
        "input_video_clips": measured.get("input_video_clips", 0),
        "input_text_chars": measured.get("input_text_chars", 0),
    }
    if measured.get("input_audio_s") is not None:
        raw["input_audio_s"] = measured["input_audio_s"]
    if measured.get("input_video_s") is not None:
        raw["input_video_s"] = measured["input_video_s"]
    basis = getattr(owner, "_telemetry_time_basis", "")
    if basis:
        raw["telemetry_time_basis"] = basis
    rate = getattr(owner, "_codec_frame_rate", None)
    if rate:
        raw["codec_frame_rate"] = rate

    text = _output_string(response, "GENERATED_TEXT") or ""
    reason = _output_string(response, "FINISH_REASON")
    audio_b64 = _output_string(response, "GENERATED_AUDIO") or ""
    raw["output_text_chars"] = len(text)
    if reason:
        raw["finish_reason"] = reason
    if audio_b64:
        try:
            audio_s = wav_duration_s(base64.b64decode(audio_b64))
        except (binascii.Error, ValueError):
            audio_s = None
        if audio_s is not None:
            raw["output_audio_s"] = audio_s

    hook = {}
    bucket = getattr(owner, "telemetry_usage", None)
    if isinstance(bucket, dict):
        supplied = bucket.get(index) or {}
        hook = {key: supplied[key] for key in _HOOK_KEYS if key in supplied}
    if "input_tokens" in hook:
        raw["input_tokens"] = int(hook["input_tokens"])
        raw["input_tokens_source"] = "exact"
    else:
        counted = count_tokens(owner, measured.get("_prompt_text") or "")
        if counted is not None:
            raw["input_tokens"] = counted
            raw["input_tokens_source"] = "text_only"
    if "output_tokens" in hook:
        raw["output_tokens"] = int(hook["output_tokens"])
        raw["output_tokens_source"] = "exact"
    else:
        counted = count_tokens(owner, text)
        if counted is not None:
            raw["output_tokens"] = counted
            raw["output_tokens_source"] = "retokenized"
    if "generate_time_s" in hook:
        raw["generate_time_s"] = round(float(hook["generate_time_s"]), 6)
    if "skipped" in hook:
        raw["skipped"] = hook["skipped"]
    if "context_overflow" in hook:
        raw["context_overflow"] = bool(hook["context_overflow"])
    return build_metrics(raw)


def _run(owner: Any, orig_execute: Callable, requests: list) -> list:
    if not getattr(owner, "_telemetry_on", False):
        responses = orig_execute(requests)
        attached = []
        for response in responses:
            try:
                attached.append(_attach_metrics(response, ""))
            except Exception:
                attached.append(response)
        return attached

    measured = [_measure_request(request) for request in requests]
    bucket = getattr(owner, "telemetry_usage", None)
    if isinstance(bucket, dict):
        bucket.clear()
    else:
        owner.telemetry_usage = {}
    started = time.perf_counter()
    responses = orig_execute(requests)
    elapsed = time.perf_counter() - started
    model = getattr(owner, "_telemetry_model", "") or "model"
    out = []
    for index, response in enumerate(responses):
        if _has_error(response):
            out.append(response)
            continue
        try:
            record = _fill_record(
                owner,
                measured[index] if index < len(measured) else {},
                response,
                elapsed,
                len(requests),
                index,
            )
            payload = json.dumps(record, separators=(",", ":"), sort_keys=True)
            rebuilt = _attach_metrics(response, payload)
            logger = getattr(_pb(), "Logger", None)
            if logger is not None:
                logger.log_info("telemetry " + payload)
            out.append(rebuilt)
        except Exception as exc:
            _warn_limited(model, f"telemetry failed for request {index}: {type(exc).__name__}: {exc}")
            out.append(response)
    return out


def instrument(cls: type) -> type:
    """Return a subclass whose ``execute`` reports metrics. Idempotent."""
    if getattr(cls, "_telemetry_instrumented", False):
        return cls

    class Instrumented(cls):
        _telemetry_instrumented = True

        def initialize(self, args):
            super().initialize(args)
            _arm(self, args)

        def execute(self, requests):
            return _run(self, super().execute, requests)

    Instrumented.__name__ = cls.__name__
    Instrumented.__qualname__ = getattr(cls, "__qualname__", cls.__name__)
    Instrumented.__module__ = cls.__module__
    return Instrumented
