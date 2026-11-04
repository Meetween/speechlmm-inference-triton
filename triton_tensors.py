#!/usr/bin/env python3
"""Triton HTTP/JSON wire format for the speechlmm_v2 backend.

One copy of the tensor builders and the infer envelope, imported by the OpenAI
gateway and by the raw-tensor debug probe so the two cannot drift.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional, Sequence

# Mirrors AUDIO_REQUIRED_MODES / VIDEO_REQUIRED_MODES in backends/speechlmm_backend.py.
AUDIO_REQUIRED_MODES = {"transcribe", "caption", "summary", "st", "s2st"}
VIDEO_REQUIRED_MODES = {"lipread", "avspeakd"}


# Mirrors BaseBackend._finish in backends/base_backend.py; an unlisted reason is a 502.
FINISH_REASONS = frozenset({"stop", "length", "empty", "empty_length"})


class TritonInferError(RuntimeError):
    """HTTP or payload error from a Triton infer endpoint."""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


def bytes_input(name: str, value: str | Sequence[str]) -> dict[str, Any]:
    data = [value] if isinstance(value, str) else list(value)
    return {"name": name, "shape": [1, len(data)], "datatype": "BYTES", "data": data}


def int_input(name: str, value: int) -> dict[str, Any]:
    return {"name": name, "shape": [1, 1], "datatype": "INT32", "data": [int(value)]}


def bool_input(name: str, value: bool) -> dict[str, Any]:
    return {"name": name, "shape": [1, 1], "datatype": "BOOL", "data": [bool(value)]}


def fp32_input(name: str, value: float) -> dict[str, Any]:
    return {"name": name, "shape": [1, 1], "datatype": "FP32", "data": [float(value)]}


def _as_b64_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [item for item in value if item]


def _envelope_parts(parts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """TEXT_PROMPT `parts`: text carries its string, media carries only its type."""
    encoded: list[dict[str, Any]] = []
    for part in parts:
        kind = str(part.get("type") or "")
        if kind == "text":
            text = part.get("text")
            encoded.append({"type": "text", "text": "" if text is None else str(text)})
        else:
            encoded.append({"type": kind})
    return encoded


def build_inputs(
    *,
    mode: str,
    prompt: str,
    system: Optional[str] = None,
    audio_b64: str | Sequence[str] | None = None,
    video_b64: str | Sequence[str] | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    repetition_penalty: float = 1.0,
    language: Optional[str] = None,
    fmt: Optional[str] = None,
    caption_chars: Optional[int] = None,
    parts: Optional[Sequence[Mapping[str, Any]]] = None,
    return_audio: bool = False,
    speaker: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Input tensors for one request. Sampling is on when temperature > 0."""
    audios = _as_b64_list(audio_b64)
    videos = _as_b64_list(video_b64)
    envelope: dict[str, Any] = {"mode": mode, "prompt": prompt}
    if system:
        envelope["system"] = system
    if language:
        envelope["language"] = language
    if fmt:
        envelope["format"] = fmt
    if caption_chars is not None:
        envelope["caption_chars"] = caption_chars
    # Text-only requests stay a plain prompt. With media, `parts` is the order.
    if parts and (audios or videos):
        envelope["parts"] = _envelope_parts(parts)
    if return_audio:
        envelope["return_audio"] = True
    if speaker:
        envelope["speaker"] = speaker
    inputs = [
        bytes_input("TEXT_PROMPT", json.dumps(envelope)),
        int_input("MAX_NEW_TOKENS", max_new_tokens),
        bool_input("DO_SAMPLE", temperature > 0.0),
        fp32_input("TEMPERATURE", temperature),
        fp32_input("REPETITION_PENALTY", repetition_penalty),
    ]
    if audios:
        inputs.append(bytes_input("AUDIO_BYTES", audios))
    if videos:
        inputs.append(bytes_input("VIDEO_BYTES", videos))
    return inputs


def infer(url: str, model: str, inputs: list[dict[str, Any]], timeout: int) -> str:
    """Text-only compatibility wrapper for existing clients."""
    return infer_result(url, model, inputs, timeout)[0]


def infer_result(
    url: str, model: str, inputs: list[dict[str, Any]], timeout: int
) -> tuple[str, str]:
    """Return text and finish reason. Audio, when present, is dropped."""
    text, reason, _audio = infer_output(url, model, inputs, timeout)
    return text, reason


def infer_output(
    url: str, model: str, inputs: list[dict[str, Any]], timeout: int
) -> tuple[str, str, Optional[str]]:
    """Return text, finish reason, and base64 WAV. Missing audio is None."""
    text, reason, audio, _metrics = infer_output_metrics(url, model, inputs, timeout)
    return text, reason, audio


def infer_output_metrics(
    url: str, model: str, inputs: list[dict[str, Any]], timeout: int
) -> tuple[str, str, Optional[str], Optional[dict[str, Any]]]:
    """Return text, finish reason, base64 WAV, and parsed METRICS (or None)."""
    payload = json.dumps({"inputs": inputs}).encode("utf-8")
    endpoint = url.rstrip("/") + f"/v2/models/{model}/infer"
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise TritonInferError(f"HTTP {exc.code} from {endpoint}: {detail}", exc.code) from exc
    except TimeoutError as exc:
        raise TritonInferError(f"timeout after {timeout}s from {endpoint}") from exc
    except urllib.error.URLError as exc:
        raise TritonInferError(f"cannot reach {endpoint}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise TritonInferError(f"invalid JSON from {endpoint}: {exc}") from exc

    outputs = body.get("outputs") or []
    if not outputs:
        raise TritonInferError(f"no outputs in response: {body}")
    by_name = {output["name"]: output.get("data") for output in outputs}
    data = by_name.get("GENERATED_TEXT")
    if not data:
        raise TritonInferError("missing GENERATED_TEXT output")
    reason_data = by_name.get("FINISH_REASON")
    if not reason_data and model in {
        "speechlmm_v2_l",
        "speechlmm_v2_xl",
        "qwen25_omni",
        "qwen3_omni",
        "seamless_m4t_v2_large",
        "qwen3_tts",
        "omnivoice",
    }:
        raise TritonInferError(
            f"missing FINISH_REASON output; update the {model} backend and config"
        )
    # Older non-SpeechLMM backends retain their existing gateway behavior.
    reason = reason_data[0] if reason_data else "stop"
    if reason not in FINISH_REASONS:
        raise TritonInferError(f"invalid FINISH_REASON: {reason!r}")
    audio_data = by_name.get("GENERATED_AUDIO")
    audio = str(audio_data[0]) if audio_data else ""
    metrics_data = by_name.get("METRICS")
    metrics_raw = str(metrics_data[0]) if metrics_data else ""
    metrics: Optional[dict[str, Any]] = None
    if metrics_raw:
        try:
            parsed = json.loads(metrics_raw)
        except json.JSONDecodeError as exc:
            raise TritonInferError(f"invalid METRICS JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise TritonInferError("invalid METRICS JSON: expected an object")
        metrics = parsed
    return str(data[0]), reason, audio or None, metrics
