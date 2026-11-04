"""Helpers that are the same in the SpeechLMM and Qwen-Omni backends.

Each backend re-exports these names so existing imports keep working.
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    try:
        return int(v) if v is not None and v != "" else default
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    try:
        return float(v) if v is not None and v != "" else default
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _model_env(model_name: str, suffix: str) -> str:
    """<MODEL>_<SUFFIX> from model_name, for example SPEECHLMM_V2_L_MODEL_PATH."""
    if not model_name:
        return ""
    return os.environ.get(f"{model_name.upper()}_{suffix}", "").strip()


def _env_bool_model(model_name: str, suffix: str, global_name: str, default: bool) -> bool:
    """Per-model ``<MODEL>_SUFFIX`` wins over the global flag."""
    per = _model_env(model_name, suffix)
    if per != "":
        return per.strip().lower() in {"1", "true", "yes", "y", "on"}
    return _env_bool(global_name, default)


class ContextOverflow(ValueError):
    """The prompt plus the output budget exceeds the model context, and the policy is reject."""


def context_limit(config: Any) -> Optional[int]:
    """Language-model ``max_position_embeddings`` of an Omni-style or plain config."""
    def field(node: Any, name: str) -> Any:
        return node.get(name) if isinstance(node, dict) else getattr(node, name, None)

    for path in (("thinker_config", "text_config"), ("text_config",), ()):
        node = config
        for name in path:
            node = field(node, name)
        value = field(node, "max_position_embeddings")
        if isinstance(value, int) and value > 0:
            return value
    return None


def context_policy(model_name: str) -> str:
    """Per-model ``<MODEL>_CONTEXT_OVERFLOW`` wins over ``CONTEXT_OVERFLOW``: warn (default) or reject."""
    value = _model_env(model_name, "CONTEXT_OVERFLOW") or os.environ.get("CONTEXT_OVERFLOW", "")
    value = value.strip().lower() or "warn"
    if value not in {"warn", "reject"}:
        raise ValueError(f"CONTEXT_OVERFLOW must be 'warn' or 'reject', not {value!r}")
    return value


def check_context(owner: Any, input_tokens: int, max_new_tokens: int) -> bool:
    """Warn, or raise ContextOverflow, when the prompt plus the output budget exceeds the context.

    Returns whether the request overflows. Nothing is checked when the limit is unknown.
    """
    limit = getattr(owner, "context_limit", None)
    if not limit or input_tokens + max_new_tokens <= limit:
        return False
    message = (
        f"{input_tokens} input tokens plus max_new_tokens={max_new_tokens} exceed the "
        f"{limit}-token context of {getattr(owner, 'model_name', None) or 'this model'}"
    )
    if getattr(owner, "context_policy", "warn") == "reject":
        raise ContextOverflow(message)
    owner._pb().Logger.log_warn(f"{message}; generating anyway (CONTEXT_OVERFLOW=warn)")
    return True


def _config_param(model_config: Dict[str, Any], key: str) -> str:
    params = model_config.get("parameters") or {}
    entry = params.get(key) or {}
    if isinstance(entry, dict):
        return str(entry.get("string_value") or "").strip()
    return str(entry).strip()


def _assemble_user_content(
    parts: Sequence[Tuple[str, str]],
    audios: Sequence[Any],
    videos: Sequence[Any],
) -> List[Dict[str, Any]]:
    """Walk `parts`, popping from the audio and video queues. Text stays where it is."""
    audio_q = list(audios)
    video_q = list(videos)
    content: List[Dict[str, Any]] = []
    for kind, text in parts:
        if kind == "text":
            content.append({"type": "text", "text": text})
        elif kind == "audio":
            if not audio_q:
                raise ValueError("parts ask for audio but none remain")
            content.append({"type": "audio", "audio": audio_q.pop(0)})
        elif kind == "video":
            if not video_q:
                raise ValueError("parts ask for video but none remain")
            content.append({"type": "video", "video": video_q.pop(0)})
        else:
            raise ValueError(f"unknown part type {kind!r}")
    if audio_q or video_q:
        raise ValueError(
            f"unused media after walking parts: {len(audio_q)} audio, {len(video_q)} video"
        )
    return content


def decode_wav(wav_bytes: bytes, processor: Any, torch_mod: Any) -> Tuple[np.ndarray, int]:
    """Decode mono audio at the loaded feature extractor's sampling rate."""
    import soundfile as sf

    wav_data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if not isinstance(wav_data, np.ndarray):
        wav_data = np.array(wav_data, dtype=np.float32)
    if wav_data.ndim > 1:
        wav_data = (
            wav_data.mean(axis=1) if wav_data.shape[0] > wav_data.shape[1] else wav_data.mean(axis=0)
        )
    wav_data = wav_data.flatten().astype(np.float32, copy=False)
    target_sr = int(processor.feature_extractor.sampling_rate)
    if sr != target_sr:
        from torchaudio.functional import resample

        wav_data = resample(torch_mod.from_numpy(wav_data), int(sr), target_sr).numpy()
    return wav_data, target_sr
