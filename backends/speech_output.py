"""Shared speech-output helpers for the SpeechLMM and Qwen-Omni backends.

No torch import at module level: the CPU tests import this file, and only the
waveform encoder and the RoPE patch need torch.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Optional, Sequence

SPEECH_MODES = {"tts", "s2st"}
OUTPUT_SAMPLE_RATE = 24000
# Qwen2.5-Omni talker prompt; 7B and L speech is unintelligible without it.
QWEN25_AUDIO_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

_Q25_DIT_ROPE_FIXED = False


def envelope_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value) if value is not None else False


def encode_wav_b64(wav: Any, sample_rate: int = OUTPUT_SAMPLE_RATE) -> str:
    """PCM16 WAV at ``sample_rate`` (24 kHz for the talkers), base64. Clips to [-1, 1]."""
    import numpy as np
    import soundfile as sf

    if isinstance(wav, (list, tuple)):
        if len(wav) != 1:
            raise ValueError(f"expected one waveform, got {len(wav)}")
        wav = wav[0]
    if hasattr(wav, "detach"):
        wav = wav.detach().float().cpu().numpy()
    samples = np.clip(np.asarray(wav, dtype=np.float32).reshape(-1), -1.0, 1.0)
    if samples.size == 0:
        raise ValueError("speech generation returned an empty waveform")
    buf = io.BytesIO()
    sf.write(buf, samples, int(sample_rate), format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def warn_custom_audio_system(owner: Any, log_warn: Any) -> None:
    """Warn once per model instance that a custom Qwen2.5 system prompt degrades audio."""
    if getattr(owner, "_warned_custom_audio_system", False):
        return
    owner._warned_custom_audio_system = True
    log_warn(
        "custom system prompt on a Qwen2.5 speech request; the talker was trained "
        "with the official Qwen prompt and the audio may be unintelligible"
    )


def resolve_speaker(name: Optional[str], available: Sequence[str]) -> str:
    """Case-insensitive match. Qwen2.5 keys are 'Ethan'; the gateway sends 'ethan'."""
    keys = [str(key) for key in available]
    wanted = (name or "ethan").strip() or "ethan"
    for key in keys:
        if key.lower() == wanted.lower():
            return key
    raise ValueError(f"Unknown speaker {wanted!r}; available: {keys}")


def _deinterleave_head_dim(x: Any) -> Any:
    import torch

    return torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1)


def _llama_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """HF #47403: deinterleave the 5.2 layout, then Llama half-split RoPE."""
    import torch

    q = _deinterleave_head_dim(q)
    k = _deinterleave_head_dim(k)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    half = q.shape[-1] // 2
    q_embed = (q * cos) + torch.cat((-q[..., half:], q[..., :half]), dim=-1) * sin
    k_embed = (k * cos) + torch.cat((-k[..., half:], k[..., :half]), dim=-1) * sin
    return q_embed, k_embed


def apply_qwen25_dit_rope_fix(module: Any = None) -> bool:
    """Patch Qwen2.5 DiT RoPE on transformers 5.2. No-op when already fixed.

    Pass ``module`` in tests. The real call patches
    ``transformers.models.qwen2_5_omni.modeling_qwen2_5_omni`` once.
    """
    global _Q25_DIT_ROPE_FIXED
    if module is None:
        if _Q25_DIT_ROPE_FIXED:
            return False
        _Q25_DIT_ROPE_FIXED = True
        try:
            from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as module
        except Exception:
            return False
    fn = getattr(module, "apply_rotary_pos_emb", None)
    code = getattr(fn, "__code__", None)
    if code is None or "rotate_half_codec" not in code.co_varnames:
        return False
    module.apply_rotary_pos_emb = _llama_apply_rotary_pos_emb
    return True


def validate_speech_request(
    mode: str,
    return_audio: bool,
    has_audio: bool,
    has_video: bool,
    has_lipread: bool,
    prompt: str,
    speech_enabled: bool,
) -> None:
    """Shared 400 rules for TTS and speech-to-speech."""
    normalized = (mode or "").strip().lower()
    if return_audio and not speech_enabled:
        raise ValueError(
            "speech output is disabled on this model; set <MODEL>_DISABLE_TALKER=0 and "
            "<MODEL>_DISABLE_CODE2WAV=0 (or SPEECHLMM_DISABLE_TALKER / SPEECHLMM_DISABLE_CODE2WAV)"
        )
    if return_audio and (has_video or has_lipread):
        raise ValueError("speech output does not support video or lipread input")
    if normalized in SPEECH_MODES and not return_audio:
        raise ValueError(
            f"mode '{normalized}' requires audio output (modalities including 'audio')"
        )
    if normalized == "tts":
        if has_audio or has_video:
            raise ValueError("mode 'tts' takes text only")
        if not (prompt or "").strip():
            raise ValueError("mode 'tts' requires a text prompt")
    if normalized == "s2st" and not has_audio:
        raise ValueError("mode 's2st' requires audio input")


def speech_generate_kwargs(
    sample: Any,
    speaker: str,
    talker_max_new_tokens: int,
    talker_do_sample: bool,
    eos_id: Any,
    thinker_max_new_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """Only thinker_/talker_ keys, plus speaker and return_audio.

    Official Qwen2.5 generate() forwards unprefixed keys into token2wav.
    """
    budget = sample.max_new_tokens if thinker_max_new_tokens is None else thinker_max_new_tokens
    kwargs: dict[str, Any] = {
        "return_audio": True,
        "speaker": speaker,
        "use_audio_in_video": False,
        "thinker_max_new_tokens": budget,
        "thinker_do_sample": sample.do_sample,
        "thinker_repetition_penalty": sample.repetition_penalty,
        "talker_do_sample": talker_do_sample,
        "talker_max_new_tokens": talker_max_new_tokens,
    }
    if sample.do_sample:
        kwargs["thinker_temperature"] = sample.temperature
    if eos_id is not None:
        kwargs["thinker_eos_token_id"] = eos_id
    return kwargs
