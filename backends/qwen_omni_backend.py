from __future__ import annotations

"""Shared Python backend for stock Qwen-Omni checkpoints.

Speaks the repository-wide infer contract: AUDIO_BYTES, VIDEO_BYTES, TEXT_PROMPT,
MAX_NEW_TOKENS, DO_SAMPLE, REPETITION_PENALTY, TEMPERATURE, at max_batch_size: 1.
Optional inputs follow Triton config; with neither audio nor video the request is
text-only (including MT). Audio and video accept several clips, ordered by `parts`; video
goes to the vision tower, and only lipread is refused.

Model-specific values come from outside this file:
  <MODEL_NAME>_MODEL_PATH  checkpoint dir or Hub id (QWEN25_OMNI_MODEL is the
                           docker-compose fallback for qwen25_omni)
  <MODEL_NAME>_DEVICE_MAP  overrides the config.pbtxt device_map parameter
  config.pbtxt parameters  device_map (default cuda:<instance>) and dtype
                           (default float16)
Processor and thinker classes are selected from the checkpoint's model_type.
"""

import base64
import io
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import triton_python_backend_utils as pb_utils

from backend_common import (
    _assemble_user_content,
    _config_param,
    _env_bool,
    _env_bool_model,
    _env_float,
    _env_int,
    _model_env,
    ContextOverflow,
    check_context,
    context_limit,
    context_policy,
    decode_wav,
)
from base_backend import BaseBackend, Result, Sample
from telemetry import output_token_count, stash_generate_usage
from speech_output import (
    QWEN25_AUDIO_SYSTEM_PROMPT,
    apply_qwen25_dit_rope_fix,
    encode_wav_b64,
    envelope_bool,
    resolve_speaker,
    speech_generate_kwargs,
    validate_speech_request,
    warn_custom_audio_system,
)


def _resolve_eos_pad(generation_config: Any, tokenizer: Any, model_id: str) -> tuple[Any, Any]:
    eos = getattr(generation_config, "eos_token_id", None) if generation_config is not None else None
    pad = getattr(generation_config, "pad_token_id", None) if generation_config is not None else None
    if eos is None:
        eos = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
    if pad is None:
        pad = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
    if pad is None:
        pad = eos
    if eos is None:
        raise RuntimeError(f"{model_id} has no eos token; generation would never stop.")
    return eos, pad


def _resolve_model_id(model_name: str) -> str:
    path = _model_env(model_name, "MODEL_PATH") or _model_env(model_name, "MODEL")
    if not path:
        raise RuntimeError(
            f"{model_name.upper()}_MODEL_PATH or {model_name.upper()}_MODEL is required "
            "(Qwen-Omni checkpoint directory or Hub id)."
        )
    return path


def _resolve_device_map(raw: str, instance_device_id: int) -> Any:
    device_map = (raw or "").strip() or f"cuda:{instance_device_id}"
    if device_map in {"auto", "balanced", "balanced_low_0", "sequential", "cpu"}:
        return device_map
    if device_map == "cuda":
        return {"": f"cuda:{instance_device_id}"}
    if device_map.startswith("cuda:"):
        return {"": device_map}
    raise ValueError(
        f"unsupported device_map '{device_map}'; expected auto, balanced, balanced_low_0, "
        "sequential, cpu, cuda, or cuda:<id>."
    )


_DTYPE_NAMES = {
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float32": "float32",
    "fp32": "float32",
}


def _resolve_dtype_name(raw: str) -> str:
    name = (raw or "").strip().lower() or "float16"
    if name not in _DTYPE_NAMES:
        raise ValueError(
            f"unsupported dtype '{raw}'; expected float16, bfloat16, or float32."
        )
    return _DTYPE_NAMES[name]


def _load_placement(args: Dict[str, Any]) -> Dict[str, Any]:
    model_name = str(args.get("model_name", "")).strip()
    try:
        model_config = json.loads(args.get("model_config") or "{}")
    except json.JSONDecodeError:
        model_config = {}
    instance_device_id = int(args.get("model_instance_device_id", 0))
    return {
        "model_name": model_name,
        "model_id": _resolve_model_id(model_name),
        "device_map": _resolve_device_map(
            _model_env(model_name, "DEVICE_MAP")
            or _config_param(model_config, "device_map"),
            instance_device_id,
        ),
        "dtype_name": _resolve_dtype_name(_config_param(model_config, "dtype")),
        "instance_device_id": instance_device_id,
    }


def _omni_family(model_type: str) -> str:
    mt = (model_type or "").lower()
    if "qwen2_5" in mt or "qwen2.5" in mt:
        return "qwen2_5_omni"
    if "qwen3_omni" in mt:
        return "qwen3_omni"
    raise RuntimeError(f"unrecognised Qwen-Omni model_type '{model_type}'.")


#: Needs a mouth-ROI encoder trained on cropped lips, which these weights do not carry.
#: Not avspeakd: that one only needs the ordinary vision tower, which they do have.
LIPREAD_MODES = {"lipread"}
LIPREAD_PLACEHOLDER = "<lipread>"


class UnsupportedRequest(ValueError):
    """A request these weights cannot serve, answered as a 400 and never as a guess."""


def _reject_unsupported_request(mode: str, prompt: str, model_name: str) -> None:
    """Refuse work this model cannot do.

    Silence is the danger here, not failure. _parse_text_prompt maps an unknown mode to
    chat, and <lipread> is not a token this tokenizer knows, so it reaches the chat
    template as literal text. The model then answers fluently from the wording of the
    prompt alone -- a plausible transcript of a video it never saw.
    """
    normalized = (mode or "").strip().lower()
    if normalized in LIPREAD_MODES:
        trigger = f"mode '{mode}'"
    elif LIPREAD_PLACEHOLDER in (prompt or ""):
        trigger = f"the {LIPREAD_PLACEHOLDER} placeholder"
    else:
        return
    raise UnsupportedRequest(
        f"{trigger} needs a mouth-ROI lipread encoder, which '{model_name}' does not have."
    )


def _triton_error(exc: BaseException) -> Any:
    """INVALID_ARG for a bad request, so the gateway reports 400 rather than 500."""
    message = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, (UnsupportedRequest, ContextOverflow)):
        code = getattr(pb_utils.TritonError, "INVALID_ARG", None)
        if code is not None:
            try:
                return pb_utils.TritonError(message, code)
            except TypeError:
                pass  # older pb_utils takes the message only
    return pb_utils.TritonError(message)


def _import_omni_classes(family: str) -> tuple[Any, Any, Any]:
    if family == "qwen2_5_omni":
        from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import (
            Qwen2_5OmniProcessor,
        )
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        return (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
            Qwen2_5OmniForConditionalGeneration,
        )
    if family == "qwen3_omni":
        from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
            Qwen3OmniMoeProcessor,
        )
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )

        return (
            Qwen3OmniMoeProcessor,
            Qwen3OmniMoeThinkerForConditionalGeneration,
            Qwen3OmniMoeForConditionalGeneration,
        )
    raise RuntimeError(f"unrecognised Qwen-Omni family '{family}'.")


# MT accepts text-only input; ST still requires speech.
AUDIO_REQUIRED_MODES = {"transcribe", "caption", "summary", "st", "s2st"}
# Preserve silence skipping for MT requests that explicitly include audio.
SILENCE_CHECK_MODES = AUDIO_REQUIRED_MODES | {"mt"}

#: Vision-tower modes. Refuse them without video, so the mode can never be answered
#: from the wording of the prompt alone.
VIDEO_REQUIRED_MODES = {"avspeakd"}


def _clean_text(text: str) -> str:
    # Some generations continue with dataset-style turns; keep first segment.
    for marker in ["\nHuman:", "\nhuman:", "\nUSER:", "\nUser:"]:
        if marker in text:
            text = text.split(marker, 1)[0]
    if "Human:" in text:
        text = text.split("Human:", 1)[0]
    return text.strip()


def _maybe_dedupe_speakers(summary: str) -> str:
    # If model starts by repeating "Speaker: ..." names, collapse to unique list once.
    # Keep it minimal and deterministic.
    lines = summary.splitlines()
    if not lines:
        return summary
    first = lines[0].strip()
    if not first.lower().startswith("speaker"):
        return summary
    # Extract comma-separated names after first colon.
    if ":" not in first:
        return summary
    _, rest = first.split(":", 1)
    names = [n.strip() for n in rest.split(",") if n.strip()]
    seen = set()
    uniq = []
    for n in names:
        k = n.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(n)
    lines[0] = "Speakers: " + ", ".join(uniq[:12])
    return "\n".join(lines).strip()


def _parse_text_prompt(text_prompt: str) -> Dict[str, Any]:
    """
    Supports either:
      1) Plain string prompt (legacy): treated as chat instruction.
      2) JSON object (string) with keys:
         - mode: "chat" | "transcribe" | "caption" | "summary"
         - prompt: string (user instruction / question)
         - format: optional string. e.g. "verbatim", "clean", "srt-like"
         - language: optional string hint, e.g. "en"
         - caption_chars: optional int (soft target chars per caption line)
    """
    s = (text_prompt or "").strip()
    if not s:
        return {"mode": "chat", "prompt": ""}

    # JSON mode (recommended for FastAPI)
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                mode = str(obj.get("mode", "chat")).strip().lower()
                prompt = obj.get("prompt", obj.get("text", ""))
                if prompt is None:
                    prompt = ""
                parsed = {
                    "mode": mode,
                    "prompt": str(prompt),
                    "system": obj.get("system"),
                    "format": obj.get("format"),
                    "language": obj.get("language"),
                    "caption_chars": obj.get("caption_chars"),
                    "return_audio": envelope_bool(obj.get("return_audio")),
                    "speaker": obj.get("speaker"),
                }
                if obj.get("parts") is not None:
                    parsed["parts"] = obj.get("parts")
                return parsed
        except Exception:
            # Fall through to legacy parsing
            pass

    # Lightweight prefix mode (human-friendly)
    # Examples:
    #   "mode:transcribe\n<instruction...>"
    #   "[transcribe]\n<instruction...>"
    lower = s.lower()
    for tag, mode in (
        ("mode:transcribe", "transcribe"),
        ("mode:caption", "caption"),
        ("mode:summary", "summary"),
        ("mode:chat", "chat"),
        ("[transcribe]", "transcribe"),
        ("[caption]", "caption"),
        ("[summary]", "summary"),
        ("[chat]", "chat"),
    ):
        if lower.startswith(tag):
            rest = s[len(tag) :].lstrip("\r\n\t :")
            return {"mode": mode, "prompt": rest}

    return {"mode": "chat", "prompt": s}


def _build_mode_instruction(spec: Dict[str, Any], has_audio: bool, has_video: bool) -> str:
    """Validate mode inputs and return only the client's instruction text."""
    mode = str(spec.get("mode", "chat") or "chat").lower()
    prompt = str(spec.get("prompt", "") or "").strip()
    if mode in AUDIO_REQUIRED_MODES and not has_audio:
        raise ValueError(f"mode '{mode}' requires audio, but AUDIO_BYTES was empty.")
    if mode in VIDEO_REQUIRED_MODES and not has_video:
        raise ValueError(f"mode '{mode}' requires video, but VIDEO_BYTES was empty.")
    return prompt


def _client_system(spec: Dict[str, Any]) -> Optional[str]:
    system = spec.get("system")
    if system is None:
        return None
    text = str(system).strip()
    return text or None


def _unlink_quietly(paths: Sequence[str]) -> None:
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass


def _b64decode(value: str, name: str) -> bytes:
    try:
        return base64.b64decode(value)
    except Exception as e:
        raise ValueError(f"Invalid base64 {name}: {type(e).__name__}: {e}") from e


def _normalize_parts(
    raw: Any, n_audio: int, n_video: int, prompt: str
) -> Tuple[Tuple[str, str], ...]:
    """Client content order, or audio, then video, then the prompt when `parts` is omitted."""
    if not raw:
        return tuple([("audio", "")] * n_audio + [("video", "")] * n_video + [("text", prompt)])
    if not isinstance(raw, list):
        raise ValueError(f"parts must be a list, got {type(raw).__name__}")
    parts: List[Tuple[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each part must be an object")
        kind = str(item.get("type") or "").strip().lower()
        if kind == "text":
            text = item.get("text")
            parts.append(("text", "" if text is None else str(text)))
        elif kind in {"audio", "video"}:
            parts.append((kind, ""))
        else:
            raise ValueError(f"unknown part type {kind!r}")
    got_audio = sum(kind == "audio" for kind, _text in parts)
    got_video = sum(kind == "video" for kind, _text in parts)
    if got_audio != n_audio or got_video != n_video:
        raise ValueError(
            f"parts describe {got_audio} audio and {got_video} video, "
            f"but the request has {n_audio} audio and {n_video} video"
        )
    return tuple(parts)


def _conversation_messages(
    spec: Dict[str, Any], audios: Sequence[Any], videos: Sequence[Any]
) -> List[Dict[str, Any]]:
    user_prompt = _build_mode_instruction(
        spec, has_audio=bool(audios), has_video=bool(videos)
    )
    messages: List[Dict[str, Any]] = []
    system = _client_system(spec)
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    if audios or videos:
        content = _assemble_user_content(
            _normalize_parts(spec.get("parts"), len(audios), len(videos), user_prompt),
            audios,
            videos,
        )
    else:
        content = [{"type": "text", "text": user_prompt}]
    messages.append({"role": "user", "content": content})
    return messages


def _wrap_caption_lines(text: str, target_chars: int) -> str:
    """
    Deterministic word-wrap for caption output. No timestamps; one caption per line.
    """
    t = (text or "").strip()
    if not t:
        return ""
    target = max(20, min(140, int(target_chars)))
    words = t.split()
    lines: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for w in words:
        add_len = len(w) if not cur else (1 + len(w))
        if cur and (cur_len + add_len) > target:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += add_len
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines).strip()


class TritonPythonModel(BaseBackend):
    supports_batching = False
    supported_modes = None
    def _pin_speech_modules(self) -> None:
        """Keep the talker and vocoder on the thinker embedding device."""
        from accelerate.hooks import remove_hook_from_module

        primary = self.omni.thinker.get_input_embeddings().weight.device
        for name in ("talker", "token2wav", "code2wav"):
            module = getattr(self.omni, name, None)
            if module is None:
                continue
            try:
                remove_hook_from_module(module, recurse=True)
            except Exception as exc:
                pb_utils.Logger.log_warn(
                    f"could not unpin {name}: {type(exc).__name__}: {exc}"
                )
            module.to(primary)
            pb_utils.Logger.log_info(f"pinned {name} to {primary}")

    def initialize(self, args: Dict[str, Any]) -> None:
        # Ensure the venv site-packages are visible.
        venv_site = os.environ.get("VENV_SITE_PACKAGES")
        if venv_site:
            import sys

            if venv_site not in sys.path:
                sys.path.insert(0, venv_site)

        import torch  # noqa: F401
        from transformers import AutoConfig

        self._torch = __import__("torch")
        placement = _load_placement(args)
        self.model_name = placement["model_name"] or "this model"
        self.context_policy = context_policy(placement["model_name"])
        self.model_id = placement["model_id"]
        device_map = placement["device_map"]
        dtype = getattr(self._torch, placement["dtype_name"])
        instance_device_id = placement["instance_device_id"]

        self.default_max_new_tokens = _env_int("MAX_NEW_TOKENS", 512)
        self.default_temperature = _env_float("TEMPERATURE", 0.0)
        self.default_do_sample = _env_bool("DO_SAMPLE", False)

        if not self._torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Triton Qwen-Omni backend.")

        config = AutoConfig.from_pretrained(self.model_id)
        family = _omni_family(getattr(config, "model_type", "") or "")
        self.omni_family = family
        processor_cls, thinker_cls, omni_cls = _import_omni_classes(family)
        self.disable_talker = _env_bool_model(
            self.model_name, "DISABLE_TALKER", "SPEECHLMM_DISABLE_TALKER", True
        )
        self.disable_code2wav = _env_bool_model(
            self.model_name, "DISABLE_CODE2WAV", "SPEECHLMM_DISABLE_CODE2WAV", True
        )
        self.speech_enabled = not self.disable_talker and not self.disable_code2wav
        self.default_speaker = (os.environ.get("SPEECHLMM_SPEAKER") or "ethan").strip() or "ethan"
        self.talker_max_new_tokens = _env_int("TALKER_MAX_NEW_TOKENS", 4096)
        self.talker_do_sample = _env_bool("TALKER_DO_SAMPLE", True)
        self.omni = None
        self.speakers: List[str] = []

        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "attn_implementation": "sdpa",
        }
        if self.speech_enabled:
            if family == "qwen2_5_omni":
                apply_qwen25_dit_rope_fix()
            self.omni = omni_cls.from_pretrained(self.model_id, **load_kwargs)
            self.model = self.omni.thinker
            if family == "qwen2_5_omni":
                self.speakers = [str(name) for name in self.omni.speaker_map]
            else:
                speaker_id = getattr(getattr(config, "talker_config", None), "speaker_id", None) or {}
                self.speakers = [str(name) for name in speaker_id]
            if not self.speakers:
                raise RuntimeError(
                    f"speech output is enabled but '{self.model_name}' has no speakers"
                )
            self._pin_speech_modules()
        else:
            # Thinker-only keeps the talker and vocoder off the GPUs.
            self.model = thinker_cls.from_pretrained(self.model_id, **load_kwargs)
        self.model.eval()
        from long_audio import install_windowed_audio
        self._long_audio = install_windowed_audio(self.model)
        self.processor = processor_cls.from_pretrained(self.model_id)
        self.eos_token_id, self.pad_token_id = _resolve_eos_pad(
            getattr(self.model, "generation_config", None),
            getattr(self.processor, "tokenizer", None),
            self.model_id,
        )
        try:
            self.device = self.model.get_input_embeddings().weight.device
            self.model_dtype = self.model.get_input_embeddings().weight.dtype
        except Exception:
            self.device = self._torch.device(f"cuda:{instance_device_id}")
            self.model_dtype = dtype

        self.context_limit = context_limit(self.model.config)
        pb_utils.Logger.log_info(
            f"Loaded {self.model_id} name={self.model_name} family={family} "
            f"dtype={placement['dtype_name']} device_map={device_map} "
            f"speech={self.speech_enabled} context={self.context_limit or 'unknown'} "
            f"overflow={self.context_policy}."
        )

        # Optional compilation (can improve throughput; adds warmup cost).
        self._compile_enabled = _env_bool("TORCH_COMPILE", False)
        if self._compile_enabled and hasattr(self._torch, "compile"):
            try:
                self.model = self._torch.compile(self.model, mode="reduce-overhead")
                pb_utils.Logger.log_info("torch.compile enabled (mode=reduce-overhead).")
            except Exception as e:
                pb_utils.Logger.log_warn(f"torch.compile failed (continuing eager): {type(e).__name__}: {e}")

        # Optional: Voice Activity Detection (skip inference on silence)
        self._vad_model = None
        self._vad_get_speech_timestamps = None
        # If values are missing/unparseable, VAD is disabled rather than guessing numbers.
        self._vad_enabled = _env_bool("VAD_ENABLED", False)
        self._vad_threshold = None
        self._vad_energy_fallback_threshold = None
        if self._vad_enabled:
            try:
                self._vad_threshold = float(os.environ["VAD_THRESHOLD"].strip())
            except Exception as e:
                pb_utils.Logger.log_warn(f"VAD disabled: invalid/missing VAD_THRESHOLD ({type(e).__name__}: {e})")
                self._vad_enabled = False
            try:
                self._vad_energy_fallback_threshold = float(os.environ["VAD_ENERGY_FALLBACK"].strip())
            except Exception as e:
                pb_utils.Logger.log_warn(
                    f"VAD disabled: invalid/missing VAD_ENERGY_FALLBACK ({type(e).__name__}: {e})"
                )
                self._vad_enabled = False
            try:
                self._vad_model, utils = self._torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    onnx=False,
                    verbose=False,
                    trust_repo=True,
                )
                self._vad_model.eval()
                (self._vad_get_speech_timestamps, _, _, *_) = utils
                pb_utils.Logger.log_info(f"VAD loaded (threshold={self._vad_threshold}).")
            except Exception as e:
                pb_utils.Logger.log_warn(f"VAD not loaded (fallback: no silence filtering): {e}")
                self._vad_model = None
                self._vad_get_speech_timestamps = None

    def _get_audio_for_vad(self, waveform: "Any", sample_rate: int):
        """
        Return a 16 kHz mono float32 tensor suitable for Silero VAD (on CPU).
        waveform: [samples] float32
        """
        VAD_SR = 16000
        w = waveform.float().cpu()
        if sample_rate == VAD_SR:
            return w, VAD_SR
        w = w.unsqueeze(0)
        # Use torchaudio if available; otherwise rely on linear interpolation.
        try:
            import torchaudio  # type: ignore

            resampled = torchaudio.functional.resample(w, sample_rate, VAD_SR)
        except Exception:
            # Fallback: crude resample using linear interpolation.
            import numpy as _np

            x = w.squeeze(0).numpy()
            if x.size < 2:
                return w.squeeze(0), VAD_SR
            new_len = int(round(x.size * (VAD_SR / float(sample_rate))))
            new_len = max(1, new_len)
            xp = _np.linspace(0.0, 1.0, num=x.size, endpoint=True)
            xq = _np.linspace(0.0, 1.0, num=new_len, endpoint=True)
            y = _np.interp(xq, xp, x).astype(_np.float32)
            resampled = self._torch.from_numpy(y).unsqueeze(0)
        return resampled.squeeze(0).cpu(), VAD_SR

    def _has_speech(self, wav_bytes: bytes) -> bool:
        """
        Return True if audio is classified as speech, False if silence.
        If VAD is disabled/unavailable, returns True (do not skip inference).
        """
        if not self._vad_enabled:
            return True
        # Required when VAD is enabled
        if self._vad_energy_fallback_threshold is None:
            return True
        try:
            import soundfile as sf  # type: ignore
            import numpy as _np
            import io as _io

            wav_data, sr = sf.read(_io.BytesIO(wav_bytes), dtype="float32")
            # Ensure 1D mono float32
            if not isinstance(wav_data, _np.ndarray):
                wav_data = _np.array(wav_data, dtype=_np.float32)
            if wav_data.ndim > 1:
                wav_data = (
                    wav_data.mean(axis=1) if wav_data.shape[0] > wav_data.shape[1] else wav_data.mean(axis=0)
                )
            wav_data = wav_data.flatten()
            if wav_data.size < int(sr * 0.25):
                return True
            waveform = self._torch.from_numpy(wav_data).float()
            audio_16k, sr_16k = self._get_audio_for_vad(waveform, int(sr))
            if audio_16k.numel() < int(sr_16k * 0.25):
                return True
            # If Silero VAD is available, use it. Otherwise fall back to energy VAD.
            if self._vad_model is not None and self._vad_get_speech_timestamps is not None:
                with self._torch.no_grad():
                    timestamps = self._vad_get_speech_timestamps(
                        audio_16k,
                        self._vad_model,
                        sampling_rate=sr_16k,
                        threshold=float(self._vad_threshold) if self._vad_threshold is not None else 0.5,
                    )
                if timestamps:
                    return True
                # If Silero returns no speech, treat as silence.
                return False

            # Energy-only fallback (very cheap, no extra deps)
            # Use PEAK, not mean: short speech bursts inside a 2s chunk can have low mean energy.
            peak = float(self._torch.max(audio_16k.abs()).item())
            return peak >= float(self._vad_energy_fallback_threshold)
        except Exception:
            return True

    def _decode_wav(self, wav_bytes: bytes) -> Tuple[np.ndarray, int]:
        return decode_wav(wav_bytes, self.processor, self._torch)

    def _decode_request_str(self, tensor: Any, idx: int) -> str:
        arr = tensor.as_numpy().reshape(-1)
        # TYPE_STRING comes as object array of bytes
        v = arr[idx]
        if isinstance(v, (bytes, bytearray)):
            return v.decode("utf-8", errors="ignore")
        return str(v)

    def _decode_request_strs(self, tensor: Any) -> List[str]:
        if tensor is None:
            return []
        arr = tensor.as_numpy().reshape(-1)
        out: List[str] = []
        for value in arr:
            if isinstance(value, (bytes, bytearray)):
                decoded = value.decode("utf-8", errors="ignore")
            else:
                decoded = str(value)
            if decoded.strip():
                out.append(decoded)
        return out

    def _write_videos(self, video_b64s: Sequence[str]) -> List[str]:
        """Spill each clip to a temp .mp4: the processor loads video by path."""
        paths: List[str] = []
        try:
            for video_b64 in video_b64s:
                blob = _b64decode(video_b64, "VIDEO_BYTES")
                tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                try:
                    tmp.write(blob)
                    tmp.flush()
                finally:
                    tmp.close()
                paths.append(tmp.name)
        except BaseException:
            _unlink_quietly(paths)
            raise
        return paths

    def _decode_generated(
        self, generated_ids: Any, prompt_len: int, eos_token_id: Any, max_new_tokens: int
    ) -> tuple[str, str]:
        # Single row on purpose: this backend stays at max_batch_size 1, so
        # returning a list of rows would be theatre.
        if generated_ids.dim() == 1:
            new_tokens = generated_ids[prompt_len:].unsqueeze(0)
        else:
            new_tokens = generated_ids[:, prompt_len:]
        text = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        text = _clean_text(text)
        tokens = new_tokens[0].tolist()
        eos_ids = set(
            []
            if eos_token_id is None
            else [eos_token_id]
            if isinstance(eos_token_id, int)
            else eos_token_id
        )
        if any(token in eos_ids for token in tokens):
            reason = "stop"
        elif len(tokens) >= max_new_tokens:
            reason = "length"
        else:
            raise RuntimeError("generation ended without EOS or token-limit exhaustion")
        self._last_output_counts = [output_token_count(tokens, eos_ids)]
        return text, reason

    def _text_response(self, text: str, finish_reason: str, audio_b64: str = "") -> Any:
        return self.build_response(Result(text, finish_reason, audio_b64))

    def map_error(self, exc: BaseException) -> Any:
        return self._pb().InferenceResponse(error=_triton_error(exc))

    def decode_request(self, index: int, request: pb_utils.InferenceRequest) -> Sample:
        audio_t = pb_utils.get_input_tensor_by_name(request, "AUDIO_BYTES")
        video_t = pb_utils.get_input_tensor_by_name(request, "VIDEO_BYTES")
        prompt_t = pb_utils.get_input_tensor_by_name(request, "TEXT_PROMPT")
        max_new_t = pb_utils.get_input_tensor_by_name(request, "MAX_NEW_TOKENS")
        do_sample_t = pb_utils.get_input_tensor_by_name(request, "DO_SAMPLE")
        rep_pen_t = pb_utils.get_input_tensor_by_name(request, "REPETITION_PENALTY")
        temp_t = pb_utils.get_input_tensor_by_name(request, "TEMPERATURE")

        spec = _parse_text_prompt(self._decode_request_str(prompt_t, 0))
        raw_speaker = spec.get("speaker")
        sample = Sample(
            index=index,
            mode=str(spec.get("mode", "chat") or "chat").lower(),
            spec=spec,
            prompt=str(spec.get("prompt") or ""),
            max_new_tokens=(
                int(max_new_t.as_numpy().reshape(-1)[0]) if max_new_t is not None else self.default_max_new_tokens
            ),
            do_sample=bool(do_sample_t.as_numpy().reshape(-1)[0]) if do_sample_t is not None else self.default_do_sample,
            temperature=float(temp_t.as_numpy().reshape(-1)[0]) if temp_t is not None else self.default_temperature,
            repetition_penalty=float(rep_pen_t.as_numpy().reshape(-1)[0]) if rep_pen_t is not None else 1.0,
            return_audio=envelope_bool(spec.get("return_audio")),
            speaker=str(raw_speaker).strip() if raw_speaker else None,
        )
        sample.extra["audio_b64s"] = self._decode_request_strs(audio_t)
        sample.extra["video_b64s"] = self._decode_request_strs(video_t)
        sample.extra["video_paths"] = []
        return sample

    def validate_extra(self, sample: Sample) -> None:
        _reject_unsupported_request(sample.mode, sample.prompt, self.model_name)
        has_lipread = sample.mode == "lipread" or LIPREAD_PLACEHOLDER in sample.prompt
        validate_speech_request(
            sample.mode,
            sample.return_audio,
            bool(sample.extra["audio_b64s"]),
            bool(sample.extra["video_b64s"]),
            has_lipread,
            sample.prompt,
            getattr(self, "speech_enabled", False),
        )

    def prepare(self, sample: Sample) -> None:
        spec = sample.spec
        if sample.return_audio and getattr(self, "omni_family", "") == "qwen2_5_omni":
            client_system = _client_system(spec)
            if not client_system:
                spec["system"] = QWEN25_AUDIO_SYSTEM_PROMPT
            elif client_system != QWEN25_AUDIO_SYSTEM_PROMPT:
                warn_custom_audio_system(self, pb_utils.Logger.log_warn)
        caption_target = 64
        if sample.mode == "caption":
            try:
                if spec.get("caption_chars") is not None:
                    caption_target = int(spec.get("caption_chars"))
            except Exception:
                caption_target = 64
        sample.extra["caption_target"] = caption_target
        sample.audio = [_b64decode(b64, "AUDIO_BYTES") for b64 in sample.extra["audio_b64s"]]

    def short_circuit(self, sample: Sample) -> Optional[Result]:
        # Skip inference on silence (VAD). Single clip only: with several,
        # silence in one says nothing about the request as a whole.
        if (
            not sample.return_audio
            and len(sample.audio) == 1
            and not sample.extra["video_b64s"]
            and sample.mode in SILENCE_CHECK_MODES
            and not self._has_speech(sample.audio[0])
        ):
            return Result("", "stop", "", {"skipped": "silence", "output_tokens": 0})
        return None

    def postprocess(self, sample: Sample, text: str) -> str:
        # Only apply speaker de-dupe on summaries; it can harm verbatim transcripts.
        if sample.mode == "summary":
            text = _maybe_dedupe_speakers(text)
        if sample.mode == "caption":
            text = _wrap_caption_lines(text, sample.extra["caption_target"])
        return text

    def cleanup(self, sample: Sample) -> None:
        _unlink_quietly(sample.extra.get("video_paths") or [])

    def generate(self, samples: List[Sample]) -> List[Result]:
        return [self._generate_one(samples[0])]

    def _generate_one(self, sample: Sample) -> Result:
        wavs = [self._decode_wav(blob)[0] for blob in sample.audio]
        video_paths = self._write_videos(sample.extra["video_b64s"])
        sample.extra["video_paths"] = video_paths
        conversations = _conversation_messages(sample.spec, wavs, video_paths)

        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            text_kwargs={"padding": False},
            audio_kwargs={"truncation": False},
        )
        inputs = inputs.to(self.device)
        inputs = {
            key: (
                value.to(self.model_dtype)
                if self._torch.is_tensor(value) and self._torch.is_floating_point(value)
                else value
            )
            for key, value in inputs.items()
        }

        prompt_len = int(inputs["input_ids"].shape[1])
        overflow = check_context(self, prompt_len, sample.max_new_tokens)
        if sample.return_audio:
            speech_kwargs = speech_generate_kwargs(
                sample,
                resolve_speaker(sample.speaker or self.default_speaker, self.speakers),
                self.talker_max_new_tokens,
                self.talker_do_sample,
                self.eos_token_id,
            )
            started = time.perf_counter()
            generated = self.omni.generate(**inputs, **speech_kwargs)
            generate_time_s = time.perf_counter() - started
            if not isinstance(generated, tuple) or len(generated) != 2:
                raise RuntimeError("speech generation did not return text and audio")
            sequences, wav = generated
            text, finish_reason = self._decode_generated(
                sequences,
                prompt_len,
                speech_kwargs.get("thinker_eos_token_id"),
                speech_kwargs["thinker_max_new_tokens"],
            )
            audio_b64 = encode_wav_b64(wav)
        else:
            gen_kwargs: Dict[str, Any] = {
                "use_cache": True,
                "max_new_tokens": sample.max_new_tokens,
                "do_sample": sample.do_sample,
                "repetition_penalty": sample.repetition_penalty,
            }
            if sample.do_sample:
                gen_kwargs["temperature"] = sample.temperature
            if self.eos_token_id is not None:
                gen_kwargs["eos_token_id"] = self.eos_token_id
            if self.pad_token_id is not None:
                gen_kwargs["pad_token_id"] = self.pad_token_id

            from long_audio import generation_logits
            started = time.perf_counter()
            with generation_logits(self.model, getattr(self, "_long_audio", False)):
                text_ids = self.model.generate(**inputs, **gen_kwargs)
            generate_time_s = time.perf_counter() - started
            text, finish_reason = self._decode_generated(
                text_ids, prompt_len, gen_kwargs.get("eos_token_id"), sample.max_new_tokens
            )
            audio_b64 = ""
        usage = stash_generate_usage(self, inputs, prompt_len, generate_time_s)
        row = dict(usage[0]) if usage else {}
        if overflow:
            row["context_overflow"] = True
        return Result(text, finish_reason, audio_b64, row)
