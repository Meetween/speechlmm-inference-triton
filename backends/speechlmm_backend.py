"""Triton Python backend for SpeechLMM v2 (native speechlmm checkpoint).

Same HTTP tensor names as qwen25_omni / voxtral, plus optional VIDEO_BYTES
(base64 MP4). `lipread` mode sends that video through the mouth-ROI encoder;
every other mode with video uses the Qwen3-Omni vision tower. Loads
SpeechLMMForConditionalGeneration via trust_remote_code + the checkpoint's auto_map.
Talker and code2wav stay off unless DISABLE_TALKER and DISABLE_CODE2WAV are both 0.
Speech output then goes through the checkpoint's rebuilt Omni shell.

One request per inference: a batch dimension above 1 is rejected, and Triton's
dynamic batcher is what groups requests together.
"""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
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
    check_context,
    context_limit,
    context_policy,
    decode_wav,
)
from base_backend import BaseBackend, Result
from telemetry import output_token_count, stash_generate_usage
from speech_output import (
    QWEN25_AUDIO_SYSTEM_PROMPT,
    encode_wav_b64,
    envelope_bool,
    resolve_speaker,
    speech_generate_kwargs,
    validate_speech_request,
    warn_custom_audio_system,
)

LIPREAD_PLACEHOLDER = "<lipread>"
LIPREAD_BOS_TOKEN = "<|lipread_start|>"
LIPREAD_PAD_TOKEN = "<|lipread_pad|>"
LIPREAD_EOS_TOKEN = "<|lipread_end|>"
LIPREAD_H = 88
LIPREAD_W = 88
LIPREAD_FPS = 25
LIPREAD_MEAN = [0.421]
LIPREAD_STD = [0.165]

AUDIO_REQUIRED_MODES = {"transcribe", "caption", "summary", "st", "s2st"}
VIDEO_REQUIRED_MODES = {"lipread", "avspeakd"}


def _uses_lipread_encoder(mode: str) -> bool:
    return (mode or "").strip().lower() == "lipread"


def _debug_mem_enabled() -> bool:
    """Opt-in: shape and allocator lines are a diagnostic, not a serving default."""

    return os.environ.get("SPEECHLMM_DEBUG_MEM", "0").strip().lower() not in {"", "0", "false", "no"}


def _shape_of(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return type(value).__name__
    return "x".join(str(int(d)) for d in shape)


def _grid_summary(grid: Any) -> str:
    """Summarise a t/h/w patch grid: raw patches and post-merge LLM tokens."""

    try:
        rows = grid.tolist()
    except Exception:
        return _shape_of(grid)
    total = 0
    parts = []
    for row in rows:
        t, h, w = (int(x) for x in row[:3])
        total += t * h * w
        parts.append(f"t={t},h={h},w={w}")
    # spatial_merge_size=2 -> 2x2 patches collapse into one LLM token.
    return f"{' '.join(parts)} patches={total} llm_tokens~{total // 4}"


def _log_request_shapes(samples: Sequence[Any], per_inputs: Sequence[Dict[str, Any]]) -> None:
    """One line per request: what the processor actually produced."""

    if not _debug_mem_enabled():
        return
    for sample, inputs in zip(samples, per_inputs):
        parts = [f"mode={getattr(sample, 'mode', '?')}"]
        for key in (
            "input_ids",
            "input_features",
            "feature_attention_mask",
            "pixel_values",
            "pixel_values_videos",
            "lipread_values",
        ):
            value = inputs.get(key)
            if value is not None:
                parts.append(f"{key}={_shape_of(value)}")
        for key in ("image_grid_thw", "video_grid_thw"):
            value = inputs.get(key)
            if value is not None:
                parts.append(f"{key}[{_grid_summary(value)}]")
        pb_utils.Logger.log_info("SpeechLMM v2 shapes: " + " ".join(parts))


def _log_cuda_memory(tag: str, *, warn: bool = False) -> None:
    """Allocated/reserved/peak per visible GPU, on one line."""

    if not _debug_mem_enabled():
        return
    try:
        import torch

        parts = []
        for i in range(torch.cuda.device_count()):
            parts.append(
                "gpu%d a=%.2f r=%.2f peak=%.2f"
                % (
                    i,
                    torch.cuda.memory_allocated(i) / 2 ** 30,
                    torch.cuda.memory_reserved(i) / 2 ** 30,
                    torch.cuda.max_memory_allocated(i) / 2 ** 30,
                )
            )
        line = f"SpeechLMM v2 cuda[{tag}] GiB: " + " | ".join(parts)
    except Exception as error:
        line = f"SpeechLMM v2 cuda[{tag}] probe failed: {type(error).__name__}: {error}"
    (pb_utils.Logger.log_warn if warn else pb_utils.Logger.log_info)(line)


def _reset_peak_memory() -> None:
    """Make the next ``peak=`` reading per-request. Only meaningful when logging."""

    if not _debug_mem_enabled():
        return
    try:
        import torch

        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
    except Exception:
        pass


def _log_cuda_summary() -> None:
    """Full allocator breakdown; only on OOM, it is verbose."""

    if not _debug_mem_enabled():
        return
    try:
        import torch

        for i in range(torch.cuda.device_count()):
            pb_utils.Logger.log_warn(
                f"SpeechLMM v2 memory_summary gpu{i}:\n"
                + torch.cuda.memory_summary(device=i, abbreviated=True)
            )
    except Exception as error:
        pb_utils.Logger.log_warn(
            f"SpeechLMM v2 memory_summary failed: {type(error).__name__}: {error}"
        )


def _log_attn_implementation(config: Any) -> None:
    """Report the attention kernel each sub-encoder ended up with."""

    parts = [f"top={getattr(config, '_attn_implementation', None)}"]
    thinker = getattr(config, "thinker_config", None)
    for name in ("text_config", "audio_config", "vision_config"):
        sub = getattr(thinker, name, None) if thinker is not None else None
        if sub is not None:
            parts.append(f"{name.split('_')[0]}={getattr(sub, '_attn_implementation', None)}")
    pb_utils.Logger.log_info("SpeechLMM v2 attn_implementation: " + " ".join(parts))


def _resolve_device_map(raw: str, instance_device_id: int) -> str:
    device_map = (raw or "").strip() or "auto"
    if device_map in {"auto", "balanced", "balanced_low_0", "sequential", "cpu"}:
        return device_map
    if device_map == "cuda":
        return f"cuda:{instance_device_id}"
    if device_map.startswith("cuda:"):
        return device_map
    raise ValueError(
        f"unsupported device_map '{device_map}'; expected auto, balanced, balanced_low_0, "
        "sequential, cpu, cuda, or cuda:<id>."
    )


def _parse_text_prompt(text_prompt: str) -> Dict[str, Any]:
    """Plain string (legacy chat) or JSON with mode/prompt/format/language."""
    s = (text_prompt or "").strip()
    if not s:
        return {"mode": "chat", "prompt": ""}

    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"TEXT_PROMPT looks like JSON but does not parse: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError(f"TEXT_PROMPT JSON must be an object, got {type(obj).__name__}.")
        prompt = obj.get("prompt", obj.get("text", ""))
        parsed = {
            "mode": str(obj.get("mode", "chat")).strip().lower(),
            "prompt": "" if prompt is None else str(prompt),
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

    lower = s.lower()
    for tag, mode in (
        ("mode:transcribe", "transcribe"),
        ("mode:caption", "caption"),
        ("mode:summary", "summary"),
        ("mode:text_summary", "text_summary"),
        ("mode:lipread", "lipread"),
        ("mode:avspeakd", "avspeakd"),
        ("mode:chat", "chat"),
        ("[transcribe]", "transcribe"),
        ("[caption]", "caption"),
        ("[summary]", "summary"),
        ("[text_summary]", "text_summary"),
        ("[lipread]", "lipread"),
        ("[avspeakd]", "avspeakd"),
        ("[chat]", "chat"),
    ):
        if lower.startswith(tag):
            rest = s[len(tag) :].lstrip("\r\n\t :")
            return {"mode": mode, "prompt": rest}

    return {"mode": "chat", "prompt": s}


def _normalize_parts(
    raw: Any, n_audio: int, n_video: int, prompt: str
) -> Tuple[Tuple[str, str], ...]:
    """Client content order, or audio, then video, then the prompt when `parts` is omitted.

    Each entry is ``(kind, text)``. Media text is empty; the Nth audio/video maps to
    that queue. Text is kept verbatim, including whitespace next to a clip.
    """
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


DEFAULT_VIDEO_FPS = 2.0


def _video_sampling_kwargs() -> Dict[str, Any]:
    """Frame sampling for the general vision tower.

    These go at the TOP level of apply_chat_template. Nesting them under
    ``video_kwargs`` is silently ignored, and the tower then keeps every
    decoded frame: a 52 s clip at a native 25 fps becomes t=658, 579k patches
    and ~145k LLM tokens, past the 65536-token context. fps=2 matches
    vision_config.tokens_per_second and brings the same clip to t=52 and
    ~11.5k tokens.

    fps and num_frames are mutually exclusive in sample_frames(), so
    SPEECHLMM_VIDEO_NUM_FRAMES wins when set: a fixed budget whatever the
    duration, for clips too long to sample at a fixed rate.
    SPEECHLMM_VIDEO_FPS=0 with no NUM_FRAMES restores take-every-frame.

    Lipread video never reaches here; it goes to the dedicated encoder.
    """

    raw_frames = os.environ.get("SPEECHLMM_VIDEO_NUM_FRAMES", "").strip()
    if raw_frames:
        try:
            num_frames = int(raw_frames)
        except ValueError:
            pb_utils.Logger.log_warn(
                f"Ignoring SPEECHLMM_VIDEO_NUM_FRAMES={raw_frames!r}; not an integer"
            )
        else:
            if num_frames > 0:
                return {"do_sample_frames": True, "num_frames": num_frames}

    raw_fps = os.environ.get("SPEECHLMM_VIDEO_FPS", "").strip()
    try:
        fps = float(raw_fps) if raw_fps else DEFAULT_VIDEO_FPS
    except ValueError:
        pb_utils.Logger.log_warn(
            f"Ignoring SPEECHLMM_VIDEO_FPS={raw_fps!r}; using {DEFAULT_VIDEO_FPS}"
        )
        fps = DEFAULT_VIDEO_FPS
    if fps <= 0:
        return {}
    return {"do_sample_frames": True, "fps": fps}


def _processor_tokenize_kwargs() -> Dict[str, Any]:
    # Flat padding=False is copied into audio_kwargs; Whisper then cannot stack
    # two different-length clips (numpy "axes don't match array").
    kwargs: Dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "text_kwargs": {"padding": False},
        "audio_kwargs": {"truncation": False},
    }
    kwargs.update(_video_sampling_kwargs())
    return kwargs


def _caption_target(caption_chars: Any) -> int:
    """Characters per caption line, clamped to what the wrapper can honour."""
    try:
        target = 64 if caption_chars is None else int(caption_chars)
    except (TypeError, ValueError):
        target = 64
    return max(32, min(120, target))


def _build_mode_instruction(spec: Dict[str, Any], has_audio: bool, has_video: bool) -> str:
    """Validate mode inputs and return only the client's instruction text."""
    mode = str(spec.get("mode", "chat") or "chat").lower()
    prompt = str(spec.get("prompt", "") or "").strip()
    if mode in AUDIO_REQUIRED_MODES and not has_audio:
        raise ValueError(f"mode '{mode}' requires audio, but AUDIO_BYTES was empty.")
    if mode in VIDEO_REQUIRED_MODES and not has_video:
        raise ValueError(f"mode '{mode}' requires video, but VIDEO_BYTES was empty.")
    if mode == "text_summary" and not prompt:
        raise ValueError(
            "mode 'text_summary' requires text to summarize, but the prompt was empty."
        )
    return prompt


def _wrap_caption_lines(text: str, target: int) -> str:
    t = (text or "").strip()
    if not t:
        return ""
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


def _lipread_token_string(n_frames: int) -> str:
    return LIPREAD_BOS_TOKEN + (LIPREAD_PAD_TOKEN * n_frames) + LIPREAD_EOS_TOKEN


def _expand_lipread_placeholder(text: str, n_frames: int) -> str:
    token_str = _lipread_token_string(n_frames)
    if LIPREAD_PLACEHOLDER in text:
        return text.replace(LIPREAD_PLACEHOLDER, token_str, 1)
    return token_str + text


def _place_lipread_tokens(
    parts: Sequence[Tuple[str, str]], n_frames: int
) -> Tuple[Tuple[str, str], ...]:
    """Expand `<lipread>` in place, or put the tokens where the video part sat."""
    token_str = _lipread_token_string(n_frames)
    has_placeholder = any(kind == "text" and LIPREAD_PLACEHOLDER in text for kind, text in parts)
    placed: List[Tuple[str, str]] = []
    expanded = False
    for kind, text in parts:
        if kind == "video":
            if not has_placeholder and not expanded:
                placed.append(("text", token_str))
                expanded = True
            continue
        if kind == "text" and not expanded and LIPREAD_PLACEHOLDER in text:
            placed.append(("text", text.replace(LIPREAD_PLACEHOLDER, token_str, 1)))
            expanded = True
            continue
        placed.append((kind, text))
    return tuple(placed)


def _thinker_model_type(config: Any) -> str:
    thinker = getattr(config, "thinker_config", None)
    if thinker is None:
        return ""
    if isinstance(thinker, dict):
        return str(thinker.get("model_type", "") or "")
    return str(getattr(thinker, "model_type", "") or "")


def _is_qwen25(config: Any) -> bool:
    backbone = str(getattr(config, "backbone_type", "") or "")
    return "qwen2_5" in backbone or "qwen2_5" in _thinker_model_type(config)


@dataclass
class LipreadProcessor:
    """Mouth-ROI video -> [T, 1, 88, 88] grayscale at 25 fps."""

    H: int = LIPREAD_H
    W: int = LIPREAD_W
    fps: int = LIPREAD_FPS
    transforms: Any = field(default=None)

    def __post_init__(self) -> None:
        from torchvision.transforms import v2

        self.transforms = v2.Compose(
            [
                v2.Grayscale(),
                v2.ToTensor(),
                v2.ToDtype(__import__("torch").float, scale=True),
                v2.Normalize(mean=LIPREAD_MEAN, std=LIPREAD_STD),
            ]
        )

    def __call__(self, video_path: str) -> "Any":
        from torchcodec.decoders import VideoDecoder
        from torchcodec.samplers import clips_at_regular_timestamps
        from torchcodec.transforms import Resize

        video_dec = VideoDecoder(video_path, transforms=[Resize((self.H, self.W))])
        video = clips_at_regular_timestamps(
            video_dec, seconds_between_clip_starts=1 / self.fps
        )
        video = self.transforms(video.data)
        return video[:, 0, :, :]


@dataclass
class _DecodedSample:
    """One Triton request after media decode, before collate/generate."""

    index: int
    mode: str
    user_prompt: str
    system_prompt: str
    use_system: bool
    wav_data: List[np.ndarray]
    lipread_video: Any
    vision_video_bytes: List[bytes]
    parts: Tuple[Tuple[str, str], ...]
    max_new_tokens: int
    do_sample: bool
    temperature: float
    repetition_penalty: float
    caption_target: int
    return_audio: bool = False
    speaker: Optional[str] = None

    def group_key(self) -> Tuple[Any, ...]:
        return (
            self.mode,
            len(self.wav_data),
            self.lipread_video is not None,
            len(self.vision_video_bytes),
            tuple(kind for kind, _text in self.parts),
            self.max_new_tokens,
            self.do_sample,
            self.temperature,
            self.repetition_penalty,
            self.return_audio,
            self.speaker,
        )


class TritonPythonModel(BaseBackend):
    supports_batching = True
    supported_modes = None
    @staticmethod
    def _preload_ffmpeg() -> None:
        """dlopen vendored FFmpeg so torchcodec works even if LD_LIBRARY_PATH was dropped."""
        import ctypes
        import sysconfig

        roots = [
            os.environ.get("FFMPEG_LIB_DIR", "/opt/ffmpeg/lib"),
            os.path.join(sysconfig.get_paths()["purelib"], "av.libs"),
        ]
        names = (
            "libavutil.so.60",
            "libavcodec.so.62",
            "libavformat.so.62",
            "libavdevice.so.62",
            "libavfilter.so.11",
            "libswscale.so.9",
            "libswresample.so.6",
        )
        for root in roots:
            if not os.path.isdir(root):
                continue
            for name in names:
                path = os.path.join(root, name)
                if os.path.exists(path):
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

    def _pin_speech_modules(self) -> None:
        """Keep the talker and vocoder on the embedding device.

        device_map=auto can park them on another GPU. Official generate() then
        mixes embedding-device hidden states with those weights.
        """
        from accelerate.hooks import remove_hook_from_module

        primary = self._primary_device()
        seen: set[int] = set()
        for owner in (self.model, getattr(self.model, "omni", None)):
            if owner is None:
                continue
            for name in ("talker", "code2wav", "token2wav"):
                module = getattr(owner, name, None)
                if module is None or id(module) in seen:
                    continue
                seen.add(id(module))
                try:
                    remove_hook_from_module(module, recurse=True)
                except Exception as exc:
                    pb_utils.Logger.log_warn(
                        f"could not unpin {name}: {type(exc).__name__}: {exc}"
                    )
                module.to(primary)
                pb_utils.Logger.log_info(f"pinned {name} to {primary}")

    def _pin_lipread_modules(self) -> None:
        """Keep the lipread encoder and adapter on the embedding device.

        Under a sharded load the root AlignDevicesHook moves the clip to the
        main device -- dispatch_model sets execution_device[""] = main_device,
        so the hook is not a no-op and the staging in _attach_lipread() does
        not survive. AutoAVSR.encode then builds its pad mask on the clip's
        device and multiplies it with encoder output from the encoder's GPU,
        which is the operation that reports two devices. Root cause and the
        reading of accelerate that establishes it: issue #48.

        Skipped when anything is offloaded: those weights are on meta, and
        module.to() would fail the whole load rather than one request.
        """
        from accelerate.hooks import remove_hook_from_module

        device_map = getattr(self.model, "hf_device_map", None) or {}
        if {str(device) for device in device_map.values()} & {"cpu", "disk"}:
            pb_utils.Logger.log_warn(
                "lipread modules left unpinned: the model is offloaded to cpu/disk"
            )
            return

        primary = self._primary_device()
        for name in ("lipread_encoder", "lipread_adapter"):
            module = getattr(self.model, name, None)
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

    def _require_speech_modules(self, config: Any) -> None:
        if getattr(self.model, "omni", None) is None:
            raise RuntimeError(
                "speech output is enabled but this checkpoint has no model.omni; "
                "re-export it with current speechlmm-v2 (scripts/export_checkpoint.py)"
            )
        if _is_qwen25(config) and not (getattr(self.model, "speaker_map", None) or {}):
            raise RuntimeError(
                "speech output is enabled but the Qwen2.5 speaker_map is empty; "
                "re-export with current speechlmm-v2 so the checkpoint carries spk_dict.pt"
            )

    def _uses_qwen25_audio_system(self) -> bool:
        if not getattr(self, "speech_enabled", False):
            return False
        return _is_qwen25(getattr(getattr(self, "model", None), "config", None))

    def _speaker_names(self) -> List[str]:
        names = list((getattr(self.model, "speaker_map", None) or {}).keys())
        if names:
            return [str(name) for name in names]
        talker_cfg = getattr(getattr(self.model, "config", None), "talker_config", None)
        speaker_id = getattr(talker_cfg, "speaker_id", None) if talker_cfg is not None else None
        if isinstance(speaker_id, dict):
            return [str(name) for name in speaker_id]
        return []

    def initialize(self, args: Dict[str, Any]) -> None:
        self._preload_ffmpeg()

        import torch
        from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer
        import transformers.dynamic_module_utils as _dmu

        # Dead-code imports in remote checkpoints (espnet, speechlmm.tokens) fail
        # the static scanner; LLaMA-Factory uses the same override.
        _dmu.check_imports = _dmu.get_relative_imports

        self._torch = torch
        self.default_max_new_tokens = _env_int("MAX_NEW_TOKENS", 512)
        self.default_temperature = _env_float("TEMPERATURE", 0.0)
        self.default_do_sample = _env_bool("DO_SAMPLE", False)
        model_name = str(args.get("model_name", "")).strip()
        self.model_name = model_name
        self.context_policy = context_policy(model_name)
        self.disable_talker = _env_bool_model(
            model_name, "DISABLE_TALKER", "SPEECHLMM_DISABLE_TALKER", True
        )
        self.disable_code2wav = _env_bool_model(
            model_name, "DISABLE_CODE2WAV", "SPEECHLMM_DISABLE_CODE2WAV", True
        )
        self.speech_enabled = not self.disable_talker and not self.disable_code2wav
        self.default_speaker = (os.environ.get("SPEECHLMM_SPEAKER") or "ethan").strip() or "ethan"
        self.talker_max_new_tokens = _env_int("TALKER_MAX_NEW_TOKENS", 4096)
        self.talker_do_sample = _env_bool("TALKER_DO_SAMPLE", True)
        try:
            model_config = json.loads(args.get("model_config") or "{}")
        except json.JSONDecodeError:
            model_config = {}

        self.model_path = (
            _model_env(model_name, "MODEL_PATH")
            or os.environ.get("SPEECHLMM_MODEL_PATH", "").strip()
        )
        if not self.model_path:
            raise RuntimeError(
                f"{model_name.upper()}_MODEL_PATH or SPEECHLMM_MODEL_PATH is required "
                "(SpeechLMM v2 checkpoint directory)."
            )
        if not os.path.isdir(self.model_path):
            raise RuntimeError(f"checkpoint path does not exist: {self.model_path}")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the SpeechLMM v2 Triton backend.")

        instance_device_id = int(args.get("model_instance_device_id", 0))
        device_map = _resolve_device_map(
            _model_env(model_name, "DEVICE_MAP")
            or _config_param(model_config, "device_map"),
            instance_device_id,
        )

        attn_impl = _model_env(model_name, "ATTN_IMPL") or None

        pb_utils.Logger.log_info(
            f"Loading SpeechLMM v2 name={model_name} from {self.model_path} "
            f"device_map={device_map} attn_impl={attn_impl or '(transformers default)'}"
        )

        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        if self.disable_talker:
            config.enable_talker = False
        if self.disable_code2wav:
            config.enable_code2wav = False

        load_kwargs: Dict[str, Any] = {}
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl

        self.model = AutoModel.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            **load_kwargs,
        )
        self.model.eval()
        _log_attn_implementation(self.model.config)
        self.context_limit = context_limit(self.model.config)
        pb_utils.Logger.log_info(
            f"context limit {self.context_limit or 'unknown'} tokens, "
            f"overflow policy {self.context_policy}"
        )
        if self.speech_enabled:
            self._require_speech_modules(config)
            self._pin_speech_modules()
        self._pin_lipread_modules()
        from long_audio import install_windowed_audio
        self._long_audio = install_windowed_audio(self.model)

        self.processor = self._load_processor(self.model_path, config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.lipread_processor = LipreadProcessor()
        self._thinker_mt = _thinker_model_type(config)

        if getattr(self.model, "lipread_encoder", None) is not None:
            vocab = self.tokenizer.get_vocab()
            missing = [
                token
                for token in (LIPREAD_BOS_TOKEN, LIPREAD_PAD_TOKEN, LIPREAD_EOS_TOKEN)
                if token not in vocab
            ]
            if missing:
                raise RuntimeError(
                    f"checkpoint has a lipread encoder but its tokenizer is missing {missing}; "
                    "video requests would be tokenized as plain text."
                )

        pb_utils.Logger.log_info(
            f"Loaded SpeechLMM v2 thinker={self._thinker_mt} "
            f"talker={getattr(config, 'enable_talker', None)} "
            f"code2wav={getattr(config, 'enable_code2wav', None)} "
            f"speech={self.speech_enabled}"
        )

    def _load_processor(self, model_path: str, config: Any) -> Any:
        from transformers import AutoProcessor

        try:
            return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            pb_utils.Logger.log_warn(
                f"AutoProcessor failed ({type(e).__name__}: {e}); selecting by thinker model_type"
            )
        thinker_mt = _thinker_model_type(config).lower()
        if "qwen2_5" in thinker_mt or "qwen2.5" in thinker_mt:
            from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import (
                Qwen2_5OmniProcessor,
            )

            return Qwen2_5OmniProcessor.from_pretrained(model_path)
        if "qwen3_omni" in thinker_mt:
            from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
                Qwen3OmniMoeProcessor,
            )

            return Qwen3OmniMoeProcessor.from_pretrained(model_path)
        raise RuntimeError(
            f"cannot select a processor: unrecognised thinker model_type '{thinker_mt}'."
        )

    def _primary_device(self) -> "Any":
        try:
            emb = self.model.get_input_embeddings()
            return emb.weight.device
        except Exception:
            return next(p.device for p in self.model.parameters() if p.device.type != "meta")

    def _lipread_device(self) -> "Any":
        encoder = getattr(self.model, "lipread_encoder", None)
        if encoder is not None:
            try:
                return next(encoder.parameters()).device
            except StopIteration:
                pass
        return self._primary_device()

    def _vision_device(self) -> "Any":
        visual = getattr(self.model, "visual", None)
        if visual is not None:
            try:
                return next(visual.parameters()).device
            except StopIteration:
                pass
        return self._primary_device()

    def _single_element(self, tensor: Any, name: str) -> Any:
        arr = tensor.as_numpy().reshape(-1)
        if arr.size != 1:
            raise ValueError(
                f"{name} carries {arr.size} elements; this backend serves one request per "
                "inference, so the batch dimension must be 1. Send separate requests and let "
                "Triton's dynamic batcher group them."
            )
        return arr[0]

    def _decode_request_str(self, tensor: Any, name: str) -> str:
        if tensor is None:
            return ""
        value = self._single_element(tensor, name)
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", errors="ignore")
        return str(value)

    def _decode_request_strs(self, tensor: Any, name: str) -> List[str]:
        if tensor is None:
            return []
        arr = tensor.as_numpy()
        if arr.ndim <= 1:
            values = arr.reshape(-1)
        elif arr.ndim == 2:
            if arr.shape[0] != 1:
                raise ValueError(
                    f"{name} carries batch {arr.shape[0]}; this backend serves one request per "
                    "inference, so the batch dimension must be 1. Send separate requests and let "
                    "Triton's dynamic batcher group them."
                )
            values = arr[0]
        else:
            raise ValueError(f"{name} has rank {arr.ndim}; expected 1 or 2")
        out: List[str] = []
        for value in values:
            if isinstance(value, (bytes, bytearray)):
                decoded = value.decode("utf-8", errors="ignore")
            else:
                decoded = str(value)
            if decoded.strip():
                out.append(decoded)
        return out

    def _decode_wav(self, wav_bytes: bytes) -> Tuple[np.ndarray, int]:
        return decode_wav(wav_bytes, self.processor, self._torch)

    def _process_lipread(self, video_bytes: bytes) -> "Any":
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            tmp.write(video_bytes)
            tmp.flush()
            return self.lipread_processor(tmp.name)

    def _to_device_dict(self, inputs: Any, device: "Any") -> Dict[str, Any]:
        if hasattr(inputs, "items"):
            items = dict(inputs)
        elif self._torch.is_tensor(inputs):
            tensor = inputs if inputs.dim() == 2 else inputs.unsqueeze(0)
            items = {"input_ids": tensor}
        else:
            raise TypeError(f"Unexpected tokenizer/processor output type: {type(inputs)}")
        out: Dict[str, Any] = {}
        for key, value in items.items():
            if self._torch.is_tensor(value):
                out[key] = value.to(device)
            else:
                out[key] = value
        if "input_ids" in out and out["input_ids"].dim() == 1:
            out["input_ids"] = out["input_ids"].unsqueeze(0)
        return out

    def _scalar_int(self, tensor: Any, default: int, name: str) -> int:
        if tensor is None:
            return default
        return int(self._single_element(tensor, name))

    def _scalar_bool(self, tensor: Any, default: bool, name: str) -> bool:
        if tensor is None:
            return default
        return bool(self._single_element(tensor, name))

    def _scalar_float(self, tensor: Any, default: float, name: str) -> float:
        if tensor is None:
            return default
        return float(self._single_element(tensor, name))

    def _decode_sample(self, index: int, request: pb_utils.InferenceRequest) -> _DecodedSample:
        audio_t = pb_utils.get_input_tensor_by_name(request, "AUDIO_BYTES")
        video_t = pb_utils.get_input_tensor_by_name(request, "VIDEO_BYTES")
        prompt_t = pb_utils.get_input_tensor_by_name(request, "TEXT_PROMPT")
        max_new_tokens = self._scalar_int(
            pb_utils.get_input_tensor_by_name(request, "MAX_NEW_TOKENS"),
            self.default_max_new_tokens,
            "MAX_NEW_TOKENS",
        )
        do_sample = self._scalar_bool(
            pb_utils.get_input_tensor_by_name(request, "DO_SAMPLE"),
            self.default_do_sample,
            "DO_SAMPLE",
        )
        temperature = self._scalar_float(
            pb_utils.get_input_tensor_by_name(request, "TEMPERATURE"),
            self.default_temperature,
            "TEMPERATURE",
        )
        repetition_penalty = self._scalar_float(
            pb_utils.get_input_tensor_by_name(request, "REPETITION_PENALTY"),
            1.0,
            "REPETITION_PENALTY",
        )
        raw_prompt = self._decode_request_str(prompt_t, "TEXT_PROMPT")
        audio_b64s = self._decode_request_strs(audio_t, "AUDIO_BYTES")
        video_b64s = self._decode_request_strs(video_t, "VIDEO_BYTES")

        spec = _parse_text_prompt(raw_prompt)
        mode = str(spec.get("mode", "chat") or "chat").lower()
        has_audio = bool(audio_b64s)
        has_video = bool(video_b64s)
        user_prompt = _build_mode_instruction(spec, has_audio=has_audio, has_video=has_video)
        return_audio = envelope_bool(spec.get("return_audio"))
        raw_speaker = spec.get("speaker")
        speaker = str(raw_speaker).strip() if raw_speaker else None
        speaker = speaker or None
        has_lipread = _uses_lipread_encoder(mode) or LIPREAD_PLACEHOLDER in str(spec.get("prompt") or "")
        validate_speech_request(
            mode,
            return_audio,
            has_audio,
            has_video,
            has_lipread,
            str(spec.get("prompt") or ""),
            getattr(self, "speech_enabled", False),
        )
        client_system = str(spec.get("system") or "").strip()
        if return_audio and self._uses_qwen25_audio_system():
            if not client_system:
                client_system = QWEN25_AUDIO_SYSTEM_PROMPT
            elif client_system != QWEN25_AUDIO_SYSTEM_PROMPT:
                warn_custom_audio_system(self, pb_utils.Logger.log_warn)
        use_system = bool(client_system)
        system_prompt = client_system
        caption_target = _caption_target(spec.get("caption_chars"))
        parts = _normalize_parts(
            spec.get("parts"), len(audio_b64s), len(video_b64s), user_prompt
        )

        wav_data: List[np.ndarray] = []
        for audio_b64 in audio_b64s:
            try:
                audio_bytes = base64.b64decode(audio_b64)
            except Exception as e:
                raise ValueError(f"Invalid base64 AUDIO_BYTES: {type(e).__name__}: {e}") from e
            wav, _sr = self._decode_wav(audio_bytes)
            wav_data.append(wav)

        lipread_video = None
        n_frames = 0
        vision_video_bytes: List[bytes] = []
        if _uses_lipread_encoder(mode) and len(video_b64s) > 1:
            raise ValueError(
                f"mode 'lipread' accepts exactly one video clip, got {len(video_b64s)}"
            )
        for video_b64 in video_b64s:
            try:
                video_bytes = base64.b64decode(video_b64)
            except Exception as e:
                raise ValueError(f"Invalid base64 VIDEO_BYTES: {type(e).__name__}: {e}") from e
            if _uses_lipread_encoder(mode):
                lipread_video = self._process_lipread(video_bytes)
                n_frames = int(lipread_video.shape[0])
            else:
                vision_video_bytes.append(video_bytes)
        if lipread_video is not None:
            parts = _place_lipread_tokens(parts, n_frames)
            user_prompt = "".join(text for kind, text in parts if kind == "text")

        return _DecodedSample(
            index=index,
            mode=mode,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            use_system=use_system,
            wav_data=wav_data,
            lipread_video=lipread_video,
            vision_video_bytes=vision_video_bytes,
            parts=parts,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            caption_target=caption_target,
            return_audio=return_audio,
            speaker=speaker,
        )

    def _tokenize_sample(self, sample: _DecodedSample) -> Any:
        vision_paths: List[str] = []
        try:
            for blob in sample.vision_video_bytes:
                tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                try:
                    tmp.write(blob)
                    tmp.flush()
                finally:
                    tmp.close()
                vision_paths.append(tmp.name)

            if not sample.wav_data and not vision_paths:
                messages: List[Dict[str, str]] = []
                if sample.use_system:
                    messages.append({"role": "system", "content": sample.system_prompt})
                messages.append({"role": "user", "content": sample.user_prompt})
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )

            conversations: List[Dict[str, Any]] = []
            if sample.use_system:
                conversations.append(
                    {"role": "system", "content": [{"type": "text", "text": sample.system_prompt}]}
                )
            user_content = _assemble_user_content(
                sample.parts,
                sample.wav_data,
                vision_paths,
            )
            conversations.append({"role": "user", "content": user_content})
            return self.processor.apply_chat_template(
                conversations,
                **_processor_tokenize_kwargs(),
            )
        finally:
            for path in vision_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _attach_lipread(self, model_inputs: Dict[str, Any], sample: _DecodedSample) -> None:
        if sample.lipread_video is None:
            return
        lip_dev = self._lipread_device()
        video = sample.lipread_video.to(lip_dev).permute(1, 0, 2, 3)
        max_length = int(sample.lipread_video.shape[0])
        lipread_mask = self._torch.zeros(
            (video.shape[0], max_length, 1),
            dtype=self._torch.uint8,
            device=lip_dev,
        )
        lipread_mask[:, :max_length, 0] = 1
        model_inputs["lipread"] = video
        model_inputs["lipread_mask"] = lipread_mask

    def _place_vision(self, model_inputs: Dict[str, Any]) -> None:
        keys = (
            "pixel_values_videos",
            "pixel_values",
            "video_grid_thw",
            "image_grid_thw",
            "video_second_per_grid",
        )
        if not any(key in model_inputs for key in keys):
            return
        device = self._vision_device()
        for key in keys:
            value = model_inputs.get(key)
            if value is not None and self._torch.is_tensor(value):
                model_inputs[key] = value.to(device)

    def _pad_and_cat(
        self, tensors: List[Any], *, rank: int, left: bool = False, pad_value: int = 0
    ) -> Any:
        """Normalize to `rank` dims, pad the last axis to the batch max, cat on the batch axis.

        Text pads on the left (the tokenizer is left-padding); audio features pad on
        the right so the real time axis stays contiguous.
        """
        normalized = []
        for tensor in tensors:
            while tensor.dim() < rank:
                tensor = tensor.unsqueeze(0)
            if tensor.dim() == rank + 1:
                # squeeze() is a no-op unless the axis is size 1, so check the result.
                tensor = tensor.squeeze(1)
            if tensor.dim() != rank:
                raise RuntimeError(
                    f"cannot normalize a tensor of shape {tuple(tensor.shape)} to rank {rank}"
                )
            normalized.append(tensor)
        max_len = max(int(t.shape[-1]) for t in normalized)
        padded = []
        for tensor in normalized:
            pad = max_len - int(tensor.shape[-1])
            if pad:
                tensor = self._torch.nn.functional.pad(
                    tensor, (pad, 0) if left else (0, pad), value=pad_value
                )
            padded.append(tensor)
        return self._torch.cat(padded, dim=0)

    def _collate(self, per_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(per_inputs) == 1:
            return per_inputs[0]
        keys = per_inputs[0].keys()
        batched: Dict[str, Any] = {}
        pad_id = int(self.tokenizer.pad_token_id or 0)
        for key in keys:
            values = [item[key] for item in per_inputs]
            if not self._torch.is_tensor(values[0]):
                batched[key] = values
                continue
            if key in {"input_ids", "attention_mask"}:
                batched[key] = self._pad_and_cat(
                    values, rank=2, left=True, pad_value=pad_id if key == "input_ids" else 0
                )
            elif key == "lipread":
                max_t = max(int(v.shape[1]) for v in values)
                padded = []
                for video in values:
                    pad = max_t - int(video.shape[1])
                    if pad:
                        video = self._torch.nn.functional.pad(video, (0, 0, 0, 0, 0, pad), value=0)
                    padded.append(video)
                batched[key] = self._torch.cat(padded, dim=0)
            elif key == "lipread_mask":
                max_t = max(int(v.shape[1]) for v in values)
                padded = []
                for mask in values:
                    pad = max_t - int(mask.shape[1])
                    if pad:
                        mask = self._torch.nn.functional.pad(mask, (0, 0, 0, pad), value=0)
                    padded.append(mask)
                batched[key] = self._torch.cat(padded, dim=0)
            elif key in {"pixel_values_videos", "pixel_values"}:
                batched[key] = self._torch.cat(
                    [v if v.dim() >= 2 else v.unsqueeze(0) for v in values], dim=0
                )
            elif key in {"video_grid_thw", "image_grid_thw"}:
                batched[key] = self._torch.cat(
                    [v if v.dim() == 2 else v.unsqueeze(0) for v in values], dim=0
                )
            elif key == "video_second_per_grid":
                batched[key] = self._torch.cat(
                    [v.reshape(-1) if v.dim() > 0 else v.unsqueeze(0) for v in values], dim=0
                )
            elif key == "feature_attention_mask":
                batched[key] = self._pad_and_cat(values, rank=2)
            elif key == "input_features":
                batched[key] = self._pad_and_cat(values, rank=3)
            else:
                raise RuntimeError(
                    f"cannot batch model input '{key}' with shape {tuple(values[0].shape)}; "
                    "add an explicit collate rule for it."
                )
        return batched

    def _cast_floats(self, model_inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v.to(self._torch.bfloat16) if self._torch.is_tensor(v) and self._torch.is_floating_point(v) else v
            for k, v in model_inputs.items()
        }

    def _decode_generated(
        self, generated_ids: Any, prompt_len: int, eos_token_id: Any, max_new_tokens: int
    ) -> List[Tuple[str, str]]:
        if generated_ids.dim() == 1:
            new_tokens = generated_ids[prompt_len:].unsqueeze(0)
        else:
            new_tokens = generated_ids[:, prompt_len:]
        if self.processor is not None and hasattr(self.processor, "batch_decode"):
            texts = self.processor.batch_decode(
                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        else:
            texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        eos_ids = set(
            [] if eos_token_id is None else
            [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
        )
        results = []
        counts = []
        for text, tokens in zip(texts, new_tokens.tolist()):
            # Search only the generated suffix. EOS also wins at the budget
            # boundary; padding after an earlier EOS belongs to a finished row.
            if any(token in eos_ids for token in tokens):
                reason = "stop"
            elif len(tokens) >= max_new_tokens:
                reason = "length"
            else:
                raise RuntimeError("generation ended without EOS or token-limit exhaustion")
            results.append((text.strip(), reason))
            counts.append(output_token_count(tokens, eos_ids))
        self._last_output_counts = counts
        return results

    def _generate_once(
        self,
        model_inputs: Dict[str, Any],
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        repetition_penalty: float,
    ) -> List[Tuple[str, str]]:
        gen_kwargs: Dict[str, Any] = {
            "use_cache": True,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is not None:
            gen_kwargs["eos_token_id"] = eos
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id
        # Current remote code defaults to speech whenever a Talker is loaded; older
        # exports reject the kwarg, so only pass it when speech is enabled.
        if getattr(self, "speech_enabled", False):
            gen_kwargs["return_audio"] = False

        from long_audio import generation_logits
        started = time.perf_counter()
        with generation_logits(self.model, getattr(self, "_long_audio", False)):
            generated_ids = self.model.generate(**model_inputs, **gen_kwargs)
        generate_time_s = time.perf_counter() - started
        prompt_len = int(model_inputs["input_ids"].shape[-1])
        decoded = self._decode_generated(
            generated_ids, prompt_len,
            gen_kwargs.get("eos_token_id", self.model.generation_config.eos_token_id),
            max_new_tokens,
        )
        stash_generate_usage(self, model_inputs, prompt_len, generate_time_s)
        return decoded

    def _is_oom(self, exc: BaseException) -> bool:
        return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

    def _generate_speech(
        self,
        model_inputs: Dict[str, Any],
        sample: _DecodedSample,
        *,
        thinker_max_new_tokens: Optional[int] = None,
    ) -> Tuple[str, str, str]:
        speaker = resolve_speaker(sample.speaker or self.default_speaker, self._speaker_names())
        kwargs = speech_generate_kwargs(
            sample,
            speaker,
            self.talker_max_new_tokens,
            self.talker_do_sample,
            getattr(self.tokenizer, "eos_token_id", None),
            thinker_max_new_tokens=thinker_max_new_tokens,
        )
        started = time.perf_counter()
        generated = self.model.generate(**model_inputs, **kwargs)
        generate_time_s = time.perf_counter() - started
        if not isinstance(generated, tuple) or len(generated) != 2:
            raise RuntimeError("speech generation did not return text and audio")
        sequences, wav = generated
        prompt_len = int(model_inputs["input_ids"].shape[-1])
        text, reason = self._decode_generated(
            sequences, prompt_len, kwargs.get("thinker_eos_token_id"), kwargs["thinker_max_new_tokens"]
        )[0]
        stash_generate_usage(self, model_inputs, prompt_len, generate_time_s)
        return text, reason, encode_wav_b64(wav)

    def _generate_group(self, samples: List[_DecodedSample]) -> List[Tuple[str, str, str]]:
        if not isinstance(getattr(self, "_pending_usage", None), dict):
            self._pending_usage = {}
        primary = self._primary_device()
        per_inputs: List[Dict[str, Any]] = []
        overflow: set = set()
        for sample in samples:
            raw = self._tokenize_sample(sample)
            model_inputs = self._to_device_dict(raw, primary)
            self._attach_lipread(model_inputs, sample)
            self._place_vision(model_inputs)
            per_inputs.append(self._cast_floats(model_inputs))
            if check_context(self, int(model_inputs["input_ids"].shape[-1]), sample.max_new_tokens):
                overflow.add(sample.index)
        _log_request_shapes(samples, per_inputs)
        _reset_peak_memory()
        _log_cuda_memory("before")
        first = samples[0]
        if first.return_audio:
            # Official Omni rejects batch > 1 with audio.
            results = []
            for sample, inputs in zip(samples, per_inputs):
                results.append(self._generate_speech_oom(inputs, sample))
                usage = (getattr(self, "_last_usage", None) or [None])[0]
                if usage:
                    self._pending_usage[sample.index] = usage
            self._flag_overflow(overflow)
            return results
        gen_kwargs = {
            "max_new_tokens": first.max_new_tokens,
            "do_sample": first.do_sample,
            "temperature": first.temperature,
            "repetition_penalty": first.repetition_penalty,
        }
        try:
            pairs = self._generate_once(self._collate(per_inputs), **gen_kwargs)
        except Exception as e:
            if not self._is_oom(e):
                raise
            _log_cuda_memory("oom", warn=True)
            _log_cuda_summary()
            self._torch.cuda.empty_cache()
            if len(samples) > 1:
                # execute() retries the group one request at a time.
                raise
            halved = max(16, first.max_new_tokens // 2)
            pb_utils.Logger.log_warn(f"CUDA OOM; retrying generate with max_new_tokens={halved}")
            pairs = self._generate_once(per_inputs[0], **{**gen_kwargs, "max_new_tokens": halved})
        _log_cuda_memory("after")
        for sample, usage in zip(samples, getattr(self, "_last_usage", []) or []):
            self._pending_usage[sample.index] = usage
        self._flag_overflow(overflow)
        return [(text, reason, "") for text, reason in pairs]

    def _flag_overflow(self, indices: set) -> None:
        """Mark requests past the context in their usage, so telemetry reports it."""
        for index in indices:
            usage = dict(self._pending_usage.get(index) or {})
            usage["context_overflow"] = True
            self._pending_usage[index] = usage

    def _generate_speech_oom(
        self, model_inputs: Dict[str, Any], sample: _DecodedSample
    ) -> Tuple[str, str, str]:
        try:
            return self._generate_speech(model_inputs, sample)
        except Exception as e:
            if not self._is_oom(e):
                raise
            self._torch.cuda.empty_cache()
            halved = max(16, sample.max_new_tokens // 2)
            pb_utils.Logger.log_warn(
                f"CUDA OOM; retrying speech generate with thinker_max_new_tokens={halved}"
            )
            return self._generate_speech(model_inputs, sample, thinker_max_new_tokens=halved)

    def _postprocess(self, sample: _DecodedSample, text: str) -> str:
        if sample.mode == "caption":
            return _wrap_caption_lines(text, sample.caption_target)
        return text

    def postprocess(self, sample: _DecodedSample, text: str) -> str:
        return self._postprocess(sample, text)

    def decode_request(self, index: int, request: pb_utils.InferenceRequest) -> _DecodedSample:
        return self._decode_sample(index, request)

    def generate(self, samples: List[_DecodedSample]) -> List[Result]:
        triples = self._generate_group(samples)
        pending = getattr(self, "_pending_usage", {}) or {}
        return [
            Result(text, reason, audio_b64, dict(pending.get(sample.index) or {}))
            for sample, (text, reason, audio_b64) in zip(samples, triples)
        ]

    def _text_response(
        self, text: str, finish_reason: str, audio_b64: str = ""
    ) -> pb_utils.InferenceResponse:
        return self.build_response(Result(text, finish_reason, audio_b64))

