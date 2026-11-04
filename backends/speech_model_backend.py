from __future__ import annotations

"""Base for speech-output models that have no chat thinker (TTS and S2ST).

Same infer contract as every backend; a subclass only loads weights and turns
one request into (text, waveform, sample_rate). The TTS text is the request
prompt, so clients send the text to speak, not an instruction.

Model-specific values come from outside the code:
  <MODEL_NAME>_MODEL_PATH     checkpoint dir or Hub id (required)
  <MODEL_NAME>_DEVICE_MAP     cuda, cuda:<id> or cpu; overrides config.pbtxt device_map
  <MODEL_NAME>_SITE_PACKAGES  optional `pip --target` sidecar prepended to sys.path
                              before the model imports (for a conflicting transformers)
  <MODEL_NAME>_SPEAKER        default voice, used when a request names none or names the
                              gateway default (ethan) that the model does not have
  <MODEL_NAME>_SEED           torch seed set before every request (default 0), so
                              sampling models give the same audio for the same input
  config.pbtxt parameters     device_map (default cuda:<instance>) and dtype
"""

import base64
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import triton_python_backend_utils as pb_utils

from backend_common import _config_param, _model_env
from base_backend import BaseBackend, Result, Sample
from speech_output import encode_wav_b64, envelope_bool, resolve_speaker, validate_speech_request

GATEWAY_DEFAULT_VOICE = "ethan"
_DTYPES = {"float16": "float16", "fp16": "float16", "bfloat16": "bfloat16",
           "bf16": "bfloat16", "float32": "float32", "fp32": "float32"}


def prepend_site_packages(model_name: str) -> Optional[str]:
    """Put <MODEL>_SITE_PACKAGES first on sys.path, before any model import."""
    site = _model_env(model_name, "SITE_PACKAGES")
    if not site:
        return None
    if not os.path.isdir(site):
        raise RuntimeError(f"{model_name.upper()}_SITE_PACKAGES={site} is not a directory")
    if site in sys.path:
        sys.path.remove(site)
    sys.path.insert(0, site)
    return site


def parse_envelope(text_prompt: str) -> Dict[str, Any]:
    """The gateway's JSON envelope; a plain string is a prompt with no mode."""
    s = (text_prompt or "").strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            return {
                "mode": str(obj.get("mode") or "chat").strip().lower(),
                "prompt": str(obj.get("prompt") or obj.get("text") or ""),
                "language": obj.get("language"),
                "return_audio": envelope_bool(obj.get("return_audio")),
                "speaker": obj.get("speaker"),
            }
    return {"mode": "chat", "prompt": s, "language": None, "return_audio": False, "speaker": None}


def resolve_placement(args: Dict[str, Any], model_config: Dict[str, Any]) -> Dict[str, Any]:
    model_name = str(args.get("model_name", "")).strip()
    model_id = _model_env(model_name, "MODEL_PATH") or _model_env(model_name, "MODEL")
    if not model_id:
        raise RuntimeError(f"{model_name.upper()}_MODEL_PATH is required (checkpoint dir or Hub id).")
    instance = int(args.get("model_instance_device_id", 0))
    device = (_model_env(model_name, "DEVICE_MAP") or _config_param(model_config, "device_map")
              or f"cuda:{instance}")
    if device == "cuda":
        device = f"cuda:{instance}"
    if device != "cpu" and not device.startswith("cuda:"):
        raise ValueError(f"unsupported device_map '{device}'; expected cuda, cuda:<id> or cpu.")
    dtype = (_config_param(model_config, "dtype") or "float16").lower()
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported dtype '{dtype}'; expected float16, bfloat16, or float32.")
    return {"model_name": model_name, "model_id": model_id, "device": device,
            "dtype_name": _DTYPES[dtype]}


def _tensor_strs(tensor: Any) -> List[str]:
    if tensor is None:
        return []
    out = []
    for value in tensor.as_numpy().reshape(-1):
        decoded = value.decode("utf-8", errors="ignore") if isinstance(value, (bytes, bytearray)) else str(value)
        if decoded.strip():
            out.append(decoded)
    return out


class SpeechModelBackend(BaseBackend):
    """Speech out only: every request must ask for audio."""

    supports_batching = False
    default_voice: Optional[str] = None

    def load_model(self, args: Dict[str, Any], model_config: Dict[str, Any]) -> None:
        placement = resolve_placement(args, model_config)
        self.model_config = model_config
        self.model_name = placement["model_name"] or "this model"
        self.site_packages = prepend_site_packages(placement["model_name"])
        self.speakers: List[str] = []
        self.speech_enabled = True
        self.seed = int(_model_env(placement["model_name"], "SEED") or 0)
        self.load_weights(placement)
        self.default_speaker = _model_env(placement["model_name"], "SPEAKER") or self.default_voice
        if self.speakers and self.default_speaker:
            self.default_speaker = resolve_speaker(self.default_speaker, self.speakers)
        pb_utils.Logger.log_info(
            f"Loaded {placement['model_id']} name={self.model_name} device={placement['device']} "
            f"dtype={placement['dtype_name']} site_packages={self.site_packages} "
            f"speaker={self.default_speaker}."
        )

    def load_weights(self, placement: Dict[str, Any]) -> None:
        raise NotImplementedError

    def decode_request(self, index: int, request: Any) -> Sample:
        prompt_t = pb_utils.get_input_tensor_by_name(request, "TEXT_PROMPT")
        prompts = _tensor_strs(prompt_t)
        spec = parse_envelope(prompts[0] if prompts else "")
        sample = Sample(
            index=index,
            mode=spec["mode"],
            spec=spec,
            prompt=spec["prompt"],
            return_audio=spec["return_audio"],
            speaker=str(spec["speaker"]).strip() if spec["speaker"] else None,
        )
        sample.extra["audio_b64s"] = _tensor_strs(pb_utils.get_input_tensor_by_name(request, "AUDIO_BYTES"))
        sample.extra["video_b64s"] = _tensor_strs(pb_utils.get_input_tensor_by_name(request, "VIDEO_BYTES"))
        return sample

    def validate_extra(self, sample: Sample) -> None:
        validate_speech_request(
            sample.mode, sample.return_audio, bool(sample.extra["audio_b64s"]),
            bool(sample.extra["video_b64s"]), False, sample.prompt, True,
        )
        if len(sample.extra["audio_b64s"]) > 1:
            raise ValueError(f"{self.model_name} takes at most one audio clip")

    def prepare(self, sample: Sample) -> None:
        try:
            sample.audio = [base64.b64decode(value) for value in sample.extra["audio_b64s"]]
        except Exception as exc:
            raise ValueError(f"Invalid base64 AUDIO_BYTES: {type(exc).__name__}: {exc}") from exc

    def pick_speaker(self, requested: Optional[str]) -> Optional[str]:
        """The request's voice, else the model default for none or the gateway default."""
        if not self.speakers:
            return None
        wanted = (requested or "").strip()
        if not wanted:
            return self.default_speaker
        try:
            return resolve_speaker(wanted, self.speakers)
        except ValueError:
            if wanted.lower() == GATEWAY_DEFAULT_VOICE and self.default_speaker:
                return self.default_speaker
            raise

    def generate(self, samples: Sequence[Sample]) -> List[Result]:
        sample = samples[0]
        torch = sys.modules.get("torch")
        if torch is not None:
            torch.manual_seed(getattr(self, "seed", 0))
        text, wav, sample_rate = self.synthesize(sample)
        return [Result(text, "stop", encode_wav_b64(wav, sample_rate))]

    def synthesize(self, sample: Sample) -> Tuple[str, np.ndarray, int]:
        raise NotImplementedError
