from __future__ import annotations

"""
Triton Python backend for Voxtral-Mini-3B-2507.

Contract matches existing SpeechLMM/Qwen25 style:
  - AUDIO_BYTES (optional): base64-encoded WAV/MP3 bytes (recommended: WAV)
  - TEXT_PROMPT: plain string, or JSON string:
      {"mode": "chat"|"transcribe"|"caption"|"summary", "prompt": "...", "language": "en"}
  - MAX_NEW_TOKENS / DO_SAMPLE / REPETITION_PENALTY / TEMPERATURE (optional)

Outputs:
  - GENERATED_TEXT (string)
"""

import base64
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import triton_python_backend_utils as pb_utils


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


def _decode_request_str(tensor: Any, idx: int) -> str:
    arr = tensor.as_numpy().reshape(-1)
    v = arr[idx]
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="ignore")
    return str(v)


def _parse_text_prompt(text_prompt: str) -> Dict[str, Any]:
    s = (text_prompt or "").strip()
    if not s:
        return {"mode": "chat", "prompt": ""}
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return {
                    "mode": str(obj.get("mode", "chat") or "chat").strip().lower(),
                    "prompt": str(obj.get("prompt", obj.get("text", "")) or ""),
                    "language": obj.get("language"),
                    "format": obj.get("format"),
                    "caption_chars": obj.get("caption_chars"),
                }
        except Exception:
            pass
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


def _wrap_caption_lines(text: str, target_chars: int) -> str:
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


class TritonPythonModel:
    def initialize(self, args: Dict[str, Any]) -> None:
        import sys

        # Optional overlay (kept in Triton-Unified) to pin small but finicky deps
        # (e.g. accelerate version) without modifying the shared external deps volume.
        overlay_site = os.environ.get("VOXTRAL_OVERLAY_SITE_PACKAGES", "")
        if overlay_site and overlay_site not in sys.path:
            sys.path.insert(0, overlay_site)

        # Add dedicated Voxtral deps first (transformers>=4.54 + mistral-common[audio], etc.)
        voxtral_site = os.environ.get("VOXTRAL_SITE_PACKAGES", "/deps/voxtral_site")
        if voxtral_site and voxtral_site not in sys.path:
            sys.path.insert(0, voxtral_site)

        # Ensure the mounted venv site-packages are visible (PyTorch, etc.)
        venv_site = os.environ.get("VENV_SITE_PACKAGES", "/venv/lib/python3.10/site-packages")
        if venv_site and venv_site not in sys.path:
            # Keep Voxtral deps ahead of the shared venv to avoid importing an older transformers.
            insert_at = 1 if (voxtral_site and sys.path and sys.path[0] == voxtral_site) else 0
            sys.path.insert(insert_at, venv_site)

        # GPU placement: respect Triton's instance device id (set by config.pbtxt instance_group).
        instance_device_id = int(args.get("model_instance_device_id", 0))

        import torch
        from transformers import AutoProcessor, VoxtralForConditionalGeneration

        self._torch = torch
        self._AutoProcessor = AutoProcessor
        self._VoxtralForConditionalGeneration = VoxtralForConditionalGeneration

        self.model_id = os.environ.get("VOXTRAL_MODEL", "mistralai/Voxtral-Mini-3B-2507")
        self.default_max_new_tokens = _env_int("MAX_NEW_TOKENS", 512)
        self.default_temperature = _env_float("TEMPERATURE", 0.2)
        self.default_do_sample = _env_bool("DO_SAMPLE", True)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Voxtral Triton backend.")

        self.device = f"cuda:{instance_device_id}"

        # Voxtral supports fp16; keep sdpa for broad GPU compatibility.
        self.processor = self._AutoProcessor.from_pretrained(self.model_id)
        self.model = self._VoxtralForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        self.model.eval()

        pb_utils.Logger.log_info(f"Loaded {self.model_id} on cuda:{instance_device_id}.")

        # Optional: Voice Activity Detection (skip inference on silence)
        self._vad_model = None
        self._vad_get_speech_timestamps = None
        self._vad_enabled = _env_bool("VAD_ENABLED", False)
        self._vad_threshold: Optional[float] = None
        self._vad_energy_fallback_threshold: Optional[float] = None
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

    def _get_audio_for_vad(self, waveform: "Any", sample_rate: int) -> Tuple["Any", int]:
        """
        Return a 16 kHz mono float32 tensor suitable for Silero VAD (on CPU).
        waveform: [samples] float32
        """
        VAD_SR = 16000
        w = waveform.float().cpu()
        if sample_rate == VAD_SR:
            return w, VAD_SR
        w = w.unsqueeze(0)
        try:
            import torchaudio  # type: ignore

            resampled = torchaudio.functional.resample(w, sample_rate, VAD_SR)
        except Exception:
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
        if self._vad_energy_fallback_threshold is None:
            return True
        try:
            import soundfile as sf  # type: ignore
            import numpy as _np
            import io as _io

            wav_data, sr = sf.read(_io.BytesIO(wav_bytes), dtype="float32")
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

            if self._vad_model is not None and self._vad_get_speech_timestamps is not None:
                with self._torch.no_grad():
                    timestamps = self._vad_get_speech_timestamps(
                        audio_16k,
                        self._vad_model,
                        sampling_rate=sr_16k,
                        threshold=float(self._vad_threshold) if self._vad_threshold is not None else 0.5,
                    )
                return bool(timestamps)

            peak = float(self._torch.max(audio_16k.abs()).item())
            return peak >= float(self._vad_energy_fallback_threshold)
        except Exception:
            return True

    def _write_audio_tempfile(self, audio_bytes: bytes) -> str:
        # Prefer writing as-is; processor can decode common formats by extension.
        # Default to .wav (works if client sends wav bytes).
        tmp_root = os.environ.get("TMPDIR", "/tmp")
        os.makedirs(tmp_root, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="voxtral_", suffix=".wav", dir=tmp_root)
        os.close(fd)
        with open(path, "wb") as f:
            f.write(audio_bytes)
        return path

    def execute(self, requests: List[pb_utils.InferenceRequest]) -> List[pb_utils.InferenceResponse]:
        responses: List[pb_utils.InferenceResponse] = []

        for request in requests:
            audio_t = pb_utils.get_input_tensor_by_name(request, "AUDIO_BYTES")
            prompt_t = pb_utils.get_input_tensor_by_name(request, "TEXT_PROMPT")
            max_new_t = pb_utils.get_input_tensor_by_name(request, "MAX_NEW_TOKENS")
            do_sample_t = pb_utils.get_input_tensor_by_name(request, "DO_SAMPLE")
            rep_pen_t = pb_utils.get_input_tensor_by_name(request, "REPETITION_PENALTY")
            temp_t = pb_utils.get_input_tensor_by_name(request, "TEMPERATURE")

            max_new_tokens = (
                int(max_new_t.as_numpy().reshape(-1)[0]) if max_new_t is not None else self.default_max_new_tokens
            )
            do_sample = (
                bool(do_sample_t.as_numpy().reshape(-1)[0]) if do_sample_t is not None else self.default_do_sample
            )
            temperature = (
                float(temp_t.as_numpy().reshape(-1)[0]) if temp_t is not None else self.default_temperature
            )
            repetition_penalty = float(rep_pen_t.as_numpy().reshape(-1)[0]) if rep_pen_t is not None else 1.0

            raw_prompt = _decode_request_str(prompt_t, 0)
            spec = _parse_text_prompt(raw_prompt)
            mode = str(spec.get("mode", "chat") or "chat").lower()

            audio_b64 = _decode_request_str(audio_t, 0) if audio_t is not None else ""
            has_audio = bool(audio_b64.strip())

            tmp_path = None
            try:
                if mode in {"transcribe", "caption", "summary"} and not has_audio:
                    raise ValueError(f"mode '{mode}' requires AUDIO_BYTES")

                if has_audio:
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                    except Exception as e:
                        raise ValueError(f"Invalid base64 AUDIO_BYTES: {type(e).__name__}: {e}") from e

                    # Skip inference on silence (VAD)
                    if mode in {"transcribe", "summary", "caption", "mt", "st"} and not self._has_speech(audio_bytes):
                        out = pb_utils.Tensor("GENERATED_TEXT", np.array([""], dtype=object))
                        responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
                        continue
                    tmp_path = self._write_audio_tempfile(audio_bytes)

                if mode in {"transcribe", "caption"}:
                    # Preferred transcription API if available.
                    lang = spec.get("language")
                    try:
                        inputs = self.processor.apply_transcription_request(
                            language=lang if isinstance(lang, str) and lang.strip() else None,
                            audio=tmp_path,
                            model_id=self.model_id,
                        )
                        inputs = inputs.to(self.device, dtype=self._torch.float16)
                        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                        decoded = self.processor.batch_decode(
                            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
                        )[0].strip()
                        text = decoded
                    except Exception:
                        # Fallback to chat template with ASR-style instruction.
                        convo = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "audio", "path": tmp_path},
                                    {
                                        "type": "text",
                                        "text": (
                                            "Generate captions for the audio. Output only the words spoken."
                                            if mode == "caption"
                                            else "Transcribe the audio exactly. Output only the words spoken."
                                        ),
                                    },
                                ],
                            }
                        ]
                        inputs = self.processor.apply_chat_template(convo)
                        inputs = inputs.to(self.device, dtype=self._torch.float16)
                        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                        text = self.processor.batch_decode(
                            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
                        )[0].strip()

                    if mode == "caption":
                        target = 60
                        try:
                            if spec.get("caption_chars") is not None:
                                target = int(spec.get("caption_chars"))
                        except Exception:
                            target = 60
                        text = _wrap_caption_lines(text, target_chars=target)

                elif mode == "summary":
                    # Voxtral does not have a dedicated summary API; route through chat template.
                    prompt = str(spec.get("prompt", "") or "")
                    convo = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio", "path": tmp_path},
                                {
                                    "type": "text",
                                    "text": (
                                        "Summarize the audio in 5–10 bullet points. "
                                        "Include key decisions and action items.\n"
                                        + (f"\nAdditional instruction:\n{prompt.strip()}" if prompt.strip() else "")
                                    ),
                                },
                            ],
                        }
                    ]
                    inputs = self.processor.apply_chat_template(convo)
                    inputs = inputs.to(self.device, dtype=self._torch.float16)
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                        repetition_penalty=repetition_penalty,
                    )
                    text = self.processor.batch_decode(
                        outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )[0].strip()

                else:
                    # Chat / audio understanding.
                    user_text = str(spec.get("prompt", "") or "")
                    if has_audio:
                        convo = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "audio", "path": tmp_path},
                                    {"type": "text", "text": user_text},
                                ],
                            }
                        ]
                    else:
                        convo = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]

                    inputs = self.processor.apply_chat_template(convo)
                    inputs = inputs.to(self.device, dtype=self._torch.float16)
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                        repetition_penalty=repetition_penalty,
                    )
                    text = self.processor.batch_decode(
                        outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )[0].strip()

                out = pb_utils.Tensor("GENERATED_TEXT", np.array([text], dtype=object))
                responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
            except Exception as e:
                err = pb_utils.TritonError(f"{type(e).__name__}: {e}")
                responses.append(pb_utils.InferenceResponse(error=err))
            finally:
                if tmp_path:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

        return responses

