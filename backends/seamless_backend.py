from __future__ import annotations

"""SeamlessM4T v2 speech-to-speech translation (mode s2st).

The target language is the envelope `language` (the gateway's speechlmm_language),
an ISO 639-1 code mapped to Seamless's three-letter codes. The prompt is ignored.
Speech is returned at the vocoder's native 16 kHz with the translated text.

config.pbtxt parameters, on top of the SpeechModelBackend ones:
  text_num_beams    beams for the text decoder (default 5)
  speech_do_sample  sample the unit decoder (default false: greedy)
"""

from typing import Any, Dict, Tuple

import numpy as np

import triton_python_backend_utils as pb_utils  # noqa: F401  (patched in tests)

from backend_common import _config_param, decode_wav
from base_backend import Sample
from speech_model_backend import SpeechModelBackend
from speech_output import envelope_bool

# ISO 639-1 -> Seamless code, for the languages the evaluation sets use.
SEAMLESS_LANGUAGES = {
    "bg": "bul", "cs": "ces", "da": "dan", "de": "deu", "el": "ell", "en": "eng",
    "es": "spa", "et": "est", "fi": "fin", "fr": "fra", "ga": "gle", "hr": "hrv",
    "hu": "hun", "it": "ita", "lt": "lit", "lv": "lvs", "mt": "mlt", "nl": "nld",
    "pl": "pol", "pt": "por", "ro": "ron", "sk": "slk", "sl": "slv", "sv": "swe",
}


def seamless_language(code: Any) -> str:
    """Accept ISO 639-1 or an already three-letter Seamless code."""
    value = str(code or "").strip().lower()
    if not value:
        raise ValueError("mode 's2st' needs a target language (speechlmm_language)")
    return SEAMLESS_LANGUAGES.get(value.split("-")[0].split("_")[0], value)


class TritonPythonModel(SpeechModelBackend):
    supported_modes = frozenset({"s2st"})

    def load_weights(self, placement: Dict[str, Any]) -> None:
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2Model

        self._torch = torch
        self.device = torch.device(placement["device"])
        self.dtype = getattr(torch, placement["dtype_name"])
        self.processor = AutoProcessor.from_pretrained(placement["model_id"])
        self.model = SeamlessM4Tv2Model.from_pretrained(
            placement["model_id"], torch_dtype=self.dtype
        ).to(self.device).eval()
        self.speech_languages = set(self.model.generation_config.vocoder_lang_code_to_id)
        self.sample_rate = int(self.model.config.sampling_rate)
        self.text_num_beams = int(_config_param(self.model_config, "text_num_beams") or 5)
        self.speech_do_sample = envelope_bool(_config_param(self.model_config, "speech_do_sample"))

    def validate_extra(self, sample: Sample) -> None:
        super().validate_extra(sample)
        target = seamless_language(sample.spec.get("language"))
        if target not in self.speech_languages:
            raise ValueError(f"SeamlessM4T v2 has no speech output for '{target}'")
        sample.extra["tgt_lang"] = target

    def synthesize(self, sample: Sample) -> Tuple[str, np.ndarray, int]:
        torch = self._torch
        wav, sr = decode_wav(sample.audio[0], self.processor, torch)
        inputs = self.processor(audio=wav, sampling_rate=sr, return_tensors="pt").to(self.device)
        inputs["input_features"] = inputs["input_features"].to(self.dtype)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                tgt_lang=sample.extra["tgt_lang"],
                return_intermediate_token_ids=True,
                text_num_beams=self.text_num_beams,
                speech_do_sample=self.speech_do_sample,
            )
        # waveform_lengths is 0-dim for a batch of one.
        lengths = out.waveform_lengths
        length = int(lengths.reshape(-1)[0]) if lengths is not None else out.waveform.shape[-1]
        waveform = out.waveform[0, :length].float().cpu().numpy()
        text = self.processor.decode(out.sequences[0], skip_special_tokens=True).strip()
        return text, waveform, self.sample_rate
