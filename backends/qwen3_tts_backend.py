from __future__ import annotations

"""Qwen3-TTS CustomVoice text-to-speech (mode tts).

Needs the qwen-tts package and its transformers 4.57.3, which the SIF does not
have: build scripts/requirements-qwen3-tts-site.txt with
scripts/build/build_model_site.sh and set <MODEL>_SITE_PACKAGES.

The prompt is the text to speak. The envelope `language` (ISO 639-1) selects the
language; the voice is a preset speaker (default ryan, <MODEL>_SPEAKER to change).
Generation follows the checkpoint's generate_config (sampling), seeded per request.
"""

from typing import Any, Dict, Tuple

import numpy as np

import triton_python_backend_utils as pb_utils  # noqa: F401  (patched in tests)

from base_backend import Sample
from speech_model_backend import SpeechModelBackend

QWEN3_TTS_LANGUAGES = {
    "de": "german", "en": "english", "es": "spanish", "fr": "french", "it": "italian",
    "ja": "japanese", "ko": "korean", "pt": "portuguese", "ru": "russian", "zh": "chinese",
}


def qwen3_tts_language(code: Any) -> str:
    value = str(code or "").strip().lower()
    if not value:
        raise ValueError("mode 'tts' needs a language (speechlmm_language) for qwen3_tts")
    return QWEN3_TTS_LANGUAGES.get(value.split("-")[0].split("_")[0], value)


class TritonPythonModel(SpeechModelBackend):
    supported_modes = frozenset({"tts"})
    default_voice = "ryan"

    def load_weights(self, placement: Dict[str, Any]) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        self.tts = Qwen3TTSModel.from_pretrained(
            placement["model_id"],
            device_map=placement["device"],
            dtype=getattr(torch, placement["dtype_name"]),
            attn_implementation="sdpa",
        )
        self.tokenizer = getattr(self.tts.processor, "tokenizer", None)
        self.speakers = [str(name) for name in (self.tts.get_supported_speakers() or [])]
        self.languages = {str(name).lower() for name in (self.tts.get_supported_languages() or [])}

    def validate_extra(self, sample: Sample) -> None:
        super().validate_extra(sample)
        language = qwen3_tts_language(sample.spec.get("language"))
        if self.languages and language not in self.languages:
            raise ValueError(f"qwen3_tts does not speak '{language}'; supported: {sorted(self.languages)}")
        sample.extra["tts_language"] = language
        sample.extra["tts_speaker"] = self.pick_speaker(sample.speaker)

    def synthesize(self, sample: Sample) -> Tuple[str, np.ndarray, int]:
        wavs, sample_rate = self.tts.generate_custom_voice(
            text=sample.prompt.strip(),
            speaker=sample.extra["tts_speaker"],
            language=sample.extra["tts_language"],
        )
        return sample.prompt, np.asarray(wavs[0], dtype=np.float32), int(sample_rate)
