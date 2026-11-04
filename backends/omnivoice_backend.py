from __future__ import annotations

"""OmniVoice text-to-speech (mode tts).

Needs omnivoice and transformers>=5.3.0: build scripts/requirements-omnivoice-site.txt
with scripts/build/build_model_site.sh and set <MODEL>_SITE_PACKAGES.

The prompt is the text to speak and the envelope `language` (ISO 639-1) its
language. There are no preset voices: without a reference clip OmniVoice picks
a voice itself, so the request voice is ignored and <MODEL>_SEED (set before
every request) fixes the choice. <MODEL>_INSTRUCT, if set, designs the voice
instead (for example "female, low pitch"). Default generation settings apply.
"""

from typing import Any, Dict, Tuple

import numpy as np

import triton_python_backend_utils as pb_utils  # noqa: F401  (patched in tests)

from backend_common import _model_env
from base_backend import Sample
from speech_model_backend import SpeechModelBackend


class TritonPythonModel(SpeechModelBackend):
    supported_modes = frozenset({"tts"})

    def load_weights(self, placement: Dict[str, Any]) -> None:
        import torch
        from omnivoice import OmniVoice

        self.tts = OmniVoice.from_pretrained(
            placement["model_id"],
            device_map=placement["device"],
            dtype=getattr(torch, placement["dtype_name"]),
        )
        self.tokenizer = getattr(self.tts, "text_tokenizer", None)
        self.languages = {str(code).lower() for code in self.tts.supported_language_ids()}
        self.instruct = _model_env(placement["model_name"], "INSTRUCT") or None

    def validate_extra(self, sample: Sample) -> None:
        super().validate_extra(sample)
        language = str(sample.spec.get("language") or "").strip().lower()
        if not language:
            raise ValueError("mode 'tts' needs a language (speechlmm_language) for omnivoice")
        if self.languages and language not in self.languages:
            raise ValueError(f"omnivoice does not speak '{language}'")
        sample.extra["tts_language"] = language

    def synthesize(self, sample: Sample) -> Tuple[str, np.ndarray, int]:
        audios = self.tts.generate(
            text=sample.prompt.strip(),
            language=sample.extra["tts_language"],
            instruct=self.instruct,
        )
        return sample.prompt, np.asarray(audios[0], dtype=np.float32).reshape(-1), int(self.tts.sampling_rate)
