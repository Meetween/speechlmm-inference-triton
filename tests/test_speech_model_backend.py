#!/usr/bin/env python3
"""SpeechModelBackend and the SeamlessM4T v2 backend, with fake pb_utils and weights."""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_REPO = Path(__file__).resolve().parents[1]

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - the SIF has it
    sf = None


def _load(name: str):
    backends = str(_REPO / "backends")
    if backends not in sys.path:
        sys.path.insert(0, backends)
    sys.modules.setdefault("triton_python_backend_utils", types.ModuleType("stub"))
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", _REPO / "backends" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Tensor:
    def __init__(self, name, data):
        self.name = name
        self._data = data

    def as_numpy(self):
        return self._data


class Response:
    def __init__(self, output_tensors=None, error=None):
        self.output_tensors = output_tensors
        self.error = error


class TritonError(Exception):
    pass


# The stub module is shared by every test file: only fill in what is missing.
pb = sys.modules.setdefault("triton_python_backend_utils", types.ModuleType("stub"))
for _name, _value in {
    "Tensor": Tensor,
    "InferenceResponse": Response,
    "TritonError": TritonError,
    "Logger": types.SimpleNamespace(log_info=lambda m: None, log_warn=lambda m: None),
    "get_input_tensor_by_name": lambda request, name: request.get(name),
}.items():
    if not hasattr(pb, _name):
        setattr(pb, _name, _value)

speech = _load("speech_model_backend")
seamless = _load("seamless_backend")
qwen3_tts = _load("qwen3_tts_backend")
omnivoice = _load("omnivoice_backend")


def _request(envelope, audio: bytes | None = None):
    request = {"TEXT_PROMPT": Tensor("TEXT_PROMPT", np.array([json.dumps(envelope).encode()], dtype=object))}
    if audio is not None:
        request["AUDIO_BYTES"] = Tensor("AUDIO_BYTES", np.array([[base64.b64encode(audio)]], dtype=object))
    return request


def _wav(seconds=0.1, rate=16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * rate), dtype=np.float32), rate, format="WAV")
    return buf.getvalue()


def _outputs(response):
    return {t.name: t.as_numpy()[0] for t in response.output_tensors}


class FakeTTS(speech.SpeechModelBackend):
    supported_modes = frozenset({"tts"})
    default_voice = "Ryan"

    def load_weights(self, placement):
        self.placement = placement
        self.speakers = ["Ryan", "Vivian"]

    def synthesize(self, sample):
        self.seen = (sample.prompt, sample.spec["language"], self.pick_speaker(sample.speaker))
        return sample.prompt, np.zeros(2400, dtype=np.float32), 24000


class SpeechModelBackendTests(unittest.TestCase):
    def _backend(self, env=None):
        backend = FakeTTS()
        with mock.patch.dict(os.environ, {"FAKE_TTS_MODEL_PATH": "/ckpt", **(env or {})}):
            backend.initialize({"model_name": "fake_tts", "model_config": "{}", "model_instance_device_id": 1})
        return backend

    def test_placement_and_default_speaker(self) -> None:
        backend = self._backend({"FAKE_TTS_SPEAKER": "vivian"})
        self.assertEqual(backend.placement["device"], "cuda:1")
        self.assertEqual(backend.placement["dtype_name"], "float16")
        self.assertEqual(backend.default_speaker, "Vivian")

    def test_sidecar_is_prepended(self) -> None:
        site = str(_REPO / "tests")
        backend = self._backend({"FAKE_TTS_SITE_PACKAGES": site})
        self.assertEqual(sys.path[0], site)
        self.assertEqual(backend.site_packages, site)
        sys.path.remove(site)

    def test_missing_sidecar_fails_loudly(self) -> None:
        with self.assertRaises(RuntimeError):
            self._backend({"FAKE_TTS_SITE_PACKAGES": "/does/not/exist"})

    @unittest.skipIf(sf is None, "soundfile missing")
    def test_tts_speaks_the_prompt_with_the_gateway_voice_mapped(self) -> None:
        backend = self._backend()
        env = {"mode": "tts", "prompt": "Hallo Welt.", "language": "de", "return_audio": True, "speaker": "ethan"}
        [response] = backend.execute([_request(env)])
        self.assertIsNone(response.error)
        out = _outputs(response)
        self.assertEqual(out["GENERATED_TEXT"], "Hallo Welt.")
        self.assertEqual(out["FINISH_REASON"], "stop")
        self.assertEqual(backend.seen, ("Hallo Welt.", "de", "Ryan"))
        _, rate = sf.read(io.BytesIO(base64.b64decode(out["GENERATED_AUDIO"])))
        self.assertEqual(rate, 24000)

    def test_unknown_voice_and_text_output_are_rejected(self) -> None:
        backend = self._backend()
        bad_voice = {"mode": "tts", "prompt": "hi", "return_audio": True, "speaker": "nobody"}
        no_audio = {"mode": "tts", "prompt": "hi", "return_audio": False}
        chat = {"mode": "chat", "prompt": "hi", "return_audio": True}
        responses = backend.execute([_request(bad_voice), _request(no_audio), _request(chat)])
        for response in responses:
            self.assertIsNotNone(response.error)


class SeamlessTests(unittest.TestCase):
    def _backend(self):
        backend = seamless.TritonPythonModel()
        backend.model_name = "seamless_m4t_v2_large"
        backend.speech_languages = {"eng", "deu"}
        return backend

    def test_language_codes(self) -> None:
        self.assertEqual(seamless.seamless_language("de"), "deu")
        self.assertEqual(seamless.seamless_language("lv"), "lvs")
        self.assertEqual(seamless.seamless_language("eng"), "eng")
        with self.assertRaises(ValueError):
            seamless.seamless_language("")

    def test_target_without_vocoder_is_refused(self) -> None:
        backend = self._backend()
        sample = backend.decode_request(0, _request(
            {"mode": "s2st", "prompt": "", "language": "hu", "return_audio": True}, b"wav"))
        with self.assertRaisesRegex(ValueError, "no speech output for 'hun'"):
            backend.validate(sample)

    def test_s2st_needs_audio_input(self) -> None:
        backend = self._backend()
        sample = backend.decode_request(0, _request({"mode": "s2st", "prompt": "x", "language": "en", "return_audio": True}))
        with self.assertRaises(ValueError):
            backend.validate(sample)

    def test_tts_mode_is_not_served(self) -> None:
        backend = self._backend()
        sample = backend.decode_request(0, _request({"mode": "tts", "prompt": "x", "language": "en", "return_audio": True}))
        with self.assertRaisesRegex(ValueError, "not supported"):
            backend.validate(sample)

    def test_valid_request_records_target(self) -> None:
        backend = self._backend()
        sample = backend.decode_request(0, _request(
            {"mode": "s2st", "prompt": "", "language": "de", "return_audio": True}, b"wav"))
        backend.validate(sample)
        self.assertEqual(sample.extra["tgt_lang"], "deu")


class FakeQwen3TTS:
    def __init__(self):
        self.calls = []

    def generate_custom_voice(self, text, speaker, language):
        self.calls.append((text, speaker, language))
        return [np.zeros(240, dtype=np.float32)], 24000


class Qwen3TTSTests(unittest.TestCase):
    def _backend(self):
        backend = qwen3_tts.TritonPythonModel()
        backend.model_name = "qwen3_tts"
        backend.tts = FakeQwen3TTS()
        backend.speakers = ["ryan", "vivian"]
        backend.default_speaker = "ryan"
        backend.languages = {"english", "german"}
        return backend

    def test_language_codes(self) -> None:
        self.assertEqual(qwen3_tts.qwen3_tts_language("de"), "german")
        self.assertEqual(qwen3_tts.qwen3_tts_language("pt-BR"), "portuguese")

    @unittest.skipIf(sf is None, "soundfile missing")
    def test_speaks_prompt_in_language_with_default_voice(self) -> None:
        backend = self._backend()
        env = {"mode": "tts", "prompt": " Guten Tag. ", "language": "de", "return_audio": True, "speaker": "ethan"}
        [response] = backend.execute([_request(env)])
        self.assertIsNone(response.error)
        self.assertEqual(backend.tts.calls, [("Guten Tag.", "ryan", "german")])

    def test_unsupported_language_is_refused(self) -> None:
        backend = self._backend()
        sample = backend.decode_request(0, _request({"mode": "tts", "prompt": "x", "language": "bg", "return_audio": True}))
        with self.assertRaisesRegex(ValueError, "does not speak 'bg'"):
            backend.validate(sample)


class FakeOmniVoice:
    sampling_rate = 24000

    def __init__(self):
        self.calls = []

    def generate(self, text, language, instruct):
        self.calls.append((text, language, instruct))
        return [np.zeros(240, dtype=np.float32)]


class OmniVoiceTests(unittest.TestCase):
    def _backend(self):
        backend = omnivoice.TritonPythonModel()
        backend.model_name = "omnivoice"
        backend.tts = FakeOmniVoice()
        backend.languages = {"en", "ga", "mt"}
        backend.instruct = None
        return backend

    @unittest.skipIf(sf is None, "soundfile missing")
    def test_ignores_the_gateway_voice_and_passes_the_language(self) -> None:
        backend = self._backend()
        env = {"mode": "tts", "prompt": "Dia duit.", "language": "ga", "return_audio": True, "speaker": "ethan"}
        [response] = backend.execute([_request(env)])
        self.assertIsNone(response.error)
        self.assertEqual(backend.tts.calls, [("Dia duit.", "ga", None)])

    def test_missing_language_is_refused(self) -> None:
        backend = self._backend()
        sample = backend.decode_request(0, _request({"mode": "tts", "prompt": "x", "return_audio": True}))
        with self.assertRaisesRegex(ValueError, "needs a language"):
            backend.validate(sample)


class ShimTests(unittest.TestCase):
    def test_speech_only_configs_declare_the_contract(self) -> None:
        for model, backend in (
            ("seamless_m4t_v2_large", "seamless_backend"),
            ("qwen3_tts", "qwen3_tts_backend"),
            ("omnivoice", "omnivoice_backend"),
        ):
            with self.subTest(model=model):
                config = (_REPO / "models" / model / "config.pbtxt").read_text()
                self.assertIn(f'name: "{model}"', config)
                for name in ("GENERATED_TEXT", "FINISH_REASON", "GENERATED_AUDIO", "METRICS"):
                    self.assertIn(f'name: "{name}"', config)
                shim = (_REPO / "models" / model / "1" / "model.py").read_text()
                self.assertIn(f"from {backend} import TritonPythonModel as _Backend", shim)


if __name__ == "__main__":
    unittest.main()
