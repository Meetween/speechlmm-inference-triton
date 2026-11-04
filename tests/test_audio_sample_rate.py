"""Audio decoding regressions; run inside the serving SIF (no GPU required)."""

import base64
import io
import json
import types
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional
except (ImportError, OSError) as exc:
    raise unittest.SkipTest(f"audio dependencies are not available: {exc}")

from test_backend_prompts import _FakeTensor, backend
from test_qwen_omni_backend import qwen as qwen_backend


def tone(rate, frequency=440):
    return (0.2 * np.sin(2 * np.pi * frequency * np.arange(rate) / rate)).astype(np.float32)


def wav_bytes(waveform, rate):
    handle = io.BytesIO()
    sf.write(handle, waveform, rate, format="WAV", subtype="FLOAT")
    return handle.getvalue()


class AudioSampleRateTests(unittest.TestCase):
    def setUp(self):
        self.model = object.__new__(backend.TritonPythonModel)
        self.model._torch = torch
        self.model.processor = types.SimpleNamespace(
            feature_extractor=types.SimpleNamespace(sampling_rate=16000)
        )

    def test_resampling_preserves_duration_and_frequency(self):
        for source_rate in (8000, 44100, 48000):
            with self.subTest(source_rate=source_rate):
                audio, rate = self.model._decode_wav(wav_bytes(tone(source_rate), source_rate))
                self.assertEqual(rate, 16000)
                self.assertEqual(audio.shape, (16000,))
                self.assertEqual(audio.dtype, np.float32)
                np.testing.assert_allclose(audio[32:-32], tone(16000)[32:-32], atol=0.002)

    def test_matching_rate_keeps_samples_exactly(self):
        original = tone(16000)
        with patch("torchaudio.functional.resample") as resample:
            audio, rate = self.model._decode_wav(wav_bytes(original, 16000))
        resample.assert_not_called()
        np.testing.assert_array_equal(audio, original)
        self.assertEqual(rate, 16000)

    def test_rate_comes_from_processor_and_stereo_is_downmixed(self):
        self.model.processor.feature_extractor.sampling_rate = 8000
        original = tone(48000)
        audio, rate = self.model._decode_wav(
            wav_bytes(np.column_stack([original, original]), 48000)
        )
        self.assertEqual(rate, 8000)
        self.assertEqual(audio.shape, (8000,))
        np.testing.assert_allclose(audio[32:-32], tone(8000)[32:-32], atol=0.002)

    def test_request_resamples_each_clip_without_reordering(self):
        self.model.default_max_new_tokens = 128
        self.model.default_do_sample = False
        self.model.default_temperature = 0.0
        clips = [(48000, 440), (8000, 880)]
        request = {
            "TEXT_PROMPT": _FakeTensor(json.dumps({"mode": "transcribe", "prompt": "Read."})),
            "AUDIO_BYTES": _FakeTensor([
                base64.b64encode(wav_bytes(tone(rate, frequency), rate)).decode()
                for rate, frequency in clips
            ]),
        }
        with patch.object(backend.pb_utils, "get_input_tensor_by_name", lambda req, name: req.get(name), create=True):
            sample = self.model._decode_sample(0, request)
        self.assertEqual(len(sample.wav_data), 2)
        for audio, (_, frequency) in zip(sample.wav_data, clips):
            self.assertEqual(audio.shape, (16000,))
            np.testing.assert_allclose(audio[32:-32], tone(16000, frequency)[32:-32], atol=0.002)


class QwenAudioSampleRateTests(unittest.TestCase):
    def setUp(self):
        self.model = object.__new__(qwen_backend.TritonPythonModel)
        self.model._torch = torch
        self.model.processor = types.SimpleNamespace(
            feature_extractor=types.SimpleNamespace(sampling_rate=16000)
        )

    def test_resampling_preserves_duration_and_frequency(self):
        for source_rate in (8000, 44100, 48000):
            with self.subTest(source_rate=source_rate):
                audio, rate = self.model._decode_wav(wav_bytes(tone(source_rate), source_rate))
                self.assertEqual(rate, 16000)
                self.assertEqual(audio.shape, (16000,))
                self.assertEqual(audio.dtype, np.float32)
                np.testing.assert_allclose(audio[32:-32], tone(16000)[32:-32], atol=0.002)

    def test_matching_rate_keeps_samples_exactly(self):
        original = tone(16000)
        with patch("torchaudio.functional.resample") as resample:
            audio, rate = self.model._decode_wav(wav_bytes(original, 16000))
        resample.assert_not_called()
        np.testing.assert_array_equal(audio, original)
        self.assertEqual(rate, 16000)

    def test_rate_comes_from_processor_and_stereo_is_downmixed(self):
        self.model.processor.feature_extractor.sampling_rate = 8000
        original = tone(48000)
        audio, rate = self.model._decode_wav(
            wav_bytes(np.column_stack([original, original]), 48000)
        )
        self.assertEqual(rate, 8000)
        self.assertEqual(audio.shape, (8000,))
        np.testing.assert_allclose(audio[32:-32], tone(8000)[32:-32], atol=0.002)


if __name__ == "__main__":
    unittest.main()
