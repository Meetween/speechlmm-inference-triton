#!/usr/bin/env python3
"""Telemetry formulas and the execute wrapper. No Triton runtime."""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import sys
import types
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

_REPO = Path(__file__).resolve().parents[1]


def _load_telemetry():
    backends = str(_REPO / "backends")
    if backends not in sys.path:
        sys.path.insert(0, backends)
    stub = sys.modules.get("triton_python_backend_utils")
    if stub is None:
        stub = types.ModuleType("triton_python_backend_utils")
        sys.modules["triton_python_backend_utils"] = stub
    spec = importlib.util.spec_from_file_location(
        "telemetry_under_test", _REPO / "backends" / "telemetry.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, stub


telemetry, pb = _load_telemetry()


class Logger:
    lines: list[str] = []
    warns: list[str] = []

    @staticmethod
    def log_info(message: str) -> None:
        Logger.lines.append(message)

    @staticmethod
    def log_warn(message: str) -> None:
        Logger.warns.append(message)


class Tensor:
    def __init__(self, name: str, data) -> None:
        self.name = name
        self._data = data

    def as_numpy(self):
        return self._data


class Response:
    def __init__(self, output_tensors=None, error=None) -> None:
        self._tensors = list(output_tensors or [])
        self.error = error

    def output_tensors(self):
        return list(self._tensors)

    def has_error(self) -> bool:
        return self.error is not None


def _install_pb() -> None:
    pb.Tensor = Tensor
    pb.InferenceResponse = Response
    pb.Logger = Logger
    pb.get_input_tensor_by_name = lambda req, name: req.get(name)

    def get_output(response, name):
        for tensor in response.output_tensors():
            if getattr(tensor, "name", None) == name:
                return tensor
        return None

    pb.get_output_tensor_by_name = get_output


def _text_tensor(name: str, value: str) -> Tensor:
    return Tensor(name, np.array([value], dtype=object))


def _ok(text: str = "hello", reason: str = "stop", audio: str = "") -> Response:
    return Response(
        [
            _text_tensor("GENERATED_TEXT", text),
            _text_tensor("FINISH_REASON", reason),
            _text_tensor("GENERATED_AUDIO", audio),
        ]
    )


def _wav_bytes(seconds: float, rate: int = 16000, channels: int = 1) -> bytes:
    frames = int(seconds * rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * frames * channels)
    return buffer.getvalue()


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode()


class _WordTokenizer:
    def __call__(self, text, add_special_tokens=False):
        if not text:
            return {"input_ids": []}
        return {"input_ids": text.split()}


class BuildMetricsTests(unittest.TestCase):
    def test_rtf_from_audio(self) -> None:
        record = telemetry.build_metrics(
            {"processing_time_s": 1.0, "input_audio_s": 4.0, "mode": "transcribe"}
        )
        self.assertEqual(record["rtf"], 0.25)

    def test_video_rtf(self) -> None:
        record = telemetry.build_metrics(
            {"processing_time_s": 2.0, "input_video_s": 8.0, "mode": "lipread"}
        )
        self.assertEqual(record["video_rtf"], 0.25)
        self.assertNotIn("rtf", record)

    def test_output_rtf_tts(self) -> None:
        record = telemetry.build_metrics(
            {"processing_time_s": 0.5, "output_audio_s": 2.0, "mode": "tts"}
        )
        self.assertEqual(record["output_rtf"], 0.25)

    def test_tokens_per_second(self) -> None:
        record = telemetry.build_metrics(
            {"processing_time_s": 2.0, "input_tokens": 10, "output_tokens": 4, "mode": "mt"}
        )
        self.assertEqual(record["input_tokens_per_s"], 5.0)
        self.assertEqual(record["output_tokens_per_s"], 2.0)

    def test_codec_estimate_only_with_frame_rate(self) -> None:
        bare = telemetry.build_metrics(
            {"processing_time_s": 1.0, "output_audio_s": 2.0, "mode": "tts"}
        )
        self.assertNotIn("codec_tokens_est", bare)
        rated = telemetry.build_metrics(
            {
                "processing_time_s": 1.0,
                "output_audio_s": 2.0,
                "codec_frame_rate": 12.5,
                "mode": "tts",
            }
        )
        self.assertEqual(rated["codec_tokens_est"], 25)
        self.assertEqual(rated["codec_tokens_per_s"], 25.0)

    def test_compression_only_summary_modes(self) -> None:
        summary = telemetry.build_metrics(
            {"processing_time_s": 1.0, "input_tokens": 9, "output_tokens": 3, "mode": "text_summary"}
        )
        self.assertEqual(summary["compression_factor"], 3.0)
        spoken = telemetry.build_metrics(
            {"processing_time_s": 1.0, "input_tokens": 9, "output_tokens": 3, "mode": "summary"}
        )
        self.assertEqual(spoken["compression_factor"], 3.0)
        chat = telemetry.build_metrics(
            {"processing_time_s": 1.0, "input_tokens": 9, "output_tokens": 3, "mode": "chat"}
        )
        self.assertNotIn("compression_factor", chat)

    def test_zero_or_missing_denominators_omitted(self) -> None:
        record = telemetry.build_metrics(
            {
                "processing_time_s": 0,
                "input_audio_s": 0,
                "input_tokens": 4,
                "output_tokens": 0,
                "mode": "text_summary",
            }
        )
        for key in ("rtf", "input_tokens_per_s", "output_tokens_per_s", "compression_factor"):
            self.assertNotIn(key, record)
        missing = telemetry.build_metrics({"mode": "transcribe", "processing_time_s": 1.0})
        self.assertNotIn("rtf", missing)

    def test_rounding(self) -> None:
        record = telemetry.build_metrics(
            {"processing_time_s": 3.0, "input_tokens": 1, "mode": "mt"}
        )
        self.assertEqual(record["input_tokens_per_s"], 0.3333)

    def test_generate_time_basis(self) -> None:
        record = telemetry.build_metrics(
            {
                "processing_time_s": 4.0,
                "generate_time_s": 1.0,
                "input_audio_s": 2.0,
                "telemetry_time_basis": "generate",
                "mode": "transcribe",
            }
        )
        self.assertEqual(record["rtf"], 0.5)
        self.assertNotIn("telemetry_time_basis", record)


class MeasurementTests(unittest.TestCase):
    def test_wav_duration_invalid_bytes_returns_none(self) -> None:
        self.assertIsNone(telemetry.wav_duration_s(b"not a wav"))

    @unittest.skipUnless(importlib.util.find_spec("soundfile"), "requires soundfile")
    def test_wav_duration_mono_and_stereo(self) -> None:
        self.assertAlmostEqual(telemetry.wav_duration_s(_wav_bytes(1.0)), 1.0, places=3)
        self.assertAlmostEqual(
            telemetry.wav_duration_s(_wav_bytes(0.5, channels=2)), 0.5, places=3
        )

    def test_count_tokens_prefers_tokenizer_then_processor(self) -> None:
        owner = SimpleNamespace(tokenizer=_WordTokenizer(), processor=None)
        self.assertEqual(telemetry.count_tokens(owner, "one two three"), 3)
        self.assertEqual(telemetry.count_tokens(owner, ""), 0)
        fallback = SimpleNamespace(
            tokenizer=None, processor=SimpleNamespace(tokenizer=_WordTokenizer())
        )
        self.assertEqual(telemetry.count_tokens(fallback, "only one"), 2)

    def test_count_tokens_absent_returns_none(self) -> None:
        self.assertIsNone(telemetry.count_tokens(SimpleNamespace(), "hello"))

    def test_output_token_count_includes_first_eos(self) -> None:
        self.assertEqual(telemetry.output_token_count([1, 9, 0], {9}), 2)
        self.assertEqual(telemetry.output_token_count([1, 2, 3], {9}), 3)


class FlagTests(unittest.TestCase):
    def test_flag_default_off(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(telemetry.telemetry_enabled("speechlmm_v2_xl"))

    def test_global_flag_on(self) -> None:
        with patch.dict(os.environ, {"TELEMETRY": "1"}, clear=True):
            self.assertTrue(telemetry.telemetry_enabled("qwen3_omni"))

    def test_per_model_overrides_global(self) -> None:
        with patch.dict(
            os.environ,
            {"TELEMETRY": "1", "SPEECHLMM_V2_XL_TELEMETRY": "0"},
            clear=True,
        ):
            self.assertFalse(telemetry.telemetry_enabled("speechlmm_v2_xl"))
            self.assertTrue(telemetry.telemetry_enabled("qwen3_omni"))
        with patch.dict(
            os.environ,
            {"TELEMETRY": "0", "QWEN3_OMNI_TELEMETRY": "yes"},
            clear=True,
        ):
            self.assertTrue(telemetry.telemetry_enabled("qwen3_omni"))
            self.assertFalse(telemetry.telemetry_enabled("speechlmm_v2_l"))


class _Backend:
    def __init__(self) -> None:
        self.tokenizer = _WordTokenizer()
        self.calls = 0

    def initialize(self, args) -> None:
        self.initialized_with = args

    def execute(self, requests):
        self.calls += 1
        responses = []
        for request in requests:
            if request.get("fail"):
                responses.append(Response(error=RuntimeError("nope")))
            else:
                text = request.get("text", "hello there")
                responses.append(_ok(text))
        return responses


class WrapperTests(unittest.TestCase):
    def setUp(self) -> None:
        _install_pb()
        Logger.lines = []
        Logger.warns = []
        self.cls = telemetry.instrument(_Backend)

    def _model(self, on: bool, model_name: str = "speechlmm_v2_xl", **config) -> _Backend:
        env = {"TELEMETRY": "1" if on else "0"}
        parameters = {}
        if "codec_frame_rate" in config:
            parameters["codec_frame_rate"] = {"string_value": str(config["codec_frame_rate"])}
        args = {
            "model_name": model_name,
            "model_config": json.dumps({"parameters": parameters}),
        }
        with patch.dict(os.environ, env, clear=False):
            model = self.cls()
            model.initialize(args)
        return model

    def _metrics(self, response: Response) -> dict:
        for tensor in response.output_tensors():
            if tensor.name == "METRICS":
                raw = tensor.as_numpy().reshape(-1)[0]
                return json.loads(raw) if raw else {}
        raise AssertionError("METRICS missing")

    def test_off_returns_same_tensors_plus_empty_metrics(self) -> None:
        model = self._model(False)
        original = _ok("hi")
        model.execute = lambda requests: [original]  # type: ignore[method-assign]
        # The wrapper calls the superclass execute, so drive it through the class.
        response = self.cls.execute(model, [{"text": "hi"}])[0]
        names = [tensor.name for tensor in response.output_tensors()]
        self.assertEqual(names, ["GENERATED_TEXT", "FINISH_REASON", "GENERATED_AUDIO", "METRICS"])
        self.assertEqual(self._metrics(response), {})
        self.assertEqual(response.output_tensors()[0].as_numpy().tolist(), ["hi"])

    def test_off_does_no_measurement(self) -> None:
        model = self._model(False)
        with patch.object(telemetry, "wav_duration_s") as wav, patch.object(telemetry, "count_tokens") as count:
            self.cls.execute(model, [{"text": "hi"}])
        wav.assert_not_called()
        count.assert_not_called()

    def test_on_adds_metrics_and_keeps_tensors(self) -> None:
        model = self._model(True)
        prompt = json.dumps({"mode": "mt", "prompt": "one two"})
        request = {"TEXT_PROMPT": _text_tensor("TEXT_PROMPT", prompt), "text": "three"}
        response = self.cls.execute(model, [request])[0]
        names = [tensor.name for tensor in response.output_tensors()]
        self.assertEqual(names[:3], ["GENERATED_TEXT", "FINISH_REASON", "GENERATED_AUDIO"])
        metrics = self._metrics(response)
        self.assertEqual(metrics["mode"], "mt")
        self.assertEqual(metrics["input_tokens"], 2)
        self.assertEqual(metrics["input_tokens_source"], "text_only")
        self.assertEqual(metrics["output_tokens"], 1)
        self.assertEqual(metrics["output_tokens_source"], "retokenized")
        self.assertEqual(metrics["batch_size"], 1)
        self.assertIn("input_tokens_per_s", metrics)

    def test_hook_overrides_generic(self) -> None:
        class Hooked(_Backend):
            def execute(self, requests):
                self.telemetry_usage[0] = {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "generate_time_s": 0.25,
                    "ignored": 1,
                }
                return super().execute(requests)

        cls = telemetry.instrument(Hooked)
        with patch.dict(os.environ, {"TELEMETRY": "1"}, clear=False):
            model = cls()
            model.initialize({"model_name": "speechlmm_v2_xl", "model_config": "{}"})
        prompt = json.dumps({"mode": "transcribe", "prompt": "one"})
        response = cls.execute(model, [{"TEXT_PROMPT": _text_tensor("TEXT_PROMPT", prompt)}])[0]
        metrics = self._metrics(response)
        self.assertEqual(metrics["input_tokens"], 10)
        self.assertEqual(metrics["input_tokens_source"], "exact")
        self.assertEqual(metrics["output_tokens"], 2)
        self.assertEqual(metrics["generate_time_s"], 0.25)
        self.assertNotIn("ignored", metrics)

    def test_error_responses_pass_through(self) -> None:
        model = self._model(True)
        response = self.cls.execute(model, [{"fail": True}])[0]
        self.assertTrue(response.has_error())
        self.assertEqual(response.output_tensors(), [])

    def test_mixed_batch_metrics_only_on_success_and_batch_size(self) -> None:
        model = self._model(True)
        prompt = json.dumps({"mode": "chat", "prompt": "one"})
        responses = self.cls.execute(
            model,
            [
                {"fail": True},
                {"TEXT_PROMPT": _text_tensor("TEXT_PROMPT", prompt), "text": "two"},
            ],
        )
        self.assertTrue(responses[0].has_error())
        metrics = self._metrics(responses[1])
        self.assertEqual(metrics["batch_size"], 2)
        self.assertEqual(metrics["output_tokens"], 1)

    def test_response_order_and_length_preserved(self) -> None:
        model = self._model(True)
        requests = [
            {"TEXT_PROMPT": _text_tensor("TEXT_PROMPT", json.dumps({"mode": "chat", "prompt": "a"})), "text": "alpha"},
            {"TEXT_PROMPT": _text_tensor("TEXT_PROMPT", json.dumps({"mode": "mt", "prompt": "b"})), "text": "beta"},
        ]
        responses = self.cls.execute(model, requests)
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0].output_tensors()[0].as_numpy().tolist(), ["alpha"])
        self.assertEqual(self._metrics(responses[1])["mode"], "mt")

    def test_metrics_failure_falls_back_to_original(self) -> None:
        model = self._model(True)
        with patch.object(telemetry, "build_metrics", side_effect=RuntimeError("boom")):
            response = self.cls.execute(model, [{"text": "hi"}])[0]
        self.assertEqual(
            [tensor.name for tensor in response.output_tensors()],
            ["GENERATED_TEXT", "FINISH_REASON", "GENERATED_AUDIO"],
        )
        self.assertTrue(any("telemetry failed" in line for line in Logger.warns))

    def test_log_line_prefix_and_valid_json(self) -> None:
        model = self._model(True)
        self.cls.execute(model, [{"text": "hello there"}])
        lines = [line for line in Logger.lines if line.startswith("telemetry {")]
        self.assertEqual(len(lines), 1)
        parsed = json.loads(lines[0][len("telemetry ") :])
        self.assertEqual(parsed["output_tokens"], 2)

    def test_instrument_is_idempotent(self) -> None:
        again = telemetry.instrument(self.cls)
        self.assertIs(again, self.cls)

    def test_usage_cleared_between_calls(self) -> None:
        class Hooked(_Backend):
            def __init__(self) -> None:
                super().__init__()
                self.rounds = 0

            def execute(self, requests):
                self.rounds += 1
                if self.rounds == 1:
                    self.telemetry_usage[0] = {"input_tokens": 7, "output_tokens": 1}
                return super().execute(requests)

        cls = telemetry.instrument(Hooked)
        with patch.dict(os.environ, {"TELEMETRY": "1"}, clear=False):
            model = cls()
            model.initialize({"model_name": "qwen25_omni", "model_config": "{}"})
        prompt = {"TEXT_PROMPT": _text_tensor("TEXT_PROMPT", json.dumps({"mode": "chat", "prompt": "a b"}))}
        first = self._metrics(cls.execute(model, [prompt])[0])
        second = self._metrics(cls.execute(model, [prompt])[0])
        self.assertEqual(first["input_tokens_source"], "exact")
        self.assertEqual(second["input_tokens_source"], "text_only")
        self.assertEqual(second["input_tokens"], 2)

    @unittest.skipUnless(importlib.util.find_spec("soundfile"), "requires soundfile")
    def test_wav_duration_multiple_clips_summed(self) -> None:
        model = self._model(True)
        clips = [_b64(_wav_bytes(1.0)), _b64(_wav_bytes(0.5))]
        request = {
            "TEXT_PROMPT": _text_tensor("TEXT_PROMPT", json.dumps({"mode": "transcribe", "prompt": "a"})),
            "AUDIO_BYTES": Tensor("AUDIO_BYTES", np.array(clips, dtype=object)),
            "text": "b",
        }
        metrics = self._metrics(self.cls.execute(model, [request])[0])
        self.assertAlmostEqual(metrics["input_audio_s"], 1.5, places=3)
        self.assertEqual(metrics["input_audio_clips"], 2)
        self.assertIn("rtf", metrics)


if __name__ == "__main__":
    unittest.main()
