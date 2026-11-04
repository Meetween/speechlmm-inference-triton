#!/usr/bin/env python3
"""Tests for the shared Qwen-Omni backend. No Triton runtime; numpy and torch only."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

import numpy as np

_REPO = Path(__file__).resolve().parents[1]


class FakeStrTensor:
    """A Triton string tensor: shape [1, N] of UTF-8 payloads."""

    def __init__(self, value: Any) -> None:
        self._values = [value] if isinstance(value, str) else list(value)

    def as_numpy(self) -> Any:
        return np.array([[value.encode() for value in self._values]], dtype=object)


def _load_qwen_backend():
    backends = str(_REPO / "backends")
    if backends not in sys.path:
        sys.path.insert(0, backends)
    sys.modules.setdefault("triton_python_backend_utils", types.ModuleType("stub"))
    spec = importlib.util.spec_from_file_location(
        "qwen_omni_backend_under_test", _REPO / "backends" / "qwen_omni_backend.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


qwen = _load_qwen_backend()


def _load_shim(model_dir: str):
    spec = importlib.util.spec_from_file_location(
        f"{model_dir}_shim_under_test", _REPO / "models" / model_dir / "1" / "model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ShimAndConfigTests(unittest.TestCase):
    def test_qwen_shims_expose_the_shared_class(self) -> None:
        for name in ("qwen25_omni", "qwen3_omni"):
            with self.subTest(name=name):
                shim = _load_shim(name)
                import qwen_omni_backend

                # The shim wraps the shared class so Triton can attach METRICS
                # without changing the class the in-process tests instantiate.
                self.assertTrue(issubclass(shim.TritonPythonModel, qwen_omni_backend.TritonPythonModel))
                self.assertIs(shim.TritonPythonModel.__bases__[0], qwen_omni_backend.TritonPythonModel)
                self.assertTrue(shim.TritonPythonModel._telemetry_instrumented)

    def test_every_model_dir_config_name_matches_the_directory(self) -> None:
        models = _REPO / "models"
        for config in sorted(models.glob("*/config.pbtxt")):
            match = re.search(r'^name:\s*"([^"]+)"', config.read_text(), re.M)
            self.assertIsNotNone(match, config)
            self.assertEqual(match.group(1), config.parent.name, config)


class EnvAndPlacementTests(unittest.TestCase):
    def test_model_path_wins_over_the_compose_fallback(self) -> None:
        env = {
            "QWEN25_OMNI_MODEL_PATH": "/ckpt",
            "QWEN25_OMNI_MODEL": "Qwen/Qwen2.5-Omni-7B",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(qwen._resolve_model_id("qwen25_omni"), "/ckpt")

    def test_compose_fallback_is_used_when_model_path_is_absent(self) -> None:
        with patch.dict(os.environ, {"QWEN25_OMNI_MODEL": "Qwen/Qwen2.5-Omni-7B"}, clear=False):
            os.environ.pop("QWEN25_OMNI_MODEL_PATH", None)
            self.assertEqual(qwen._resolve_model_id("qwen25_omni"), "Qwen/Qwen2.5-Omni-7B")

    def test_missing_both_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("QWEN25_OMNI_MODEL_PATH", None)
            os.environ.pop("QWEN25_OMNI_MODEL", None)
            with self.assertRaises(RuntimeError) as caught:
                qwen._resolve_model_id("qwen25_omni")
        self.assertIn("QWEN25_OMNI_MODEL_PATH", str(caught.exception))
        self.assertIn("QWEN25_OMNI_MODEL", str(caught.exception))

    def test_absent_parameters_default_to_instance_cuda_and_float16(self) -> None:
        with patch.dict(os.environ, {"QWEN25_OMNI_MODEL": "Qwen/Qwen2.5-Omni-7B"}, clear=False):
            os.environ.pop("QWEN25_OMNI_MODEL_PATH", None)
            os.environ.pop("QWEN25_OMNI_DEVICE_MAP", None)
            placement = qwen._load_placement(
                {"model_name": "qwen25_omni", "model_instance_device_id": "1"}
            )
        self.assertEqual(placement["device_map"], {"": "cuda:1"})
        self.assertEqual(placement["dtype_name"], "float16")

    def test_config_parameters_and_env_device_map_override(self) -> None:
        config = {
            "parameters": {
                "device_map": {"string_value": "cuda:0"},
                "dtype": {"string_value": "bfloat16"},
            }
        }
        with patch.dict(
            os.environ,
            {
                "QWEN25_OMNI_MODEL_PATH": "/ckpt",
                "QWEN25_OMNI_DEVICE_MAP": "auto",
            },
            clear=False,
        ):
            placement = qwen._load_placement(
                {
                    "model_name": "qwen25_omni",
                    "model_config": json.dumps(config),
                    "model_instance_device_id": "0",
                }
            )
        self.assertEqual(placement["device_map"], "auto")
        self.assertEqual(placement["dtype_name"], "bfloat16")

    def test_qwen3_config_parameters_are_auto_and_bfloat16(self) -> None:
        text = (_REPO / "models" / "qwen3_omni" / "config.pbtxt").read_text()
        self.assertIn('string_value: "auto"', text)
        self.assertIn('string_value: "bfloat16"', text)
        self.assertIn("KIND_MODEL", text)
        self.assertEqual(qwen._resolve_dtype_name("bfloat16"), "bfloat16")
        self.assertEqual(qwen._resolve_device_map("auto", 0), "auto")

    def test_eos_comes_from_generation_config_then_tokenizer_then_raises(self) -> None:
        gen = types.SimpleNamespace(eos_token_id=9, pad_token_id=0)
        tok = types.SimpleNamespace(eos_token_id=8, pad_token_id=1)
        self.assertEqual(qwen._resolve_eos_pad(gen, tok, "m"), (9, 0))
        self.assertEqual(
            qwen._resolve_eos_pad(types.SimpleNamespace(eos_token_id=None, pad_token_id=None), tok, "m"),
            (8, 1),
        )
        self.assertEqual(
            qwen._resolve_eos_pad(None, types.SimpleNamespace(eos_token_id=8, pad_token_id=None), "m"),
            (8, 8),
        )
        with self.assertRaises(RuntimeError) as caught:
            qwen._resolve_eos_pad(None, types.SimpleNamespace(eos_token_id=None, pad_token_id=None), "ckpt")
        self.assertIn("ckpt", str(caught.exception))
        self.assertIn("eos", str(caught.exception))

    def test_omni_family_from_model_type(self) -> None:
        self.assertEqual(qwen._omni_family("qwen2_5_omni"), "qwen2_5_omni")
        self.assertEqual(qwen._omni_family("qwen3_omni_moe"), "qwen3_omni")
        with self.assertRaises(RuntimeError):
            qwen._omni_family("llama")


class ParseTextPromptParityTests(unittest.TestCase):
    CASES = (
        ("", {"mode": "chat", "prompt": ""}),
        ("hello", {"mode": "chat", "prompt": "hello"}),
        (
            '{"mode": "transcribe", "prompt": "Write it down."}',
            {
                "mode": "transcribe",
                "prompt": "Write it down.",
                "system": None,
                "format": None,
                "language": None,
                "caption_chars": None,
                "return_audio": False,
                "speaker": None,
            },
        ),
        (
            '{"mode": "caption", "text": "Cap.", "format": "clean", "language": "it", "caption_chars": 40}',
            {
                "mode": "caption",
                "prompt": "Cap.",
                "system": None,
                "format": "clean",
                "language": "it",
                "caption_chars": 40,
                "return_audio": False,
                "speaker": None,
            },
        ),
        (
            '{"mode": "unknown", "prompt": "go"}',
            {
                "mode": "unknown",
                "prompt": "go",
                "system": None,
                "format": None,
                "language": None,
                "caption_chars": None,
                "return_audio": False,
                "speaker": None,
            },
        ),
        (
            '{"mode": "chat", "prompt": null}',
            {
                "mode": "chat",
                "prompt": "",
                "system": None,
                "format": None,
                "language": None,
                "caption_chars": None,
                "return_audio": False,
                "speaker": None,
            },
        ),
        ("mode:transcribe\nWrite it down.", {"mode": "transcribe", "prompt": "Write it down."}),
        ("[caption]\nCap.", {"mode": "caption", "prompt": "Cap."}),
        ("{not json}", {"mode": "chat", "prompt": "{not json}"}),
    )

    def test_envelopes_match_the_pre_refactor_parser(self) -> None:
        for raw, expected in self.CASES:
            with self.subTest(raw=raw):
                self.assertEqual(qwen._parse_text_prompt(raw), expected)


class RefuseUnsupportedRequestTests(unittest.TestCase):
    NAME = "qwen25_omni"

    def test_only_lipread_is_refused(self) -> None:
        cases = (
            ("lipread", "read it", True),
            ("chat", "say <lipread> please", True),
            ("avspeakd", "who speaks", False),
            ("chat", "describe this", False),
            ("transcribe", "write it down", False),
        )
        for mode, prompt, refuse in cases:
            with self.subTest(mode=mode, prompt=prompt):
                if not refuse:
                    qwen._reject_unsupported_request(mode, prompt, self.NAME)
                    continue
                with self.assertRaises(qwen.UnsupportedRequest) as caught:
                    qwen._reject_unsupported_request(mode, prompt, self.NAME)
                message = str(caught.exception)
                self.assertIn("mouth-ROI", message)
                self.assertIn(self.NAME, message)
                # The refusal explains this model, and does not route clients elsewhere.
                self.assertNotIn("speechlmm", message.lower())

    def test_triton_error_uses_invalid_arg_for_unsupported_requests(self) -> None:
        class TritonError(Exception):
            INVALID_ARG = 3

            def __init__(self, message, code=None):
                super().__init__(message)
                self.code = code

        qwen.pb_utils.TritonError = TritonError
        err = qwen._triton_error(qwen.UnsupportedRequest("nope"))
        self.assertIsInstance(err, TritonError)
        self.assertEqual(err.code, 3)
        other = qwen._triton_error(ValueError("boom"))
        self.assertIsInstance(other, TritonError)
        self.assertIsNone(other.code)

    def test_triton_error_falls_back_when_code_is_not_accepted(self) -> None:
        class OldTritonError(Exception):
            INVALID_ARG = 3

            def __init__(self, message):
                super().__init__(message)

        qwen.pb_utils.TritonError = OldTritonError
        err = qwen._triton_error(qwen.UnsupportedRequest("nope"))
        self.assertIsInstance(err, OldTritonError)

    def test_execute_returns_an_error_response_instead_of_raising(self) -> None:
        class TritonError(Exception):
            INVALID_ARG = 3

            def __init__(self, message, code=None):
                super().__init__(message)
                self.code = code

        class Response:
            def __init__(self, output_tensors=None, error=None):
                self.output_tensors = output_tensors
                self.error = error

        class FakeTensor:
            def __init__(self, value):
                self._value = value

            def as_numpy(self):
                import numpy as np

                if isinstance(self._value, (list, tuple)):
                    return np.array([list(self._value)], dtype=object)
                return np.array([self._value])

        qwen.pb_utils.TritonError = TritonError
        qwen.pb_utils.InferenceResponse = lambda **kw: Response(**kw)
        qwen.pb_utils.get_input_tensor_by_name = lambda req, name: req.get(name)
        model = object.__new__(qwen.TritonPythonModel)
        model.model_name = "qwen25_omni"
        model.default_max_new_tokens = 8
        model.default_do_sample = False
        model.default_temperature = 0.0
        request = {
            "TEXT_PROMPT": FakeTensor('{"mode": "lipread", "prompt": "read it"}'),
            "VIDEO_BYTES": FakeTensor("QQ=="),
        }
        responses = model.execute([request])
        self.assertEqual(len(responses), 1)
        self.assertIsNotNone(responses[0].error)
        self.assertEqual(responses[0].error.code, 3)
        self.assertIn("mouth-ROI", str(responses[0].error))


class TranslationModeTests(unittest.TestCase):
    def test_speech_modes_still_require_audio(self) -> None:
        for mode in ("transcribe", "caption", "summary", "st"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "requires audio"):
                qwen._conversation_messages({"mode": mode, "prompt": "Translate."}, [], [])

    def test_mt_executes_without_audio_and_keeps_optional_audio_silence_check(self) -> None:
        import torch

        class Inputs(dict):
            def to(self, *_args, **_kwargs):
                return self

        prompt = "Translate this text into Italian.\n\nGood morning."
        for name in ("qwen25_omni", "qwen3_omni"):
            for silent_audio in (False, True):
                with self.subTest(model=name, silent_audio=silent_audio):
                    seen = []
                    vad_calls = []

                    def apply_chat_template(conversations, **kwargs):
                        seen.extend(conversations)
                        return Inputs(input_ids=torch.tensor([[1, 2]]))

                    model = object.__new__(qwen.TritonPythonModel)
                    model.model_name = name
                    model.default_max_new_tokens = 4
                    model.default_do_sample = False
                    model.default_temperature = 0.0
                    model.device = "cpu"
                    model.model_dtype = torch.float32
                    model.eos_token_id = 9
                    model.pad_token_id = None
                    model._torch = torch
                    model._has_speech = lambda blob: vad_calls.append(blob) or False
                    model.processor = SimpleNamespace(
                        apply_chat_template=apply_chat_template,
                        batch_decode=lambda rows, **kw: ["Buongiorno."],
                    )
                    model.model = SimpleNamespace(
                        generate=lambda **kw: torch.tensor([[1, 2, 7, 9]])
                    )
                    model.telemetry_usage = {}
                    request = {"TEXT_PROMPT": FakeStrTensor(json.dumps({"mode": "mt", "prompt": prompt}))}
                    if silent_audio:
                        request["AUDIO_BYTES"] = FakeStrTensor(base64.b64encode(b"silence").decode())
                    with patch.object(qwen.pb_utils, "Tensor", lambda name, data: (name, data.tolist()), create=True), \
                         patch.object(qwen.pb_utils, "InferenceResponse", lambda **kw: kw, create=True), \
                         patch.object(qwen.pb_utils, "get_input_tensor_by_name", lambda req, name: req.get(name), create=True):
                        responses = model.execute([request])
                    self.assertEqual(responses[0]["output_tensors"], [
                        ("GENERATED_TEXT", ["" if silent_audio else "Buongiorno."]),
                        ("FINISH_REASON", ["stop"]),
                        ("GENERATED_AUDIO", [""]),
                    ])
                    self.assertEqual(vad_calls, [b"silence"] if silent_audio else [])
                    self.assertEqual(seen, [] if silent_audio else [
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
                    ])
                    if silent_audio:
                        self.assertEqual(
                            model.telemetry_usage[0],
                            {"skipped": "silence", "output_tokens": 0},
                        )


class VerbatimPromptTests(unittest.TestCase):
    def test_every_mode_passes_the_prompt_through_verbatim(self) -> None:
        prompt = "Transcribe the following speech."
        for mode in ("transcribe", "caption", "summary", "chat"):
            with self.subTest(mode=mode):
                spec = {"mode": mode, "prompt": prompt}
                self.assertEqual(
                    qwen._build_mode_instruction(spec, has_audio=mode != "chat", has_video=False),
                    prompt,
                )
                messages = qwen._conversation_messages(
                    spec, [] if mode == "chat" else ["wav"], []
                )
                user_text = "".join(
                    part["text"] for part in messages[-1]["content"] if part["type"] == "text"
                )
                self.assertEqual(user_text, prompt)
                self.assertNotIn("Additional instruction", user_text)
                self.assertNotIn("Do not summarize", user_text)

    def test_absent_prompt_is_not_replaced_with_a_default(self) -> None:
        for mode in ("transcribe", "caption", "summary", "chat"):
            with self.subTest(mode=mode):
                spec = {"mode": mode, "prompt": ""}
                self.assertEqual(
                    qwen._build_mode_instruction(spec, has_audio=mode != "chat", has_video=False), ""
                )

    def test_envelope_system_is_the_only_system_message(self) -> None:
        spec = {"mode": "transcribe", "prompt": "Write it down.", "system": "Be terse."}
        messages = qwen._conversation_messages(spec, ["wav"], [])
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"][0]["text"], "Be terse.")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])

    def test_no_system_message_when_the_envelope_omits_it(self) -> None:
        spec = {"mode": "chat", "prompt": "Hi."}
        messages = qwen._conversation_messages(spec, [], [])
        self.assertEqual([m["role"] for m in messages], ["user"])


class MediaPartsTests(unittest.TestCase):
    """Video reaches the vision tower, and several clips keep the client's order."""

    def test_parts_decide_the_content_order(self) -> None:
        spec = {
            "mode": "avspeakd",
            "prompt": "Who speaks?",
            "parts": [
                {"type": "text", "text": "Who speaks? "},
                {"type": "video"},
                {"type": "audio"},
            ],
        }
        content = qwen._conversation_messages(spec, ["wav"], ["/tmp/a.mp4"])[-1]["content"]
        self.assertEqual(
            [(part["type"], part.get("audio") or part.get("video") or part["text"]) for part in content],
            [("text", "Who speaks? "), ("video", "/tmp/a.mp4"), ("audio", "wav")],
        )

    def test_audio_then_text_is_unchanged(self) -> None:
        spec = {
            "mode": "chat",
            "prompt": "after",
            "parts": [{"type": "audio"}, {"type": "text", "text": "after"}],
        }
        content = qwen._conversation_messages(spec, ["wav"], [])[-1]["content"]
        self.assertEqual(
            [(part["type"], part.get("audio") or part.get("text")) for part in content],
            [("audio", "wav"), ("text", "after")],
        )

    def test_default_order_is_audio_then_video_and_text_last(self) -> None:
        spec = {"mode": "chat", "prompt": "Describe."}
        content = qwen._conversation_messages(spec, ["w1", "w2"], ["/tmp/a.mp4"])[-1]["content"]
        self.assertEqual(
            [part["type"] for part in content], ["audio", "audio", "video", "text"]
        )

    def test_part_counts_must_match_the_clips(self) -> None:
        for raw in ([{"type": "audio"}, {"type": "audio"}], [{"type": "audio"}, {"type": "lipread"}]):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                qwen._normalize_parts(raw, 1, 1, "")

    def test_json_envelope_keeps_parts(self) -> None:
        spec = qwen._parse_text_prompt(
            '{"mode": "chat", "prompt": "Input: ", "parts": '
            '[{"type": "text", "text": "Input: "}, {"type": "audio"}]}'
        )
        self.assertEqual(spec["parts"][0]["text"], "Input: ")
        self.assertEqual(spec["parts"][1], {"type": "audio"})

    def test_avspeakd_without_video_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            qwen._build_mode_instruction(
                {"mode": "avspeakd", "prompt": "Who speaks?"}, has_audio=True, has_video=False
            )
        self.assertIn("requires video", str(caught.exception))

    def test_clips_are_written_to_disk_and_removed_after_the_request(self) -> None:
        model = object.__new__(qwen.TritonPythonModel)
        paths = model._write_videos([base64.b64encode(b"mp4-bytes").decode()])
        try:
            self.assertTrue(paths[0].endswith(".mp4"))
            with open(paths[0], "rb") as handle:
                self.assertEqual(handle.read(), b"mp4-bytes")
        finally:
            qwen._unlink_quietly(paths)
        self.assertFalse(os.path.exists(paths[0]))

    def test_undecodable_video_leaves_nothing_behind(self) -> None:
        model = object.__new__(qwen.TritonPythonModel)
        written: List[str] = []
        real = tempfile.NamedTemporaryFile

        def spy(*args, **kwargs):
            handle = real(*args, **kwargs)
            written.append(handle.name)
            return handle

        with patch.object(tempfile, "NamedTemporaryFile", spy):
            with self.assertRaises(ValueError) as caught:
                model._write_videos([base64.b64encode(b"ok").decode(), "!!not base64!!"])
        self.assertIn("VIDEO_BYTES", str(caught.exception))
        self.assertEqual([path for path in written if os.path.exists(path)], [])

    def test_execute_sends_video_to_the_processor_and_cleans_up(self) -> None:
        import numpy as np
        import torch

        class Inputs(dict):
            def to(self, *_args, **_kwargs):
                return self

        seen: Dict[str, Any] = {}

        def apply_chat_template(conversations, **_kwargs):
            seen["conversations"] = conversations
            seen["existed"] = [
                os.path.exists(part["video"])
                for part in conversations[-1]["content"]
                if part["type"] == "video"
            ]
            return Inputs(input_ids=torch.tensor([[1, 2]]))

        model = object.__new__(qwen.TritonPythonModel)
        model.model_name = "qwen3_omni"
        model.default_max_new_tokens = 4
        model.default_do_sample = False
        model.default_temperature = 0.0
        model.device = "cpu"
        model.model_dtype = torch.float32
        model.eos_token_id = 9
        model.pad_token_id = None
        model._torch = torch
        model.processor = SimpleNamespace(
            apply_chat_template=apply_chat_template,
            batch_decode=lambda rows, **kw: ["speaker 1"],
        )
        model.model = SimpleNamespace(
            generate=lambda **kw: torch.tensor([[1, 2, 7, 9]])
        )
        model._decode_wav = lambda blob: (np.zeros(4, dtype="float32"), 16000)

        request = {
            "TEXT_PROMPT": FakeStrTensor('{"mode": "avspeakd", "prompt": "Who speaks?"}'),
            "AUDIO_BYTES": FakeStrTensor([base64.b64encode(b"wav").decode()]),
            "VIDEO_BYTES": FakeStrTensor([base64.b64encode(b"mp4").decode()]),
        }
        with patch.object(qwen.pb_utils, "Tensor", lambda name, data: (name, data.tolist()), create=True), \
             patch.object(qwen.pb_utils, "InferenceResponse", lambda **kw: kw, create=True), \
             patch.object(qwen.pb_utils, "get_input_tensor_by_name", lambda req, name: req.get(name), create=True):
            responses = model.execute([request])

        self.assertEqual(
            responses[0]["output_tensors"],
            [("GENERATED_TEXT", ["speaker 1"]), ("FINISH_REASON", ["stop"]), ("GENERATED_AUDIO", [""])],
        )
        content = seen["conversations"][-1]["content"]
        self.assertEqual([part["type"] for part in content], ["audio", "video", "text"])
        self.assertEqual(seen["existed"], [True])
        video_path = [part["video"] for part in content if part["type"] == "video"][0]
        self.assertFalse(os.path.exists(video_path))


if __name__ == "__main__":
    unittest.main()
