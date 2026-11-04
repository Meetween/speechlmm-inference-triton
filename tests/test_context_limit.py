#!/usr/bin/env python3
"""Context-length check: config lookup, warn/reject policy, and both backends' generate paths."""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_backend_prompts import backend
from test_qwen_omni_backend import FakeStrTensor, qwen

import backend_common as common

try:
    import torch
except ImportError:
    torch = None


class Logger:
    def __init__(self):
        self.warnings = []

    def log_warn(self, message):
        self.warnings.append(message)

    def log_info(self, message):
        pass


class TritonError(Exception):
    INVALID_ARG = 3

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class ContextLimitTests(unittest.TestCase):
    def test_reads_the_language_model_limit_of_each_config_shape(self) -> None:
        omni = SimpleNamespace(
            thinker_config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=32768)),
            max_position_embeddings=999,
        )
        self.assertEqual(common.context_limit(omni), 32768)
        as_dict = SimpleNamespace(thinker_config={"text_config": {"max_position_embeddings": 65536}})
        self.assertEqual(common.context_limit(as_dict), 65536)
        thinker = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=4096))
        self.assertEqual(common.context_limit(thinker), 4096)
        self.assertEqual(common.context_limit(SimpleNamespace(max_position_embeddings=2048)), 2048)
        self.assertIsNone(common.context_limit(SimpleNamespace()))

    def test_policy_defaults_to_warn_and_the_model_setting_wins(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(common.context_policy("speechlmm_v2_l"), "warn")
        with patch.dict(os.environ, {"CONTEXT_OVERFLOW": "reject"}, clear=True):
            self.assertEqual(common.context_policy("speechlmm_v2_l"), "reject")
        with patch.dict(
            os.environ,
            {"CONTEXT_OVERFLOW": "reject", "SPEECHLMM_V2_L_CONTEXT_OVERFLOW": "warn"},
            clear=True,
        ):
            self.assertEqual(common.context_policy("speechlmm_v2_l"), "warn")
        with patch.dict(os.environ, {"CONTEXT_OVERFLOW": "truncate"}, clear=True):
            with self.assertRaisesRegex(ValueError, "'warn' or 'reject'"):
                common.context_policy("speechlmm_v2_l")

    def test_check_counts_the_output_budget_and_follows_the_policy(self) -> None:
        logger = Logger()
        owner = SimpleNamespace(
            context_limit=100, context_policy="warn", model_name="m", _pb=lambda: SimpleNamespace(Logger=logger)
        )
        self.assertFalse(common.check_context(owner, 90, 10))
        self.assertEqual(logger.warnings, [])
        self.assertTrue(common.check_context(owner, 90, 11))
        self.assertEqual(len(logger.warnings), 1)
        self.assertIn("90 input tokens plus max_new_tokens=11 exceed the 100-token context of m", logger.warnings[0])
        owner.context_policy = "reject"
        with self.assertRaises(common.ContextOverflow):
            common.check_context(owner, 90, 11)
        owner.context_limit = None
        self.assertFalse(common.check_context(owner, 10**6, 10))

    def test_base_backend_reports_overflow_as_invalid_argument(self) -> None:
        model = backend.TritonPythonModel()
        with patch.object(backend.pb_utils, "TritonError", TritonError, create=True), \
             patch.object(backend.pb_utils, "InferenceResponse", lambda **kw: kw, create=True):
            response = model.map_error(common.ContextOverflow("too long"))
            other = model.map_error(RuntimeError("boom"))
        self.assertEqual(response["error"].code, 3)
        self.assertIsNone(other["error"].code)


@unittest.skipIf(torch is None, "requires torch from the serving image")
class SpeechLmmContextTests(unittest.TestCase):
    def _model(self, *, limit, policy, input_len):
        model = backend.TritonPythonModel()
        model.context_limit = limit
        model.context_policy = policy
        model.model_name = "speechlmm_v2_l"
        model.calls = 0
        model._primary_device = lambda: "cpu"
        model._tokenize_sample = lambda sample: {"input_ids": torch.ones(1, input_len, dtype=torch.long)}
        model._to_device_dict = lambda raw, device: raw
        model._attach_lipread = lambda *args, **kwargs: None
        model._place_vision = lambda *args, **kwargs: None
        model._cast_floats = lambda raw: raw
        model._collate = lambda rows: rows[0]

        def generate_once(inputs, **kwargs):
            model.calls += 1
            model._last_usage = [{"input_tokens": input_len, "output_tokens": 1}]
            return [("hi", "stop")]

        model._generate_once = generate_once
        return model

    def _sample(self):
        from test_speech_output import _sample

        return _sample(mode="transcribe", return_audio=False, max_new_tokens=8)

    def test_warn_generates_and_flags_the_usage(self) -> None:
        logger = Logger()
        model = self._model(limit=100, policy="warn", input_len=95)
        with patch.object(backend.pb_utils, "Logger", logger, create=True):
            results = model._generate_group([self._sample()])
        self.assertEqual(results, [("hi", "stop", "")])
        self.assertEqual(model.calls, 1)
        self.assertIs(model._pending_usage[0]["context_overflow"], True)
        self.assertEqual(len(logger.warnings), 1)

    def test_within_the_limit_nothing_is_flagged(self) -> None:
        logger = Logger()
        model = self._model(limit=100, policy="reject", input_len=92)
        with patch.object(backend.pb_utils, "Logger", logger, create=True):
            model._generate_group([self._sample()])
        self.assertNotIn("context_overflow", model._pending_usage[0])
        self.assertEqual(logger.warnings, [])

    def test_reject_fails_before_generating(self) -> None:
        model = self._model(limit=100, policy="reject", input_len=95)
        with self.assertRaises(common.ContextOverflow):
            model._generate_group([self._sample()])
        self.assertEqual(model.calls, 0)


@unittest.skipIf(torch is None, "requires torch from the serving image")
class QwenContextTests(unittest.TestCase):
    def _run(self, *, policy, input_len):
        class Inputs(dict):
            def to(self, *_args, **_kwargs):
                return self

        calls = []
        model = object.__new__(qwen.TritonPythonModel)
        model.model_name = "qwen25_omni"
        model.context_limit = 10
        model.context_policy = policy
        model.default_max_new_tokens = 4
        model.default_do_sample = False
        model.default_temperature = 0.0
        model.device = "cpu"
        model.model_dtype = torch.float32
        model.eos_token_id = 9
        model.pad_token_id = None
        model._torch = torch
        model.processor = SimpleNamespace(
            apply_chat_template=lambda conversations, **kw: Inputs(
                input_ids=torch.ones(1, input_len, dtype=torch.long)
            ),
            batch_decode=lambda rows, **kw: ["ok"],
        )
        model.model = SimpleNamespace(
            generate=lambda **kw: calls.append(kw) or torch.ones(1, input_len + 1, dtype=torch.long) * 9
        )
        model.telemetry_usage = {}
        logger = Logger()
        request = {"TEXT_PROMPT": FakeStrTensor(json.dumps({"mode": "mt", "prompt": "Translate."}))}
        with patch.object(qwen.pb_utils, "Tensor", lambda name, data: (name, data.tolist()), create=True), \
             patch.object(qwen.pb_utils, "InferenceResponse", lambda **kw: kw, create=True), \
             patch.object(qwen.pb_utils, "TritonError", TritonError, create=True), \
             patch.object(qwen.pb_utils, "Logger", logger, create=True), \
             patch.object(qwen.pb_utils, "get_input_tensor_by_name", lambda req, name: req.get(name), create=True):
            response = model.execute([request])[0]
        return response, calls, model.telemetry_usage, logger

    def test_warn_generates_and_reports_overflow_in_telemetry(self) -> None:
        response, calls, usage, logger = self._run(policy="warn", input_len=8)
        self.assertNotIn("error", response)
        self.assertEqual(len(calls), 1)
        self.assertIs(usage[0]["context_overflow"], True)
        self.assertEqual(len(logger.warnings), 1)

    def test_reject_answers_invalid_argument_without_generating(self) -> None:
        response, calls, _usage, _logger = self._run(policy="reject", input_len=8)
        self.assertEqual(calls, [])
        self.assertEqual(response["error"].code, 3)
        self.assertIn("exceed the 10-token context of qwen25_omni", str(response["error"]))

    def test_short_request_is_untouched(self) -> None:
        response, calls, usage, logger = self._run(policy="reject", input_len=4)
        self.assertNotIn("error", response)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("context_overflow", usage.get(0, {}))
        self.assertEqual(logger.warnings, [])


if __name__ == "__main__":
    unittest.main()
