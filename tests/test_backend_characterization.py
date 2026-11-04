#!/usr/bin/env python3
"""Pin observable execute() behavior of both backends, using fake models.

These tests are the regression net for the BaseBackend migration. They check
outputs, error text and the kwargs passed to generate, not private call order.
"""

from __future__ import annotations

import base64
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_backend_prompts import backend
from test_qwen_omni_backend import FakeStrTensor, qwen

try:
    import torch
except ImportError:
    torch = None


def _pb_patch(module):
    return (
        patch.object(module.pb_utils, "Tensor", lambda name, data: (name, data.tolist()), create=True),
        patch.object(module.pb_utils, "InferenceResponse", lambda **kw: SimpleNamespace(**kw), create=True),
        patch.object(module.pb_utils, "TritonError", lambda message, code=None: RuntimeError(message), create=True),
        patch.object(module.pb_utils, "get_input_tensor_by_name", lambda req, name: req.get(name), create=True),
        patch.object(module.pb_utils, "Logger", SimpleNamespace(log_warn=lambda *a, **k: None, log_info=lambda *a, **k: None), create=True),
    )


def _enter(test, patches):
    for item in patches:
        item.start()
        test.addCleanup(item.stop)


def _prompt(mode: str, prompt: str, **extra) -> dict:
    payload = {"mode": mode, "prompt": prompt, **extra}
    return {"TEXT_PROMPT": FakeStrTensor(json.dumps(payload))}


def _error_of(response):
    return getattr(response, "error", None)


def _text_of(response) -> str:
    tensors = response.output_tensors
    for name, data in tensors:
        if name == "GENERATED_TEXT":
            return data[0]
    raise AssertionError(tensors)


@unittest.skipIf(torch is None, "requires torch from the serving image")
class SpeechLmmExecuteTests(unittest.TestCase):
    def setUp(self) -> None:
        _enter(self, _pb_patch(backend))
        self.calls = []
        weight = torch.zeros(1)
        model = backend.TritonPythonModel()
        model._torch = torch
        model.speech_enabled = False
        model.default_max_new_tokens = 8
        model.default_do_sample = False
        model.default_temperature = 0.0
        model._long_audio = False

        def generate(**kwargs):
            self.calls.append(kwargs)
            ids = kwargs["input_ids"]
            extra = torch.tensor([[7, 9]]).repeat(ids.shape[0], 1)
            return torch.cat([ids, extra], dim=-1)

        model.model = SimpleNamespace(
            generate=generate,
            get_input_embeddings=lambda: SimpleNamespace(weight=weight),
            generation_config=SimpleNamespace(eos_token_id=9),
        )
        seen = {}

        def apply_chat_template(messages, **kwargs):
            seen["messages"] = messages
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

        model.tokenizer = SimpleNamespace(
            eos_token_id=9,
            pad_token_id=0,
            apply_chat_template=apply_chat_template,
            batch_decode=lambda rows, **kw: ["hello world"] * len(rows),
        )
        model.processor = None
        model._uses_qwen25_audio_system = lambda: False
        self.model = model
        self.seen = seen

    def test_chat_text_returns_decoded_text_and_kwargs(self) -> None:
        response = self.model.execute([_prompt("chat", "Say hi")])[0]
        self.assertIsNone(_error_of(response))
        self.assertEqual(_text_of(response), "hello world")
        self.assertEqual(self.calls[0]["max_new_tokens"], 8)
        self.assertFalse(self.calls[0]["do_sample"])
        self.assertEqual(self.seen["messages"][-1]["content"], "Say hi")

    def test_text_summary_empty_prompt_is_an_error(self) -> None:
        response = self.model.execute([_prompt("text_summary", "")])[0]
        self.assertIn("text_summary", str(_error_of(response)))
        self.assertEqual(self.calls, [])

    def test_transcribe_without_audio_is_an_error(self) -> None:
        response = self.model.execute([_prompt("transcribe", "Transcribe.")])[0]
        self.assertIn("requires audio", str(_error_of(response)))

    def test_caption_is_wrapped(self) -> None:
        self.model.tokenizer.batch_decode = lambda rows, **kw: ["alpha beta " * 12]
        request = _prompt("caption", "Caption.", caption_chars=8)
        # Caption requires audio. Feed a decoded wav by stubbing _decode_wav and a fake clip.
        request["AUDIO_BYTES"] = FakeStrTensor(base64.b64encode(b"wav").decode())
        self.model._decode_wav = lambda blob: (torch.zeros(4).numpy(), 16000)
        self.model.processor = SimpleNamespace(
            apply_chat_template=lambda conversations, **kw: {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            },
            batch_decode=lambda rows, **kw: ["alpha beta " * 12],
            feature_extractor=SimpleNamespace(sampling_rate=16000),
        )
        response = self.model.execute([request])[0]
        self.assertIsNone(_error_of(response))
        self.assertIn("\n", _text_of(response))

    def test_two_same_key_requests_share_one_generate(self) -> None:
        self.model.execute([_prompt("chat", "a"), _prompt("chat", "b")])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["input_ids"].shape[0], 2)

    def test_response_order_matches_requests(self) -> None:
        self.model.tokenizer.batch_decode = lambda rows, **kw: [f"row{i}" for i in range(len(rows))]
        responses = self.model.execute([_prompt("mt", "left"), _prompt("mt", "right")])
        self.assertEqual([_text_of(item) for item in responses], ["row0", "row1"])


@unittest.skipIf(torch is None, "requires torch from the serving image")
class QwenExecuteTests(unittest.TestCase):
    def setUp(self) -> None:
        _enter(self, _pb_patch(qwen))
        self.seen = {}

        class Inputs(dict):
            def to(self, *_args, **_kwargs):
                return self

        def apply_chat_template(conversations, **kwargs):
            self.seen["conversations"] = conversations
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
        model._long_audio = False
        model.speech_enabled = False
        model.omni_family = "qwen3_omni"
        model.processor = SimpleNamespace(
            apply_chat_template=apply_chat_template,
            batch_decode=lambda rows, **kw: ["Buongiorno."],
        )
        model.model = SimpleNamespace(generate=lambda **kw: torch.tensor([[1, 2, 7, 9]]))
        model._has_speech = lambda blob: True
        self.model = model

    def test_mt_keeps_the_client_prompt(self) -> None:
        prompt = "Translate this text into Italian.\n\nGood morning."
        response = self.model.execute([_prompt("mt", prompt)])[0]
        self.assertIsNone(_error_of(response))
        self.assertEqual(_text_of(response), "Buongiorno.")
        self.assertEqual(self.seen["conversations"][-1]["content"][0]["text"], prompt)

    def test_lipread_is_rejected(self) -> None:
        request = _prompt("lipread", "read it")
        request["VIDEO_BYTES"] = FakeStrTensor("QQ==")
        response = self.model.execute([request])[0]
        self.assertIsNotNone(_error_of(response))
        self.assertIn("mouth-ROI", str(_error_of(response)))

    def test_silence_skips_generate(self) -> None:
        self.model._has_speech = lambda blob: False
        self.model.telemetry_usage = {}
        request = _prompt("transcribe", "Transcribe.")
        request["AUDIO_BYTES"] = FakeStrTensor(base64.b64encode(b"silence").decode())
        response = self.model.execute([request])[0]
        self.assertEqual(_text_of(response), "")
        self.assertEqual(self.model.telemetry_usage[0]["skipped"], "silence")
        self.assertNotIn("conversations", self.seen)


if __name__ == "__main__":
    unittest.main()
