"""Speech-output paths on the SpeechLMM and Qwen backends, with fake models."""

from __future__ import annotations

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


def _sample(**overrides):
    fields = dict(
        index=0,
        mode="tts",
        user_prompt="Say this.",
        system_prompt="",
        use_system=False,
        wav_data=[],
        lipread_video=None,
        vision_video_bytes=[],
        parts=(),
        max_new_tokens=8,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.0,
        caption_target=64,
        return_audio=True,
        speaker="ethan",
    )
    fields.update(overrides)
    return backend._DecodedSample(**fields)


@unittest.skipIf(torch is None, "requires torch from the serving image")
class SpeechLmmSpeechTests(unittest.TestCase):
    def _model(self, *, speech_enabled: bool, generate):
        model = backend.TritonPythonModel()
        model.speech_enabled = speech_enabled
        model.default_speaker = "ethan"
        model.talker_max_new_tokens = 32
        model.talker_do_sample = True
        model.tokenizer = SimpleNamespace(eos_token_id=9, pad_token_id=0)
        model.processor = SimpleNamespace(batch_decode=lambda rows, **kw: [" hi " for _ in rows])
        model.model = SimpleNamespace(
            generate=generate,
            generation_config=SimpleNamespace(eos_token_id=9),
            speaker_map={"Ethan": 0},
            config=SimpleNamespace(talker_config=None),
        )
        model._primary_device = lambda: "cpu"
        model._tokenize_sample = lambda sample: {"input_ids": torch.tensor([[1]])}
        model._to_device_dict = lambda raw, device: raw
        model._attach_lipread = lambda *args, **kwargs: None
        model._place_vision = lambda *args, **kwargs: None
        model._cast_floats = lambda raw: raw
        return model

    def test_text_path_passes_return_audio_false_only_when_speech_is_enabled(self) -> None:
        seen = {}

        def generate(**kwargs):
            seen.clear()
            seen.update(kwargs)
            return torch.tensor([[4, 9]])

        disabled = self._model(speech_enabled=False, generate=generate)
        disabled._generate_once(
            {"input_ids": torch.tensor([[4]])},
            max_new_tokens=1,
            do_sample=False,
            temperature=0,
            repetition_penalty=1,
        )
        self.assertNotIn("return_audio", seen)

        enabled = self._model(speech_enabled=True, generate=generate)
        enabled._generate_once(
            {"input_ids": torch.tensor([[4]])},
            max_new_tokens=1,
            do_sample=False,
            temperature=0,
            repetition_penalty=1,
        )
        self.assertIs(seen["return_audio"], False)

    def test_speech_group_generates_once_per_request(self) -> None:
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return torch.tensor([[4, 9]]), torch.zeros(8)

        model = self._model(speech_enabled=True, generate=generate)
        results = model._generate_group([_sample(index=0), _sample(index=1, speaker="Ethan")])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(results), 2)
        for kwargs in calls:
            self.assertIs(kwargs["return_audio"], True)
            self.assertEqual(kwargs["speaker"], "Ethan")
            self.assertNotIn("use_cache", kwargs)
            self.assertNotIn("pad_token_id", kwargs)
            self.assertTrue(kwargs["thinker_max_new_tokens"])
        self.assertTrue(results[0][2])
        self.assertEqual(results[0][0], "hi")

    def test_text_response_emits_empty_audio(self) -> None:
        model = backend.TritonPythonModel()
        with patch.object(backend.pb_utils, "Tensor", lambda name, data: (name, data.tolist()), create=True), \
             patch.object(backend.pb_utils, "InferenceResponse", lambda **kw: kw, create=True):
            result = model._text_response("Hi", "stop")
        self.assertEqual(
            result["output_tensors"],
            [("GENERATED_TEXT", ["Hi"]), ("FINISH_REASON", ["stop"]), ("GENERATED_AUDIO", [""])],
        )


class QwenSpeechTests(unittest.TestCase):
    def _model(self):
        model = object.__new__(qwen.TritonPythonModel)
        model.model_name = "qwen25_omni"
        model.default_max_new_tokens = 8
        model.default_do_sample = False
        model.default_temperature = 0.0
        model.speech_enabled = True
        model.omni_family = "qwen2_5_omni"
        model.speakers = ["Ethan", "Chelsie"]
        model.default_speaker = "ethan"
        model.talker_max_new_tokens = 16
        model.talker_do_sample = True
        model.eos_token_id = 9
        model.pad_token_id = 0
        return model

    def _patch_pb(self):
        class Response:
            def __init__(self, output_tensors=None, error=None):
                self.output_tensors = output_tensors
                self.error = error

        class TritonError(Exception):
            def __init__(self, message, code=None):
                super().__init__(message)
                self.code = code

        qwen.pb_utils.TritonError = TritonError
        qwen.pb_utils.InferenceResponse = lambda **kw: Response(**kw)
        qwen.pb_utils.Tensor = lambda name, data: (name, data.tolist())
        qwen.pb_utils.get_input_tensor_by_name = lambda req, name: req.get(name)

    def test_speech_path_calls_omni_generate_with_the_resolved_speaker(self) -> None:
        import torch

        calls = {}

        def omni_generate(**kwargs):
            calls.update(kwargs)
            return torch.tensor([[1, 2, 9]]), torch.zeros(4)

        model = self._model()
        model.device = "cpu"
        model.model_dtype = torch.float32
        model._torch = torch
        seen = {}

        class Inputs(dict):
            def to(self, *_args, **_kwargs):
                return self

        def apply_chat_template(conversations, **_kwargs):
            seen["conversations"] = conversations
            return Inputs(input_ids=torch.tensor([[1, 2]]))

        model.processor = SimpleNamespace(
            apply_chat_template=apply_chat_template,
            batch_decode=lambda rows, **kw: [" hi "],
        )
        model.omni = SimpleNamespace(generate=omni_generate)
        model.model = SimpleNamespace(generate=lambda **kw: (_ for _ in ()).throw(AssertionError("thinker")))
        self._patch_pb()
        request = {
            "TEXT_PROMPT": FakeStrTensor(
                json.dumps({"mode": "tts", "prompt": "Say this.", "return_audio": True, "speaker": "ethan"})
            ),
        }
        responses = model.execute([request])
        self.assertIsNone(responses[0].error)
        self.assertEqual(calls["speaker"], "Ethan")
        self.assertIs(calls["return_audio"], True)
        self.assertEqual(
            seen["conversations"][0]["content"][0]["text"],
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
            "capable of perceiving auditory and visual inputs, as well as generating text and speech.",
        )
        self.assertNotIn("use_cache", calls)
        self.assertNotIn("pad_token_id", calls)
        names = [name for name, _data in responses[0].output_tensors]
        self.assertEqual(names, ["GENERATED_TEXT", "FINISH_REASON", "GENERATED_AUDIO"])
        self.assertTrue(responses[0].output_tensors[2][1][0])

    def test_vad_is_bypassed_when_audio_is_requested(self) -> None:
        import torch

        called = {"omni": 0, "vad": 0}

        def omni_generate(**kwargs):
            called["omni"] += 1
            return torch.tensor([[1, 9]]), torch.zeros(4)

        class Inputs(dict):
            def to(self, *_args, **_kwargs):
                return self

        model = self._model()
        model.device = "cpu"
        model.model_dtype = torch.float32
        model._torch = torch
        model._has_speech = lambda _blob: called.__setitem__("vad", called["vad"] + 1) or False
        model._decode_wav = lambda blob: (torch.zeros(4).numpy(), 16000)
        model.processor = SimpleNamespace(
            apply_chat_template=lambda conversations, **kw: Inputs(input_ids=torch.tensor([[1]])),
            batch_decode=lambda rows, **kw: ["hi"],
        )
        model.omni = SimpleNamespace(generate=omni_generate)
        model.model = SimpleNamespace(generate=lambda **kw: (_ for _ in ()).throw(AssertionError("thinker")))
        self._patch_pb()
        request = {
            "TEXT_PROMPT": FakeStrTensor(
                json.dumps({"mode": "s2st", "prompt": "", "return_audio": True, "speaker": "ethan"})
            ),
            "AUDIO_BYTES": FakeStrTensor("QQ=="),
        }
        responses = model.execute([request])
        self.assertIsNone(responses[0].error)
        self.assertEqual(called["omni"], 1)
        self.assertEqual(called["vad"], 0)

    def test_speech_disabled_returns_a_clear_error(self) -> None:
        model = self._model()
        model.speech_enabled = False
        self._patch_pb()
        request = {
            "TEXT_PROMPT": FakeStrTensor(
                json.dumps({"mode": "tts", "prompt": "Say this.", "return_audio": True})
            ),
        }
        responses = model.execute([request])
        self.assertIsNotNone(responses[0].error)
        self.assertIn("disabled", str(responses[0].error))


if __name__ == "__main__":
    unittest.main()
