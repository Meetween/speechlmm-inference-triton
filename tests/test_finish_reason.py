"""Generation termination and backend-to-gateway metadata regressions."""
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_backend_prompts import backend
from test_qwen_omni_backend import qwen
from triton_tensors import infer, infer_result, TritonInferError

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires torch from the serving image")
class GenerationFinishTests(unittest.TestCase):
    def setUp(self):
        self.model = backend.TritonPythonModel()
        self.model.processor = SimpleNamespace(batch_decode=lambda rows, **kw: [" Hi " for _ in rows])

    def test_mixed_batch_eos_padding_and_limit(self):
        # EOS in the prompt must not end the generated continuation.
        rows = torch.tensor([[9, 1, 9, 0], [9, 1, 2, 3], [9, 1, 2, 9]])
        self.assertEqual(self.model._decode_generated(rows, 1, 9, 3),
                         [("Hi", "stop"), ("Hi", "length"), ("Hi", "stop")])

    def test_multiple_eos_ids_and_eos_as_padding(self):
        rows = torch.tensor([[1, 8, 9, 9], [1, 2, 3, 4]])
        self.assertEqual(self.model._decode_generated(rows, 1, [8, 9], 3),
                         [("Hi", "stop"), ("Hi", "length")])

    def test_single_token_budget_and_no_eos_configuration(self):
        self.assertEqual(self.model._decode_generated(torch.tensor([4, 5]), 1, None, 1),
                         [("Hi", "length")])

    def test_unexplained_early_termination_is_not_reported_as_stop(self):
        with self.assertRaises(RuntimeError):
            self.model._decode_generated(torch.tensor([[4, 5]]), 1, 9, 10)

    def test_generated_counts_stop_at_eos(self):
        rows = torch.tensor([[9, 1, 9, 0], [9, 1, 2, 3]])
        self.model._decode_generated(rows, 1, 9, 3)
        self.assertEqual(self.model._last_output_counts, [2, 3])

    def test_generate_uses_effective_retry_budget(self):
        self.model.tokenizer = SimpleNamespace(eos_token_id=9, pad_token_id=0)
        self.model.model = SimpleNamespace(
            generation_config=SimpleNamespace(eos_token_id=9),
            generate=lambda **kw: torch.tensor([[4, 5]]))
        self.assertEqual(self.model._generate_once(
            {"input_ids": torch.tensor([[4]])}, max_new_tokens=1,
            do_sample=False, temperature=0, repetition_penalty=1), [("Hi", "length")])

    def test_backend_returns_both_tensors(self):
        with patch.object(backend.pb_utils, "Tensor", lambda name, data: (name, data.tolist()), create=True), \
             patch.object(backend.pb_utils, "InferenceResponse", lambda **kw: kw, create=True):
            result = self.model._text_response("Hi", "length")
        self.assertEqual(
            result["output_tensors"],
            [("GENERATED_TEXT", ["Hi"]), ("FINISH_REASON", ["length"]), ("GENERATED_AUDIO", [""])],
        )


@unittest.skipIf(torch is None, "requires torch from the serving image")
class QwenGenerationFinishTests(unittest.TestCase):
    def setUp(self):
        self.model = qwen.TritonPythonModel()
        self.model.processor = SimpleNamespace(batch_decode=lambda rows, **kw: [" Hi " for _ in rows])

    def test_eos_is_stop_and_budget_exhaustion_is_length(self):
        rows = torch.tensor([[9, 1, 9, 0], [9, 1, 2, 3]])
        self.assertEqual(self.model._decode_generated(rows[0], 1, 9, 3), ("Hi", "stop"))
        self.assertEqual(self.model._last_output_counts, [2])
        self.assertEqual(self.model._decode_generated(rows[1], 1, 9, 3), ("Hi", "length"))
        self.assertEqual(self.model._last_output_counts, [3])

    def test_backend_returns_both_tensors(self):
        with patch.object(qwen.pb_utils, "Tensor", lambda name, data: (name, data.tolist()), create=True), \
             patch.object(qwen.pb_utils, "InferenceResponse", lambda **kw: kw, create=True):
            result = self.model._text_response("Hi", "length")
        self.assertEqual(
            result["output_tensors"],
            [("GENERATED_TEXT", ["Hi"]), ("FINISH_REASON", ["length"]), ("GENERATED_AUDIO", [""])],
        )


class TransportFinishTests(unittest.TestCase):
    def response(self, reason="length"):
        # Deliberately put metadata first: output order is not the contract.
        outputs = [{"name": "GENERATED_TEXT", "data": ["Hi"]}]
        if reason is not None:
            outputs.insert(0, {"name": "FINISH_REASON", "data": [reason]})
        return io.BytesIO(json.dumps({"outputs": outputs}).encode())

    def test_named_outputs_and_text_only_compatibility(self):
        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: self.response()):
            self.assertEqual(infer_result("http://localhost", "speechlmm_v2_xl", [], 1), ("Hi", "length"))
            self.assertEqual(infer("http://localhost", "speechlmm_v2_xl", [], 1), "Hi")

    def test_missing_or_invalid_metadata_fails_for_speechlmm(self):
        for reason in [None, "wrong"]:
            with self.subTest(reason=reason), patch("urllib.request.urlopen", return_value=self.response(reason)):
                with self.assertRaises(TritonInferError):
                    infer_result("http://localhost", "speechlmm_v2_xl", [], 1)

    def test_empty_reasons_reach_the_caller(self):
        for reason in ["empty", "empty_length"]:
            with self.subTest(reason=reason), patch("urllib.request.urlopen", return_value=self.response(reason)):
                self.assertEqual(
                    infer_result("http://localhost", "speechlmm_v2_xl", [], 1), ("Hi", reason))

    def test_legacy_backend_compatibility(self):
        with patch("urllib.request.urlopen", return_value=self.response(None)):
            self.assertEqual(infer_result("http://localhost", "voxtral_mini_3b_2507", [], 1), ("Hi", "stop"))

    def test_qwen_omni_requires_finish_reason(self):
        with patch("urllib.request.urlopen", return_value=self.response(None)):
            with self.assertRaises(TritonInferError):
                infer_result("http://localhost", "qwen25_omni", [], 1)

    def test_gateway_preserves_reason(self):
        try:
            from gateway.openai_server import ChatCompletionRequest, chat_completions
        except ImportError:
            self.skipTest("requires gateway dependencies")
        for reason in ["stop", "length"]:
            with self.subTest(reason=reason), patch("gateway.openai_server.triton_infer", return_value=("Hi", reason, None)):
                response = chat_completions(ChatCompletionRequest(
                    model="speechlmm_v2_xl", messages=[{"role": "user", "content": "Hello"}]))
                choice = response["choices"][0]
                self.assertEqual(choice["finish_reason"], reason)
                self.assertEqual(choice["message"]["content"], "Hi")
