#!/usr/bin/env python3
"""BaseBackend contract, exercised with a fake text backend and a fake pb_utils."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]


def _load():
    backends = str(_REPO / "backends")
    if backends not in sys.path:
        sys.path.insert(0, backends)
    sys.modules.setdefault("triton_python_backend_utils", types.ModuleType("stub"))
    spec = importlib.util.spec_from_file_location(
        "base_backend_under_test", _REPO / "backends" / "base_backend.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load()
pb = sys.modules["triton_python_backend_utils"]


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

    def has_error(self):
        return self.error is not None


class TritonError(Exception):
    def __init__(self, message):
        super().__init__(message)


pb.Tensor = Tensor
pb.InferenceResponse = Response
pb.TritonError = TritonError
pb.Logger = types.SimpleNamespace(log_warn=lambda message: warnings.append(message), log_info=lambda message: None)
warnings: list[str] = []


class EchoBackend(base.BaseBackend):
    supported_modes = frozenset({"chat"})
    supports_batching = False

    def __init__(self):
        self.loaded = False
        self.generated = []
        self.cleaned = []

    def load_model(self, args, model_config):
        self.loaded = True
        self.model_config = model_config

    def decode_request(self, index, request):
        spec = json.loads(request["TEXT_PROMPT"].as_numpy().reshape(-1)[0])
        return base.Sample(index=index, mode=str(spec.get("mode") or "chat"), prompt=str(spec.get("prompt") or ""))

    def generate(self, samples):
        self.generated.append([sample.prompt for sample in samples])
        if samples[0].prompt == "boom" and len(samples) >= 1:
            raise RuntimeError("boom")
        return [base.Result(sample.prompt or "echo", "stop") for sample in samples]

    def cleanup(self, sample):
        self.cleaned.append(sample.index)


class BatchBackend(EchoBackend):
    supports_batching = True

    def group_key(self, sample):
        return ("all",)

    def generate(self, samples):
        self.generated.append([sample.prompt for sample in samples])
        if len(samples) > 1:
            raise RuntimeError("batch failed")
        return [base.Result(sample.prompt, "stop") for sample in samples]


def _prompt(mode: str, prompt: str):
    return {"TEXT_PROMPT": Tensor("TEXT_PROMPT", np.array([json.dumps({"mode": mode, "prompt": prompt})], dtype=object))}


class BaseBackendTests(unittest.TestCase):
    def test_cannot_instantiate_without_generate(self) -> None:
        class Incomplete(base.BaseBackend):
            def decode_request(self, index, request):
                return base.Sample(index=index, mode="chat")

        with self.assertRaises(TypeError):
            Incomplete()

    def test_minimal_subclass_end_to_end(self) -> None:
        model = EchoBackend()
        model.initialize({"model_config": "{\"parameters\": {\"device_map\": {\"string_value\": \"auto\"}}}"})
        self.assertTrue(model.loaded)
        self.assertEqual(model.model_config["parameters"]["device_map"]["string_value"], "auto")
        response = model.execute([_prompt("chat", "hi")])[0]
        self.assertEqual(response.output_tensors[0].name, "GENERATED_TEXT")
        self.assertEqual(response.output_tensors[0].as_numpy().tolist(), ["hi"])
        self.assertEqual(model.cleaned, [0])

    def test_unknown_mode_is_rejected(self) -> None:
        model = EchoBackend()
        response = model.execute([_prompt("lipread", "x")])[0]
        self.assertIsNotNone(response.error)
        self.assertIn("not supported", str(response.error))

    def test_batching_groups_and_fallback(self) -> None:
        warnings.clear()
        model = BatchBackend()
        responses = model.execute([_prompt("chat", "a"), _prompt("chat", "b")])
        self.assertEqual(responses[0].output_tensors[0].as_numpy().tolist(), ["a"])
        self.assertEqual(responses[1].output_tensors[0].as_numpy().tolist(), ["b"])
        self.assertEqual(model.generated[0], ["a", "b"])
        self.assertEqual(model.generated[1:], [["a"], ["b"]])
        self.assertTrue(any("retrying one request at a time" in line for line in warnings))

    def test_short_circuit_skips_generate(self) -> None:
        class Skip(EchoBackend):
            def short_circuit(self, sample):
                if sample.prompt == "skip":
                    return base.Result("skipped", "stop")
                return None

        model = Skip()
        response = model.execute([_prompt("chat", "skip")])[0]
        self.assertEqual(response.output_tensors[0].as_numpy().tolist(), ["skipped"])
        self.assertEqual(model.generated, [])

    def test_cleanup_called_on_error(self) -> None:
        model = EchoBackend()
        response = model.execute([_prompt("chat", "boom")])[0]
        self.assertIn("boom", str(response.error))
        self.assertEqual(model.cleaned, [0])

    def _empty(self, reason):
        class Empty(EchoBackend):
            def generate(self, samples):
                return [base.Result("   ", reason)]

        return Empty().execute([_prompt("chat", "hi")])[0]

    def test_empty_generation_is_recorded_not_an_error(self) -> None:
        response = self._empty("stop")
        self.assertIsNone(response.error)
        self.assertEqual(response.output_tensors[0].as_numpy().tolist(), [""])
        self.assertEqual(response.output_tensors[1].as_numpy().tolist(), ["empty"])

    def test_empty_generation_keeps_the_termination_axis(self) -> None:
        response = self._empty("length")
        self.assertIsNone(response.error)
        self.assertEqual(response.output_tensors[1].as_numpy().tolist(), ["empty_length"])

    def test_short_circuit_empty_is_not_tagged(self) -> None:
        class Silent(EchoBackend):
            def short_circuit(self, sample):
                return base.Result("", "stop")

        response = Silent().execute([_prompt("chat", "hi")])[0]
        self.assertIsNone(response.error)
        self.assertEqual(response.output_tensors[1].as_numpy().tolist(), ["stop"])

    def test_non_empty_generation_keeps_its_reason(self) -> None:
        class Truncated(EchoBackend):
            def generate(self, samples):
                return [base.Result(sample.prompt, "length") for sample in samples]

        response = Truncated().execute([_prompt("chat", "hi")])[0]
        self.assertIsNone(response.error)
        self.assertEqual(response.output_tensors[0].as_numpy().tolist(), ["hi"])
        self.assertEqual(response.output_tensors[1].as_numpy().tolist(), ["length"])

    def test_order_preserved(self) -> None:
        model = EchoBackend()
        responses = model.execute([_prompt("chat", "first"), _prompt("chat", "second")])
        self.assertEqual(responses[0].output_tensors[0].as_numpy().tolist(), ["first"])
        self.assertEqual(responses[1].output_tensors[0].as_numpy().tolist(), ["second"])

    def test_pb_utils_resolved_from_subclass_module(self) -> None:
        model = EchoBackend()
        seen = {}

        class Mark(Tensor):
            pass

        module = sys.modules[type(model).__module__]
        # EchoBackend is defined in this test module. Point its pb_utils at a marker.
        module.pb_utils = types.SimpleNamespace(
            Tensor=lambda name, data: Mark(name, data),
            InferenceResponse=Response,
            TritonError=TritonError,
            Logger=pb.Logger,
        )
        try:
            response = model.execute([_prompt("chat", "hi")])[0]
        finally:
            del module.pb_utils
        self.assertIsInstance(response.output_tensors[0], Mark)
        self.assertTrue(seen == {})


if __name__ == "__main__":
    unittest.main()
