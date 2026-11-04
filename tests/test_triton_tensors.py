#!/usr/bin/env python3
"""Tests for the Triton wire format. Stdlib only, so any Python 3.9+ runs them."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triton_tensors import (  # noqa: E402
    TritonInferError,
    build_inputs,
    infer_output,
    infer_output_metrics,
    infer_result,
)


def _by_name(inputs: list[dict]) -> dict[str, object]:
    return {item["name"]: item["data"][0] for item in inputs}


class BuildInputsTests(unittest.TestCase):
    def test_envelope_omits_unset_optional_keys(self) -> None:
        envelope = json.loads(_by_name(build_inputs(mode="chat", prompt="hi"))["TEXT_PROMPT"])
        self.assertEqual(envelope, {"mode": "chat", "prompt": "hi"})

    def test_envelope_carries_a_client_system_prompt(self) -> None:
        inputs = build_inputs(mode="chat", prompt="hi", system="Be terse.")
        envelope = json.loads(_by_name(inputs)["TEXT_PROMPT"])
        self.assertEqual(envelope["system"], "Be terse.")

    def test_envelope_carries_language_format_and_caption_chars(self) -> None:
        inputs = build_inputs(
            mode="caption",
            prompt="",
            language="de",
            fmt="clean",
            caption_chars=42,
        )
        envelope = json.loads(_by_name(inputs)["TEXT_PROMPT"])
        self.assertEqual(envelope["language"], "de")
        self.assertEqual(envelope["format"], "clean")
        self.assertEqual(envelope["caption_chars"], 42)

    def test_sampling_follows_temperature(self) -> None:
        greedy = _by_name(build_inputs(mode="chat", prompt="hi", temperature=0.0))
        sampled = _by_name(build_inputs(mode="chat", prompt="hi", temperature=0.7))
        self.assertFalse(greedy["DO_SAMPLE"])
        self.assertTrue(sampled["DO_SAMPLE"])
        self.assertAlmostEqual(float(sampled["TEMPERATURE"]), 0.7, places=5)

    def test_media_tensors_only_present_when_supplied(self) -> None:
        text_only = {item["name"] for item in build_inputs(mode="chat", prompt="hi")}
        self.assertNotIn("AUDIO_BYTES", text_only)
        self.assertNotIn("VIDEO_BYTES", text_only)
        both = {
            item["name"]
            for item in build_inputs(
                mode="avspeakd", prompt="", audio_b64="QQ==", video_b64="Qg=="
            )
        }
        self.assertIn("AUDIO_BYTES", both)
        self.assertIn("VIDEO_BYTES", both)

    def test_every_tensor_uses_the_batch_one_shape(self) -> None:
        for item in build_inputs(mode="transcribe", prompt="", audio_b64="QQ=="):
            self.assertEqual(item["shape"], [1, 1], item["name"])

    def test_list_form_emits_one_by_n(self) -> None:
        audio = next(
            item
            for item in build_inputs(
                mode="chat", prompt="hi", audio_b64=["QQ==", "Qg=="]
            )
            if item["name"] == "AUDIO_BYTES"
        )
        self.assertEqual(audio["shape"], [1, 2])
        self.assertEqual(audio["data"], ["QQ==", "Qg=="])

    def test_parts_are_omitted_without_media(self) -> None:
        envelope = json.loads(
            _by_name(
                build_inputs(
                    mode="chat",
                    prompt="hi",
                    parts=[{"type": "text", "text": "hi"}],
                )
            )["TEXT_PROMPT"]
        )
        self.assertNotIn("parts", envelope)
        self.assertNotIn("media_order", envelope)

    def test_parts_keep_text_beside_a_single_clip(self) -> None:
        parts = [{"type": "text", "text": "Input: "}, {"type": "audio"}]
        envelope = json.loads(
            _by_name(
                build_inputs(
                    mode="transcribe",
                    prompt="Input: ",
                    audio_b64="QQ==",
                    parts=parts,
                )
            )["TEXT_PROMPT"]
        )
        self.assertEqual(envelope["parts"], parts)

    def test_parts_are_included_for_several_clips(self) -> None:
        parts = [
            {"type": "audio"},
            {"type": "video"},
            {"type": "text", "text": "Pair them."},
            {"type": "audio"},
        ]
        envelope = json.loads(
            _by_name(
                build_inputs(
                    mode="chat",
                    prompt="Pair them.",
                    audio_b64=["QQ==", "Qg=="],
                    video_b64=["Uw=="],
                    parts=parts,
                )
            )["TEXT_PROMPT"]
        )
        self.assertEqual(envelope["parts"], parts)

    def test_speech_keys_are_omitted_until_requested(self) -> None:
        envelope = json.loads(_by_name(build_inputs(mode="chat", prompt="hi"))["TEXT_PROMPT"])
        self.assertNotIn("return_audio", envelope)
        self.assertNotIn("speaker", envelope)

    def test_speech_keys_are_added_when_set(self) -> None:
        envelope = json.loads(
            _by_name(
                build_inputs(mode="tts", prompt="hi", return_audio=True, speaker="ethan")
            )["TEXT_PROMPT"]
        )
        self.assertIs(envelope["return_audio"], True)
        self.assertEqual(envelope["speaker"], "ethan")


class InferOutputTests(unittest.TestCase):
    def _response(self, outputs: list[dict]) -> "io.BytesIO":
        import io

        return io.BytesIO(json.dumps({"outputs": outputs}).encode())

    def test_generated_audio_is_optional(self) -> None:
        from unittest.mock import patch

        outputs = [
            {"name": "FINISH_REASON", "data": ["stop"]},
            {"name": "GENERATED_TEXT", "data": ["Hi"]},
        ]
        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: self._response(outputs)):
            self.assertEqual(
                infer_output("http://localhost", "speechlmm_v2_l", [], 1),
                ("Hi", "stop", None),
            )
            self.assertEqual(
                infer_result("http://localhost", "speechlmm_v2_l", [], 1),
                ("Hi", "stop"),
            )

    def test_empty_generated_audio_is_none(self) -> None:
        from unittest.mock import patch

        outputs = [
            {"name": "GENERATED_TEXT", "data": ["Hi"]},
            {"name": "FINISH_REASON", "data": ["stop"]},
            {"name": "GENERATED_AUDIO", "data": [""]},
        ]
        with patch("urllib.request.urlopen", return_value=self._response(outputs)):
            self.assertEqual(
                infer_output("http://localhost", "qwen25_omni", [], 1)[2],
                None,
            )

    def test_generated_audio_is_returned(self) -> None:
        from unittest.mock import patch

        outputs = [
            {"name": "GENERATED_TEXT", "data": ["Hi"]},
            {"name": "FINISH_REASON", "data": ["length"]},
            {"name": "GENERATED_AUDIO", "data": ["QQ=="]},
        ]
        with patch("urllib.request.urlopen", return_value=self._response(outputs)):
            self.assertEqual(
                infer_output("http://localhost", "speechlmm_v2_xl", [], 1),
                ("Hi", "length", "QQ=="),
            )

    def test_metrics_parsed(self) -> None:
        from unittest.mock import patch

        payload = '{"mode":"transcribe","rtf":0.5}'
        outputs = [
            {"name": "GENERATED_TEXT", "data": ["Hi"]},
            {"name": "FINISH_REASON", "data": ["stop"]},
            {"name": "METRICS", "data": [payload]},
        ]
        with patch("urllib.request.urlopen", return_value=self._response(outputs)):
            text, reason, audio, metrics = infer_output_metrics(
                "http://localhost", "speechlmm_v2_l", [], 1
            )
        self.assertEqual((text, reason, audio), ("Hi", "stop", None))
        self.assertEqual(metrics, {"mode": "transcribe", "rtf": 0.5})

    def test_metrics_missing_is_none(self) -> None:
        from unittest.mock import patch

        outputs = [
            {"name": "GENERATED_TEXT", "data": ["Hi"]},
            {"name": "FINISH_REASON", "data": ["stop"]},
        ]
        with patch("urllib.request.urlopen", return_value=self._response(outputs)):
            self.assertIsNone(infer_output_metrics("http://localhost", "speechlmm_v2_l", [], 1)[3])

    def test_metrics_empty_is_none(self) -> None:
        from unittest.mock import patch

        outputs = [
            {"name": "GENERATED_TEXT", "data": ["Hi"]},
            {"name": "FINISH_REASON", "data": ["stop"]},
            {"name": "METRICS", "data": [""]},
        ]
        with patch("urllib.request.urlopen", return_value=self._response(outputs)):
            self.assertIsNone(infer_output_metrics("http://localhost", "qwen3_omni", [], 1)[3])

    def test_metrics_invalid_json_raises(self) -> None:
        from unittest.mock import patch

        outputs = [
            {"name": "GENERATED_TEXT", "data": ["Hi"]},
            {"name": "FINISH_REASON", "data": ["stop"]},
            {"name": "METRICS", "data": ["{"]},
        ]
        with patch("urllib.request.urlopen", return_value=self._response(outputs)):
            with self.assertRaises(TritonInferError) as caught:
                infer_output_metrics("http://localhost", "speechlmm_v2_xl", [], 1)
        self.assertIn("invalid METRICS JSON", str(caught.exception))

    def test_infer_output_signature_unchanged(self) -> None:
        from unittest.mock import patch

        outputs = [
            {"name": "GENERATED_TEXT", "data": ["Hi"]},
            {"name": "FINISH_REASON", "data": ["stop"]},
            {"name": "METRICS", "data": ['{"mode":"chat"}']},
        ]
        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: self._response(outputs)):
            self.assertEqual(
                infer_output("http://localhost", "speechlmm_v2_l", [], 1),
                ("Hi", "stop", None),
            )
            self.assertEqual(infer_result("http://localhost", "speechlmm_v2_l", [], 1), ("Hi", "stop"))


if __name__ == "__main__":
    unittest.main()
