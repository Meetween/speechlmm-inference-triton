#!/usr/bin/env python3
"""Tests for the backend's prompt rules. Stdlib only, so any Python 3.9+ runs them.

The client owns the wording: a supplied prompt is passed through verbatim and the
server never invents a missing instruction. These tests load backends/speechlmm_backend.py
directly with triton_python_backend_utils stubbed, so no Triton runtime is needed.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

_REPO = Path(__file__).resolve().parents[1]


def _load_backend():
    """Import the backend module without a Triton runtime."""
    backends = str(_REPO / "backends")
    if backends not in sys.path:
        sys.path.insert(0, backends)
    sys.modules.setdefault("triton_python_backend_utils", types.ModuleType("stub"))
    spec = importlib.util.spec_from_file_location(
        "speechlmm_backend_under_test", _REPO / "backends" / "speechlmm_backend.py"
    )
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves __module__ through sys.modules, so register before exec.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


backend = _load_backend()


def _instruction(mode: str, prompt: str = "", **spec: object) -> str:
    return backend._build_mode_instruction(
        {"mode": mode, "prompt": prompt, **spec},
        has_audio=mode in backend.AUDIO_REQUIRED_MODES or bool(spec.get("has_audio")),
        has_video=mode in backend.VIDEO_REQUIRED_MODES,
    )


class SuppliedPromptWinsTests(unittest.TestCase):
    """A prompt from the client is the user message, with nothing prepended."""

    def test_every_mode_passes_the_prompt_through_verbatim(self) -> None:
        prompt = "Transcribe the following speech."
        for mode in (
            "transcribe",
            "caption",
            "summary",
            "text_summary",
            "mt",
            "st",
            "lipread",
            "avspeakd",
            "chat",
        ):
            with self.subTest(mode=mode):
                self.assertEqual(_instruction(mode, prompt), prompt)

    def test_the_old_additional_instruction_wrapper_is_gone(self) -> None:
        text = _instruction("transcribe", "Transcribe the following speech.")
        self.assertNotIn("Additional instruction", text)
        self.assertNotIn("Do not summarize", text)

    def test_hints_do_not_leak_into_a_supplied_prompt(self) -> None:
        text = _instruction("transcribe", "Write it down.", language="it", format="clean")
        self.assertEqual(text, "Write it down.")

    def test_surrounding_whitespace_is_stripped(self) -> None:
        self.assertEqual(_instruction("chat", "  hi  "), "hi")


class NoFallbackTests(unittest.TestCase):
    """Missing instructions remain missing, regardless of mode or hints."""

    def test_modes_do_not_supply_missing_user_instructions(self) -> None:
        for mode in ("chat", "transcribe", "caption", "summary", "mt", "st", "lipread", "avspeakd"):
            for prompt in ("", "   "):
                with self.subTest(mode=mode, prompt=prompt):
                    self.assertEqual(_instruction(mode, prompt, language="it", format="clean", caption_chars=99), "")


class RequirementTests(unittest.TestCase):
    """Refusals that must not degrade into a weaker request."""

    def test_audio_modes_reject_a_missing_waveform(self) -> None:
        for mode in sorted(backend.AUDIO_REQUIRED_MODES):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError) as caught:
                    backend._build_mode_instruction(
                        {"mode": mode, "prompt": "go"}, has_audio=False, has_video=False
                    )
                self.assertIn("requires audio", str(caught.exception))

    def test_video_modes_reject_a_missing_clip(self) -> None:
        for mode in sorted(backend.VIDEO_REQUIRED_MODES):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError) as caught:
                    backend._build_mode_instruction(
                        {"mode": mode, "prompt": "go"}, has_audio=True, has_video=False
                    )
                self.assertIn("requires video", str(caught.exception))

    def test_text_summary_has_no_default_because_the_prompt_is_the_content(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _instruction("text_summary")
        self.assertIn("requires text to summarize", str(caught.exception))


class VideoRoutingTests(unittest.TestCase):
    def test_only_lipread_uses_the_mouth_roi_encoder(self) -> None:
        self.assertTrue(backend._uses_lipread_encoder("lipread"))
        self.assertFalse(backend._uses_lipread_encoder("avspeakd"))
        self.assertFalse(backend._uses_lipread_encoder("chat"))
        self.assertFalse(backend._uses_lipread_encoder("transcribe"))


class SystemPromptTests(unittest.TestCase):
    def test_envelope_carries_a_client_system_prompt(self) -> None:
        spec = backend._parse_text_prompt('{"mode": "chat", "system": "Be terse."}')
        self.assertEqual(spec["system"], "Be terse.")

    def test_system_is_absent_for_the_tagged_and_plain_forms(self) -> None:
        self.assertIsNone(backend._parse_text_prompt("mode:transcribe rest").get("system"))
        self.assertIsNone(backend._parse_text_prompt("just text").get("system"))

class _FakeTensor:
    """Minimal stand-in for a Triton input tensor of one request."""

    def __init__(self, value: object) -> None:
        self._value = value

    def as_numpy(self):  # noqa: ANN201 - mirrors the Triton API
        import numpy as np

        if isinstance(self._value, (list, tuple)):
            return np.array([list(self._value)], dtype=object)
        return np.array([self._value])


def _decode(envelope: str, *, with_video: bool = False, with_audio: bool = False) -> object:
    """Run the real _decode_sample for a request carrying `envelope`."""
    model = object.__new__(backend.TritonPythonModel)
    model.default_max_new_tokens = 128
    model.default_do_sample = False
    model.default_temperature = 0.0
    request = {"TEXT_PROMPT": _FakeTensor(envelope.encode("utf-8"))}
    if with_audio:
        request["AUDIO_BYTES"] = _FakeTensor(b"QQ==")
        model._decode_wav = lambda _bytes: ([0.0], 16000)
    if with_video:
        request["VIDEO_BYTES"] = _FakeTensor(b"QQ==")
        model._process_lipread = lambda _bytes: types.SimpleNamespace(shape=(4,))
    backend.pb_utils.get_input_tensor_by_name = lambda req, name: req.get(name)
    return model._decode_sample(0, request)


class DecodeSampleSystemTests(unittest.TestCase):
    """The system turn the model actually receives."""

    def test_client_system_prompt_wins(self) -> None:
        sample = _decode('{"mode": "chat", "prompt": "hi", "system": "Be terse."}')
        self.assertTrue(sample.use_system)
        self.assertEqual(sample.system_prompt, "Be terse.")

    def test_absent_system_stays_absent(self) -> None:
        sample = _decode('{"mode": "chat", "prompt": "hi"}')
        self.assertFalse(sample.use_system)
        self.assertEqual(sample.system_prompt, "")

    def test_blank_system_is_treated_as_absent(self) -> None:
        sample = _decode('{"mode": "chat", "prompt": "hi", "system": "   "}')
        self.assertFalse(sample.use_system)
        self.assertEqual(sample.system_prompt, "")

    def test_video_modes_still_default_to_no_system_turn(self) -> None:
        sample = _decode('{"mode": "lipread", "prompt": "read it"}', with_video=True)
        self.assertFalse(sample.use_system)
        self.assertEqual(sample.system_prompt, "")

    def test_video_modes_honour_an_explicit_system_prompt(self) -> None:
        sample = _decode(
            '{"mode": "lipread", "prompt": "read it", "system": "Be terse."}',
            with_video=True,
        )
        self.assertTrue(sample.use_system)
        self.assertEqual(sample.system_prompt, "Be terse.")

    def test_qwen25_speech_fills_the_official_system_prompt(self) -> None:
        model = object.__new__(backend.TritonPythonModel)
        model.default_max_new_tokens = 128
        model.default_do_sample = False
        model.default_temperature = 0.0
        model.speech_enabled = True
        model.model = types.SimpleNamespace(
            config=types.SimpleNamespace(backbone_type="qwen2_5_omni", thinker_config=None)
        )
        request = {
            "TEXT_PROMPT": _FakeTensor(
                b'{"mode": "tts", "prompt": "Say this.", "return_audio": true}'
            )
        }
        backend.pb_utils.get_input_tensor_by_name = lambda req, name: req.get(name)
        sample = model._decode_sample(0, request)
        self.assertTrue(sample.use_system)
        self.assertIn("virtual human", sample.system_prompt)

    def test_client_system_prompt_still_wins_for_qwen25_speech(self) -> None:
        model = object.__new__(backend.TritonPythonModel)
        model.default_max_new_tokens = 128
        model.default_do_sample = False
        model.default_temperature = 0.0
        model.speech_enabled = True
        model.model = types.SimpleNamespace(
            config=types.SimpleNamespace(backbone_type="qwen2_5_omni", thinker_config=None)
        )
        request = {
            "TEXT_PROMPT": _FakeTensor(
                b'{"mode": "tts", "prompt": "Say this.", "return_audio": true, "system": "Be terse."}'
            )
        }
        backend.pb_utils.get_input_tensor_by_name = lambda req, name: req.get(name)
        warnings: list[str] = []
        logger = types.SimpleNamespace(log_warn=warnings.append)
        with patch.object(backend.pb_utils, "Logger", logger, create=True):
            sample = model._decode_sample(0, request)
            model._decode_sample(1, request)
        self.assertEqual(sample.system_prompt, "Be terse.")
        self.assertEqual(len(warnings), 1)
        self.assertIn("official Qwen prompt", warnings[0])

    def test_supplied_prompt_reaches_the_user_turn_verbatim(self) -> None:
        sample = _decode('{"mode": "chat", "prompt": "just this"}')
        self.assertEqual(sample.user_prompt, "just this")


class UserContentAssemblyTests(unittest.TestCase):
    def test_processor_kwargs_do_not_disable_audio_padding(self) -> None:
        kwargs = backend._processor_tokenize_kwargs()
        self.assertNotIn("padding", kwargs)
        self.assertEqual(kwargs["text_kwargs"]["padding"], False)
        self.assertEqual(kwargs["audio_kwargs"]["truncation"], False)

    def test_content_follows_parts(self) -> None:
        parts = backend._assemble_user_content(
            (
                ("audio", ""),
                ("video", ""),
                ("text", "Pair them."),
                ("audio", ""),
                ("video", ""),
            ),
            ["a1", "a2"],
            ["v1", "v2"],
        )
        self.assertEqual(
            parts,
            [
                {"type": "audio", "audio": "a1"},
                {"type": "video", "video": "v1"},
                {"type": "text", "text": "Pair them."},
                {"type": "audio", "audio": "a2"},
                {"type": "video", "video": "v2"},
            ],
        )

    def test_text_before_audio_stays_before_audio(self) -> None:
        normalized = backend._normalize_parts(
            [{"type": "text", "text": "Input: "}, {"type": "audio"}],
            1,
            0,
            "Input:",
        )
        content = backend._assemble_user_content(normalized, ["wav"], [])
        self.assertEqual([part["type"] for part in content], ["text", "audio"])
        self.assertEqual(content[0]["text"], "Input: ")

    def test_audio_then_text_is_unchanged(self) -> None:
        normalized = backend._normalize_parts(
            [{"type": "audio"}, {"type": "text", "text": "after"}],
            1,
            0,
            "after",
        )
        content = backend._assemble_user_content(normalized, ["wav"], [])
        self.assertEqual(
            [(part["type"], part.get("text", part.get("audio"))) for part in content],
            [("audio", "wav"), ("text", "after")],
        )

    def test_omitted_parts_keep_media_before_text(self) -> None:
        normalized = backend._normalize_parts(None, 1, 1, "Describe.")
        content = backend._assemble_user_content(normalized, ["wav"], ["vid"])
        self.assertEqual([part["type"] for part in content], ["audio", "video", "text"])
        self.assertEqual(content[-1]["text"], "Describe.")

    def test_part_counts_must_match_the_tensors(self) -> None:
        with self.assertRaises(ValueError):
            backend._normalize_parts([{"type": "audio"}, {"type": "audio"}], 1, 1, "")

    def test_lipread_tokens_replace_the_video_part(self) -> None:
        normalized = backend._normalize_parts(
            [
                {"type": "text", "text": "Before "},
                {"type": "video"},
                {"type": "text", "text": " after"},
            ],
            0,
            1,
            "Before  after",
        )
        placed = backend._place_lipread_tokens(normalized, 2)
        self.assertEqual([kind for kind, _text in placed], ["text", "text", "text"])
        self.assertEqual(placed[0][1], "Before ")
        self.assertIn(backend.LIPREAD_BOS_TOKEN, placed[1][1])
        self.assertNotIn("<lipread>", placed[1][1])
        self.assertEqual(placed[2][1], " after")

    def test_lipread_placeholder_expands_in_the_text_part(self) -> None:
        placed = backend._place_lipread_tokens((("text", "See <lipread> now"), ("video", "")), 2)
        self.assertEqual([kind for kind, _text in placed], ["text"])
        self.assertIn(backend.LIPREAD_BOS_TOKEN, placed[0][1])
        self.assertNotIn("<lipread>", placed[0][1])

    def test_envelope_parts_place_text_before_audio(self) -> None:
        import json

        sample = _decode(
            json.dumps(
                {
                    "mode": "transcribe",
                    "prompt": "Input: ",
                    "parts": [{"type": "text", "text": "Input: "}, {"type": "audio"}],
                }
            ),
            with_audio=True,
        )
        model = object.__new__(backend.TritonPythonModel)
        seen: list = []

        def tokenize(messages, **kwargs):
            seen.extend(messages)
            return {}

        model.processor = types.SimpleNamespace(apply_chat_template=tokenize)
        model.tokenizer = model.processor
        model._tokenize_sample(sample)
        content = seen[-1]["content"]
        self.assertEqual([part["type"] for part in content], ["text", "audio"])
        self.assertEqual(content[0]["text"], "Input: ")

    def test_lipread_refuses_several_clips(self) -> None:
        model = object.__new__(backend.TritonPythonModel)
        model.default_max_new_tokens = 128
        model.default_do_sample = False
        model.default_temperature = 0.0
        request = {
            "TEXT_PROMPT": _FakeTensor(b'{"mode": "lipread", "prompt": "read it"}'),
            "VIDEO_BYTES": _FakeTensor([b"QQ==", b"Qg=="]),
        }
        backend.pb_utils.get_input_tensor_by_name = lambda req, name: req.get(name)
        with self.assertRaises(ValueError) as caught:
            model._decode_sample(0, request)
        self.assertIn("exactly one", str(caught.exception))


if __name__ == "__main__":
    unittest.main()


class TokenizedConversationTests(unittest.TestCase):
    def capture(self, mode, *, prompt="", system=None):
        import json
        envelope = {"mode": mode, "prompt": prompt, "language": "it", "format": "clean"}
        if system is not None:
            envelope["system"] = system
        sample = _decode(json.dumps(envelope),
                         with_audio=mode in backend.AUDIO_REQUIRED_MODES,
                         with_video=mode in backend.VIDEO_REQUIRED_MODES)
        model = object.__new__(backend.TritonPythonModel)
        seen = []
        def tokenize(messages, **kwargs):
            seen.extend(messages)
            return {}
        model.tokenizer = types.SimpleNamespace(apply_chat_template=tokenize)
        model.processor = types.SimpleNamespace(apply_chat_template=tokenize)
        model._tokenize_sample(sample)
        return sample, seen

    def text(self, message):
        content = message["content"]
        return content if isinstance(content, str) else "".join(
            part["text"] for part in content if part["type"] == "text")

    def test_every_mode_tokenizes_only_supplied_instructions(self):
        for mode in ("chat", "transcribe", "caption", "summary", "text_summary", "mt", "st", "lipread", "avspeakd"):
            with self.subTest(mode=mode):
                sample, messages = self.capture(mode, prompt="Client instruction.")
                self.assertEqual([m["role"] for m in messages], ["user"])
                expected = "Client instruction."
                if mode == "lipread":
                    expected = backend._expand_lipread_placeholder(expected, 4)
                self.assertEqual(self.text(messages[0]), expected)
                self.assertFalse(sample.use_system)

    def test_explicit_system_is_preserved_for_all_modes(self):
        for mode in ("chat", "transcribe", "caption", "summary", "text_summary", "mt", "st", "lipread", "avspeakd"):
            with self.subTest(mode=mode):
                _, messages = self.capture(mode, prompt="Client instruction.", system="Client system.")
                self.assertEqual([m["role"] for m in messages], ["system", "user"])
                self.assertEqual(self.text(messages[0]), "Client system.")

    def test_missing_and_blank_messages_never_trigger_instructions(self):
        for mode in ("chat", "transcribe", "caption", "summary", "mt", "st", "lipread", "avspeakd"):
            for system in (None, "", "   "):
                with self.subTest(mode=mode, system=system):
                    _, messages = self.capture(mode, system=system)
                    self.assertEqual([m["role"] for m in messages], ["user"])
                    expected = backend._expand_lipread_placeholder("", 4) if mode == "lipread" else ""
                    self.assertEqual(self.text(messages[0]), expected)

    def test_achap_transcription_mode_keeps_chaptering_instruction(self):
        prompt = "Transcribe the meeting and divide it into topical chapters. Return Markdown titles followed by transcripts."
        _, messages = self.capture("transcribe", prompt=prompt)
        self.assertEqual(len(messages), 1)
        self.assertEqual(self.text(messages[0]), prompt)
        self.assertEqual(messages[0]["content"][0]["type"], "audio")
