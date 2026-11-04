#!/usr/bin/env python3
"""Tests for the gateway's request parsing - no Triton server needed.

fastapi and pydantic live in the SIF, so run these with the container
interpreter:

  singularity exec "${SIF}" python3.12 -m unittest discover -s tests
"""

from __future__ import annotations

import base64
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote_from_bytes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from fastapi import HTTPException

    from gateway.openai_server import (
        ChatCompletionRequest,
        _derive_mode,
        _load_url_bytes,
        _parse_messages,
        chat_completions,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - depends on the interpreter
    raise unittest.SkipTest(f"gateway dependencies are not importable here: {exc}")


def _user(*parts: dict) -> list[dict]:
    return [{"role": "user", "content": list(parts)}]


class DeriveModeTests(unittest.TestCase):
    def test_media_selects_the_mode(self) -> None:
        self.assertEqual(_derive_mode(False, False, None), "chat")
        self.assertEqual(_derive_mode(True, False, None), "transcribe")
        self.assertEqual(_derive_mode(False, True, None), "lipread")
        self.assertEqual(_derive_mode(True, True, None), "avspeakd")

    def test_explicit_mode_wins(self) -> None:
        self.assertEqual(_derive_mode(True, False, "summary"), "summary")
        self.assertEqual(_derive_mode(False, False, " TEXT_SUMMARY "), "text_summary")


class ParseMessagesTests(unittest.TestCase):
    def test_string_content_becomes_the_prompt(self) -> None:
        prompt, system, audio, video, _order = _parse_messages(
            [{"role": "user", "content": "hello"}]
        )
        self.assertEqual(prompt, "hello")
        self.assertIsNone(system)
        self.assertEqual(audio, [])
        self.assertEqual(video, [])

    def test_audio_url_is_read_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "clip.wav"
            clip.write_bytes(b"RIFFDATA")
            prompt, _system, audio, video, _order = _parse_messages(
                _user(
                    {"type": "text", "text": "Transcribe."},
                    {"type": "audio_url", "audio_url": {"url": clip.as_uri()}},
                )
            )
        self.assertEqual(prompt, "Transcribe.")
        self.assertEqual(base64.b64decode(audio[0]), b"RIFFDATA")
        self.assertEqual(video, [])

    def test_input_audio_passes_base64_through(self) -> None:
        _, _, audio, _, _order = _parse_messages(
            _user({"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}})
        )
        self.assertEqual(audio, ["QUJD"])

    def test_audio_part_without_a_url_is_rejected(self) -> None:
        """The dangerous case: this used to become a text-only chat with a 200."""
        with self.assertRaises(HTTPException) as caught:
            _parse_messages(
                _user(
                    {"type": "text", "text": "Transcribe."},
                    {"type": "audio_url", "audio_url": {}},
                )
            )
        self.assertEqual(caught.exception.status_code, 400)

    def test_input_audio_without_data_is_rejected(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            _parse_messages(_user({"type": "input_audio", "input_audio": {"data": ""}}))
        self.assertEqual(caught.exception.status_code, 400)

    def test_several_audio_and_video_parts_keep_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.mp4"
            second = Path(tmp) / "b.mp4"
            first.write_bytes(b"VID1")
            second.write_bytes(b"VID2")
            prompt, _system, audio, video, parts = _parse_messages(
                _user(
                    {"type": "input_audio", "input_audio": {"data": "QUJD"}},
                    {"type": "video_url", "video_url": {"url": first.as_uri()}},
                    {"type": "input_audio", "input_audio": {"data": "REVG"}},
                    {"type": "video_url", "video_url": {"url": second.as_uri()}},
                    {"type": "text", "text": "Pair them."},
                )
            )
        self.assertEqual(prompt, "Pair them.")
        self.assertEqual(audio, ["QUJD", "REVG"])
        self.assertEqual([base64.b64decode(item) for item in video], [b"VID1", b"VID2"])
        self.assertEqual(
            parts,
            [
                {"type": "audio"},
                {"type": "video"},
                {"type": "audio"},
                {"type": "video"},
                {"type": "text", "text": "Pair them."},
            ],
        )

    def test_text_then_audio_keeps_the_trailing_space(self) -> None:
        prompt, _system, _audio, _video, parts = _parse_messages(
            _user(
                {"type": "text", "text": "Translate.\nInput: "},
                {"type": "input_audio", "input_audio": {"data": "QUJD"}},
            )
        )
        self.assertEqual(prompt, "Translate.\nInput: ")
        self.assertEqual(
            parts,
            [
                {"type": "text", "text": "Translate.\nInput: "},
                {"type": "audio"},
            ],
        )

    def test_audio_then_text_keeps_the_audio_first(self) -> None:
        _prompt, _system, _audio, _video, parts = _parse_messages(
            _user(
                {"type": "input_audio", "input_audio": {"data": "QUJD"}},
                {"type": "text", "text": "after"},
            )
        )
        self.assertEqual(
            parts,
            [{"type": "audio"}, {"type": "text", "text": "after"}],
        )

    def test_adjacent_text_parts_are_joined_with_a_newline(self) -> None:
        prompt, _system, _audio, _video, parts = _parse_messages(
            _user(
                {"type": "text", "text": "  hello  "},
                {"type": "text", "text": "  world  "},
            )
        )
        self.assertEqual(prompt, "hello\nworld")
        self.assertEqual(parts, [{"type": "text", "text": "hello\nworld"}])

    def test_whitespace_between_media_is_kept(self) -> None:
        _prompt, _system, _audio, _video, parts = _parse_messages(
            _user(
                {"type": "input_audio", "input_audio": {"data": "QUJD"}},
                {"type": "text", "text": " "},
                {"type": "input_audio", "input_audio": {"data": "REVG"}},
            )
        )
        self.assertEqual(
            parts,
            [
                {"type": "audio"},
                {"type": "text", "text": " "},
                {"type": "audio"},
            ],
        )

    def test_too_many_audio_parts_is_rejected(self) -> None:
        from gateway.openai_server import MAX_AUDIO_PARTS

        parts = [
            {"type": "input_audio", "input_audio": {"data": "QUJD"}}
            for _ in range(MAX_AUDIO_PARTS + 1)
        ]
        with self.assertRaises(HTTPException) as caught:
            _parse_messages(_user(*parts))
        self.assertEqual(caught.exception.status_code, 400)

    def test_unknown_part_type_is_rejected(self) -> None:
        with self.assertRaises(HTTPException):
            _parse_messages(_user({"type": "image_url", "image_url": {"url": "file:///x.png"}}))

    def test_system_message_is_passed_through(self) -> None:
        """The client owns the prompts, so a system turn must survive."""
        prompt, system, _, _, _order = _parse_messages(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ]
        )
        self.assertEqual(system, "You are helpful.")
        self.assertEqual(prompt, "hi")

    def test_system_text_parts_are_joined(self) -> None:
        _, system, _, _, _order = _parse_messages(
            [
                {"role": "system", "content": [{"type": "text", "text": "Be terse."}]},
                {"role": "user", "content": "hi"},
            ]
        )
        self.assertEqual(system, "Be terse.")

    def test_media_in_a_system_message_is_rejected(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            _parse_messages(
                [
                    {
                        "role": "system",
                        "content": [{"type": "input_audio", "input_audio": {"data": "QUJD"}}],
                    },
                    {"role": "user", "content": "hi"},
                ]
            )
        self.assertEqual(caught.exception.status_code, 400)

    def test_conversation_history_is_still_rejected(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            _parse_messages(
                [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            )
        self.assertEqual(caught.exception.status_code, 400)


class ExtensionFieldTests(unittest.TestCase):
    """A typo in extra_body must not fall back to mode derivation with a 200."""

    def _request(self, **extra) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="speechlmm_v2_l",
            messages=[{"role": "user", "content": "hi"}],
            **extra,
        )

    def test_misspelled_mode_is_rejected(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            chat_completions(self._request(speechlmm_mdoe="transcribe"))
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("speechlmm_mdoe", caught.exception.detail)

    def test_unrelated_openai_fields_are_tolerated(self) -> None:
        """Real SDKs send top_p and friends; only speechlmm_* keys are policed."""
        body = self._request(top_p=0.9, n=1)
        self.assertEqual(
            sorted(k for k in body.model_extra if k.startswith("speechlmm_")), []
        )

    def test_stream_is_rejected_before_anything_else(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            chat_completions(self._request(stream=True))
        self.assertEqual(caught.exception.status_code, 400)


class LoadUrlBytesTests(unittest.TestCase):
    def test_base64_data_url(self) -> None:
        url = "data:audio/wav;base64," + base64.b64encode(b"RIFF").decode("ascii")
        self.assertEqual(_load_url_bytes(url), b"RIFF")

    def test_percent_encoded_data_url(self) -> None:
        url = "data:text/plain," + quote_from_bytes(b"a b")
        self.assertEqual(_load_url_bytes(url), b"a b")

    def test_undecodable_base64_is_a_400_not_a_500(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            _load_url_bytes("data:audio/wav;base64,hello")
        self.assertEqual(caught.exception.status_code, 400)

    def test_missing_file_is_a_400(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            _load_url_bytes("file:///nonexistent/clip.wav")
        self.assertEqual(caught.exception.status_code, 400)

    def test_remote_urls_are_refused_by_default(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            _load_url_bytes("https://example.invalid/clip.wav")
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("GATEWAY_ALLOW_REMOTE_MEDIA", caught.exception.detail)

    def test_unsupported_scheme_is_a_400(self) -> None:
        with self.assertRaises(HTTPException):
            _load_url_bytes("ftp://example.invalid/clip.wav")


class SpeechOutputGatewayTests(unittest.TestCase):
    def _body(self, **extra) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="speechlmm_v2_l",
            messages=[{"role": "user", "content": "Say this."}],
            speechlmm_mode="tts",
            **extra,
        )

    def test_bad_modalities_audio_without_modalities_and_format_are_400(self) -> None:
        cases = [
            dict(modalities=["video"]),
            dict(audio={"voice": "ethan"}),
            dict(modalities=["text", "audio"], audio={"format": "mp3"}),
        ]
        for extra in cases:
            with self.subTest(extra=extra):
                with self.assertRaises(HTTPException) as caught:
                    chat_completions(self._body(**extra))
                self.assertEqual(caught.exception.status_code, 400)

    def test_message_audio_shape(self) -> None:
        with patch("gateway.openai_server.triton_infer", return_value=("hello", "stop", "QQ==")):
            response = chat_completions(
                self._body(modalities=["text", "audio"], audio={"voice": "ethan", "format": "wav"})
            )
        message = response["choices"][0]["message"]
        self.assertEqual(message["content"], "hello")
        self.assertEqual(message["audio"]["data"], "QQ==")
        self.assertEqual(message["audio"]["transcript"], "hello")
        self.assertTrue(message["audio"]["id"].startswith("audio_"))
        self.assertGreater(message["audio"]["expires_at"], response["created"])

    def test_missing_audio_is_502(self) -> None:
        with patch("gateway.openai_server.triton_infer", return_value=("hello", "stop", None)):
            with self.assertRaises(HTTPException) as caught:
                chat_completions(
                    self._body(modalities=["text", "audio"], audio={"voice": "ethan", "format": "wav"})
                )
        self.assertEqual(caught.exception.status_code, 502)

    def test_text_only_response_is_unchanged(self) -> None:
        with patch("gateway.openai_server.triton_infer", return_value=("Hi", "stop", None)):
            response = chat_completions(
                ChatCompletionRequest(
                    model="speechlmm_v2_l",
                    messages=[{"role": "user", "content": "Hello"}],
                )
            )
        message = response["choices"][0]["message"]
        self.assertEqual(message, {"role": "assistant", "content": "Hi"})
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertNotIn("usage", response)
        self.assertNotIn("metrics", response)

    def test_response_keys_unchanged_without_metrics(self) -> None:
        with patch("gateway.openai_server.triton_infer", return_value=("Hi", "stop", None)):
            response = chat_completions(
                ChatCompletionRequest(
                    model="speechlmm_v2_l",
                    messages=[{"role": "user", "content": "Hello"}],
                )
            )
        self.assertEqual(
            set(response.keys()),
            {"id", "object", "created", "model", "choices"},
        )

    def test_usage_and_metrics_when_backend_sends_metrics(self) -> None:
        metrics = {
            "mode": "transcribe",
            "input_tokens": 10,
            "output_tokens": 4,
            "rtf": 0.2,
        }
        with patch(
            "gateway.openai_server.triton_infer",
            return_value=("Hi", "stop", None, metrics),
        ):
            response = chat_completions(
                ChatCompletionRequest(
                    model="speechlmm_v2_xl",
                    messages=[{"role": "user", "content": "Hello"}],
                    speechlmm_mode="transcribe",
                )
            )
        self.assertEqual(
            response["usage"],
            {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        )
        self.assertEqual(response["metrics"], metrics)

    def test_usage_omitted_when_token_counts_missing(self) -> None:
        metrics = {"mode": "chat", "rtf": 1.0}
        with patch(
            "gateway.openai_server.triton_infer",
            return_value=("Hi", "stop", None, metrics),
        ):
            response = chat_completions(
                ChatCompletionRequest(
                    model="speechlmm_v2_l",
                    messages=[{"role": "user", "content": "Hello"}],
                )
            )
        self.assertNotIn("usage", response)
        self.assertEqual(response["metrics"], metrics)


if __name__ == "__main__":
    unittest.main()
