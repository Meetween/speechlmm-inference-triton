"""CPU tests for backends/speech_output.py. soundfile comes from the serving image."""

from __future__ import annotations

import base64
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backends"))

from speech_output import (  # noqa: E402
    OUTPUT_SAMPLE_RATE,
    apply_qwen25_dit_rope_fix,
    encode_wav_b64,
    resolve_speaker,
    speech_generate_kwargs,
    validate_speech_request,
)


class WavTests(unittest.TestCase):
    def test_round_trip_is_24khz_pcm_and_clips(self) -> None:
        import numpy as np
        import soundfile as sf

        wav = np.array([0.0, 0.5, 2.0, -2.0], dtype=np.float32)
        raw = base64.b64decode(encode_wav_b64(wav))
        data, rate = sf.read(io.BytesIO(raw), dtype="float32")
        self.assertEqual(rate, OUTPUT_SAMPLE_RATE)
        self.assertEqual(rate, 24000)
        self.assertAlmostEqual(float(data[1]), 0.5, places=3)
        self.assertAlmostEqual(float(data[2]), 1.0, places=3)
        self.assertAlmostEqual(float(data[3]), -1.0, places=3)


class SpeakerTests(unittest.TestCase):
    def test_match_is_case_insensitive(self) -> None:
        self.assertEqual(resolve_speaker("ethan", ["Chelsie", "Ethan"]), "Ethan")
        self.assertEqual(resolve_speaker("ETHAN", ["ethan", "chelsie"]), "ethan")

    def test_unknown_speaker_lists_the_valid_names(self) -> None:
        with self.assertRaises(ValueError) as caught:
            resolve_speaker("bob", ["Ethan", "Chelsie"])
        self.assertIn("Ethan", str(caught.exception))
        self.assertIn("Chelsie", str(caught.exception))


class ValidateTests(unittest.TestCase):
    def _ok(self, **kwargs) -> None:
        defaults = dict(
            mode="chat",
            return_audio=False,
            has_audio=False,
            has_video=False,
            has_lipread=False,
            prompt="hi",
            speech_enabled=True,
        )
        defaults.update(kwargs)
        validate_speech_request(**defaults)

    def test_allowed_requests(self) -> None:
        self._ok()
        self._ok(mode="tts", return_audio=True, prompt="Say this.")
        self._ok(mode="s2st", return_audio=True, has_audio=True, prompt="")

    def test_rejected_requests(self) -> None:
        cases = [
            dict(return_audio=True, speech_enabled=False),
            dict(return_audio=True, has_video=True),
            dict(return_audio=True, has_lipread=True),
            dict(mode="tts", return_audio=False),
            dict(mode="s2st", return_audio=False, has_audio=True),
            dict(mode="tts", return_audio=True, has_audio=True),
            dict(mode="tts", return_audio=True, prompt="  "),
            dict(mode="s2st", return_audio=True, has_audio=False),
        ]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    self._ok(**case)


class KwargsTests(unittest.TestCase):
    def test_only_prefixed_keys_plus_speaker_and_audio_flags(self) -> None:
        sample = SimpleNamespace(
            max_new_tokens=8, do_sample=False, temperature=0.7, repetition_penalty=1.1
        )
        kwargs = speech_generate_kwargs(sample, "Ethan", 32, True, 9)
        allowed = {"speaker", "return_audio", "use_audio_in_video"}
        self.assertTrue(all(key.startswith(("thinker_", "talker_")) or key in allowed for key in kwargs))
        self.assertNotIn("use_cache", kwargs)
        self.assertNotIn("pad_token_id", kwargs)
        self.assertNotIn("thinker_temperature", kwargs)
        self.assertEqual(kwargs["speaker"], "Ethan")
        self.assertIs(kwargs["return_audio"], True)
        self.assertEqual(kwargs["thinker_eos_token_id"], 9)
        self.assertEqual(kwargs["talker_max_new_tokens"], 32)

    def test_temperature_is_included_only_when_sampling(self) -> None:
        sample = SimpleNamespace(
            max_new_tokens=8, do_sample=True, temperature=0.7, repetition_penalty=1.0
        )
        kwargs = speech_generate_kwargs(sample, "ethan", 16, True, None)
        self.assertEqual(kwargs["thinker_temperature"], 0.7)
        self.assertNotIn("thinker_eos_token_id", kwargs)


class RopeFixTests(unittest.TestCase):
    def test_broken_function_is_replaced_and_a_fixed_one_is_left_alone(self) -> None:
        def broken(q, k, cos, sin, unsqueeze_dim=1):
            rotate_half_codec = 0
            return q, k, rotate_half_codec

        def fixed(q, k, cos, sin, unsqueeze_dim=1):
            return q, k

        broken_mod = SimpleNamespace(apply_rotary_pos_emb=broken)
        self.assertTrue(apply_qwen25_dit_rope_fix(broken_mod))
        self.assertIsNot(broken_mod.apply_rotary_pos_emb, broken)

        fixed_mod = SimpleNamespace(apply_rotary_pos_emb=fixed)
        self.assertFalse(apply_qwen25_dit_rope_fix(fixed_mod))
        self.assertIs(fixed_mod.apply_rotary_pos_emb, fixed)


if __name__ == "__main__":
    unittest.main()
