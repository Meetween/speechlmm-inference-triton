"""Numerical and lifecycle regressions for serving memory optimizations."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backends"))
from long_audio import _windowed_attention, generation_logits, install_windowed_audio

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires torch")
class LongAudioTests(unittest.TestCase):
    def test_ragged_windows_match_dense_block_mask(self):
        torch.manual_seed(4)
        attention = torch.nn.Module()
        attention.num_heads = 2
        attention.scaling = 0.5
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(attention, name, torch.nn.Linear(8, 8))
        attention.eval()
        # Includes a batch boundary at 64 equal windows and a final short one.
        for lengths in ([5, 5, 3], [2] * 70 + [1]):
            bounds = torch.tensor([0] + list(__import__('itertools').accumulate(lengths)))
            hidden = torch.randn(sum(lengths), 8)
            mask = torch.full((sum(lengths), sum(lengths)), float('-inf'))
            for start, end in zip(bounds[:-1], bounds[1:]):
                mask[start:end, start:end] = 0
            def project(layer):
                return layer(hidden).reshape(-1, 2, 4).transpose(0, 1)
            dense = torch.nn.functional.scaled_dot_product_attention(
                project(attention.q_proj), project(attention.k_proj), project(attention.v_proj),
                attn_mask=mask, scale=attention.scaling,
            )
            expected = attention.out_proj(dense.transpose(0, 1).reshape(-1, 8))
            actual = _windowed_attention(attention, hidden, bounds)
            torch.testing.assert_close(actual, expected)
            with self.assertRaises(ValueError):
                _windowed_attention(attention, hidden, bounds, attention_mask=mask)

    def test_logits_preserve_last_position_and_restore_after_error(self):
        model = torch.nn.Module()
        model.lm_head = torch.nn.Linear(8, 17)
        model.eval()
        hidden = torch.randn(2, 7, 8)
        expected = model.lm_head(hidden)
        with self.assertRaisesRegex(RuntimeError, "request failed"):
            with generation_logits(model):
                actual = model.lm_head(hidden)
                self.assertEqual(actual.shape, (2, 1, 17))
                torch.testing.assert_close(actual, expected[:, -1:, :])
                raise RuntimeError("request failed")
        torch.testing.assert_close(model.lm_head(hidden), expected)
        self.assertEqual(len(model.lm_head._forward_pre_hooks), 0)
        model.train()
        with self.assertRaisesRegex(RuntimeError, "eval mode"):
            with generation_logits(model):
                pass

    def test_unrelated_architecture_is_untouched(self):
        model = torch.nn.Linear(3, 3)
        self.assertFalse(install_windowed_audio(model))

    def test_accelerate_forward_hook_is_preserved(self):
        Attention = type("Qwen2_5OmniAudioAttention", (), {})
        Encoder = type("Qwen2_5OmniAudioEncoder", (), {})
        attention = Attention()
        wrapper = object()
        attention.forward = wrapper
        attention._old_forward = object()
        attention._hf_hook = object()
        encoder = Encoder()
        encoder.layers = [SimpleNamespace(self_attn=attention)]
        encoder._prepare_attention_mask = lambda *args: None
        model = SimpleNamespace(audio_tower=encoder, training=False)
        self.assertTrue(install_windowed_audio(model))
        self.assertIs(attention.forward, wrapper)
        self.assertIs(attention._old_forward.__func__, _windowed_attention)
        self.assertTrue(install_windowed_audio(model))


if __name__ == '__main__':
    unittest.main()
