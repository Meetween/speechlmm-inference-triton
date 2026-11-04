"""Bounded-memory Qwen2.5 audio attention and generation-only logits.

The audio encoder already isolates windows with a block-diagonal mask. Evaluate
those windows directly instead of materializing the quadratic global mask.
No audio is dropped and the language model still receives one full sequence.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType


def _windowed_attention(self, hidden_states, cu_seqlens=None, attention_mask=None, **kwargs):
    import torch
    from torch.nn import functional as F

    if self.training:
        raise RuntimeError("Windowed audio serving attention requires eval mode")
    if attention_mask is not None:
        raise ValueError("Windowed audio attention cannot ignore a custom mask")
    length, heads = hidden_states.shape[0], self.num_heads
    bounds = cu_seqlens.tolist()
    if not bounds or bounds[0] != 0 or bounds[-1] != length or any(
        right <= left for left, right in zip(bounds, bounds[1:])
    ):
        raise ValueError("Invalid audio attention window boundaries")
    q = self.q_proj(hidden_states).reshape(length, heads, -1)
    k = self.k_proj(hidden_states).reshape(length, heads, -1)
    v = self.v_proj(hidden_states).reshape(length, heads, -1)
    outputs = []
    i = 0
    while i < len(bounds) - 1:
        width = bounds[i + 1] - bounds[i]
        j = i + 1
        while j < len(bounds) - 1 and j - i < 64 and bounds[j + 1] - bounds[j] == width:
            j += 1
        begin, end, batch = bounds[i], bounds[j], j - i
        def windows(tensor):
            return tensor[begin:end].reshape(batch, width, heads, -1).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            windows(q), windows(k), windows(v), attn_mask=None,
            dropout_p=0.0, is_causal=False, scale=self.scaling,
        )
        outputs.append(out.transpose(1, 2).reshape(end - begin, -1))
        i = j
    return self.out_proj(torch.cat(outputs, dim=0))


def _no_global_mask(self, inputs, cu_seqlens):
    return None


def install_windowed_audio(model):
    """Patch only the known Qwen2.5 encoder; leave other architectures alone.

    Replace Accelerate's wrapped callable, not its placement hook. Refuse an
    unexpected attention implementation rather than silently changing semantics.
    Returns whether this model uses the supported encoder.
    """
    encoder = getattr(model, "audio_tower", None)
    if encoder is None or type(encoder).__name__ != "Qwen2_5OmniAudioEncoder":
        return False
    if getattr(encoder, "_windowed_serving", False):
        return True
    if model.training:
        raise RuntimeError("Call model.eval() before installing serving attention")
    attentions = [layer.self_attn for layer in encoder.layers]
    for attention in attentions:
        if type(attention).__name__ != "Qwen2_5OmniAudioAttention":
            raise RuntimeError("Unsupported Qwen2.5 audio attention implementation")
    if not hasattr(encoder, "_prepare_attention_mask"):
        raise RuntimeError("Unsupported Qwen2.5 audio encoder mask API")
    for attention in attentions:
        key = "_old_forward" if hasattr(attention, "_hf_hook") else "forward"
        setattr(attention, key, MethodType(_windowed_attention, attention))
    encoder._prepare_attention_mask = MethodType(_no_global_mask, encoder)
    encoder._windowed_serving = True
    return True


@contextmanager
def generation_logits(model, enabled=True):
    """Project only the last position during this backend's text generation.

    Older remote-code SpeechLMM forwards do not accept logits_to_keep. A scoped
    lm_head hook provides the same saving without editing checkpoint code. It
    is removed on success or failure; ordinary forwards and training keep their
    original output shape. Triton executes requests serially per model instance.
    """
    if not enabled:
        yield
        return
    if model.training:
        raise RuntimeError("Generation logits optimization requires eval mode")
    def last_position(module, args):
        return (args[0][:, -1:, :],) + args[1:]
    handle = model.lm_head.register_forward_pre_hook(last_position)
    try:
        yield
    finally:
        handle.remove()
