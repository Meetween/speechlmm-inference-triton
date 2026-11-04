"""Template for Triton Python backends.

Subclasses implement ``generate``. Request decoding, grouping, the per-request
fallback, response tensors and telemetry hook fields live here, on the same
pattern as ``CorpusPreparer``: the pipeline is shared, the model call is not.

``pb_utils`` is taken from the subclass module, so tests that patch
``backend.pb_utils`` or ``qwen.pb_utils`` still affect this code.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from backend_common import ContextOverflow
from telemetry import record_usage


@dataclass
class Sample:
    """One request after the shared decode step. Backends may subclass it."""

    index: int
    mode: str
    spec: Dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    system: str = ""
    audio: List[bytes] = field(default_factory=list)
    video: List[bytes] = field(default_factory=list)
    parts: tuple = ()
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    return_audio: bool = False
    speaker: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    text: str
    finish_reason: str
    audio_b64: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)


class BaseBackend(ABC):
    """Fixed ``execute`` pipeline. Override the hooks, not ``execute``."""

    supported_modes: Optional[frozenset] = None
    supports_batching: bool = False

    def _pb(self):
        module = sys.modules.get(type(self).__module__)
        pb = getattr(module, "pb_utils", None) if module is not None else None
        if pb is None:
            import triton_python_backend_utils as pb
        return pb

    def initialize(self, args: Dict[str, Any]) -> None:
        raw = (args or {}).get("model_config") or "{}"
        try:
            import json

            model_config = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except json.JSONDecodeError:
            model_config = {}
        self.load_model(args or {}, model_config)

    def load_model(self, args: Dict[str, Any], model_config: Dict[str, Any]) -> None:
        """Load weights. The SpeechLMM and Qwen classes still load in ``initialize``."""

    @abstractmethod
    def decode_request(self, index: int, request: Any) -> Any:
        raise NotImplementedError

    def validate(self, sample: Any) -> None:
        modes = type(self).supported_modes
        if modes is not None and getattr(sample, "mode", None) not in modes:
            raise ValueError(f"mode '{sample.mode}' is not supported by {type(self).__name__}")
        self.validate_extra(sample)

    def validate_extra(self, sample: Any) -> None:
        return None

    def prepare(self, sample: Any) -> None:
        return None

    def short_circuit(self, sample: Any) -> Optional[Result]:
        return None

    def group_key(self, sample: Any) -> tuple:
        if not type(self).supports_batching:
            return (sample.index,)
        custom = getattr(sample, "group_key", None)
        if callable(custom):
            return tuple(custom())
        return (getattr(sample, "mode", ""),)

    @abstractmethod
    def generate(self, samples: Sequence[Any]) -> List[Result]:
        raise NotImplementedError

    def postprocess(self, sample: Any, text: str) -> str:
        return text

    def cleanup(self, sample: Any) -> None:
        return None

    def map_error(self, exc: BaseException) -> Any:
        pb = self._pb()
        message = f"{type(exc).__name__}: {exc}"
        # INVALID_ARG makes the gateway answer 400 for a request that is too long.
        code = getattr(pb.TritonError, "INVALID_ARG", None)
        if isinstance(exc, ContextOverflow) and code is not None:
            try:
                return pb.InferenceResponse(error=pb.TritonError(message, code))
            except TypeError:
                pass  # older pb_utils takes the message only
        return pb.InferenceResponse(error=pb.TritonError(message))

    def build_response(self, result: Result) -> Any:
        import numpy as np

        pb = self._pb()
        out = pb.Tensor("GENERATED_TEXT", np.array([result.text], dtype=object))
        reason = pb.Tensor("FINISH_REASON", np.array([result.finish_reason], dtype=object))
        audio = pb.Tensor("GENERATED_AUDIO", np.array([result.audio_b64], dtype=object))
        return pb.InferenceResponse(output_tensors=[out, reason, audio])

    #: stop / length come from ``generate``; an empty text turns them into
    #: empty (EOS, nothing decoded) / empty_length (budget spent on nothing).
    EMPTY_REASONS = {"stop": "empty"}

    def _empty_reason(self, reason: str) -> str:
        return self.EMPTY_REASONS.get(reason, f"empty_{reason}")

    def _finish(self, sample: Any, result: Result, *, decoded: bool = True) -> Any:
        text = self.postprocess(sample, result.text)
        reason = result.finish_reason
        # A short_circuit answer (Qwen's VAD on silence) is empty on purpose.
        if decoded and not str(text).strip():
            # Recorded rather than raised, so the sample stays in the scored set.
            reason, text = self._empty_reason(reason), ""
        record_usage(self, sample.index, result.usage)
        return self.build_response(Result(text, reason, result.audio_b64, result.usage))

    def execute(self, requests: List[Any]) -> List[Any]:
        pb = self._pb()
        n = len(requests)
        responses: List[Any] = [None] * n
        pending: List[Any] = []
        for index, request in enumerate(requests):
            sample = None
            keep = False
            try:
                sample = self.decode_request(index, request)
                self.validate(sample)
                self.prepare(sample)
                early = self.short_circuit(sample)
                if early is not None:
                    responses[index] = self._finish(sample, early, decoded=False)
                else:
                    pending.append(sample)
                    keep = True
            except Exception as exc:
                responses[index] = self.map_error(exc)
            finally:
                if sample is not None and not keep:
                    self.cleanup(sample)

        groups: Dict[tuple, List[Any]] = {}
        for sample in pending:
            groups.setdefault(self.group_key(sample), []).append(sample)

        for group in groups.values():
            try:
                self._assign_group(responses, group, self.generate(group))
            except Exception as exc:
                if len(group) == 1:
                    responses[group[0].index] = self.map_error(exc)
                else:
                    pb.Logger.log_warn(
                        f"batched generate failed for {len(group)} requests "
                        f"({type(exc).__name__}: {exc}); retrying one request at a time"
                    )
                    for sample in group:
                        try:
                            self._assign_group(responses, [sample], self.generate([sample]))
                        except Exception as inner:
                            responses[sample.index] = self.map_error(inner)
            finally:
                for sample in group:
                    self.cleanup(sample)

        empty = RuntimeError("empty response")
        return [item if item is not None else self.map_error(empty) for item in responses]

    def _assign_group(self, responses: List[Any], group: Sequence[Any], results: Sequence[Result]) -> None:
        if len(results) != len(group):
            raise RuntimeError(
                f"batched generate returned {len(results)} texts for {len(group)} requests"
            )
        for sample, result in zip(group, results):
            responses[sample.index] = self._finish(sample, result)
