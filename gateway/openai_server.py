#!/usr/bin/env python3
"""OpenAI-compatible gateway in front of the SpeechLMM Triton backend.

Public surface matches vLLM-Omni enough for /v1/chat/completions:
text, audio_url / input_audio, and video_url content parts. Triton stays
the batching engine; this process only translates the HTTP shape.

The client owns the prompts: a `system` message is passed through, and the
backend never supplies system or user instructions when a turn is omitted. Conversation
history is rejected, and anything the gateway cannot parse is a 400 rather than
a silently downgraded text-only request.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.parse import unquote, unquote_to_bytes, urlparse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from triton_tensors import TritonInferError, build_inputs
from triton_tensors import infer_output_metrics as triton_infer

TRITON_URL = os.environ.get("TRITON_URL", "http://127.0.0.1:8100").rstrip("/")
TRITON_TIMEOUT = int(os.environ.get("TRITON_TIMEOUT", "1800"))
ALLOW_REMOTE_MEDIA = os.environ.get("GATEWAY_ALLOW_REMOTE_MEDIA", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MAX_MEDIA_BYTES = int(os.environ.get("GATEWAY_MAX_MEDIA_BYTES", str(256 * 1024 * 1024)))
MAX_AUDIO_PARTS = int(os.environ.get("GATEWAY_MAX_AUDIO_PARTS", "8"))
MAX_VIDEO_PARTS = int(os.environ.get("GATEWAY_MAX_VIDEO_PARTS", "8"))

app = FastAPI(title="SpeechLMM OpenAI gateway", version="0.1")


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[Any]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream: bool = False
    temperature: Optional[float] = 0.0
    speechlmm_mode: Optional[str] = None
    speechlmm_language: Optional[str] = None
    speechlmm_format: Optional[str] = None
    speechlmm_caption_chars: Optional[int] = None
    modalities: Optional[list[str]] = None
    audio: Optional[dict[str, Any]] = None


#: Declared extension fields. Unknown `speechlmm_*` keys land in `model_extra` and
#: would otherwise be ignored, so a typo would silently fall back to mode derivation.
SPEECHLMM_FIELDS = tuple(
    name for name in ChatCompletionRequest.model_fields if name.startswith("speechlmm_")
)


def _load_url_bytes(url: str) -> bytes:
    if url.startswith("data:"):
        meta, _, payload = url[len("data:") :].partition(",")
        if not payload:
            raise HTTPException(status_code=400, detail="empty data: URL")
        if meta.rstrip().endswith(";base64"):
            try:
                return base64.b64decode(payload)
            except binascii.Error as exc:
                raise HTTPException(
                    status_code=400, detail=f"invalid base64 in data: URL: {exc}"
                ) from exc
        return unquote_to_bytes(payload)
    parsed = urlparse(url)
    if parsed.scheme in {"", "file"}:
        path = Path(unquote(parsed.path) if parsed.scheme == "file" else url)
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"file not found: {path}")
        return path.read_bytes()
    if parsed.scheme in {"http", "https"}:
        if not ALLOW_REMOTE_MEDIA:
            raise HTTPException(
                status_code=400,
                detail="remote media is disabled; set GATEWAY_ALLOW_REMOTE_MEDIA=1 to allow "
                "the gateway to fetch http(s) URLs",
            )
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read(MAX_MEDIA_BYTES + 1)
        except urllib.error.URLError as exc:
            raise HTTPException(status_code=400, detail=f"failed to fetch {url}: {exc}") from exc
        if len(data) > MAX_MEDIA_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"media at {url} exceeds GATEWAY_MAX_MEDIA_BYTES ({MAX_MEDIA_BYTES})",
            )
        return data
    raise HTTPException(status_code=400, detail=f"unsupported URL scheme: {parsed.scheme or 'none'}")


def _part_url(part: dict[str, Any], *keys: str) -> str:
    node: Any = part
    for key in keys:
        if not isinstance(node, dict):
            return ""
        node = node.get(key)
    return str(node or "").strip()


def _media_b64(part: dict[str, Any], ptype: str, keys: tuple[str, ...]) -> str:
    """Base64 payload of one audio/video content part; 400 if it carries nothing usable."""
    if ptype == "input_audio":
        data = str((part.get("input_audio") or {}).get("data") or "").strip()
        if not data:
            raise HTTPException(status_code=400, detail="input_audio part has no 'data'")
        return data
    for key in (*keys, "url"):
        url = _part_url(part, key, "url") or _part_url(part, key)
        if url:
            return base64.b64encode(_load_url_bytes(url)).decode("ascii")
    raise HTTPException(
        status_code=400,
        detail=f"'{ptype}' part carries no usable url; expected {{\"{keys[0]}\": "
        '{"url": "file:///path"}}',
    )


def _derive_mode(has_audio: bool, has_video: bool, explicit: str | None) -> str:
    if explicit:
        return explicit.strip().lower()
    if has_audio and has_video:
        return "avspeakd"
    if has_video:
        return "lipread"
    if has_audio:
        return "transcribe"
    return "chat"


def _finalize_user_parts(atoms: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Keep whitespace that touches media; strip the rest; join adjacent text with newlines."""
    kept: list[tuple[str, str]] = []
    for index, (kind, text) in enumerate(atoms):
        if kind != "text":
            kept.append((kind, ""))
            continue
        prev_media = index > 0 and atoms[index - 1][0] != "text"
        next_media = index + 1 < len(atoms) and atoms[index + 1][0] != "text"
        if not prev_media:
            text = text.lstrip()
        if not next_media:
            text = text.rstrip()
        if text:
            kept.append(("text", text))
    parts: list[dict[str, Any]] = []
    for kind, text in kept:
        if kind == "text" and parts and parts[-1]["type"] == "text":
            parts[-1] = {"type": "text", "text": parts[-1]["text"] + "\n" + text}
        elif kind == "text":
            parts.append({"type": "text", "text": text})
        else:
            parts.append({"type": kind})
    return parts


def _parse_messages(
    messages: list[Any],
) -> tuple[str, Optional[str], list[str], list[str], list[dict[str, Any]]]:
    systems: list[str] = []
    audio_b64: list[str] = []
    video_b64: list[str] = []
    atoms: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise HTTPException(status_code=400, detail="each message must be an object")
        role = str(message.get("role") or "user").strip().lower()
        if role not in {"user", "system"}:
            raise HTTPException(
                status_code=400,
                detail=f"role '{role}' is not supported: this endpoint takes no conversation "
                "history, only 'system' and 'user' messages",
            )
        content = message.get("content")
        if isinstance(content, str):
            text = content.strip()
            if text:
                if role == "system":
                    systems.append(text)
                else:
                    atoms.append(("text", text))
            continue
        if not isinstance(content, list):
            raise HTTPException(
                status_code=400, detail="message content must be a string or a list of parts"
            )
        for part in content:
            if not isinstance(part, dict):
                raise HTTPException(status_code=400, detail="each content part must be an object")
            ptype = str(part.get("type") or "").lower()
            if ptype == "text":
                raw = part.get("text")
                text = "" if raw is None else str(raw)
                if role == "system":
                    stripped = text.strip()
                    if stripped:
                        systems.append(stripped)
                else:
                    atoms.append(("text", text))
                continue
            if role == "system":
                raise HTTPException(
                    status_code=400,
                    detail=f"a system message carries text only, not '{ptype}'",
                )
            if ptype in {"audio_url", "audio", "input_audio"}:
                if len(audio_b64) >= MAX_AUDIO_PARTS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"at most {MAX_AUDIO_PARTS} audio parts per request "
                        "(GATEWAY_MAX_AUDIO_PARTS)",
                    )
                audio_b64.append(_media_b64(part, ptype, ("audio_url", "audio")))
                atoms.append(("audio", ""))
            elif ptype in {"video_url", "video"}:
                if len(video_b64) >= MAX_VIDEO_PARTS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"at most {MAX_VIDEO_PARTS} video parts per request "
                        "(GATEWAY_MAX_VIDEO_PARTS)",
                    )
                video_b64.append(_media_b64(part, ptype, ("video_url", "video")))
                atoms.append(("video", ""))
            else:
                raise HTTPException(
                    status_code=400, detail=f"unsupported content part type '{ptype}'"
                )
    parts = _finalize_user_parts(atoms)
    prompt = "".join(part["text"] for part in parts if part["type"] == "text")
    return prompt, "\n".join(systems) or None, audio_b64, video_b64, parts


def _triton_infer(
    model: str,
    mode: str,
    prompt: str,
    audio_b64: Optional[str] | Sequence[str],
    video_b64: Optional[str] | Sequence[str],
    max_tokens: int,
    temperature: float,
    system: Optional[str] = None,
    language: Optional[str] = None,
    fmt: Optional[str] = None,
    caption_chars: Optional[int] = None,
    parts: Optional[Sequence[dict[str, Any]]] = None,
    return_audio: bool = False,
    speaker: Optional[str] = None,
) -> tuple[str, str, Optional[str], Optional[dict[str, Any]]]:
    inputs = build_inputs(
        mode=mode,
        prompt=prompt,
        system=system,
        audio_b64=audio_b64,
        video_b64=video_b64,
        max_new_tokens=max_tokens,
        temperature=temperature,
        language=language,
        fmt=fmt,
        caption_chars=caption_chars,
        parts=parts,
        return_audio=return_audio,
        speaker=speaker,
    )
    try:
        result = triton_infer(TRITON_URL, model, inputs, TRITON_TIMEOUT)
    except TritonInferError as exc:
        raise HTTPException(status_code=exc.status or 502, detail=str(exc)) from exc
    # Tests and older callers patch triton_infer with the 3-tuple infer_output shape.
    if len(result) == 3:
        text, reason, audio = result
        return text, reason, audio, None
    return result


@app.get("/health")
def health() -> dict[str, str]:
    try:
        with urllib.request.urlopen(f"{TRITON_URL}/v2/health/ready", timeout=5):
            pass
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail=f"Triton not ready: {exc}") from exc
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    req = urllib.request.Request(
        f"{TRITON_URL}/v2/repository/index",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            index = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=503, detail=f"Triton repository index unavailable: {exc}"
        ) from exc
    data = []
    for entry in index:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if name and entry.get("state") == "READY":
            data.append({"id": name, "object": "model", "owned_by": "speechlmm"})
    return {"object": "list", "data": data}


def _speech_options(body: ChatCompletionRequest) -> tuple[bool, Optional[str]]:
    """OpenAI audio output: modalities include 'audio', voice becomes the speaker."""
    modalities = body.modalities
    audio = body.audio
    allowed = {"text", "audio"}
    if modalities is not None:
        bad = [item for item in modalities if str(item).lower() not in allowed]
        if bad:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported modalities {bad}; supported: text, audio",
            )
        want_audio = any(str(item).lower() == "audio" for item in modalities)
    else:
        want_audio = False
    if audio is not None and not want_audio:
        raise HTTPException(
            status_code=400,
            detail="audio is only valid when modalities includes 'audio'",
        )
    if not want_audio:
        return False, None
    if audio is None:
        return True, None
    if not isinstance(audio, dict):
        raise HTTPException(status_code=400, detail="audio must be an object")
    fmt = audio.get("format")
    if fmt is not None and str(fmt).lower() != "wav":
        raise HTTPException(status_code=400, detail="audio.format must be 'wav'")
    voice = audio.get("voice")
    speaker = str(voice).strip() if voice else None
    return True, speaker or None


@app.post("/v1/chat/completions")
def chat_completions(body: ChatCompletionRequest) -> dict[str, Any]:
    if body.stream:
        raise HTTPException(
            status_code=400,
            detail="stream=true is not supported; Triton's Python backend needs decoupled mode.",
        )
    unknown = sorted(k for k in (body.model_extra or {}) if k.startswith("speechlmm_"))
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"unknown extension field(s) {unknown}; supported: {list(SPEECHLMM_FIELDS)}",
        )
    prompt, system, audio_b64, video_b64, parts = _parse_messages(body.messages)
    mode = _derive_mode(bool(audio_b64), bool(video_b64), body.speechlmm_mode)
    return_audio, speaker = _speech_options(body)
    text, finish_reason, generated_audio, metrics = _triton_infer(
        model=body.model,
        mode=mode,
        prompt=prompt,
        audio_b64=audio_b64,
        video_b64=video_b64,
        max_tokens=body.max_completion_tokens or body.max_tokens or 128,
        temperature=float(body.temperature or 0.0),
        system=system,
        language=body.speechlmm_language,
        fmt=body.speechlmm_format,
        caption_chars=body.speechlmm_caption_chars,
        parts=parts,
        return_audio=return_audio,
        speaker=speaker,
    )
    if return_audio and not generated_audio:
        raise HTTPException(
            status_code=502,
            detail="speech output requested but the model returned no audio",
        )
    created = int(time.time())
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if return_audio:
        message["audio"] = {
            "id": f"audio_{uuid.uuid4().hex[:24]}",
            "data": generated_audio,
            "expires_at": created + 3600,
            "transcript": text,
        }
    response: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": created,
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }
    if metrics:
        prompt_tokens = metrics.get("input_tokens")
        completion_tokens = metrics.get("output_tokens")
        if prompt_tokens is not None and completion_tokens is not None:
            response["usage"] = {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens) + int(completion_tokens),
            }
        response["metrics"] = metrics
    return response
