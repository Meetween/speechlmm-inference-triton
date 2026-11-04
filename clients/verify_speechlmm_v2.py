#!/usr/bin/env python3
"""Verification ladder for the SpeechLMM gateway once it is serving.

Drives the OpenAI-compatible surface with the official SDK, so passing here means
a real client works: health, model list, chat, ASR, lipread, avspeakd, multi-clip,
the rejections that must not silently degrade into text-only chat, batch invariance
and throughput. Needs a live endpoint.

  python clients/verify_speechlmm_v2.py --url http://127.0.0.1:8000
  python clients/verify_speechlmm_v2.py --audio clip.wav --video mouth.mp4 \\
      --vsr-reference /tmp/vsr_ref.txt
  # With both --audio and --video, also runs avspeakd.

To bypass the gateway and post raw Triton tensors instead, use
clients/infer_speechlmm_v2.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Sequence

from openai import APIStatusError, BadRequestError, OpenAI

PONG = "Reply with the single word: pong"


def _health(url: str) -> None:
    endpoint = url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(endpoint, timeout=30):
            pass
    except urllib.error.URLError as exc:
        raise SystemExit(f"gateway not healthy at {endpoint}: {exc}") from exc


def _chat(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    mode: Optional[str] = None,
    audio: Optional[Path] = None,
    video: Optional[Path] = None,
    system: Optional[str] = None,
    max_tokens: int = 32,
) -> str:
    content: list[dict[str, Any]] = []
    if prompt:
        content.append({"type": "text", "text": prompt})
    if audio is not None:
        content.append({"type": "audio_url", "audio_url": {"url": audio.resolve().as_uri()}})
    if video is not None:
        content.append({"type": "video_url", "video_url": {"url": video.resolve().as_uri()}})
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        extra_body={"speechlmm_mode": mode} if mode else None,
    )
    return completion.choices[0].message.content or ""


def _chat_parts(
    client: OpenAI,
    model: str,
    prompt: str,
    parts: Sequence[tuple[str, Path]],
    *,
    mode: Optional[str] = None,
    max_tokens: int = 32,
) -> str:
    content: list[dict[str, Any]] = []
    for kind, path in parts:
        if kind == "audio":
            content.append({"type": "audio_url", "audio_url": {"url": path.resolve().as_uri()}})
        elif kind == "video":
            content.append({"type": "video_url", "video_url": {"url": path.resolve().as_uri()}})
        else:
            raise ValueError(f"unknown media kind {kind!r}")
    if prompt:
        content.append({"type": "text", "text": prompt})
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        extra_body={"speechlmm_mode": mode} if mode else None,
    )
    return completion.choices[0].message.content or ""


def _expect_400(label: str, call) -> bool:
    """A request the gateway must refuse instead of answering with a weaker mode."""
    try:
        call()
    except BadRequestError:
        print(f"  {label}: 400 as expected")
        return True
    except APIStatusError as exc:
        print(f"  {label}: expected 400, got {exc.status_code}", file=sys.stderr)
        return False
    print(f"  {label}: expected 400, got a completion", file=sys.stderr)
    return False


def _run_n(
    client: OpenAI, model: str, n: int, concurrency: int, audio: Optional[Path] = None
) -> list[str]:
    """Run n identical requests. With audio, this is what batches the feature tensors."""
    if audio is None:
        call = lambda: _chat(client, model, PONG)  # noqa: E731
    else:
        call = lambda: _chat(  # noqa: E731
            client, model, "", mode="transcribe", audio=audio, max_tokens=64
        )
    texts: list[str] = [""] * n
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(call): i for i in range(n)}
        for future in as_completed(futures):
            texts[futures[future]] = future.result()
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Gateway root, not /v1.")
    parser.add_argument("--model", default="speechlmm_v2_xl")
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument(
        "--av-video",
        type=Path,
        default=None,
        help="General (non-mouth-crop) video for avspeakd. Falls back to --video.",
    )
    parser.add_argument(
        "--av-audio",
        type=Path,
        default=None,
        help="Audio paired with --av-video. Falls back to --audio.",
    )
    parser.add_argument(
        "--av-audio-2",
        type=Path,
        default=None,
        help="Second audio clip for the multi-clip gate. Defaults to duplicating --av-audio.",
    )
    parser.add_argument(
        "--av-video-2",
        type=Path,
        default=None,
        help="Second video clip for the multi-clip gate. Defaults to duplicating --av-video.",
    )
    parser.add_argument(
        "--vsr-reference",
        type=Path,
        default=None,
        help="Text file with the expected lipread output for the same --video.",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--invariance-n", type=int, default=8)
    args = parser.parse_args()

    print("=== health ===")
    _health(args.url)
    print("ready")

    client = OpenAI(
        base_url=f"{args.url.rstrip('/')}/v1", api_key="not-used", timeout=args.timeout
    )

    print("=== model list ===")
    served = [model.id for model in client.models.list().data]
    print(json.dumps(served))
    if args.model not in served:
        print(f"{args.model} is not loaded", file=sys.stderr)
        return 1

    print("=== chat smoke ===")
    print(_chat(client, args.model, PONG))

    print("=== rejections ===")
    checks = [
        _expect_400(
            "audio part with no url",
            lambda: client.chat.completions.create(
                model=args.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe."},
                            {"type": "audio_url", "audio_url": {}},
                        ],
                    }
                ],
                max_tokens=16,
            ),
        ),
        _expect_400(
            "assistant role",
            lambda: client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "user", "content": PONG},
                    {"role": "assistant", "content": "pong"},
                ],
                max_tokens=16,
            ),
        ),
        _expect_400(
            "stream=true",
            lambda: client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": PONG}],
                max_tokens=16,
                stream=True,
            ),
        ),
    ]
    if not all(checks):
        print("REJECTIONS_FAILED", file=sys.stderr)
        return 1
    print("REJECTIONS_OK")

    print("=== client system prompt ===")
    # The client owns the prompts: a system message must reach the model, not a 400.
    supplied = _chat(
        client,
        args.model,
        PONG,
        system="You are a terse assistant. Answer with a single lowercase word.",
    )
    print(supplied)
    if not supplied.strip():
        print("SYSTEM_PASSTHROUGH_FAILED empty completion", file=sys.stderr)
        return 1
    print("SYSTEM_PASSTHROUGH_OK")

    if args.audio is not None:
        print(f"=== asr {args.audio} ===")
        print(_chat(client, args.model, "", mode="transcribe", audio=args.audio, max_tokens=128))

    if args.video is not None:
        print(f"=== lipread {args.video} ===")
        vsr = _chat(
            client,
            args.model,
            "Transcribe the content of the video",
            mode="lipread",
            video=args.video,
            max_tokens=128,
        )
        print(vsr)
        if args.vsr_reference is not None:
            reference = args.vsr_reference.read_text(encoding="utf-8").strip()
            if vsr.strip() != reference:
                print("VSR_PARITY_FAILED", file=sys.stderr)
                print(f"gateway={vsr!r}", file=sys.stderr)
                print(f"reference={reference!r}", file=sys.stderr)
                return 1
            print("VSR_PARITY_OK")

    if args.audio is not None and (args.av_video is not None or args.video is not None):
        av_audio = args.av_audio or args.audio
        av_video = args.av_video or args.video
        print(f"=== avspeakd audio={av_audio} video={av_video} ===")
        print(
            _chat(
                client,
                args.model,
                "Identify who is speaking and transcribe.",
                mode="avspeakd",
                audio=av_audio,
                video=av_video,
                max_tokens=256,
            )
        )

        av_audio_2 = args.av_audio_2 or av_audio
        av_video_2 = args.av_video_2 or av_video
        print(
            f"=== multi-clip "
            f"audio={av_audio} video={av_video} "
            f"audio_2={av_audio_2} video_2={av_video_2} ==="
        )
        multi = _chat_parts(
            client,
            args.model,
            "Identify who is speaking and transcribe.",
            [
                ("audio", av_audio),
                ("video", av_video),
                ("audio", av_audio_2),
                ("video", av_video_2),
            ],
            mode="avspeakd",
            max_tokens=256,
        )
        print(multi)
        if not multi.strip():
            print("MULTI_CLIP_FAILED empty completion", file=sys.stderr)
            return 1
        print("MULTI_CLIP_OK")

    n = args.invariance_n
    # Text covers the input_ids/attention_mask collate path; audio is the only thing
    # that batches input_features and feature_attention_mask.
    variants: list[tuple[str, Optional[Path]]] = [("text", None)]
    if args.audio is not None:
        variants.append(("audio", args.audio))
    for label, media in variants:
        print(f"=== batch invariance {label} n={n} (sequential vs concurrent) ===")
        sequential = _run_n(client, args.model, n, 1, media)
        concurrent = _run_n(client, args.model, n, n, media)
        if sequential != concurrent:
            print(f"BATCH_INVARIANCE_FAILED {label}", file=sys.stderr)
            print(json.dumps({"sequential": sequential, "concurrent": concurrent}, indent=2))
            return 1
        print(f"BATCH_INVARIANCE_OK {label}")

    print("=== throughput ===")
    for concurrency in (1, 4, n):
        started = time.perf_counter()
        _run_n(client, args.model, n, concurrency)
        elapsed = time.perf_counter() - started
        print(
            f"concurrency={concurrency} n={n} elapsed_sec={elapsed:.2f} "
            f"samples_per_sec={n / elapsed:.3f}"
        )

    print("VERIFY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
