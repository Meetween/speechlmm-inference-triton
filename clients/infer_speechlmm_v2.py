#!/usr/bin/env python3
"""Raw-tensor debug probe for the speechlmm_v2 Triton backend.

Bypasses the OpenAI gateway and posts Triton's JSON infer payload directly to
POST /v2/models/<name>/infer on 8100, which is how you tell a gateway bug from
a model bug. Normal clients should use the gateway on 8000.

Examples:
  python clients/infer_speechlmm_v2.py --url http://127.0.0.1:8100 --mode chat --prompt "pong"
  python clients/infer_speechlmm_v2.py --mode transcribe --audio clip.wav --prompt "Transcribe the audio."
  python clients/infer_speechlmm_v2.py --mode lipread --video mouth_crop.mp4 --prompt "Transcribe the silent speech."
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triton_tensors import (  # noqa: E402
    AUDIO_REQUIRED_MODES,
    VIDEO_REQUIRED_MODES,
    TritonInferError,
    build_inputs,
    infer,
)

MODES = [
    "chat",
    "transcribe",
    "caption",
    "summary",
    "text_summary",
    "lipread",
    "avspeakd",
    "mt",
    "st",
]


def _b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8100")
    parser.add_argument("--model", default="speechlmm_v2_xl")
    parser.add_argument("--mode", default="chat", choices=MODES)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--language", default=None, help="Content language; target language for mt/st.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    if args.mode in AUDIO_REQUIRED_MODES and args.audio is None:
        raise SystemExit(f"--audio is required for mode {args.mode}")
    if args.mode in VIDEO_REQUIRED_MODES and args.video is None:
        raise SystemExit(f"--video is required for mode {args.mode}")

    inputs = build_inputs(
        mode=args.mode,
        prompt=args.prompt,
        audio_b64=_b64_file(args.audio) if args.audio is not None else None,
        video_b64=_b64_file(args.video) if args.video is not None else None,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        language=args.language,
    )
    started = time.perf_counter()
    try:
        text = infer(args.url, args.model, inputs, args.timeout)
    except TritonInferError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"latency_sec={time.perf_counter() - started:.2f}", file=sys.stderr)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
