#!/usr/bin/env python3
"""Call the SpeechLMM gateway with the official OpenAI SDK.

  python examples/client/hello_speechlmm.py --model speechlmm_v2_l
  python examples/client/hello_speechlmm.py --audio clip.wav
  python examples/client/hello_speechlmm.py --video mouth_crop.mp4 --mode lipread
  python examples/client/hello_speechlmm.py --prompt "Give this audio: " --audio clip.wav \
      --suffix " please summarize it."

Content is sent in order: --prompt, the audio, the video, then --suffix. Spaces at
the edges of --prompt and --suffix are kept, so they sit right next to the clips.

Media are passed as file:// URLs, so the gateway reads them straight off the
shared filesystem instead of base64 round-tripping them through the request.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openai import OpenAI


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="speechlmm_v2_xl")
    parser.add_argument("--prompt", default="Reply with the single word: pong")
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--suffix", default=None, help="Text sent after the media.")
    parser.add_argument("--mode", default=None, help="Override mode derivation, e.g. transcribe.")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    content: list[dict] = [{"type": "text", "text": args.prompt}]
    if args.audio is not None:
        content.append(
            {"type": "audio_url", "audio_url": {"url": args.audio.resolve().as_uri()}}
        )
    if args.video is not None:
        content.append(
            {"type": "video_url", "video_url": {"url": args.video.resolve().as_uri()}}
        )
    if args.suffix:
        content.append({"type": "text", "text": args.suffix})

    client = OpenAI(base_url=f"{args.url.rstrip('/')}/v1", api_key="not-used")
    completion = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": content}],
        max_tokens=args.max_tokens,
        extra_body={"speechlmm_mode": args.mode} if args.mode else None,
    )
    print(completion.choices[0].message.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
