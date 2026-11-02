#!/usr/bin/env python3
"""
live_v1/test_human.py — Standalone mic check.

Run this BEFORE a session to verify:
  1. Microphone access is granted (macOS will ask once).
  2. Whisper loads.
  3. VAD detects when you sing/speak.
  4. Whisper transcribes correctly.

Usage:
    python3 -m live_v1.test_human
    python3 -m live_v1.test_human --whisper-model tiny      # faster
    python3 -m live_v1.test_human --duration 60             # listen 60s

Sing one or two tones. Press Ctrl+C to stop.
The transcript will be printed in real time.

If macOS does not prompt for microphone access automatically, go to:
  System Settings → Privacy & Security → Microphone
and enable access for "Terminal" (or whichever shell you are using).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass


@dataclass
class _FakeField:
    """A drop-in for orchestrator.Field that just prints events."""
    events: list

    def append(self, e):
        self.events.append(e)


def main() -> None:
    p = argparse.ArgumentParser(description="Microphone + Whisper sanity check")
    p.add_argument("--whisper-model", default="base", help="tiny / base / small / medium / large-v3")
    p.add_argument("--duration", type=float, default=30.0, help="How long to listen (s)")
    args = p.parse_args()

    from live_v1.human import HumanVoice, HUMAN_DEPS_OK, HUMAN_DEPS_ERROR
    if not HUMAN_DEPS_OK:
        print(f"Missing dependencies: {HUMAN_DEPS_ERROR}")
        print("Install: pip install sounddevice openai-whisper numpy")
        sys.exit(1)

    # Stub orchestrator.Event class for the field.append call inside HumanVoice
    import live_v1.orchestrator  # noqa: F401  (ensures Event is importable)

    fake = _FakeField(events=[])
    hv = HumanVoice(
        voice_id="human_test",
        field=fake,
        whisper_model_name=args.whisper_model,
        verbose=True,
    )
    hv.start()
    print()
    print(f"Listening for {args.duration:.0f} seconds. Sing or speak. Ctrl+C to stop early.")
    print()
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\n(interrupted)")
    hv.stop()

    print()
    print(f"Captured {len(fake.events)} utterance(s):")
    print()
    print(f"  {'#':>2}  {'dur':>5}  content")
    print(f"  {'-'*2}  {'-'*5}  {'-'*60}")
    for i, e in enumerate(fake.events, 1):
        print(f"  {i:>2}  {e.breath_chosen_s:>4.1f}s  {e.content}")
    print()
    print("Each line includes (when sung): nearest note, frequency in Hz,")
    print("[steadiness], and Whisper transcription of words.")


if __name__ == "__main__":
    main()
