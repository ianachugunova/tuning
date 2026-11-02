#!/usr/bin/env python3
"""
live_v1/test_channels.py — quick channel-by-channel sanity check.

For each channel listed in concert.py's VOICE_CHANNELS, play:
  1. A 2-second sine wave at a distinct pitch
  2. A spoken "channel N" label (TTS via macOS `say`)

Hear it from one speaker at a time — confirms that USB output N really
reaches physical speaker N through the M32R matrix.

USAGE (default, sweeps all configured channels in order):
    python3 -m live_v1.test_channels

  --device  "M32R"      override OUTPUT_DEVICE (use --list-devices in concert.py
                        to find the exact name macOS sees)
  --channels 1,2,3      override channel list (1-based)
  --duration 2.0        seconds per channel (default 2)
  --pitch 440           base pitch in Hz (each next channel goes up a minor third)
  --loop                repeat forever (Ctrl-C to stop)
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Tuning_Meditation_LLMs.live_v1 import concert as concert_mod      # type: ignore
else:
    from . import concert as concert_mod


def _sine(freq: float, duration_s: float, sr: int = 48000) -> np.ndarray:
    """Mono sine with short fade-in/out to avoid clicks."""
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    audio = 0.4 * np.sin(2 * np.pi * freq * t).astype("float32")
    fade_n = min(int(0.04 * sr), n // 8)
    if fade_n > 0:
        ramp = np.linspace(0.0, 1.0, fade_n, dtype="float32")
        audio[:fade_n] *= ramp
        audio[-fade_n:] *= ramp[::-1]
    return audio


def _tts_to_wav(text: str, voice: str = "Samantha") -> tuple[np.ndarray, int]:
    """Synthesise `text` via macOS `say`, return (mono float32 audio, sr)."""
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
        aiff_path = tmp.name
    try:
        subprocess.run(
            ["say", "-v", voice, "-o", aiff_path, text],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wav_path = aiff_path.replace(".aiff", ".wav")
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@48000", aiff_path, wav_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
        audio_i16 = np.frombuffer(raw, dtype="<i2")
        if wf.getnchannels() == 2:
            audio_i16 = audio_i16.reshape(-1, 2).mean(axis=1)
        audio = (audio_i16.astype("float32") / 32768.0) * 0.7
        Path(wav_path).unlink(missing_ok=True)
        return audio.astype("float32"), sr
    finally:
        Path(aiff_path).unlink(missing_ok=True)


def _play_on(channel: int, audio: np.ndarray, sr: int, device: object) -> None:
    """Play mono `audio` on ONE 1-based channel of `device`."""
    import sounddevice as sd
    sd.play(audio, sr, device=device, mapping=[channel])
    sd.wait()


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-channel routing sanity check")
    ap.add_argument("--device", default=None,
                    help='Override OUTPUT_DEVICE (e.g. "M32R" or numeric index).')
    ap.add_argument("--channels", default=None,
                    help="Comma-separated 1-based channels to test "
                         "(default: read VOICE_CHANNELS from concert.py).")
    ap.add_argument("--duration", type=float, default=2.0,
                    help="Sine duration per channel in seconds (default 2.0).")
    ap.add_argument("--pitch", type=float, default=440.0,
                    help="Base sine pitch in Hz (default A4=440).")
    ap.add_argument("--loop", action="store_true",
                    help="Repeat the sweep forever (Ctrl-C to stop).")
    ap.add_argument("--no-speak", action="store_true",
                    help="Skip the spoken 'channel N' label, sine only.")
    args = ap.parse_args()

    device = args.device if args.device is not None else concert_mod.OUTPUT_DEVICE
    if args.channels:
        channels = [int(c.strip()) for c in args.channels.split(",") if c.strip()]
        labels = [(f"voice_test_{c:02d}", c) for c in channels]
    else:
        labels = list(concert_mod.VOICE_CHANNELS.items())
        labels.sort(key=lambda kv: kv[1])
        if not labels:
            print("[test_channels] VOICE_CHANNELS is empty in concert.py — "
                  "either fill it in or pass --channels 1,2,3,...")
            sys.exit(1)

    print(f"[test_channels] device={device!r}, "
          f"channels={[c for _, c in labels]}, duration={args.duration}s")
    print("[test_channels] press Ctrl-C to stop.\n")

    # Each channel gets a distinct pitch so the listener can identify by ear
    # if two speakers happen to play at once (cross-bleed in the matrix).
    semitone_steps = [0, 3, 5, 8, 10, 12, 15, 17, 20, 22, 24, 27]

    pass_n = 0
    try:
        while True:
            pass_n += 1
            print(f"--- pass {pass_n} ---")
            for idx, (vid, ch) in enumerate(labels):
                step = semitone_steps[idx % len(semitone_steps)]
                freq = args.pitch * (2 ** (step / 12.0))
                note_name = _semi_to_note(step)
                print(f"  ch {ch:2d}  ({vid:<14}) → {freq:.1f} Hz [{note_name}]")
                sine = _sine(freq, args.duration)
                try:
                    _play_on(ch, sine, 48000, device)
                except Exception as exc:
                    print(f"    [error] {exc}")
                    continue
                if not args.no_speak:
                    try:
                        tts_audio, tts_sr = _tts_to_wav(f"channel {ch}")
                        _play_on(ch, tts_audio, tts_sr, device)
                    except Exception as exc:
                        print(f"    [tts error] {exc}")
                time.sleep(0.4)
            if not args.loop:
                break
            print()
    except KeyboardInterrupt:
        print("\n[test_channels] stopped.")


def _semi_to_note(step: int) -> str:
    """Helper to print a rough note name for log readability."""
    names = ["A4", "C5", "D5", "F5", "G5", "A5", "C6", "D6", "F6", "G6", "A6", "C7", "D7"]
    return names[step] if step < len(names) else f"+{step}st"


if __name__ == "__main__":
    main()
