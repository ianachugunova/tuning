"""Generate pitch-detection fixtures and the Python reference answers.

The browser and live_v1/human.py must hear the same note in the same sound,
or a singer's phone and the room microphone will disagree about what was sung.
This writes synthetic signals plus human.py's own answer for each, so
test_pitch.mjs can check the JavaScript port against the real thing.

    python3 make_fixtures.py && node test_pitch.mjs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Relative to this file, not to the name of the folder the project sits in.
# A clone is named after the repository, so importing through the directory
# name works on the author's laptop and nowhere else — including CI.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_v1.human import (  # noqa: E402
    SAMPLE_RATE,
    _estimate_pitch,
    _hz_to_note,
    _stability_label,
)

SR = SAMPLE_RATE
DUR = 1.0
rng = np.random.default_rng(20260824)


def t() -> np.ndarray:
    return np.arange(int(SR * DUR), dtype=np.float32) / SR


def sine(hz: float, amp: float = 0.3) -> np.ndarray:
    return (amp * np.sin(2 * np.pi * hz * t())).astype(np.float32)


def voice(hz: float, amp: float = 0.3) -> np.ndarray:
    """A sung vowel is not a sine — give it a harmonic stack."""
    x = np.zeros(int(SR * DUR), dtype=np.float32)
    for k, weight in enumerate([1.0, 0.5, 0.33, 0.25, 0.16], start=1):
        x += weight * np.sin(2 * np.pi * hz * k * t())
    return (amp * x / np.max(np.abs(x))).astype(np.float32)


def vibrato(hz: float, depth: float = 0.03, rate: float = 5.5) -> np.ndarray:
    phase = 2 * np.pi * hz * (t() + depth / (2 * np.pi * rate) * np.sin(2 * np.pi * rate * t()))
    return (0.3 * np.sin(phase)).astype(np.float32)


def glide(f0: float, f1: float) -> np.ndarray:
    k = (f1 - f0) / DUR
    phase = 2 * np.pi * (f0 * t() + 0.5 * k * t() ** 2)
    return (0.3 * np.sin(phase)).astype(np.float32)


CASES: list[tuple[str, np.ndarray]] = [
    ("sine_110",        sine(110.0)),
    ("sine_220",        sine(220.0)),
    ("sine_305",        sine(305.0)),
    ("sine_440",        sine(440.0)),
    ("sine_523",        sine(523.25)),
    ("sine_880",        sine(880.0)),
    ("voice_147",       voice(146.83)),   # D3, a low sung tone
    ("voice_196",       voice(196.0)),    # G3
    ("voice_392",       voice(392.0)),    # G4
    ("vibrato_220",     vibrato(220.0)),
    ("glide_200_300",   glide(200.0, 300.0)),
    ("noise",           (0.3 * rng.standard_normal(int(SR * DUR))).astype(np.float32)),
    ("silence",         np.zeros(int(SR * DUR), dtype=np.float32)),
    ("quiet_220",       sine(220.0, amp=0.004)),   # under the frame silence gate
    ("voice_plus_noise", (voice(220.0) + 0.02 * rng.standard_normal(int(SR * DUR))).astype(np.float32)),
]


def main() -> None:
    out = []
    for name, audio in CASES:
        med, std, voiced = _estimate_pitch(audio, SR)
        out.append({
            "name": name,
            "sample_rate": SR,
            "audio": [float(v) for v in audio],
            "python": {
                "median_hz": med,
                "std_hz": std,
                "voiced_fraction": voiced,
                "note": _hz_to_note(med),
                "stability": _stability_label(std, med),
            },
        })
        print(f"{name:18} {med:8.2f} Hz  std {std:6.2f}  voiced {voiced:.2f}  "
              f"{_hz_to_note(med)} [{_stability_label(std, med)}]")

    path = Path(__file__).with_name("fixtures.json")
    path.write_text(json.dumps(out))
    print(f"\nwrote {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
