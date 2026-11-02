"""Is the tone in the room a throat or a pitch? Measured, not guessed.

    python music_probe/tone_or_throat.py

Written to answer whether turning `THROAT` off had actually reached the card or
whether a warm container was still singing the old code, which is not a thing the
logs say and not a thing worth deciding by ear over a phone speaker. Spectral
flatness separates them by three orders of magnitude: the note on `THROAT` in
`choir_app.py` records 0.0012 for a bare oscillator against 0.24 to 0.77 for a
vowel held in a voice's own throat.

One limit, worth knowing before reading the column: this measures the whole
breath, tone and spoken words together, and speech is broadband. A voice reading
an Italian sentence therefore looks like a throat whatever its tone is made of.
Read the voices that sing a bare vowel — that is where the tone is the file.
"""

import io
import json
import sys
import time
import urllib.request

import numpy as np
import soundfile as sf

PORTAL = "https://tuning-grayarea--tuning-portal-web.modal.run"


def flatness(x: np.ndarray, sr: int) -> float:
    """Geometric mean of the spectrum over its arithmetic mean.

    One number for "is this one frequency or a body of them": a pure tone puts
    everything in one bin and drives the geometric mean to nothing, while the
    formants of a held vowel spread energy and pull the two means together.
    """
    if x.ndim > 1:
        x = x.mean(axis=1)
    # The middle of the breath, away from the fades at either end.
    mid = x[len(x) // 4: 3 * len(x) // 4] if len(x) > 8 else x
    spec = np.abs(np.fft.rfft(mid * np.hanning(len(mid)))) + 1e-12
    return float(np.exp(np.mean(np.log(spec))) / np.mean(spec))


def main() -> None:
    field = json.load(urllib.request.urlopen(f"{PORTAL}/field"))
    now = time.time()
    tones = [e for e in field["recent"] if e.get("audio") and e.get("hz")]
    if not tones:
        sys.exit("no rendered breaths in the field right now")

    print(f"{'age':>7}  {'voice':16s} {'Hz':>7}  {'flatness':>9}  what it is")
    for e in tones[-8:]:
        wav = urllib.request.urlopen(f"{PORTAL}/audio/{e['audio']}").read()
        x, sr = sf.read(io.BytesIO(wav))
        f = flatness(x, sr)
        # Wide berth around both measured populations rather than a midpoint:
        # anything in between is worth looking at by ear instead of being
        # labelled by a threshold that was never measured.
        what = ("a pitch" if f < 0.01 else
                "a throat" if f > 0.1 else "neither, look at it")
        print(f"{int(now - e['ts']):6d}s  {e['name']:16s} {e['hz']:7.1f}  "
              f"{f:9.4f}  {what}")


if __name__ == "__main__":
    main()
