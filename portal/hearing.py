"""What a voice with ears is given, when reading is not an option.

A deaf voice is handed `summarise_field` — the last few breaths as a paragraph.
A voice with ears has to be handed the sound, and the sound does not exist
anywhere as one thing: the machines' breaths are wav files on the volume, and
the people are numbers and only numbers, because no recording of a person is
kept or sent at any point in this work. This module makes the one recording out
of both.

Two things it does that a paragraph cannot.

**It sounds people from their numbers.** A phone sends a pitch, a duration and
the line the pitch walked; that is the whole of what `/breath` accepts and the
promise it is written around. It is also enough to build a tone, and every
listener's browser already builds one — `soundHuman` in `pitch.html`. The same
tone is built here, so a machine with ears can hear the people in the room
without any voice having left the phone that sang. Without this the ears would
hear the other machines and be deaf to the people, which is the exact inverse of
why they are in the piece.

**It keeps time.** A paragraph is a list; a field is a room. Two voices that
overlapped are laid over one another here, at the offsets they actually had, and
a voice that entered after a long silence is preceded by that silence. An ear
asked how many voices are holding a note can only answer from this — and in the
one comparison we have, counting was the thing the best ear could do and naming
was the thing none of them could.

Numpy and `score.Note`, nothing else: no torch, no soundfile, no volume. It runs
in CI, which is the only place the arithmetic here is ever checked.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np

from score import Note

SR = 24_000                  # Kokoro's rate, Mimi's rate, and the field's

# Loud enough to be the subject of the recording, quiet enough that two of them
# together do not clip. The choir's own wavs arrive at their own level and are
# not touched; only the tones built here have a level to choose.
HUMAN_PEAK = 0.30

# Eased at both ends. A square edge is the loudest thing in any field and this
# field is a meditation — and an encoder fed a click hears the click.
EDGE_S = 0.08

# Bounds on a breath rebuilt from numbers. A duration is a measurement from
# somebody's phone and can arrive absurd; a tone of an hour would be a denial of
# service against a listening window, and one of a millisecond is a click.
MIN_TONE_S = 0.05
MAX_TONE_S = 30.0


def human_tone(hz: float, dur_s: float, contour: Sequence[float] | None = None,
               hop_s: float = 0.0, sr: int = SR) -> np.ndarray:
    """A person, sounded from their numbers.

    The contour is the point. A median pitch and a length make a tuning fork;
    what tells a listener — machine or otherwise — that a voice is a voice is
    that the pitch moves while it sounds. `hop_s` is how far apart the contour's
    samples were taken, so a trace that covers less than the whole breath holds
    its last value rather than being stretched over the gap.
    """
    hz = float(hz or 0.0)
    if hz <= 0:
        return np.zeros(0, dtype="float32")
    dur = min(MAX_TONE_S, max(MIN_TONE_S, float(dur_s or 0.0)))
    n = int(dur * sr)
    if n <= 0:
        return np.zeros(0, dtype="float32")

    pts = [float(p) for p in (contour or []) if p and float(p) > 0]
    t = np.arange(n, dtype="float64") / sr
    if len(pts) > 1:
        span = len(pts) * hop_s if hop_s > 0 else dur
        at = np.linspace(0.0, span, len(pts))
        # `interp` holds the end values outside the range, which is exactly the
        # wanted behaviour for a trace shorter than the breath it describes.
        freq = np.interp(t, at, pts)
    else:
        freq = np.full(n, hz, dtype="float64")

    # Integrated rather than multiplied: `sin(2π f t)` with a changing f jumps
    # phase every sample and buzzes. The glide has to accumulate.
    wave = np.sin(2 * np.pi * np.cumsum(freq) / sr)

    edge = min(int(EDGE_S * sr), n // 2)
    if edge > 0:
        ramp = np.linspace(0.0, 1.0, edge)
        wave[:edge] *= ramp
        wave[-edge:] *= ramp[::-1]
    return (HUMAN_PEAK * wave).astype("float32")


def tone_of(note: Note, load_wav: Callable[[str], np.ndarray] | None = None,
            sr: int = SR) -> np.ndarray:
    """One breath as sound: the wav that was kept, or the tone its numbers imply.

    A machine's breath was rendered once and the recording is on the volume;
    playing it again is both cheaper and more honest than re-synthesising it.
    A person's breath has no recording anywhere and never will, so it is built.
    """
    if note.audio and load_wav is not None:
        try:
            wav = np.asarray(load_wav(note.audio), dtype="float32")
            if wav.ndim > 1:                       # a stereo file, mixed down
                wav = wav.mean(axis=1).astype("float32")
            if wav.size:
                return wav
        except Exception:
            # A missing or unreadable wav is a voice that is not heard this
            # round, not a round that fails. This runs unattended for a month.
            pass
    return human_tone(note.pitch_hz or 0.0, note.sound_s,
                      note.contour_hz, note.contour_hop_s, sr)


def field_audio(notes: Iterable[Note], window_s: float, end_ts: float,
                load_wav: Callable[[str], np.ndarray] | None = None,
                sr: int = SR) -> np.ndarray:
    """The last `window_s` of the field as one recording, ending at `end_ts`.

    Everything is laid at the offset its onset gives it, so what overlapped
    overlaps and what followed follows. A breath that began before the window
    is included from wherever the window opens — a voice already holding a note
    when you walk in is a voice you can hear, and cutting it would tell the ear
    that the room was empty until somebody started.

    Silence is content here. The gaps are the piece, and an ear given a
    concatenation of breaths with the rests squeezed out would be answering a
    question about a different room.
    """
    n = int(max(0.1, float(window_s)) * sr)
    out = np.zeros(n, dtype="float32")
    start = float(end_ts) - float(window_s)

    for note in notes:
        if note.kind != "tone":
            continue
        piece = tone_of(note, load_wav, sr)
        if piece.size == 0:
            continue
        at = int(round((float(note.t) - start) * sr))
        if at >= n:
            continue
        if at < 0:                     # began before the window opened
            piece = piece[-at:]
            at = 0
            if piece.size == 0:
                continue
        end = min(n, at + piece.size)
        out[at:end] += piece[:end - at]

    peak = float(np.abs(out).max()) if out.size else 0.0
    if peak > 1.0:
        # Two voices at once must not clip into distortion — an encoder hears
        # clipping as noise and would be told there was noise in the room.
        out /= peak
    return out
