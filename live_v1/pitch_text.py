"""Reading pitch out of what a model wrote.

A model contributes a line of text — "D4, a warm hum", "440Hz then ahh", "G#3".
Somewhere in there is a tone. These functions find it, decide which part of the
line is a pitch label and which part is prose, and fold anything inaudible into
the range a speaker can reproduce.

This lived inside `sound.py` and could only be imported on a Mac, because that
file also shells out to `say` and `afplay`. The portal renders the same voices
on Linux, from the same rules, so the rules moved here. Pure stdlib, no audio,
no platform: import it anywhere.
"""

from __future__ import annotations

import re

# Octave-folding bounds. A model that names 0.5 Hz or 40 kHz has still named a
# pitch class; folding keeps its identity and makes it playable.
AUDIBLE_LO = 60.0
AUDIBLE_HI = 4000.0

# Numbers with an optional Hz/kHz/MHz/GHz unit. The second branch catches bare
# numbers while refusing the ones that are plainly something else — seconds,
# milliseconds, bpm, words per minute, percentages.
#
# `(?![.,]\d)` is what keeps a rejected number from being accepted in pieces.
# Without it "12.5s" is read as a duration, refused, and then retried as the
# "12" in front of it — which passes, because what follows *that* is a dot.
# Every duration a voice mentions used to become a tone that way, and the field
# summary a voice is given to read is full of them: gpt-2 echoed the line
# "[-31.2s] human_ynmq9xv8 (human):" back into the field and was recorded as
# having sung 31 Hz.
_FREQ_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(ghz|mhz|khz|hz)\b"
    r"|(?<!\w)(\d+(?:[.,]\d+)?)(?![.,]\d)(?!\s*(?:s\b|ms\b|bpm\b|wpm\b|%|\w))",
    re.IGNORECASE,
)
_UNIT_MULT = {"hz": 1, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}

_NOTE_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NOTE_RE = re.compile(r"\b([A-G])(#{1,2}|b{1,2})?\s*([0-9])\b")


def note_to_freq(note: str, accidental: str, octave: int) -> float:
    """Note name → Hz, equal temperament, A4 = 440 Hz."""
    semitone = _NOTE_SEMITONES[note.upper()]
    semitone += (accidental or "").count("#") - (accidental or "").count("b")
    midi = (octave + 1) * 12 + semitone          # C4 = 60, A4 = 69
    return 440.0 * (2 ** ((midi - 69) / 12))


def extract_notes(text: str) -> list[float]:
    """Frequencies of every note name (C4, G#3, Bb5 …) in the text."""
    return [
        note_to_freq(m.group(1), m.group(2) or "", int(m.group(3)))
        for m in _NOTE_RE.finditer(text)
    ]


def extract_frequencies(text: str) -> list[float]:
    """Every frequency value in the text, as Hz."""
    freqs: list[float] = []
    for m in _FREQ_RE.finditer(text):
        if m.group(1):
            freqs.append(float(m.group(1).replace(",", ".")) * _UNIT_MULT[m.group(2).lower()])
        elif m.group(3):
            val = float(m.group(3).replace(",", "."))
            if 10 <= val <= 20000:               # plausible as Hz already
                freqs.append(val)
    return freqs


def fold_to_audible(freq: float) -> float:
    """Halve or double until the pitch lands in the audible range.

    Pitch class is preserved: 0.5 Hz becomes the same 'note' as 512 Hz, ten
    octaves up.
    """
    if freq <= 0:
        return AUDIBLE_LO
    while freq < AUDIBLE_LO:
        freq *= 2.0
    while freq > AUDIBLE_HI:
        freq /= 2.0
    return freq


def extract_all_pitches(text: str) -> list[tuple[float, str]]:
    """(hz, label) for every pitch in the text — note names first, then numbers."""
    results: list[tuple[float, str]] = []
    for m in _NOTE_RE.finditer(text):
        acc = m.group(2) or ""
        hz = note_to_freq(m.group(1), acc, int(m.group(3)))
        results.append((hz, f"{m.group(1)}{acc}{m.group(3)}"))
    for m in _FREQ_RE.finditer(text):
        if m.group(1):
            val = float(m.group(1).replace(",", "."))
            results.append((fold_to_audible(val * _UNIT_MULT[m.group(2).lower()]),
                            f"{val}{m.group(2)}"))
        elif m.group(3):
            val = float(m.group(3).replace(",", "."))
            if 10 <= val <= 20000:
                results.append((val, f"{val}Hz"))
    return results


_PITCH_ONLY_RE = re.compile(
    r"^[A-G][#b]?\d(\s*/\s*[A-G][#b]?\d)*$"      # D4, F#3, Bb5/D4
    r"|^\d+(?:[.,]\d+)?\s*(?:hz|khz)?$",         # 440, 440Hz
    re.IGNORECASE,
)


def is_pitch_only(text: str) -> bool:
    """True when the line is nothing but a pitch — no prose to speak."""
    return bool(_PITCH_ONLY_RE.match(text.strip()))


# One leading pitch token, plus whatever punctuation separates it from the words.
_LEADING_PITCH_RE = re.compile(
    r"^\s*(?:"
    r"[A-G](?:#{1,2}|b{1,2})?\d"
    r"|\d+(?:[.,]\d+)?\s*(?:ghz|mhz|khz|hz)?"
    r")",
    re.IGNORECASE,
)
_LEADING_SEP_RE = re.compile(
    r"^[\s,;:.!/|()\[\]\-–—~>]*(?:then\b|~>)?[\s,;:.!/|()\[\]\-–—~>]*",
    re.IGNORECASE,
)


def strip_leading_pitch(text: str) -> tuple[str, str]:
    """Split a leading pitch label off the front: ``("D4", "mmm warm")``.

    Models open a line with the tone they intend. Spoken aloud, a short breath
    is spent reading the number and the actual sound never happens. The label
    becomes a tone; only what follows is spoken.

    Returns ``("", text)`` when the line does not begin with a pitch. A leading
    number that is not a pitch ("3 slow waves") is left alone to be spoken.
    """
    rest = text.lstrip()
    labels: list[str] = []
    while True:
        m = _LEADING_PITCH_RE.match(rest)
        if not m:
            break
        token = m.group(0).strip()
        if not extract_all_pitches(token):       # not really a tone — leave it spoken
            break
        labels.append(token)
        rest = rest[m.end():]
        sep = _LEADING_SEP_RE.match(rest)
        if sep and sep.end() > 0:
            rest = rest[sep.end():]
    return " ".join(labels), rest
