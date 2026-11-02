"""The exam's arithmetic, kept free of torch so it can be tested without a card.

The exam asks one question: did a lesson change the voice, or is the difference
between Tuesday and Wednesday only a different room and a different roll of the
dice? Two measurements answer it, and both hold the room still.

**Recognition.** A model assigns a probability to every next word. Given the exact
field a voice heard and the exact breath it then sang, we read how probable that
breath was to the voice — with the lesson and without it. The number is shown as
*choices per word* (perplexity): how many words the voice was, in effect, choosing
between at each step. A voice that has learned its own past answers hesitates
less on them: 14 choices per word before school, 3 after. No sampling is involved,
so the dice play no part.

**Fixed tickets.** The same frozen field, many breaths each, with and without the
lesson. Here the dice are the point: one answer proves nothing, thirty answers make
a distribution, and two distributions can be compared — how varied they are, how
often one line dominates, how often the voice turns to a person, how much of what
it says is something it already said in the log.

**Hearing.** Since 15 October a voice that tunes is asked to name the note it is
taking (`choir_app.name_your_pitch`), and every TUNE line records both the note
it named and the note the part it answered was holding. The distance between
the two, in cents and with octaves set aside — the piece's own rule is that an
octave is the same pitch — is the one number in this project that is about
listening rather than speaking. Fifty cents is a quarter-tone, the width within
which two singers are heard as on the same note.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import learning

try:
    from live_v1.pitch_text import extract_all_pitches
except ImportError:                                   # run from inside portal/
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from live_v1.pitch_text import extract_all_pitches

HIT_CENTS = 50.0

# The sentence the room adds to its prompt (`choir_app.name_your_pitch`), copied
# because the exam's image ships neither the choir nor modal. A test in
# test_choir.py keeps the two the same.
NAME_YOUR_PITCH = (
    "\nIf you TUNE, begin \"content\" with the note you are taking — the "
    "one the part you answer is holding — as its name and then its "
    "frequency in hertz in parentheses, then your words. If you sing "
    "NEW, you may name your own pitch the same way.")


def cents_apart(named_hz, target_hz) -> float | None:
    """How far a named pitch lies from the one it answered, octaves set aside.

    0 is a unison or an octave; 600 is a tritone, the farthest two pitch classes
    can be. None when either side is missing.
    """
    try:
        a, b = float(named_hz), float(target_hz)
    except (TypeError, ValueError):
        return None
    if not (a > 0 and b > 0):
        return None
    c = (1200.0 * math.log2(a / b)) % 1200.0
    return min(c, 1200.0 - c)


def describe_hearing(pairs: Iterable[tuple[float | None, float | None]]) -> dict:
    """Named-against-target pairs from TUNE breaths, as a few plain numbers."""
    pairs = list(pairs)
    n = len(pairs)
    if not n:
        return {"tunes": 0}
    apart = [c for c in (cents_apart(a, b) for a, b in pairs) if c is not None]
    return {
        "tunes": n,
        # The share of tunings where the voice named a note at all.
        "named": len(apart) / n,
        "cents_median": statistics.median(apart) if apart else None,
        # Within a quarter-tone of the note that was there to take.
        "hit": (sum(1 for c in apart if c <= HIT_CENTS) / len(apart)) if apart else None,
    }


def named_pitch(text: str) -> float | None:
    """The first pitch in a breath's words, as the runtime reads it."""
    found = extract_all_pitches(said_in(text))
    return float(found[0][0]) if found else None


def tuned_to(text: str) -> str | None:
    found = re.search(r'"tuning_to"\s*:\s*"([^"]+)"', str(text or ""))
    return found.group(1) if found else None


def choices_per_word(mean_loss: float | None) -> float | None:
    """Perplexity: the number of equally likely words the voice is choosing between."""
    if mean_loss is None or not math.isfinite(mean_loss):
        return None
    return math.exp(min(mean_loss, 50.0))


def spread(rows: list, count: int) -> list:
    """`count` rows across the whole list, first and last included."""
    return learning._spread(rows, count)


def said_in(text: str) -> str:
    """The words of a breath, from JSON when the voice answered in the protocol."""
    text = str(text or "").strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict) and "content" in value:
                return " ".join(str(value.get("content") or "").split())
        except json.JSONDecodeError:
            pass
    found = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text, flags=re.S)
    if found:
        return " ".join(found.group(1).split())
    return " ".join(text.split())


def turned_to_a_person(text: str) -> bool:
    """A TUNE aimed at a human id, in a protocol answer."""
    text = str(text or "")
    if not re.search(r'"action"\s*:\s*"TUNE"', text, flags=re.I):
        return False
    return bool(re.search(r'"tuning_to"\s*:\s*"human', text, flags=re.I))


def _key(text: str) -> str:
    return " ".join(str(text or "").split()).casefold()


def describe_takes(takes: Iterable[str], known_lines: set[str] | None = None,
                   pitches: dict[str, float] | None = None) -> dict:
    """What a set of breaths on one ticket looks like, as plain shares.

    `pitches` is what each part in the ticket was holding, by voice_id; given,
    the TUNE takes are also read as hearing — the note named against the note
    of the part tuned to.
    """
    takes = list(takes)
    words = [said_in(t) for t in takes]
    keys = [_key(w) for w in words if w]
    n = len(takes)
    if not n:
        return {"takes": 0}
    common = Counter(keys).most_common(1)
    known = known_lines or set()
    hearing = {}
    if pitches:
        pairs = [(named_pitch(t), pitches.get(tuned_to(t) or ""))
                 for t in takes
                 if re.search(r'"action"\s*:\s*"TUNE"', str(t), flags=re.I)
                 and pitches.get(tuned_to(t) or "")]
        hearing = {"hearing": describe_hearing(pairs)}
    return {
        **hearing,
        "takes": n,
        "empty": sum(1 for w in words if not w) / n,
        # How many different things it said, out of how many breaths.
        "different": len(set(keys)) / n,
        # How much of the time the single most frequent line came out.
        "top_line": (common[0][1] / n) if common else 0.0,
        "top_line_text": common[0][0][:160] if common else "",
        # Said word for word somewhere in its own log already.
        "from_its_past": (sum(1 for k in keys if k in known) / n) if known else None,
        "to_a_person": sum(1 for t in takes if turned_to_a_person(t)) / n,
        "chars": sum(len(w) for w in words) / n,
        "samples": words[:6],
    }


def ticket_heard(voice, chronology: list[dict], times: list[float], at: float,
                 limit: int = 16) -> list[dict]:
    """The sixteen events before `at`, other voices only — the school's fallback
    window for lines older than `heard`."""
    import bisect

    end = bisect.bisect_left(times, at)
    return [r for r in chronology[max(0, end - limit * 2):end]
            if str(r.get("voice_id") or "") != voice.voice_id][-limit:]


def ticket_pitches(heard: list[dict]) -> dict[str, float]:
    """What each part in a frozen window was last holding, by voice_id."""
    out: dict[str, float] = {}
    for r in heard:
        hz = r.get("pitch_hz", r.get("hz"))
        try:
            hz = float(hz or 0)
        except (TypeError, ValueError):
            continue
        if hz > 0:
            out[str(r.get("voice_id") or "")] = hz
    return out


def ticket_prompt(voice, score: dict, chronology: list[dict], times: list[float],
                  own: list[dict], at: float, limit: int = 16,
                  notes: bool = True) -> str:
    """The prompt the school would build for a breath taken at `at`.

    The voice's previous line is the last thing it really sang before `at`.
    `notes` shows the parts' pitches, as the room has since 15 October.
    """
    heard = ticket_heard(voice, chronology, times, at, limit)
    before = [r for r in own if learning.event_time(r) < at]
    row = {"timestamp": at, "notes_shown": notes,
           "previous_content": str(before[-1].get("content") or "") if before else ""}
    prompt = learning.build_prompt(voice, score, row, heard)
    if notes and getattr(voice, "instruct", True):
        prompt += NAME_YOUR_PITCH
    return prompt
