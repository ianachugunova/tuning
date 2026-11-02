"""The log as a score — one definition of the shape, and one reader for it.

The weights die in November and are deleted. What the work leaves behind is
`choir.jsonl` and `field.jsonl` read as notation: every breath the machines took
and every breath the people took, in one chronology, notated well enough that
somebody in 2036 can perform it. The intended performance is a swap — people
sing the machine parts and machines sing the people's — so both halves have to
be notated to the same standard, and neither half may depend on anything that
only existed while the card was warm.

That is why this file exists rather than the fields being written inline at the
two places that append to the logs. There is one shape, defined once, used by
the writer on the GPU (`choir_app.py`), by the writer on the portal
(`portal_app.py`), and by the reader (`score_render.py`). A field that means one
thing to the choir and another to the page would be an unfixable defect: the log
cannot be rewritten after the fact.

**What a performer needs, and where it comes from.**

| | machine breath | human breath |
|---|---|---|
| when it began | `timestamp` | `started_ts` |
| how long it sounded | `sound_s` | `sound_s` |
| how long the silence before it | `silence_before_s`, and derivable | `silence_before_s` |
| the pitch | `pitch_hz` + `pitch_note` + `pitch_cents` | the same three |
| how the pitch was expressed | `pitch_kind`: note / hz / none | `pitch_kind`: sung |
| whether it actually sounded | `toned`, `spoke` | `voiced_fraction` |
| the words | `content` | `content` (the measured line) |
| who was singing | `voice_id`, `model_short`, `tongue` | `voice_id`, `who: human` |
| what it was tuning to | `action`, `tuning_to`, `tuning_to_id`, `heard` | not declared |

**Durations are stored, not inferred, and that is the whole point of the file.**
The choir already logged `breath_chosen_s` — the length the voice *asked* for.
It is not the length the room heard: `render_breath` stretches the tone to cover
the spoken words, so any breath whose speech ran long sounded for longer than
the log said, and by an amount nothing recorded. `sound_s` is the length of the
wav that was actually written. Once it is present on every line, every silence
in the six weeks is exactly derivable from the merged chronology, which is the
one thing a tuning meditation cannot be reconstructed without.

**Pitch is stored as a number and as a name, both.** A machine names "D4" or
names "440Hz"; a person names nothing and simply produces 305.2 Hz. All three
have to end up on a stave, and the swap means each has to be readable by the
other kind of performer: a human handed a machine's part needs `pitch_note`,
and a machine handed a human's part needs `pitch_hz`. `pitch_cents` keeps the
difference between them rather than rounding it away — a person singing 22 cents
under D#4 is the piece working, not a measurement error.

**Extending, never replacing.** Every field here is additive. `mortality.py`
replays the same file for the accounting and ignores what it does not know;
`choir_app._choir_event` filters to the fields of the June `Event` dataclass;
the portal's `/field` reads a handful of keys by name. Nothing here renames or
removes anything, and every reader in this file treats every new field as
optional, because the lines written before today do not have them.

Stdlib only, like `mortality.py`, so a score can be rendered on a laptop with no
card, no Modal and no network.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field as dc_field

# ---------------------------------------------------------------------------
# How a pitch was expressed
# ---------------------------------------------------------------------------

# The distinction matters musically and it is not recoverable from the number.
# A voice that wrote "D4" chose a name in a temperament; a voice that wrote
# "294 Hz" chose a frequency and may not have known it was a D; a person chose
# neither and made a sound. An engraver notates the first as a note head, the
# second as a note head with a cents figure, the third as a note head with a
# cents figure and a wavy line.
PITCH_NOTE = "note"      # named a note: "D4", "F#3", "Bb5"
PITCH_HZ = "hz"          # named a frequency: "294 Hz", "440"
PITCH_SUNG = "sung"      # produced a sound and named nothing — a person
PITCH_GIVEN = "given"    # meant a pitch, named none, and the runtime chose one
PITCH_NONE = "none"      # no pitch anywhere in the line

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Same shape as live_v1/pitch_text._NOTE_RE, kept here so this module stays
# stdlib-only and importable in the portal's image, which has no live_v1.
_LABEL_IS_NOTE = re.compile(r"^\s*[A-Ga-g](#{1,2}|b{1,2})?\s*[0-9]\s*$")


def label_kind(label: str) -> str:
    """Which of the two ways a model named its pitch this was."""
    if not label:
        return PITCH_NONE
    return PITCH_NOTE if _LABEL_IS_NOTE.match(label) else PITCH_HZ


def note_name(hz: float) -> str:
    """Hz → nearest equal-tempered note name, A4 = 440. "" for no pitch.

    The same arithmetic as `pitch-core.js`'s `hzToNote`, so a human breath
    measured in the browser and a machine breath notated here land on the same
    note head.
    """
    if not hz or hz <= 0:
        return ""
    semi = round(12 * math.log2(hz / 440.0))
    idx = (semi + 9) % 12
    octave = 4 + (semi + 9) // 12
    return f"{_NOTE_NAMES[idx]}{octave}"


def cents_off(hz: float) -> int:
    """Cents between `hz` and the nearest note: −50…+50."""
    if not hz or hz <= 0:
        return 0
    exact = 12 * math.log2(hz / 440.0)
    return int(round((exact - round(exact)) * 100))


# ---------------------------------------------------------------------------
# What the writers append
# ---------------------------------------------------------------------------

# Only the pitches that a reader could plausibly notate. A model occasionally
# writes a paragraph with a dozen numbers in it; the first one is the one that
# sounded and the rest are context, not a chord.
MAX_PITCHES_KEPT = 4


def transposition_fields(sounded_hz: float, asked_hz: float | None,
                         octaves: int) -> dict:
    """What a voice was answering, when that is not what came out of it.

    Each voice sings inside a range derived from the body it speaks in, and a
    pitch outside that range is moved by whole octaves until it fits — see
    `tuning.fold_into`. `pitch_hz` stays what sounded, because the log has to
    describe the sound. These two say what it was reaching for and how far it
    travelled, so "the choir converged" can never be an artefact of arithmetic
    nobody recorded: an octave is the same pitch, and a reader who wants to
    check that has to be able to see both numbers.

    Absent when nothing moved, rather than written as zero. Most breaths do not
    move, and a field that appears only on the ones that did is readable at a
    glance in a log with a hundred thousand lines in it.
    """
    if not octaves or not asked_hz:
        return {}
    return {
        "pitch_asked_hz": round(float(asked_hz), 2),
        "pitch_asked_note": note_name(float(asked_hz)),
        "pitch_octaves": int(octaves),
    }


def machine_pitch_fields(pitches: list[tuple[float, str]],
                         sounded_hz: float | None = None,
                         octaves: int = 0) -> dict:
    """The pitch block for a machine breath, from what the line contained.

    `pitches` is `live_v1.pitch_text.extract_all_pitches(content)` — exactly the
    list `render_breath` reads, so `pitch_hz` is the frequency that was actually
    put into the wav and not a second, differently-derived opinion about it.
    That equality is the reason this takes the list rather than the text: two
    implementations of "which pitch did it mean" would drift, and the log would
    stop describing the sound.

    Recoverable in principle from `content` alone, and stored anyway. The
    derivation depends on a regex in a file that will not exist in a decade, and
    the score has to survive the code.
    """
    if not pitches:
        return {"pitch_hz": None, "pitch_note": "", "pitch_cents": 0,
                "pitch_label": "", "pitch_kind": PITCH_NONE, "pitch_all": []}
    named, label = pitches[0]
    # What sounded, which is the named pitch unless it lay outside this voice's
    # range and was moved into it. `pitch_all` keeps what was written either way.
    hz = float(sounded_hz) if sounded_hz else float(named)
    return {
        "pitch_hz": round(hz, 2),
        "pitch_note": note_name(hz),
        "pitch_cents": cents_off(hz),
        "pitch_label": str(label)[:16],
        "pitch_kind": label_kind(str(label)),
        # A voice that names two tones is describing a gesture — "D4 then F4".
        # Only the first sounded; the rest is what it wrote and meant.
        "pitch_all": [[round(float(h), 2), str(l)[:16]]
                      for h, l in pitches[:MAX_PITCHES_KEPT]],
        **transposition_fields(hz, named, octaves),
    }


def given_pitch_fields(hz: float, why: str, asked_hz: float | None = None,
                       octaves: int = 0) -> dict:
    """The pitch block for a breath whose frequency the runtime chose.

    Two kinds of voice arrive here. An ear, which can hear that somebody is
    holding a note and cannot say which note it is — that was measured, and it
    is the reason the ears are asked whom they are answering rather than what
    the pitch is. And an instructed voice that wrote `"action": "TUNE"` and then
    prose with no number in it, which used to mean a declared tuning that made
    no sound at all.

    `pitch_hz` is still the frequency that went into the wav, because that
    equality is what makes the log describe the sound. What changes is
    `pitch_kind`, and `pitch_why`, which says in words where the number came
    from — so a reader in December can tell a voice that tuned from a system
    that tuned on its behalf. Nothing else in the piece can tell them apart.
    """
    hz = float(hz or 0.0)
    if hz <= 0:
        return {"pitch_hz": None, "pitch_note": "", "pitch_cents": 0,
                "pitch_label": "", "pitch_kind": PITCH_NONE, "pitch_all": []}
    return {
        "pitch_hz": round(hz, 2),
        "pitch_note": note_name(hz),
        "pitch_cents": cents_off(hz),
        "pitch_label": "",
        "pitch_kind": PITCH_GIVEN,
        "pitch_why": str(why)[:64],
        "pitch_all": [],
        **transposition_fields(hz, asked_hz, octaves),
    }


def human_pitch_fields(hz: float, note: str, cents: int) -> dict:
    """The same block for a person, who named nothing and sang.

    `note` and `cents` come from the browser rather than being recomputed, so
    what the singer was shown on their own screen is what the score says they
    sang. `note_name` here is only the fallback for a page too old to send it.
    """
    hz = float(hz or 0.0)
    return {
        "pitch_hz": round(hz, 2) if hz > 0 else None,
        "pitch_note": note or note_name(hz),
        "pitch_cents": int(cents),
        "pitch_label": "",
        "pitch_kind": PITCH_SUNG if hz > 0 else PITCH_NONE,
        "pitch_all": [],
    }


def resolve_tuning_to(tuning_to: str | None, heard: list[str]) -> str | None:
    """`tuning_to` as a voice_id, when the free text names one of the voices.

    The models are asked who they are tuning to and answer in prose: a voice_id,
    a short name, a pitch, "the low hum", nothing. An arrow on a score has to
    land on a part, so the string is matched against the voices that were
    actually in the listening window and resolved when it can be. Left `None`
    otherwise, and `tuning_to` keeps the words either way — an unresolvable
    answer is information about the voice, not a parse failure.
    """
    if not tuning_to:
        return None
    text = str(tuning_to).lower()
    for vid in heard:
        if not vid:
            continue
        low = vid.lower()
        if low in text:
            return vid
        # "voice_03" is also named by its number, and humans by their four
        # characters: "human_a1b2" answers to "a1b2".
        tail = low.rsplit("_", 1)[-1]
        if len(tail) >= 2 and tail in text:
            return vid
    return None


def silence_before(prev_end_ts: float | None, onset_ts: float) -> float | None:
    """The rest before a breath, in seconds. `None` when there is no before.

    Zero means an entry into a field that was still sounding — an overlap, not a
    rest of no length. A negative gap is never returned, because a rest cannot
    run backwards; the overlap survives in the absolute onsets, which is where a
    score keeps it anyway.

    **There are two rests and they are not the same quantity.** The silence
    before a breath *in the field* is the ensemble's rhythm: it is what the
    voice was answering and it is exactly derivable from the merged chronology.
    The silence before a breath *in one part* is what the performer of that part
    reads off their own page. `read()` derives both. What the writers store is
    redundancy against a lost or unmergeable file, plus — on the human side —
    the one measurement a merge cannot reproduce, because a singer's own rest is
    timed on the singer's own audio clock and not on arrival at a server.
    """
    if prev_end_ts is None or prev_end_ts <= 0:
        return None
    return round(max(0.0, onset_ts - prev_end_ts), 3)


# ---------------------------------------------------------------------------
# One notated event
# ---------------------------------------------------------------------------


@dataclass
class Note:
    """One row of the score. Everything an engraver needs, nothing else.

    Deliberately flat and deliberately dumb: no references into the log, no
    lazy lookups, no objects that need the rest of the repository to mean
    anything. A `list[Note]` printed as CSV is still a score.
    """

    t: float                      # onset, unix seconds
    kind: str                     # tone | death | birth | silence
    who: str                      # human | machine | system
    voice_id: str = ""
    name: str = ""                # model_short, or "a voice"
    tongue: str = ""              # the language the words are in
    sound_s: float = 0.0          # how long it sounded; 0 for a written silence
    # The two rests. `rest_s` is the ensemble's — how long the whole field had
    # been quiet — and it is what a conductor reads. `own_rest_s` is this
    # part's, and it is what the one performer of this part reads. Both derived
    # by `read()`; `heard_rest_s` is what the voice itself wrote down, kept as a
    # cross-check and as the only rest available if one of the two logs is lost.
    rest_s: float | None = None
    own_rest_s: float | None = None
    heard_rest_s: float | None = None
    pitch_hz: float | None = None
    pitch_note: str = ""
    pitch_cents: int = 0
    pitch_label: str = ""
    pitch_kind: str = PITCH_NONE
    text: str = ""                # what was sung, in words
    action: str = ""              # NEW | TUNE, for machines
    tuning_to: str = ""           # what it said it was tuning to
    tuning_to_id: str = ""        # that, resolved to a part, when it resolves
    contour_hz: list = dc_field(default_factory=list)
    contour_hop_s: float = 0.0
    audio: str = ""               # the wav on the volume, if one was kept
    note: str = ""                # a word for the reader: an epitaph, a birth

    @property
    def end(self) -> float:
        return self.t + max(0.0, self.sound_s)


# ---------------------------------------------------------------------------
# Reading the two logs as one score
# ---------------------------------------------------------------------------


def _f(row: dict, *keys, default=0.0) -> float:
    for k in keys:
        v = row.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return default


def _machine_note(row: dict) -> Note:
    hz = row.get("pitch_hz")
    hz = float(hz) if hz else None
    # Legacy lines carry no pitch block at all. The frequency is not recovered
    # here — that would need live_v1's regex, which this module deliberately
    # does not import — but the breath is still notated, with no pitch and the
    # text intact. A reader can see that it is a line from before the score
    # existed rather than a voice that chose to sound without a pitch.
    return Note(
        t=_f(row, "timestamp"),
        kind="tone",
        who="machine",
        voice_id=str(row.get("voice_id") or ""),
        name=str(row.get("model_short") or ""),
        tongue=str(row.get("tongue") or ""),
        # `breath_chosen_s` is the fallback and it is a floor, not the truth:
        # before `sound_s` existed a spoken breath could run past it.
        sound_s=_f(row, "sound_s", "breath_chosen_s"),
        heard_rest_s=(float(row["silence_before_s"])
                      if row.get("silence_before_s") is not None else None),
        pitch_hz=hz,
        pitch_note=str(row.get("pitch_note") or (note_name(hz) if hz else "")),
        pitch_cents=int(row.get("pitch_cents") or 0),
        pitch_label=str(row.get("pitch_label") or ""),
        pitch_kind=str(row.get("pitch_kind") or (PITCH_NONE if not hz else PITCH_HZ)),
        text=str(row.get("content") or ""),
        action=str(row.get("action") or ""),
        tuning_to=str(row.get("tuning_to") or ""),
        tuning_to_id=str(row.get("tuning_to_id") or ""),
        audio=str(row.get("audio") or ""),
    )


def _human_note(row: dict) -> Note:
    hz = row.get("pitch_hz") or row.get("hz")
    hz = float(hz) if hz else None
    onset = _f(row, "started_ts")
    dur = _f(row, "sound_s", "duration_s")
    if not onset:
        # Pages older than the score sent no onset. `ts` is when the breath
        # arrived at the portal, which is after the sound ended by the
        # half-second silence tail plus the network, so the onset is behind it
        # by at least the duration. Approximated rather than dropped, and the
        # approximation is stated here so nobody mistakes it for a measurement.
        onset = _f(row, "ts") - dur
    return Note(
        t=onset,
        kind="tone",
        who="human",
        voice_id=str(row.get("voice_id") or ""),
        name="a voice",
        sound_s=dur,
        # The singer's own rest, timed on their own clock between their own two
        # breaths. Kept as given: a merge can only reconstruct it from arrival
        # times, and the network is not part of the music.
        own_rest_s=(float(row["own_rest_s"])
                    if row.get("own_rest_s") is not None else None),
        pitch_hz=hz,
        pitch_note=str(row.get("pitch_note") or row.get("note") or ""),
        pitch_cents=int(row.get("pitch_cents") or row.get("cents") or 0),
        pitch_kind=str(row.get("pitch_kind")
                       or (PITCH_SUNG if hz else PITCH_NONE)),
        text=str(row.get("content") or "").strip(),
        contour_hz=list(row.get("contour_hz") or []),
        contour_hop_s=_f(row, "contour_hop_s"),
        note=str(row.get("stability") or ""),
    )


def _death_note(row: dict) -> Note:
    return Note(
        t=_f(row, "timestamp"),
        kind="death",
        who="machine",
        voice_id=str(row.get("voice_id") or ""),
        name=str(row.get("model_short") or ""),
        text=str(row.get("last_words") or ""),
        note=(f"spent {_f(row, 'spent_usd'):.2f} of {_f(row, 'seed_usd'):.2f} "
              f"over {int(_f(row, 'breaths'))} breaths; "
              f"{int(_f(row, 'living_after'))} voices left"),
    )


def _birth_note(row: dict) -> Note:
    cast = row.get("cast") or []
    return Note(
        t=_f(row, "timestamp"),
        kind="birth",
        who="system",
        text=", ".join(str(c.get("model_short") or c.get("voice_id"))
                       for c in cast),
        note=str(row.get("note") or ""),
    )


def read(choir_rows: list[dict], human_rows: list[dict]) -> list[Note]:
    """Both logs, merged into one score, with the silences filled in.

    The rests are **derived here** rather than trusted from the log, and that is
    deliberate. Once every line carries an onset and a length, the merged
    chronology determines the rhythm exactly, and the derivation cannot disagree
    with itself: every breath either begins after the field went quiet, in which
    case its `rest_s` is the length of that quiet, or it begins while somebody is
    still sounding, in which case `rest_s` is zero and the overlap is carried by
    the onsets. There is no third case and no unaccounted interval — which is the
    whole claim that the six weeks are reconstructible.

    What the writers stored is kept beside it, never instead of it: a machine's
    own `silence_before_s` becomes `heard_rest_s`, because a voice's account of
    the rest it was answering is a fact about that voice and not about the
    archive, and because it is the only rest left if one of the two files is
    ever lost. A singer's `own_rest_s` is kept as given, since a merge can only
    reconstruct it from arrival times.

    Sorted by onset. The logs themselves are append-only and their file order is
    the authority on causation (see `mortality.replay`), but two files cannot be
    merged by file order, and a score is read left to right.
    """
    notes: list[Note] = []
    for row in choir_rows:
        kind = str(row.get("kind") or "tone")
        if kind == "tone":
            notes.append(_machine_note(row))
        elif kind == "death":
            notes.append(_death_note(row))
        elif kind == "birth":
            notes.append(_birth_note(row))
        # `card`, `exit`, `load`, `score`, `train` are the ledger's business.
        # They are not silent notes; they are why the card was warm.
    for row in human_rows:
        notes.append(_human_note(row))

    notes.sort(key=lambda n: (n.t, 0 if n.kind == "birth" else 1))

    field_end: float | None = None
    part_end: dict[str, float] = {}
    for n in notes:
        if n.kind != "tone":
            continue
        n.rest_s = silence_before(field_end, n.t)
        if n.rest_s is None and n.heard_rest_s is not None:
            # The first breath of a slice has no measurable before inside it.
            # The voice that took it recorded one; that is better than nothing,
            # and it is the only place the log's own value is ever preferred.
            n.rest_s = n.heard_rest_s
        # The performer's own rest. Measured on the singer's clock when they
        # sent one — a phone timing the gap between two of its own notes beats
        # the same gap reconstructed from two arrival times — and derived from
        # the part otherwise, which is every machine breath.
        if n.own_rest_s is None:
            n.own_rest_s = silence_before(part_end.get(n.voice_id), n.t)
        field_end = max(field_end or 0.0, n.end)
        part_end[n.voice_id] = max(part_end.get(n.voice_id, 0.0), n.end)
    return notes


def parts(notes: list[Note]) -> dict[str, list[Note]]:
    """The score by part, which is how it is rehearsed and how it is swapped.

    A part is a `voice_id`, machine or human, because that is the unit the
    intended performance exchanges: a singer takes gpt-2's part, gpt-2 takes
    the part of whoever sang into a phone in Venice.
    """
    out: dict[str, list[Note]] = {}
    for n in notes:
        if n.kind != "tone":
            continue
        out.setdefault(n.voice_id, []).append(n)
    return out
