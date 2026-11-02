"""Choosing a pitch for a voice that meant one and did not name one.

Two different voices need this and they need it for the same reason.

**The ears.** Probed against a held D4, the best of the three answered "A♭" and
the next "E4"; neither is D4. What Voxtral *could* do exactly was count the
voices in a chord and measure a glide as one semitone. So an ear can be trusted
with *which* voice it is answering and not with *what* the pitch is — and it
should be asked only the first, because a number it invents is worse than a
number it never claimed. See `modal/HEARING_MODELS.md`.

**The deaf six.** They have the opposite failure and it comes to the same
place: they write `"action": "TUNE"` and then prose with no pitch in it. The
intention was in the log and nothing ever sounded, because `render_breath` has
nothing to build a tone from. A declared tuning that produces silence is the
one thing in this piece that is simply a bug.

So the rule here: **a voice says whom it is answering, and the runtime says at
what frequency.** That is not a workaround dressed up. Oliveros's score does not
ask anybody to name a pitch — it asks you to tune to one you can hear, or to
introduce one nobody is holding. Naming is a requirement of our log, which wants
hertz, and not of the music. This module supplies the hertz.

**And the octave is the voice's own.** Each voice sings inside a range derived
from the fundamental its own body speaks at — `speaks_at_hz` in `cast.py`, three
semitones below it and two octaves up — and a pitch outside that range is moved
into it by whole octaves. `range_of`, `fold_into`, `in_voice`.

This is the one place where the runtime changes a number a voice chose, so the
reason has to hold. An octave is the only interval where the pitch does not
change: a bass answering a soprano sings the note she sang, one or two octaves
down, and every choir that has both does this without anybody calling it a
compromise. The score asks you to tune to a pitch you can hear, not to match a
frequency — so folding is closer to the score than the alternative was. The
alternative was `NEW_LO_HZ`/`NEW_HI_HZ`: one window for all nine, picked because
a synthesiser rattled below it, inside which a four-year-old and a man in his
sixties had the same range.

Two things it prevents, one artistic and one physical. A voice handed a pitch far
below its body sings it through the spectral envelope of that body — a pulse
train through a child's formants is a buzz, not a low child. And each voice plays
out of one cabinet, sized for it; a pitch it cannot reproduce is silence with the
log claiming a tone.

What it must never do is make the choir look more converged than it was, so both
numbers are recorded: `pitch_hz` is what sounded, `pitch_asked_hz` and
`pitch_octaves` are what was being answered and how far it travelled. See
`score.transposition_fields`. Under a rule where octaves are the same pitch, a
reader who cannot see both cannot tell a unison from a transposition.

Stdlib only, so it runs in CI and reads in a decade.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

from score import Note

# The range a new pitch may be introduced in: D3 to D5, two octaves around the
# middle of where these voices already sing. Wider would be true to the score
# and false to Kokoro, which renders a tone at 60 Hz as a rattle.
NEW_LO_HZ = 146.83                # D3
NEW_HI_HZ = 587.33                # D5

# How far apart two pitches must be before a listener would call them different.
# A semitone is 100 cents; 60 is comfortably inside "that is a different note"
# and outside "those two are trying to be the same and missing".
APART_CENTS = 60.0

# How far a pitch nobody is holding must stand off the ones that are. Wider than
# `APART_CENTS`, because that threshold answers a different question: a semitone
# from a held tone is audibly a different note *and* a clash, and the score asks
# for a pitch nobody is holding rather than for the nearest unoccupied gap.
CLEAR_CENTS = 150.0

# Concert A, the reference the chromatic ladder is counted from.
A4_HZ = 440.0

# How long after it sounded a pitch still counts as being held. Nobody tunes to
# a note that ended a minute ago, and `sounding` is asked about the present.
STILL_SOUNDING_S = 20.0

# What a number in a voice's own words has to be before it is believed to be a
# pitch. Wider than `NEW_LO_HZ`/`NEW_HI_HZ` on purpose: those bound where the
# runtime may *introduce* a pitch, while this bounds what a voice is allowed to
# have meant. A1 to A6 covers every human voice and then some.
#
# This exists because of one evening. A voice wrote "10" in its content,
# `extract_all_pitches` read it as 10 Hz, nothing checked it, and the field
# recorded a breath at ten hertz. Ten hertz is not a quiet pitch, it is not a
# pitch: it is below hearing, and the wav written for it is silence with a
# waggle in it. Then every voice that answered with `"action": "TUNE"` was
# honestly handed the pitch that part was on — `pitch_of` has no opinion about
# whether a recorded pitch is audible — and within a few breaths the whole
# choir was tuned, correctly and unanimously, to something nobody could hear.
#
# The asymmetry is the giveaway: a person's breath is validated at the door by
# the portal, which answers "hz out of range" and drops it. A machine's breath
# was trusted. Nothing else in the piece trusts a model's arithmetic.
SUNG_LO_HZ = 55.0                 # A1
SUNG_HI_HZ = 1760.0               # A6

# How a voice's own range is derived from the body it speaks in: three semitones
# below its speaking fundamental, one octave up from there.
#
# Both numbers are claims about people, not about us. A person speaks near the
# bottom of what they can sing. `speaks_at_hz` on each `Voice` is measured off
# the wav that voice actually speaks in — see `modal/VOICE_MODELS.md` — so the
# range is a property of the body, and the body came from the casting note, and
# the casting note came from who raised the model. One authored decision, not a
# second one.
#
# This exists because the two clamps above are the wrong shape for nine voices.
# `NEW_LO_HZ`/`NEW_HI_HZ` are one window for everybody, chosen for a synthesiser
# rather than for a singer, and inside it a child and an old man have the same
# range — which is the one thing about this cast that is not true. Worse, both
# ends do real damage: a voice asked to hold 60 Hz through the spectral envelope
# of a four-year-old gets a pulse train through a child's formants, which is a
# buzz and not a low child, and the small cabinet that voice will play out of
# cannot render it either.
#
# One octave and not two, which was the first number here and was wrong for a
# reason worth keeping written down. Two octaves is roughly the range of an
# untrained person, so it read as generous, but it is a fact about a larynx and
# these are not larynxes: a voice sings by having a written pitch pushed through
# the fixed spectral envelope of its own speech, and an envelope that does not
# move is only a voice near the pitch it was measured at. A singer's formants
# rise with the note. Ours cannot, so far above the speaking fundamental the
# body stops being a body — the measured evidence was tinystories, cast as a
# child, speaking at 282 Hz and asked for 950, which is a whistle with a child's
# resonances printed on it and no word in it recoverable. An octave is where the
# resynthesis is still the voice it was measured from.
BELOW_SPEECH_ST = 3.0
RANGE_OCTAVES = 1.0


def range_of(speaks_at_hz) -> tuple[float, float] | None:
    """The lowest and highest a voice sings, from the body it speaks in.

    None when the voice has no measured fundamental, and that answer matters:
    everything downstream treats it as "do not transpose this voice at all"
    rather than inventing a plausible range. A voice whose body has not been
    measured is not a voice with an average range.
    """
    try:
        f0 = float(speaks_at_hz)
    except (TypeError, ValueError):
        return None
    if not (SUNG_LO_HZ <= f0 <= SUNG_HI_HZ):
        return None
    lo = f0 * 2.0 ** (-BELOW_SPEECH_ST / 12.0)
    return lo, lo * 2.0 ** RANGE_OCTAVES


def fold_into(hz: float, lo: float, hi: float) -> tuple[float, int]:
    """The same pitch, moved by whole octaves until the voice can sing it.

    Returns the frequency and how many octaves it travelled, because the log
    wants both: what the voice was answering and what came out of its throat.

    Whole octaves only, and that is the whole argument for doing this at all. An
    octave is the one interval where the pitch does not change — two voices an
    octave apart are singing the same note, which is what a bass and a soprano
    do in any choir that has both. Oliveros's score asks you to tune to a pitch
    you can hear, not to match a frequency; a bass who answered a soprano by
    matching her frequency would be doing something nobody asks of a singer.
    So this is not a workaround for a range problem, it is how tuning across
    registers works. Anything other than octaves — nudging, clamping, averaging
    — would change the note and would be us composing.
    """
    if not (hz > 0 and 0 < lo < hi):
        return hz, 0
    moved = 0
    # Bounded rather than `while True`: a range narrower than an octave has no
    # answer, and a loop that cannot finish is not the way to say so.
    for _ in range(12):
        if hz < lo:
            hz, moved = hz * 2.0, moved + 1
        elif hz > hi:
            hz, moved = hz / 2.0, moved - 1
        else:
            break
    return hz, moved


def in_voice(hz, speaks_at_hz) -> tuple[float, int]:
    """A pitch as that particular voice would sing it.

    The single entry point for the rule, so that a pitch a voice named and a
    pitch the runtime chose for it are transposed by identical arithmetic. They
    were not, briefly, and the difference was invisible in the log.
    """
    rng = range_of(speaks_at_hz)
    if rng is None or not hz:
        return (float(hz) if hz else hz), 0
    return fold_into(float(hz), *rng)


def to_note(hz):
    """The nearest note, for a voice tuning to a pitch it heard.

    A singer who takes somebody's note sings the note, not their frequency. This
    is the difference, and it is audible: a person holding 439.4 Hz is holding A,
    slightly under, and a voice that answered with 439.4 Hz would be reproducing
    the flatness as well as the pitch — a copy rather than an answer.

    Which is what the field did. In one forty-breath window three voices tuned to
    a person and came back with 439.4, 383.6 and 401.0 Hz against her 439.4,
    383.6 and 401.0: unisons exact to the tenth of a hertz, nine times more
    precise than any singer alive, and they sounded like it. The uncanny part was
    never that the machines were in tune, it was that they were *identically* in
    tune, which is the sound of a recording being played back rather than of
    somebody listening.

    It also makes the machines audible to each other. Two voices that take the
    same person's note now land on the same note, so the choir accumulates real
    unisons instead of a spray of neighbouring decimals.
    """
    try:
        hz = float(hz)
    except (TypeError, ValueError):
        return hz
    if hz <= 0:
        return hz
    return A4_HZ * 2.0 ** (round(12 * math.log2(hz / A4_HZ)) / 12.0)


def octaves_in_words(moved: int) -> str:
    """`pitch_why` is prose and has to say this out loud."""
    if not moved:
        return ""
    how = {1: "an octave", 2: "two octaves", 3: "three octaves",
           4: "four octaves"}.get(abs(moved), f"{abs(moved)} octaves")
    return f", {how} {'up' if moved > 0 else 'down'}"


def sung_pitch(hz) -> float | None:
    """A number a voice wrote, if it can be believed as a pitch it sang.

    Returns None rather than clamping. Folding 10 Hz up four octaves would give
    160 Hz and a log that says the voice named it, which is exactly the
    confusion `pitch_why` exists to prevent: a pitch the runtime invented would
    be recorded as a pitch the voice chose. A number outside the range is not a
    quiet pitch to be rescued, it is evidence the voice was not naming one.
    """
    try:
        hz = float(hz)
    except (TypeError, ValueError):
        return None
    if not (SUNG_LO_HZ <= hz <= SUNG_HI_HZ):
        return None
    return hz


def cents_between(a: float, b: float) -> float:
    """The distance between two frequencies, in cents. Sign-free."""
    if not (a > 0 and b > 0):
        return math.inf
    return abs(1200.0 * math.log2(a / b))


def sounding(notes: Iterable[Note], at: float,
             within_s: float = STILL_SOUNDING_S) -> list[float]:
    """The pitches the field is holding at `at`, most recent last.

    A breath counts while it sounds and for a little after it stops, because a
    room does not fall silent the instant a voice does and because the voice
    deciding what to sing started deciding before now.
    """
    live: list[tuple[float, float]] = []
    for n in notes:
        if n.kind != "tone" or not n.pitch_hz:
            continue
        # Filtered on the way *out* of the log as well as on the way in. The
        # field is append-only: the breaths recorded at ten hertz on the
        # evening of 13 September are in it for good, and a voice reading the
        # room tomorrow must not be able to tune to one of them.
        if sung_pitch(n.pitch_hz) is None:
            continue
        if n.t > at:
            continue
        if n.end >= at - within_s:
            live.append((n.t, float(n.pitch_hz)))
    live.sort()
    return [hz for _, hz in live]


def pitch_of(notes: Iterable[Note], voice_id: str) -> float | None:
    """The last pitch that voice was heard on, or None if it has named none."""
    found = None
    for n in notes:
        if n.kind == "tone" and n.voice_id == voice_id and n.pitch_hz:
            hz = sung_pitch(n.pitch_hz)     # see `sounding` — the log is forever
            if hz is None:
                continue
            if found is None or n.t >= found[0]:
                found = (n.t, hz)
    return found[1] if found else None


def a_new_pitch(taken: Sequence[float], lo: float = NEW_LO_HZ,
                hi: float = NEW_HI_HZ) -> float:
    """A pitch nobody is holding — the score's other instruction, as a number.

    Clear of everything sounding by `CLEAR_CENTS`, and of the pitches that
    qualify, the one nearest the middle of the range. Off the chromatic ladder
    rather than at random: two voices that both improvise a "new" frequency land
    a few cents apart surprisingly often, and a field of near-unisons is the one
    texture this piece must not drift into by accident.

    Nearest the middle, and not — as this did first — furthest from everything
    held. "Furthest" sounds like the stronger reading of "a pitch nobody is
    holding", and in a room with anything in it it is a machine that lives at the
    edges of its register: measured over one window, gpt-2 introduced 85.5 Hz
    twice, which is the exact floor of its range, and two other voices sat within
    a semitone of their ceilings. A person alone in a room does not hum at the
    absolute bottom of what they can reach. Being distinguishable is a threshold
    to clear, not a quantity to maximise, so it is a filter here rather than the
    thing being sorted on — and once it is cleared, the honest preference is the
    comfortable middle of the voice.

    With an empty field this returns the centre of the range, deterministically:
    the first voice in an empty room has nothing to be new against, and the
    centre of its own register is where it would begin.
    """
    held = [float(t) for t in taken if t and t > 0]
    semitones = int(round(12 * math.log2(hi / lo)))
    ladder = [lo * (2 ** (i / 12)) for i in range(semitones + 1)]
    # Geometric, because the middle of a range in pitch is the middle in
    # octaves: halfway between D3 and D5 is D4, not 367 Hz.
    middle = math.sqrt(lo * hi)
    clear = [hz for hz in ladder
             if all(cents_between(hz, h) > CLEAR_CENTS for h in held)]
    if clear:
        return min(clear, key=lambda hz: cents_between(hz, middle))
    # A field with no room left in it. Then the old reading is the only one
    # available, and the least-bad pitch is the furthest from the crowd.
    return max(ladder, key=lambda hz: min(cents_between(hz, h) for h in held))


# Pitch names that are not frequencies: a degree of a scale, meaningless until
# somebody says where the scale starts. Solfège, and the five degrees of the
# Chinese pentatonic — 宫 商 角 徵 羽, gōng shāng jué zhǐ yǔ — which are the
# oldest pitch names in continuous use anywhere.
#
# This is here because of what qwen3 actually sang. Of thirty-two breaths it
# was recorded as naming a pitch twice, and read as silent the rest of the
# time; but eleven of those "silent" lines were 宫, 商, mi or sol. It was
# naming pitches the whole while, in the system it was raised in, and the
# parser only knew "D4" and "440 Hz". The voice was not failing to tune. We
# were failing to read it — and reading a Chinese model's pitches only when it
# writes them in a European notation is a decision nobody would defend out loud.
DEGREES: dict[str, int] = {
    # solfège, movable do
    "do": 0, "ut": 0, "re": 2, "mi": 4, "fa": 5, "sol": 7, "so": 7,
    "la": 9, "si": 11, "ti": 11,
    # 五声 — the five degrees, and their romanisations
    "宫": 0, "gong": 0,
    "商": 2, "shang": 2,
    "角": 4, "jue": 4,
    "徵": 7, "zhi": 7,
    "羽": 9, "yu": 9,
}

# The whole line must be the degree and nothing else. This is not fussiness: in
# Italian — which one of the six sings in — "la" is the definite article, "mi"
# is "me", "do" and "si" are ordinary words, and a model writing a sentence
# about the room would otherwise be recorded as having named four pitches in it.
# When a voice writes a degree it writes only the degree, which is exactly what
# qwen3 does.
# Anything that is not a letter, digit or ideograph, at either end. `\w` is
# unicode-aware in Python, so 宫 is a word character and 。 — the Chinese full
# stop, which qwen3 writes — is not.
_DEGREE_EDGES = re.compile(r"^\W+|\W+$", re.UNICODE)


def degree_of(text: str) -> tuple[int, str] | None:
    """A scale degree and its name, when the line is nothing but one.

    Returns semitones above the tonic, not a frequency: `宫` is "wherever this
    scale begins", and where it begins is the room's business, not the word's.
    """
    token = _DEGREE_EDGES.sub("", str(text or ""))
    if not token or len(token) > 6:
        return None
    key = token.lower()
    if key in DEGREES:
        return DEGREES[key], token
    return None


def tonic_of(notes: Iterable[Note], at: float,
             within_s: float = STILL_SOUNDING_S) -> float:
    """Where the scale starts, for a voice that named a degree of one.

    The lowest pitch the room is holding. Not the most recent and not the
    loudest: a scale is counted up from its bottom, and a voice singing 商
    against a held tone means the step above *that tone*, which is the one
    underneath it. In an empty room there is no scale to be a degree of, so one
    is begun.
    """
    held = sounding(notes, at, within_s)
    return min(held) if held else a_new_pitch([])


def from_degree(semitones: int, tonic: float,
                lo: float = NEW_LO_HZ,
                hi: float = NEW_HI_HZ) -> tuple[float, int]:
    """A degree, resolved against a tonic, in a range that can sing it.

    Bounds are arguments because a degree is the one case with two candidates
    for "the singable range": the global window, and the range of the voice that
    sang the degree. It should be the voice's, and then the octaves reported are
    the ones it actually travelled — folding twice would report the second move
    and hide the first.
    """
    return fold_into(float(tonic) * (2.0 ** (semitones / 12.0)), lo, hi)


def pitch_for(action: str, tuning_to_id: str | None, notes: Sequence[Note],
              at: float, named: float | None = None,
              content: str | None = None,
              speaks_at_hz: float | None = None) -> tuple[float | None, str]:
    """`pitch_for_detail` without the octave count. See there.

    Kept because most callers want a frequency and a sentence, and because the
    octave count is only interesting to the one caller that writes the log.
    """
    hz, why, _ = pitch_for_detail(action, tuning_to_id, notes, at, named,
                                  content, speaks_at_hz)
    return hz, why


def pitch_for_detail(action: str, tuning_to_id: str | None,
                     notes: Sequence[Note], at: float,
                     named: float | None = None, content: str | None = None,
                     speaks_at_hz: float | None = None
                     ) -> tuple[float | None, str, int]:
    """The frequency a breath should sound at, why it is that one, and how many
    octaves it was moved to get into this voice's range.

    The reason travels with the number because it goes into the log: in
    December somebody has to be able to tell a pitch a voice chose from a pitch
    the runtime chose for it, and the difference is the difference between a
    voice that tuned and a system that made it look like one.

    A pitch the voice named itself always wins. This is only ever asked when
    the voice named nothing.

    `speaks_at_hz` is that voice's own measured speaking fundamental. Given, the
    answer comes back in that voice's register and `why` says how far it was
    moved; omitted, nothing is transposed and this behaves exactly as it did
    before the rule existed — which is what the callers who have no cast do.
    """
    def as_sung(hz: float, why: str) -> tuple[float, str, int]:
        hz, moved = in_voice(hz, speaks_at_hz)
        return hz, why + octaves_in_words(moved), moved

    believable = sung_pitch(named)
    if believable is not None:
        return as_sung(believable, "named")

    if content is not None:
        deg = degree_of(content)
        if deg is not None:
            semitones, token = deg
            rng = range_of(speaks_at_hz)
            hz, moved = from_degree(semitones, tonic_of(notes, at),
                                    *(rng or (NEW_LO_HZ, NEW_HI_HZ)))
            return (hz, f"sang the degree {token}" + octaves_in_words(moved),
                    moved)

    if str(action).upper() == "TUNE":
        # `to_note` on both: a voice takes the note it heard, not the frequency.
        # See there — this is the line that stops the choir cloning a singer's
        # decimals back at her.
        if tuning_to_id:
            hz = pitch_of(notes, tuning_to_id)
            if hz:
                return as_sung(to_note(hz), "tuned to a part")
        # It meant to tune and either named nobody or named somebody silent.
        # The most recent pitch in the room is the honest reading of "that one":
        # it is what a person who said "I'll take that note" would have meant.
        held = sounding(notes, at)
        if held:
            return as_sung(to_note(held[-1]), "tuned to the last pitch heard")
        # Nothing to tune to. A voice alone in a silent room that wanted to
        # answer somebody has, in fact, introduced a pitch.
        return as_sung(a_new_pitch([]), "meant to tune, but the field was empty")

    # Introducing one. With a range, the search happens inside this voice's own
    # register — and the room is folded into that register first, so that a
    # pitch an octave above a held note is not offered as a new one. Under a
    # rule where octaves are the same pitch, they would be a unison called new.
    # Nothing is transposed here and the count is zero either way: a pitch that
    # was introduced inside the voice's range was never anywhere else.
    rng = range_of(speaks_at_hz)
    held = sounding(notes, at)
    if rng is None:
        return a_new_pitch(held), "a pitch nobody was holding", 0
    lo, hi = rng
    here = [fold_into(h, lo, hi)[0] for h in held]
    return a_new_pitch(here, lo, hi), "a pitch nobody was holding", 0
