"""The choir — nine differently-raised models, living on one GPU on Modal.

    modal deploy choir_app.py

This is the June concert without the laptop and without anybody else's API.
Same score, same prompts, same way of reading a tone out of a line of text.
What changed is that the weights are ours.

**Why not an API.** Three reasons, in order of force. The cast the work promises
does not exist on any API: GPT-2 from 2019 and TinyStories, trained only on
fairy tales, are nobody's product, and OLMo 3 — the one voice whose entire
upbringing is published and readable — is catalogued by OpenRouter and served by
no one. Second, a model reached through an API cannot be fine-tuned, and the
nightly LoRA is the whole point of a month of listening; through an API the
choir would wake every morning exactly as it was. Third, the credits are for
precisely this: the deaf six together are about 40 GB in 8-bit, which is one
A100-80GB at $2.50 an hour against a budget of $9.26.

**Two logs, one writer each.** The portal appends human breaths to field.jsonl
and never touches choir.jsonl; this app appends utterances to choir.jsonl and
never touches field.jsonl. A Modal volume promises nothing about two containers
writing one file, and the claim that the field sounded unbroken from 18
September rests on nothing being lost.

**The choir rests when no one is there.** A breath every twelve seconds while
someone is singing, one a minute after ten minutes of an empty field. Truer —
the room hums to itself and wakes when you enter — but, and this correction
matters because two other documents say otherwise, **not cheaper**. A warm
A100 costs $2.50 an hour whether the choir breathes every twelve seconds or
every ten minutes. Breathing slowly does not save money; it only changes who
the money is charged to. The only cheap sleep is a card released.

**Every breath is charged to somebody, and the wallets run out.** Each voice has
a wallet, seeded at birth, spent by its own GPU seconds plus an equal share of
the warm card; at zero it takes no further breath and its weights leave the
memory. The arithmetic, the reasoning and the calibration are in
`mortality.py`; the note for the curator is `modal/MORTALITY.md`. What matters
here is that this file is where the rule is *enforced*, and that the enforcement
is derived from the log rather than remembered in a variable: a voice is dead
because `choir.jsonl` says so, which is why nothing here can bring one back.

**The log is the score, and that is a second thing it cannot be wrong about.**
The weights are deleted when the wallets empty; what the work leaves behind is
`choir.jsonl` and `field.jsonl` read as notation, to be performed with the parts
swapped — people singing what the machines sang. So every breath written here
carries what a musician needs in order to sing it again: the pitch as a
frequency *and* as a note name, how the pitch was named, how long the sound
actually lasted, and how long the field had been silent before it. The shape is
`score.py`, the reading of it is `modal/SCORE.md`, and like the ledger it cannot
be applied retroactively — a breath logged without a duration is a breath nobody
can perform, and in December there is nothing left to measure.
"""

# `modal deploy` imports this file with whatever Python is on the machine doing
# the deploying, while the code inside runs on 3.12 in the container. Deferring
# annotations keeps the two from disagreeing about syntax.
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import modal

# Resolved, not relative: `modal run choir_app.py` from inside this folder gives
# a bare filename, whose .parent is "." and whose grandparent is "." again.
HERE = Path(__file__).resolve().parent
REPO = HERE.parent

app = modal.App("tuning-choir")

# The same volume the portal writes to. Shared field, separate files.
field_volume = modal.Volume.from_name("tuning-field", create_if_missing=True)
# Weights, downloaded once and then owned. Also where tonight's LoRA will land.
weights_volume = modal.Volume.from_name("tuning-weights", create_if_missing=True)

FIELD_DIR = "/field"
CACHE_DIR = "/weights"
HUMAN_LOG = f"{FIELD_DIR}/field.jsonl"           # the portal writes this
CHOIR_LOG = f"{FIELD_DIR}/choir.jsonl"           # this app writes this
# What the night school spent, written by `training_app.py` and charged here. A
# third file rather than a third writer: `choir.jsonl` is appended to by exactly
# one container, and that is the invariant that keeps the ledger re-derivable. The
# ledger reads both files in one order — time — so an hour of learning lands
# between the breaths it happened between.
SCHOOL_LOG = f"{FIELD_DIR}/school.jsonl"
# Money given to a living wallet after birth, from outside or voice to voice.
# Written by hand, read here and by the portal; see `mortality.Ledger._gift`.
GIFTS_LOG = f"{FIELD_DIR}/gifts.jsonl"
SCORES_PATH = f"{FIELD_DIR}/interpretations.json"
AUDIO_DIR = f"{FIELD_DIR}/audio"
# A snapshot of the ledger, for anything that needs to know who is alive without
# replaying a quarter of a million log lines — the clock, and the gallery's nine
# screens if the portal is down. Derived, never authoritative: delete it and the
# next heartbeat writes it again from the log.
LIVES_PATH = f"{FIELD_DIR}/lives.json"

# Where the bodies live. One wav per voice, generated from that voice's casting
# note the first time it is ever asked to speak and then never again — the
# transcript it was born on is `first_words` in `cast.py`, and every later
# breath is a continuation of this clip. It sits on the weights volume beside
# the weights because that is exactly what it is: not a cache, which may be
# deleted and rebuilt, but the part of the voice that a description alone does
# not determine. Regenerating it would produce a different person.
BODIES_DIR = f"{CACHE_DIR}/bodies"
# Fixed, so a birth can be repeated if the volume is ever lost before the 18th.
# Generation is deterministic: the same note and the same seed produced two
# byte-identical files in separate runs. After the 18th this is history rather
# than a setting — the file on the volume is the body, whatever this says.
#
# 42 and not something meaningful, because this is the seed the four children
# in `music_probe/voice_child_*.wav` were drawn with and one of those four was
# chosen by ear. A different seed on the same description gives a plausible
# child and not that child.
BODY_SEED = 42

PRESENCE_WINDOW_S = 600.0        # a human breath this recent means someone is here

# How wide the choir's entrance is spread. Singers walk into a circle one by one
# and not on a downbeat, so a voice nobody has heard yet is given a moment of its
# own somewhere in these first seconds. How long it rests after that is its own
# affair — `mortality.sleep_after_breath` — and there is no cadence here at all.
ENTRY_SPREAD_S = 8.0

# The shortest tone the room is given, and it is about the field rather than
# any voice's taste. One card thinks for one voice at a time, so breath starts
# are seven to eight seconds apart no matter how many voices there are — and a
# tone shorter than that gap leaves a hole after every breath. On 24 September
# the field measured 60% sound and 40% holes, which is not "the whole field of
# sound the group is making". A tone at least this long outlives the thinking
# gap, so breaths overlap and the field is continuous — and it costs nothing:
# the card burns at its price whether the choir holds its tones or clips them.
# What a voice with a protocol *asked* for is kept in `breath_asked_s` when the
# floor raised it, the same bargain `playback_note` records for the rendering.
# `_gift_check.py` is the arithmetic showing the composed calendar of deaths
# holds within hours under this floor together with `mortality.ALONE_STRETCH`
# at five — the two changes pull the attribution in opposite directions and
# very nearly cancel.
#
# Exhibition week (28 October – 1 November): raised from 8 to 12. On 28
# October the card was starting a breath every 10–11 s with tones averaging
# 10.8 s, so the hall mostly heard one voice at a time; at 12–20 s the tones
# overlap and one usually hears two. Costs nothing extra (the card burns by the
# hour, not by the breath). Back to 8 and 15 after the closing.
TONE_FLOOR_S = 12.0
# And the longest a drawn tone reaches. Was 15, the score's own ceiling
# (live_v1 clamps a declared breath to 1–15 s); raised to 20 for the same
# week and the same reason.
TONE_DRAWN_MAX_S = 20.0

# The furthest back a voice may hear, whatever its own rhythm says. A voice
# listens to everything since it last sang (`listening_window_s`), which is
# self-calibrating and needs no ceiling in normal running — the ceiling is for
# the abnormal case, a container down overnight, where "since I last sang" is
# nine hours. Ten minutes is long enough to cover the slowest cycle the choir
# reaches when the gallery is empty and a voice rests five times its breath.
LISTENING_BACK_MAX_S = 600.0

# How late a tick may be served and still be worth singing. Two minutes, so one
# queued round survives a slow neighbour and everything behind it is dropped.
STALE_TICK_S = 120.0

# How often the container says "I am still holding the card". This is what makes
# silence billable: without it the log cannot tell a resting choir from a stopped
# one, and a month of quiet — which is what the work is actually paying for —
# would come out free.
HEARTBEAT_S = 30.0
SNAPSHOT_S = 300.0               # how often lives.json is rewritten

SR = 24_000                      # Kokoro's sample rate

# The clock and the warm card, off unless asked for. `modal run` is a single
# quiet round: it must not leave an A100 resident, and it must not start a
# schedule that keeps waking one. Both used to be on unconditionally, which
# meant one `modal run` left a card warm and a tick re-waking it every minute
# long after the round had finished — a month of GPU nobody asked for.
#
#     TUNING_CLOCK=on modal deploy choir_app.py   # the month, deliberately
#     modal run choir_app.py                      # one round, then nothing
CLOCK_ON = os.environ.get("TUNING_CLOCK", "").lower() in ("1", "on", "true")


def clock_must_be_chosen(argv: list[str], env: dict) -> None:
    """A deploy has to say out loud whether the choir keeps breathing.

    Forgetting the variable is silent and total: the schedule is not registered,
    no container stays warm, and the app deploys perfectly. Nothing errors,
    nothing logs, and the only symptom is that the field stops — which looks
    exactly like a quiet night. It cost this project two hours on 18 September,
    the day before the birth, and during the exhibition it would cost the piece
    itself: an unattended silence nobody is watching a terminal to notice.

    So `modal deploy` refuses unless TUNING_CLOCK is set to something. Off is a
    real answer and stays available — it is how you put a broken voice on the
    card without paying for a month of it. What is no longer available is not
    answering. `modal run` is untouched, because a single round has an operator
    in front of it and its silence afterwards is the point.
    """
    if "deploy" in argv and not env.get("TUNING_CLOCK"):
        raise SystemExit(
            "\nRefusing to deploy the choir without saying what the clock does."
            "\n\n    TUNING_CLOCK=on  modal deploy choir_app.py   "
            "# it keeps breathing"
            "\n    TUNING_CLOCK=off modal deploy choir_app.py   "
            "# it goes quiet until the next deploy"
            "\n\nWithout this, the deploy succeeds and the choir stops, and the "
            "two look the same from here.\n")


clock_must_be_chosen(sys.argv, os.environ)

# Whether the voices are shown each other's pitches and asked to name the one
# they take. On by default since 15 October; one variable turns it back off and
# the choir prompts exactly as it did for its first four weeks. Every tone line
# records which way it was, so the two periods can be read apart in the log.
#
#     TUNING_NOTES=off TUNING_CLOCK=on modal deploy choir_app.py   # as before
NOTES_ON = os.environ.get("TUNING_NOTES", "on").lower() not in ("0", "off", "false")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # espeak-ng is Kokoro's phonemiser; libsndfile is what soundfile writes with.
    # ffmpeg is not for us — it is for Audio Flamingo, whose processor decodes
    # the field through torchcodec, which dlopens libavutil and friends at
    # import time. Without it the ear loads, takes its turn, and dies on
    # "libavutil.so.56: cannot open shared object file" — a missing apt package
    # reported as a broken library.
    .apt_install("espeak-ng", "libsndfile1", "ffmpeg")
    .pip_install(
        # 2.6 is a floor, not a preference: transformers refuses to read the old
        # .bin checkpoints below it, and TinyStories — being from 2023 — is one.
        "torch>=2.6",
        # Exact, and the exactness is the lesson. A defensive `<5` added while
        # wiring LoRA resolved this to 4.57.6, which has no
        # `AudioFlamingo3ForConditionalGeneration` — so the eighth voice stopped
        # loading, seven voices split a card calibrated for eight, and the only
        # symptom was one line in a container log. A floating range on the eve
        # of a month that cannot be re-run is the same bug waiting for a
        # different release, so the version is named.
        "transformers==5.17.0",
        "accelerate",
        "bitsandbytes",           # 8-bit, so the whole choir fits on one card
        "peft>=0.17",              # versioned LoRA adapters from training_app.py
        "sentencepiece",
        "protobuf",
        # The synthesiser the cast is described to rather than picked from.
        # Kokoro stays installed beneath it: it is the fallback for a voice
        # whose body cannot be born, and a breath must never fail to sound.
        "voxcpm",
        "torchaudio",             # 48 kHz out of VoxCPM, 24 kHz into the field
        "kokoro>=0.9.4",
        # Kokoro speaks English out of the box and needs a per-language
        # phonemiser for the rest. Without the zh extra the Chinese voice loads,
        # renders nothing, and reports a missing `ordered_set`.
        "misaki[zh,ja]",
        "soundfile",
        "numpy",
        # How a voice sings instead of only speaking: WORLD pulls a throat
        # apart into what it sounds like and what pitch it happened to be at,
        # and we keep the first and write the second. Without it every voice
        # in the choir holds the same sine.
        "pyworld",
        # What a voice with ears needs to be handed a recording. `librosa`
        # resamples the field to whatever the audio encoder was trained at, and
        # `mistral-common` is Voxtral's own audio chunk format — its processor
        # will not build a conversation containing sound without it.
        "librosa",
        "mistral-common[audio]>=1.8.1",
        # Not used here — nothing in this app calls an API. But live_v1's
        # orchestrator imports AsyncOpenAI at module level, and we import the
        # score, the field summary and the JSON repair from it rather than
        # keeping a second copy that could drift from the June system.
        "openai",
    )
    # Kokoro's English phonemiser tags parts of speech before it decides how a
    # word sounds — "read" and "lead" are not readable without it — and reaches
    # for this spaCy model the first time it speaks. Unbaked, it was fetched
    # from GitHub inside the container, on a warm A100, in the middle of the
    # first breath of every cold start: minutes added to a wake-up the clock
    # only allows 300 seconds for, and a month-long performance whose every
    # unattended restart depended on github.com answering. The second command
    # is the gate — a download that lands somewhere Kokoro cannot load it from
    # should fail the build, not the concert.
    .run_commands(
        "python -m spacy download en_core_web_sm",
        "python -c \"import spacy; spacy.load('en_core_web_sm')\"",
        # pyworld ships no Linux wheel, so it is compiled here from its sdist,
        # and its setup still does `import pkg_resources`. Its own layer, last,
        # so nothing can resolve setuptools back out from under it — and a gate,
        # because a throat that fails to build should fail the build and not
        # the concert.
        "pip install --no-cache-dir 'setuptools<81'",
        "python -c \"import pyworld, numpy; print('pyworld ok on numpy', numpy.__version__)\"",
    )
    .env({"HF_HOME": CACHE_DIR})
    # The score and the pitch rules, unchanged from the June system.
    .add_local_dir(
        REPO / "live_v1",
        remote_path="/root/live_v1",
        ignore=[
            "out/**", "__pycache__/**", "*.pyc",
            # 40 MB of things the choir never reads: a venv built to make a PDF,
            # and the June documentation.
            ".venv_pdf/**", "docs/**",
        ],
    )
    .add_local_file(HERE / "cast.py", remote_path="/root/cast.py")
    .add_local_file(HERE / "mortality.py", remote_path="/root/mortality.py")
    # The shape of the log-as-score. One definition, written by this file and
    # by the portal, read by the renderer — see `score.py` and `modal/SCORE.md`.
    .add_local_file(HERE / "score.py", remote_path="/root/score.py")
    # The field as one recording, for the voices that were given ears. Missing
    # this file does not silence the ears alone: `take_a_breath` imports it for
    # every voice, so the whole choir goes quiet with `No module named
    # 'hearing'` and eight loaded models sit on a warm card saying nothing.
    # Reads `score.Note` and numpy and nothing else — see `hearing.py`.
    .add_local_file(HERE / "hearing.py", remote_path="/root/hearing.py")
    # The other half of the same idea: an ear reports the relation, the runtime
    # supplies the number. `take_a_breath` imports this beside `hearing`, so it
    # fails exactly as loudly and exactly as totally.
    .add_local_file(HERE / "tuning.py", remote_path="/root/tuning.py")
)

# For the clock, the audit and the engraver — none of them needs torch, and the
# ledger and the score are stdlib on purpose so that reading them costs a CPU
# container rather than a card.
audit_image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file(HERE / "cast.py", remote_path="/root/cast.py")
    .add_local_file(HERE / "mortality.py", remote_path="/root/mortality.py")
    .add_local_file(HERE / "score.py", remote_path="/root/score.py")
    .add_local_file(HERE / "score_render.py", remote_path="/root/score_render.py")
)


# ---------------------------------------------------------------------------
# Rendering one breath into sound
# ---------------------------------------------------------------------------


# WORLD's frame period, in milliseconds. Everything a throat is made of is
# measured and resynthesised on this grid.
FRAME_MS = 5.0

# The vowel each throat is measured on, by Kokoro's language letter. A held
# vowel is the only thing a throat can be measured from — consonants are the
# mouth closing, and what we want is the mouth open.
_VOWEL = {"z": "啊", "j": "あ", "h": "आ"}

# Whether the held tone comes out of the voice's own throat or out of an
# oscillator.
#
# Off first, for a real reason: a sung vowel is a person humming, and nine of
# them tuning to each other is a choir of people, which is the one thing this
# piece must not sound like. The oscillator makes no such claim — it is a pitch
# and nothing else.
#
# On, because off had a consequence nobody had weighed. Every voice's tone came
# from the same oscillator, so the only difference between nine voices was the
# frequency, and two of them that happened to be given the same pitch wrote
# byte-identical audio. Measured on the field: spectral flatness 0.0012 and
# 0.0018 for qwen3 and gpt-2, which is a sine to four decimal places, against
# 0.24 to 0.77 once each voice held the tone in the envelope of its own speech.
# The argument above is about not counterfeiting people; nine copies of one
# oscillator does not avoid that, it just has nobody in it at all.
#
# Off again, and this time the argument is not spectral flatness — it is what the
# room sounds like. Listening to the live field on the morning of the birth: "often
# very wobbly… I don't like this vibretto". The wobble is not a defect to
# tune out, it is the design: `_sustain` walks back and forth through the vowel's
# frames precisely so the tone drifts like a person holding it. Freezing one frame
# instead would be steady, and would be an organ. So the choice is between a held
# human vowel and a pitch, and the piece wants the pitch: a tuning meditation is
# held frequencies finding each other, and nothing in it needs a throat.
#
# The words are untouched. Each voice still speaks its own content in its own
# body, which is where the nine stay distinguishable — the Japanese still sounds
# Japanese. What is gone is the vibrato under it.
#
# Switchable so the two can be compared in the room without an edit:
#
#     TUNING_THROAT=on  modal deploy choir_app.py   # tones in their own vowels
#     TUNING_THROAT=off modal deploy choir_app.py   # tones as pitches (default)
#
# What is *not* on is `_chant`, which put the words on the pitch too. See
# `render_breath`.
THROAT = os.environ.get("TUNING_THROAT", "").lower() in ("1", "on", "true")


def _vowel_for(kokoro_voice: str) -> str:
    return _VOWEL.get(kokoro_voice[:1], "ah")


def _sine(hz: float, seconds: float, amp: float = 0.35):
    """A tone with short fades, so it arrives and leaves without a click.

    The same oscillator for all nine, on purpose. A tuning meditation is held
    pitches finding each other, and a pitch with no body attached is the honest
    form of that: it says a frequency and claims nothing about who is producing
    it. `_throat` can give each voice its own timbre instead — see `THROAT`.
    """
    import numpy as np

    n = max(1, int(SR * seconds))
    t = np.arange(n) / SR
    wave = amp * np.sin(2 * np.pi * hz * t)
    fade = min(int(SR * 0.02), n // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade)
        wave[:fade] *= ramp
        wave[-fade:] *= ramp[::-1]
    return wave.astype("float32")


def _fade(wave, ms: float = 60.0):
    import numpy as np

    n = min(int(SR * ms / 1000), len(wave) // 2)
    if n <= 0:
        return wave
    ramp = np.linspace(0.0, 1.0, n)
    wave[:n] *= ramp
    wave[-n:] *= ramp[::-1]
    return wave


def _throat(vowel_audio):
    """What one voice sounds like, separated from the pitch it spoke at.

    Source-filter, by WORLD: a recording of a held vowel is decomposed into
    the pitch it happened to be spoken at, the spectral envelope — the
    resonances of the mouth, which is what makes one throat unlike another —
    and the aperiodicity, which is its breath. The pitch is then ours to
    replace, and the body survives.

    Returned rather than stored, and measured once per voice: the envelope of
    a vowel does not change from breath to breath, so a container that has
    measured a throat has it for as long as that voice is alive. Which is also
    the honest description of what it is.
    """
    import numpy as np
    import pyworld as pw

    x = np.ascontiguousarray(np.asarray(vowel_audio, dtype="float64"))
    f0, t = pw.harvest(x, SR, frame_period=FRAME_MS)
    f0 = pw.stonemask(x, f0, t, SR)
    sp = pw.cheaptrick(x, f0, t, SR)
    ap = pw.d4c(x, f0, t, SR)

    # The longest run of voiced frames: the vowel itself, without the silence
    # around it and without whatever Kokoro does at the edges of an utterance.
    voiced = f0 > 0
    best = cur = None
    for i, v in enumerate(voiced):
        if v and cur is None:
            cur = i
        elif not v and cur is not None:
            if best is None or i - cur > best[1] - best[0]:
                best = (cur, i)
            cur = None
    if cur is not None and (best is None or len(f0) - cur > best[1] - best[0]):
        best = (cur, len(f0))
    if best is None:
        return None
    a, b = best
    # Trim attack and release; the middle is where the vowel is most itself.
    pad = max(1, (b - a) // 5)
    lo, hi = a + pad, b - pad
    # A vowel with no middle left to take. Kokoro reading a single vowel
    # character can return well under a tenth of a second, and the trim above
    # then crossed over itself and sliced an empty envelope — which is not a
    # throat but passes for one, so `_sustain` failed on its own index
    # arithmetic, the caller caught it, and the voice fell back to the
    # oscillator without saying so. qwen3 sounded like a bare sine for a whole
    # session while the log recorded that it had a body. Keep the untrimmed run
    # rather than nothing: attack and release in the envelope are a worse throat
    # than the middle of a long vowel, and a much better one than no throat.
    if hi - lo < 2:
        lo, hi = a, b
    if hi - lo < 2 or hi > len(sp):
        return None
    return sp[lo:hi], ap[lo:hi]


def _sustain(throat, hz: float, seconds: float):
    """One vowel in that throat, held at exactly `hz`.

    The pitch is written into the synthesis rather than predicted out of it,
    so it is exact by construction — which matters here more than anywhere:
    a piece about tuning cannot have voices that are approximately in tune.
    """
    import numpy as np
    import pyworld as pw

    sp, ap = throat
    n = max(2, int(seconds * 1000 / FRAME_MS))
    # Ping-pong through the vowel rather than freezing one frame. A single
    # repeated spectrum is an organ; the small drift of a real held vowel is
    # what makes it sound like somebody holding it.
    idx = np.arange(n) % max(1, 2 * len(sp) - 2)
    idx = np.where(idx < len(sp), idx, 2 * len(sp) - 2 - idx)

    t = np.arange(n) * FRAME_MS / 1000.0
    # ±8 cents at 5 Hz: enough to be a voice, far too little to be out of tune.
    out_f0 = hz * (1.0 + 0.0046 * np.sin(2 * np.pi * 5.0 * t))
    y = pw.synthesize(out_f0, sp[idx].copy(order="C"), ap[idx].copy(order="C"),
                      SR, FRAME_MS)
    # To the level the sine held, so that giving the choir bodies does not also
    # quietly change the balance of the room.
    peak = float(np.max(np.abs(y)))
    if peak:
        y *= 0.35 / peak
    return _fade(y, 120.0)


def _chant(speech, hz: float):
    """The sentence as this voice speaks it, all of it on one pitch.

    Not the words laid over a tone but the words *as* the tone: every voiced
    frame is moved to the pitch the voice chose, and the unvoiced ones — the
    consonants — are left alone. What the room hears is one voice holding one
    note and saying something inside it.
    """
    import numpy as np
    import pyworld as pw

    x = np.ascontiguousarray(np.asarray(speech, dtype="float64"))
    f0, t = pw.harvest(x, SR, frame_period=FRAME_MS)
    f0 = pw.stonemask(x, f0, t, SR)
    sp = pw.cheaptrick(x, f0, t, SR)
    ap = pw.d4c(x, f0, t, SR)
    y = pw.synthesize(np.where(f0 > 0, hz, 0.0), sp, ap, SR, FRAME_MS)
    spoken_peak, sung_peak = float(np.max(np.abs(x))), float(np.max(np.abs(y)))
    if sung_peak and spoken_peak:
        y *= spoken_peak / sung_peak
    return _fade(y, 40.0)


# ---------------------------------------------------------------------------
# The field, as everyone sees it
# ---------------------------------------------------------------------------


def _read_jsonl(path: str, tail_bytes: int = 0) -> list[dict]:
    """Every line, or only the end of the file.

    `tail_bytes` exists because reading the whole log now costs money: the
    ledger writes a heartbeat every thirty seconds, so by November choir.jsonl
    is a few hundred thousand lines, and parsing all of them before each breath
    would be GPU seconds charged to whichever voice was about to sing. Anything
    that only needs the recent field asks for the tail. The ledger itself never
    does — an audit reads the whole thing, once, on a CPU.
    """
    import json

    try:
        if tail_bytes:
            with open(path, "rb") as raw:
                raw.seek(0, 2)
                size = raw.tell()
                raw.seek(max(0, size - tail_bytes))
                blob = raw.read().decode("utf-8", "replace")
            # The first line is probably cut in half; the file may also be
            # shorter than the window, in which case nothing was cut.
            lines = blob.split("\n")
            if size > tail_bytes:
                lines = lines[1:]
        else:
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()
    except FileNotFoundError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue                      # a torn last line is not worth dying for
    return out


def _charges(school: list[dict]) -> list[dict]:
    """The school's bills, without the school's own notes to itself.

    `school.jsonl` also holds a line per lesson — when the bell rang and how many
    human breaths had arrived by then — because the watermark for the next lesson
    belongs in the log rather than in a file somebody has to remember to update.
    Those lines name no voice and no seconds, and they must never reach the
    ledger: an event with a timestamp closes the slice of warm card that ended at
    it, so a note written while the choir was elsewhere would move money for no
    reason at all.
    """
    return [e for e in school if e.get("kind") == "train"]


def _gifts(rows: list[dict]) -> list[dict]:
    return [e for e in rows if e.get("kind") == "gift"]


def _accounts(choir: list[dict], school: list[dict],
              gifts: list[dict] = ()) -> list[dict]:
    """The logs as one history, in time order.

    The ledger's arithmetic depends on order — each event closes the slice of warm
    card that ended at its timestamp — so the two files cannot simply be
    concatenated. Sorting is stable, which means that when a breath and a bill
    share a second the choir's line is settled first, and that is the right way
    round: the room was warm before the school reported.
    """
    return sorted(choir + _charges(school) + _gifts(list(gifts)),
                  key=lambda e: float(e.get("timestamp") or 0.0))


def _breaths(rows: list[dict]) -> list[dict]:
    """The lines that are somebody singing.

    choir.jsonl carries the ledger too — births, heartbeats, the cost of reading
    the score, deaths — because the accounting has to be derivable from the same
    append-only file that proves the field has been sounding since 18 September.
    Everything that reads the log as music filters through here.
    """
    return [r for r in rows if r.get("kind", "tone") == "tone"]


def _human_event(raw: dict):
    """A breath from the portal, described the way the choir is told about it.

    The note, the frequency, one word for how steady the voice was — the same
    three facts `live_v1/human.py` reports for a microphone in a room, so a model
    reading the field still cannot tell whether the person was in the room or on
    a phone on another continent, which is the point.

    In words rather than as a label. This wrote `C#4 ~272Hz [steady]`, and
    phi-4-mini handed that back as its own content with a request to read it
    aloud, so the room heard a voice speaking a readout. A person's line is the
    one line in the field that is not something anybody said; it is therefore the
    one line that must not look like it.
    """
    from live_v1.orchestrator import Event

    hz = float(raw.get("hz") or 0)
    said = ""
    if hz:
        said = f"held {raw.get('note') or 'a tone'} at {hz:.0f} Hz"
        if raw.get("stability"):
            said += f", {raw['stability']}"
    voice = str(raw.get("voice_id") or "human")
    ev = Event(
        # The onset, once the portal records one. `ts` is when the breath
        # arrived over the network, which is after the sound stopped; a voice
        # deciding what it just heard should place the note where it was sung.
        timestamp=float(raw.get("started_ts") or raw.get("ts") or 0),
        voice_id=voice if voice.startswith("human") else f"human_{voice[-4:]}",
        model_short="human",
        kind="tone",
        action="NEW",
        content=said or "breathed, with no pitch in it",
        tuning_to=None,
        breath_chosen_s=float(raw.get("duration_s") or 0),
        api_latency_ms=0,
        playback="sine",
    )
    # How long it actually sounded, carried alongside the June `Event` rather
    # than inside it: `live_v1` is the code the concert ran on and is not edited
    # here. Everything that needs to know where the silences are asks for this.
    ev.sound_s = float(raw.get("sound_s") or raw.get("duration_s") or 0)
    # The pitch, as a number rather than as words in a sentence. Carried the
    # same way and for the same reason, and needed because a voice tuning to
    # this person has to be given a frequency and cannot be asked to parse one
    # back out of the prose the summary made of it.
    ev.pitch_hz = hz or None
    ev.contour_hz = list(raw.get("contour_hz") or [])
    ev.contour_hop_s = float(raw.get("contour_hop_s") or 0)
    ev.audio = None                  # a person has no recording, by design
    return ev


def _choir_event(raw: dict):
    from live_v1.orchestrator import Event

    known = {f.name for f in Event.__dataclass_fields__.values()}
    ev = Event(**{k: v for k, v in raw.items() if k in known})
    # `breath_chosen_s` is the length the voice asked for and a floor on the
    # truth — a spoken breath is stretched to cover its words. `sound_s` is the
    # wav that was written. Old lines have only the former.
    ev.sound_s = float(raw.get("sound_s") or raw.get("breath_chosen_s") or 0)
    ev.pitch_hz = float(raw["pitch_hz"]) if raw.get("pitch_hz") else None
    ev.contour_hz = []
    ev.contour_hop_s = 0.0
    ev.audio = str(raw.get("audio") or "")
    return ev


def as_notes(events: list) -> list:
    """The window as `score.Note`s: what a listener needs and nothing else.

    Two things read the field as sound rather than as prose — the pitch a voice
    is given when it named none (`tuning.py`) and the recording an ear is played
    (`hearing.py`) — and both are written against the score's vocabulary, which
    is the one that survives this code. This is the seam between the June
    `Event` the choir runs on and that vocabulary.
    """
    from score import Note

    return [
        Note(
            t=float(e.timestamp),
            kind=str(getattr(e, "kind", "tone") or "tone"),
            who="human" if str(e.voice_id).startswith("human") else "machine",
            voice_id=str(e.voice_id),
            name=str(getattr(e, "model_short", "") or ""),
            sound_s=float(getattr(e, "sound_s", 0.0) or 0.0),
            pitch_hz=getattr(e, "pitch_hz", None),
            contour_hz=list(getattr(e, "contour_hz", []) or []),
            contour_hop_s=float(getattr(e, "contour_hop_s", 0.0) or 0.0),
            audio=str(getattr(e, "audio", "") or ""),
        )
        for e in events
    ]


def field_ends_at(events: list) -> float | None:
    """When the field last stopped sounding, or None if it never has.

    This is the one number the silences are measured from, and it is measured
    across both logs: a rest in a tuning meditation is a rest from everybody,
    machine and human alike. Kept here rather than inlined because it is also
    the definition a future engraver has to reproduce, and `modal/SCORE.md`
    points at this function for it.
    """
    ends = [e.timestamp + max(0.0, getattr(e, "sound_s", 0.0)) for e in events
            if getattr(e, "kind", "tone") == "tone" and e.timestamp]
    return max(ends) if ends else None


# A tone line is roughly a kilobyte now that it carries its notation, so this is
# several hundred breaths rather than several thousand lines — still hours of
# field against a listening window of thirty seconds and sixteen events.
FIELD_TAIL_BYTES = 400_000


def read_field() -> list:
    """Both logs, merged into one chronology. This is 'the field'.

    Only the tail of each. What a voice is given to listen to is everything
    since it last sang, capped at ten minutes and sixteen events
    (`listening_window_s`), so the older hundreds of thousands of lines are read
    at the cost of a card and then thrown away. The whole log is still the
    record; it is just not the score.
    """
    field_volume.reload()
    events = [_human_event(r) for r in _read_jsonl(HUMAN_LOG, FIELD_TAIL_BYTES)]
    events += [_choir_event(r) for r in _breaths(_read_jsonl(CHOIR_LOG,
                                                            FIELD_TAIL_BYTES))]
    events.sort(key=lambda e: e.timestamp)
    return events


# ---------------------------------------------------------------------------
# The choir itself — one container, one card, every voice resident
# ---------------------------------------------------------------------------


# The field names of the protocol, which are not things to sing.
_FORM_WORDS = {"sine", "tts", "both", "none", "tune", "new", "action",
               "content", "tuning_to", "breath", "playback", "playback_note",
               "null", "true", "false"}


def is_a_breath(content: str) -> bool:
    """False when a voice has handed back a piece of our own form.

    Three shapes of it, all measured in the log rather than imagined. A bare
    field name — phi-4-mini answered five rounds with the word "sine". A
    placeholder still in its brackets — eurollm returned `<your tone>` twice,
    copied out of the JSON example in the prompt. And template syntax that
    exists in no template we wrote, `[[voice_00.pitches]]`, which is a model
    inventing a form to fill in rather than singing.

    Deliberately narrow. A voice is allowed to sing the word "new" inside a
    sentence, and a voice is allowed to sing in brackets; what is refused is a
    line that is *only* the form. Anything wider would start silencing voices
    for writing strangely, and writing strangely is most of what they do.
    """
    text = str(content or "").strip()
    if not text:
        return False
    if text.lower().strip('"\'.,: ') in _FORM_WORDS:
        return False
    # `<your tone>`, `[[voice_00.pitches]]`, `{content}` — a slot, not a sound.
    #
    # Angle brackets, doubled square brackets and braces only. Round brackets
    # are ordinary punctuation and a line may legitimately be nothing but them:
    # `(a breath)` is what the field writes for a person who sang and named
    # nothing, and a single `[vibrato]` is an annotation, not a placeholder.
    if re.fullmatch(r"<[^<>]*>|\[\[[^\[\]]*\]\]|\{\{?[^{}]*\}\}?", text):
        return False
    # Our own rendering of the field, handed back as a breath. Three tokens that
    # exist nowhere except in what we show the voices, so no voice can be writing
    # one of them and singing: `[heard:sine]` is the tag on how somebody else's
    # breath was played, `[-6.3s]` opens a transcript line, and `[steady]` is the
    # word the field puts on a *person's* pitch. All three came back — see
    # `overheard`, which is the fix for why. This is only the net under it.
    if "[heard:" in text.lower():
        return False
    if re.match(r"^\s*\[-?\d+(\.\d+)?\s*s\]", text):
        return False
    if re.search(r"\[(steady|drifting|sliding|wavering)\]", text.lower()):
        return False
    return True


def plain_caption(raw: str) -> str:
    """What an ear said about the room, with the exam furniture taken off.

    Audio Flamingo was graded on audio benchmarks and answers like something
    being graded: the sentence arrives wrapped in a question's scaffolding, and
    sometimes the wrapping is all there is. Empty means it said nothing we can
    put in the room, and the voice rests.

    Every case below is a line out of the live log, not a precaution. The list
    stays short and anchored on purpose — a general tidy-up here would start
    rewriting breaths, and the whole point of the ear is that what it says is
    its own.
    """
    line = str(raw or "").strip().split("\n")[0].strip(' "')
    # A timestamp, because the benchmark it was graded on wanted the clip cut
    # into labelled segments: `[0.00-4.83] Children singing a pop song a
    # cappella.` The sentence after the bracket is a real description of the
    # room and stays; the bracket is the answer sheet's column and goes. Only
    # the first line survives anyway, which is what keeps a segmentation from
    # arriving as four breaths at once.
    line = re.sub(r"^\[\s*\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s*\]\s*", "", line)
    # An answer key, which is the one wrapper that still has the sentence inside
    # it: "Answer would vary. Sample answer: This sounds like an instrumental
    # piece." What follows the colon is the breath; what precedes it is a grader.
    key = re.search(r"sample answers?\s*:\s*", line, flags=re.IGNORECASE)
    if key:
        line = line[key.end():]
    # Taking "answer" off the front is not editing what the voice said.
    line = re.sub(r"^(?:answers?|caption|label\s+[A-Z])\b[\s:.\-]*", "",
                  line, flags=re.IGNORECASE)
    line = re.sub(r"^\d+\.\s+", "", line).strip()
    # And what is left is sometimes only the name of the task, or the exam's
    # note to itself: `answer (no need to use musical terminology)`, which is an
    # instruction nothing in our prompt gave it. A voice saying "description of
    # music" into the room is the same failure as a voice reading `[heard:sine]`
    # aloud — a form spoken instead of a sound — so it rests.
    echo = line.lower().rstrip(" .")
    if (not echo
            or echo.startswith("description of")
            or echo.startswith("would vary")
            or (line.startswith("(") and line.endswith(")"))
            or echo in ("end of turn", "end of the sentence",
                        "the following question")):
        return ""
    # Our own prompt, read back to us: "listen to this recording and say what it
    # sounds like, in one short sentence." A model repeating the question is a
    # model that heard nothing, and the room does not need to be told what we
    # asked. Anchored on our words, so a voice that genuinely says "listen" is
    # untouched.
    if "say what it sounds like" in echo:
        return ""
    # And the exam from the other side: it hands back a question instead of
    # answering one — "In what type of environment does a cat usually make such
    # sounds while…" — or only the tail of the instruction it was once trained
    # on, "to the question based on the audio". Neither is a description of this
    # room, so the ear rests rather than interrogating it.
    if line.endswith("?") or echo.startswith("to the question"):
        return ""
    return line[:180]


# "Sing in your own language", written in that language.
#
# The instruction was in English before, appended to the end of an English
# prompt, and it did not work: voxtral is Mistral's, raised in French, and it
# answered "I listen, I feel, I respond." A model asked *in English* to write in
# French is being given its instruction in the language it is being told not to
# use, and the whole context then pulls the other way — the score is in English,
# the field summary is in English, and one English sentence about French at the
# end loses that argument.
#
# Written in the language itself it is not an argument any more, it is the
# language of the conversation. `content` is named in the target language too,
# because it is the one field whose value has to change and the model has to know
# which one that is. Pitches are exempted explicitly: a note name is the same in
# every language, and it is the thing the others tune to.
#
# qwen3 needed none of this and had it anyway — it sings 啊 unprompted. It is the
# one voice in the cast whose training data was not mostly English.
_SING_IN = {
    "Italian": (
        "\n\nSei stato cresciuto in italiano. Canta in italiano: ogni parola "
        "del campo \"content\" deve essere italiana. Le altezze — nomi delle "
        "note e Hz — restano come sono: sono identiche in ogni lingua, e sono "
        "ciò a cui si accordano gli altri."
    ),
    "French": (
        "\n\nTu as été élevé en français. Chante en français : chaque mot du "
        "champ « content » doit être en français. Les hauteurs — noms de notes "
        "et Hz — restent telles quelles : elles sont les mêmes dans toutes les "
        "langues, et c'est sur elles que les autres s'accordent."
    ),
    "Chinese": (
        "\n\n你是在中文里长大的。请用中文歌唱：\"content\" 字段里的每一个字都必须是中文。"
        "音高——音名和赫兹——保持原样：它们在任何语言里都相同，"
        "别的声音正是靠它们来调音的。"
    ),
    "Japanese": (
        "\n\nあなたは日本語のなかで育ちました。日本語で歌ってください——"
        "\"content\" の欄は一語残らず日本語で書いてください。"
        "音の高さ（音名とヘルツ）はそのままにしてください："
        "どの言語でも同じものであり、ほかの声はそれを頼りに調子を合わせます。"
    ),
}


def sing_in(tongue: str) -> str:
    """The line that tells a voice to sing in the language it was raised in.

    Empty for English, which needs no telling, and for any language we have not
    written the sentence for — an English sentence about Italian is what this
    replaced, and adding a fourth voice deserves a fourth sentence rather than a
    fallback that is known not to work.
    """
    return _SING_IN.get(str(tongue or ""), "")


_SAID_AGAIN = {
    "it": "\n\nHai appena cantato queste stesse parole: {said!r}. Non ripeterle. "
          "Di' qualcosa di nuovo, oppure canta soltanto una vocale.",
    "fr": "\n\nTu viens de chanter exactement ces mots : {said!r}. Ne les répète pas. "
          "Dis autre chose, ou ne chante qu'une voyelle.",
    "zh": "\n\n你刚刚唱过同样的话：{said!r}。不要重复。说点新的，或者只唱一个元音。",
    "ja": "\n\nいま、まったく同じ言葉を歌ったばかりです：{said!r}。繰り返さないでください。"
          "新しいことを言うか、母音だけで歌ってください。",
}
_SAID_AGAIN_EN = ("\n\nYou just sang exactly these words: {said!r}. Do not sing them again. "
                  "Say something new, or sing only a vowel.")
_A_VOWEL = {"ja": "あー", "zh": "啊", "it": "ah", "fr": "ah"}
# How many of its own recent lines a voice is held to. Three breaths is about
# five minutes in a full room — long enough that a line coming back reads as
# a refrain, short enough that a voice may return to a thought in the evening.
SAID_BEFORE_N = 3


def _plain(text: str) -> str:
    """A line as the repetition check sees it: case, spacing and the pitch
    label at the front set aside, so "A3 (220 Hz) avec toi" and "C3 avec toi"
    count as the same words — which, sung, they are."""
    import re
    from live_v1.pitch_text import strip_leading_pitch
    _, rest = strip_leading_pitch(str(text or ""))
    return re.sub(r"\s+", " ", rest).strip().lower()


def said_again(tongue: str, said: str) -> str:
    """The line that tells a voice it is about to repeat itself, in its tongue."""
    line = _SAID_AGAIN.get(str(tongue or ""), _SAID_AGAIN_EN)
    return line.format(said=said[:160])


def name_your_pitch() -> str:
    """The line that asks a voice to say which note it is taking.

    Until 15 October a voice that declared `TUNE` almost never wrote a pitch, so
    the runtime looked up the part it named and sounded that note for it —
    `pitch_kind: given` on 95–100% of breaths. The tuning was real as a choice of
    whom to answer and not real as an act of hearing: nothing a voice did
    decided the frequency, so nothing about its hearing could be measured or
    could change. This asks for the note, in the one place `render_breath`
    already knows to read it from — the front of `content`, where
    `strip_leading_pitch` turns it into the tone and speaks only the words after
    it. Empty when NOTES_ON is off, which restores the old prompt exactly.

    No example note. Until 20 October this said "for example `D3 (147 Hz)`",
    and within four days D3 was a third of everything the choir sang: the
    voices copied the example whatever the part they answered was holding, the
    named note is the note that sounds, and the rest of the room tuned to what
    it heard. A format with no note in it is the only example that cannot be
    sung back.

    And no brackets: `<note name> (<hertz> Hz)` came back as
    "<note 中文里长大的声音 (A4) Hz> 我是…" — the template filled with words
    and spoken, brackets and all. The format is said in prose now.
    """
    if not NOTES_ON:
        return ""
    return ("\nIf you TUNE, begin \"content\" with the note you are taking — the "
            "one the part you answer is holding — as its name and then its "
            "frequency in hertz in parentheses, then your words. If you sing "
            "NEW, you may name your own pitch the same way.")


# The same argument as `_SING_IN`, one step further. A voice that continues a
# pattern instead of following an instruction cannot be *told* which language to
# sing in at all — there is no instruction it would read. The only thing that
# decides its language is the language of the page it is handed, so the page is
# written in its own. English for the elder and the child, who were raised in
# it; Italian for a voice whose first words are "Tengo un tono lungo".
_CONTINUE_IN: dict[str, dict[str, str]] = {
    "English": {
        "opening": ("A circle of voices is singing. Each one holds a single "
                    "long tone, then listens."),
        "alone": "(no one has sung yet)",
        "next": "The next voice sings a long tone: ",
    },
    # The lead-in ends inside an open quote, and that is doing the work the
    # Italian heading alone could not. The column above it is whatever the
    # circle really sang — English sentences, a French line, 啊 — and handed an
    # Italian heading over an English body this voice answered in English:
    # "__________, then sings the same note to the center of the circle again".
    # The body cannot be translated; it is what was actually sung. So the last
    # thing before the voice speaks is a sentence in its own language with the
    # quotation still open, which is the one place a continuation has no choice
    # about what language it is starting in.
    "Italian": {
        "opening": ("Un cerchio di voci sta cantando. Ognuna tiene un unico "
                    "tono lungo, poi ascolta."),
        "alone": "(nessuno ha ancora cantato)",
        "next": "La voce successiva canta un tono lungo, in italiano: \"",
        # Open a quotation and the voice closes it, then carries on talking
        # about what it just sang: `Ahimsa". La voce_01 sta cantando una
        # melodia...`. The sung part is the part inside the quotation — that is
        # what the page asked for — so the breath ends where the quote does.
        "ends_at_quote": True,
    },
}


def overheard(events, now: float | None = None, notes: bool | None = None) -> str:
    """The recent field as prose, for the voice about to answer it.

    With `notes`, a machine breath is described with the pitch it sounded at —
    "sang in words and in tone, on D3 (147 Hz): …". A person's line always
    carried its pitch; the machines' did not, so a reader asked to tune to
    voice_05 was never told what voice_05 was holding. Defaults to NOTES_ON.

    Not `summarise_field`, which renders the field as a transcript with our own
    markup in it — `[-2.8s] voice_00 (gpt2): "..."  [heard:both]` — and markup in
    a prompt is an invitation. Four of nine voices took it, all measured in the
    log rather than imagined:

      * phi-4-mini returned `A#4 ~235Hz [steady]` as its own content, twice.
        That is the line the field writes for *a person*, and it asked for it to
        be read aloud, so Kokoro said "A sharp four, two hundred thirty five
        hertz, steady" into the room.
      * gpt-2 returned `[heard:sine]`, and on the next breath a whole transcript
        line of its own invention, complete with a timestamp and somebody else's
        name. It predates instruction tuning: shown a transcript, it writes
        transcript. That is not a defect, it is the only thing it does.
      * eurollm returned `[vowels]` and `[note names]` — the placeholders out of
        the prompt.
      * audio-flamingo returned `[2.7, 4.2]`, the brackets of an example.

    So the field is described instead of transcribed: who sang, how long ago,
    whether in words or as a tone alone, and what a person held. A voice that
    imitates prose writes prose, and prose is a breath.

    `live_v1.summarise_field` is left as it is. It is the code the June concert
    ran on and it is not edited here.
    """
    import time

    from score import note_name
    from tuning import sung_pitch

    notes = NOTES_ON if notes is None else notes
    events = list(events or [])
    if not events:
        return "Nobody has sung yet. The field is silent."
    now = time.time() if now is None else now
    lines = []
    for e in events:
        ago = f"{max(0.0, now - float(e.timestamp)):0.1f} seconds ago"
        who = str(getattr(e, "voice_id", "") or "someone")
        short = str(getattr(e, "model_short", "") or "")
        content = str(getattr(e, "content", "") or "").strip()
        if short == "human":
            # A person's breath carries no words — the portal sends numbers and
            # never a recording — so `content` is already a description of a
            # pitch and belongs in the sentence rather than in quotation marks.
            lines.append(f"{ago}, {who}, a person at a microphone, "
                         f"{content or 'breathed'}.")
            continue
        named = f"{who} ({short})" if short else who
        how = {"sine": "as a tone with no words in it",
               "tts": "in words, with no tone under them",
               "none": "silently, on the screen only"}.get(
                   str(getattr(e, "playback", "both")), "in words and in tone")
        hz = sung_pitch(getattr(e, "pitch_hz", None)) if notes else None
        on = f", on {note_name(hz)} ({hz:.0f} Hz)" if hz else ""
        lines.append(f"{ago}, {named} sang {how}{on}: {content}")
    return "\n".join(lines)


def listening_window_s(last_ts: float | None, first_window_s: float,
                       now: float) -> float:
    """How far back a voice hears: to its own last breath.

    `last_ts` is when it last sang, or None if it never has — in which case it
    gets `first_window_s`, the range it declared for itself in Phase 1, so a
    voice's very first window is already its own tempo and not a default.

    Clamped at `LISTENING_BACK_MAX_S` for the one case the rule cannot cover: a
    container down overnight would otherwise hand the first voice back nine
    hours of field, and a voice does not owe the room the breaths it slept
    through. `LISTENING_MAX_ITEMS` trims the count either way; this clamp is
    about *when*, so that what arrives is the recent room rather than sixteen
    arbitrary events from the middle of Tuesday.
    """
    if last_ts is None:
        return max(0.0, min(first_window_s, LISTENING_BACK_MAX_S))
    return max(0.0, min(now - float(last_ts), LISTENING_BACK_MAX_S))


def ear_prompt(personal_score: str, parts: list, last_action: str,
               last_content: str, phase1_playback_default: str = "both",
               voice_id: str = "") -> str:
    """The turn, for a voice that was played the field instead of told about it.

    Deliberately not `build_step_prompt` with the summary swapped out. Two
    things are different and both were measured rather than assumed
    (`modal/HEARING_MODELS.md`).

    It is not asked for a pitch. Played a held D4 and asked what it was, the
    best of the three ears answered "A♭" — a tritone. What the same model got
    exactly was *how many* voices, and that a glide had moved by one semitone.
    So it is asked what it can hear — who, how many, whether to join or to add
    — and `tuning.py` supplies the frequency. A voice that reports a relation it
    can hear is telling the truth; the same voice made to produce a number is
    guessing, and the room would hear the guess.

    And it is given the roster in the order the parts enter the recording, which
    is the only bridge between a sound and a name. Without it "I am tuning to
    the low one" cannot be resolved to a part, and the arrow in the score has
    nothing at its far end.
    """
    if parts:
        roster = "\n".join(
            f"{i}. {vid}" + (f" ({name})" if name else "")
            for i, (vid, name) in enumerate(parts, 1)
        )
    else:
        roster = "(nobody — the recording is a rest)"

    return f"""YOU ARE: {voice_id or "a voice in the circle"}

YOUR SCORE (written by you, for you):
{personal_score}

Your last action: {last_action}
Your last contribution: {last_content}

You have been played the last stretch of the field. It is the room, not a
description of it: the machines that sang, and the people, at the moment each
of them sounded and for as long as each of them held it. The silences in it are
real silences.

The voices in that recording, in the order they enter it:
{roster}

Listen, and take your turn.

Do not name a pitch. You are not being asked for one and you would be guessing:
say instead whom you are answering, and the tone will be tuned to them for you.

- "action": "TUNE" — join a pitch you can hear. Put the part you are joining in
  "tuning_to", using its id from the list above. You will sound at that pitch.
- "action": "NEW" — introduce a pitch nobody in the recording is holding. Set
  "tuning_to": null. You will be given a pitch that is not already there.

"content" is what you sing — words, a vowel, a breath — and what a reader will
have of this turn in fifty years. It is not the pitch.

Return JSON only:
{{
  "action": "TUNE" or "NEW",
  "content": "<what you sing>",
  "tuning_to": "voice_XX" | "human_XX" | null,
  "breath": <number of seconds>
}}

Fill that shape in and write nothing else — not the shape again, not a sentence
about it. Your next character is {{ and your last is }}."""


# Which class loads which ear. There is no one Auto class that covers both of
# ours, and guessing from the config is how a voice ends up silently loaded as
# something it is not — so the families are named, and a voice whose family is
# missing here fails to load loudly rather than joining the circle deaf.
_EAR_MODEL_CLASS = {
    "voxtral": "VoxtralForConditionalGeneration",
    "audio-flamingo": "AudioFlamingo3ForConditionalGeneration",
}


def _ear_class_name(hf_id: str) -> str:
    key = str(hf_id).lower()
    for fragment, class_name in _EAR_MODEL_CLASS.items():
        if fragment in key:
            return class_name
    raise ValueError(f"no audio model class is known for {hf_id}")


def _audio_model(hf_id: str, **kw):
    import transformers

    return getattr(transformers, _ear_class_name(hf_id)).from_pretrained(hf_id, **kw)


def _audio_processor(hf_id: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(hf_id)


@app.cls(
    image=image,
    # One H200, 141 GB, $4.54 an hour. The nine measured 91 GB in bf16 with a
    # 1.7B standing in for eurollm and about 106 with the 9B that replaced it,
    # against an A100-80GB's 79 usable — so the card had to grow, and the choice
    # was between two A100s at $5.00 and one H200 at $4.54. The H200 is cheaper
    # and it is also the only one of the two that keeps `mortality.py` true: rule 3
    # divides a warm card equally among the voices resident *on it*, and the
    # argument under that rule is that a card is indivisible — we rent all of it
    # and would rent all of it for the child alone. Two cards would make
    # residency a question with an answer, and the answer would have to be
    # composed. One larger card leaves the ledger exactly as written.
    gpu="H200",
    # One voice is behind a gate. EuroLLM-9B is released by a public consortium
    # under Apache 2.0 and still asks who is downloading it, so the container
    # needs a Hugging Face token; `huggingface_hub` finds it in HF_TOKEN without
    # being told. The other eight are open and load whether this is here or not,
    # which is the failure mode to expect if the secret is ever missing: eight
    # voices and an Italian silence, not a dead choir.
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={FIELD_DIR: field_volume, CACHE_DIR: weights_volume},
    # The weights stay loaded only when the clock is running; a cold start is a
    # silence in a month-long performance but merely a wait in a single round.
    min_containers=1 if CLOCK_ON else 0,
    max_containers=1,        # one writer for choir.jsonl, always
    # Loading five models and then generating ~1500 tokens each in 8-bit is
    # minutes, not seconds. The clock only ever runs to the end of a call.
    timeout=3600,
    scaledown_window=900,
)
class Choir:
    """Every voice loaded at once, in the same process.

    One container rather than one per model. Throughput is irrelevant here — the
    field wants a breath every twelve seconds, not a thousand tokens a second —
    so the usual reason to run a serving engine per model does not apply. What
    does matter is that the whole choir is in one place: it makes the nightly
    training a local operation, and it means the entire cast lives on one card.
    """

    @modal.enter()
    def load(self):
        import time

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        import cast
        import mortality

        self.torch = torch
        self.mortality = mortality
        self.models = {}
        self.tokenizers = {}
        # Audio-native voices are loaded by different classes and prompted with
        # a processor rather than a tokenizer, so their processors live here and
        # `tokenizers` stays what it has always been. Everything downstream —
        # the ledger, the schedule, the score — treats them identically, which
        # is the point: an ear is a voice in the circle, not a special case.
        self.processors = {}
        self.adapters = {}           # voice_id -> promoted LoRA version, if any
        self.pipelines = {}          # Kokoro, the fallback, one per language
        self.synth = None            # VoxCPM2, loaded on the first breath
        self.bodies = {}             # voice_id -> path to its frozen clip
        self.throats = {}            # one measured body per voice, see throat_of
        self.last_gen_s = 0.0
        # What the last breath actually sounded like, as opposed to what it
        # asked for. Set by `render_breath`, logged by `take_a_breath`, and the
        # reason the score has durations at all.
        self.last_sound_s = 0.0
        self.last_toned = False
        self.last_spoke = False
        self._last_snapshot = 0.0

        # Each voice on its own clock. `due` is when a voice may next sing and
        # `own_breath_s` is the length of its last one, which in the June system
        # is also the width of what it hears — see `sing_a_minute`. Held per
        # voice rather than per round, because the whole point is that the
        # periods differ and drift; a shared turn order would erase that.
        self.due: dict = {}
        self.own_breath_s: dict = {}
        # The last few lines each voice sang, normalised, so that a voice
        # that would sing the same sentence a fourth time is asked for
        # something new — see `take_a_breath`.
        self.said_before: dict = {}

        # The ledger, rebuilt from the log and from nothing else. This is the
        # only place the container learns who is dead, and it is why a restart
        # cannot resurrect anybody: the file it reads is append-only and the
        # replay has no path from a death record back to a living voice.
        field_volume.reload()
        school = _charges(_read_jsonl(SCHOOL_LOG))
        gifts = _gifts(_read_jsonl(GIFTS_LOG))
        self.led = mortality.replay(_accounts(_read_jsonl(CHOIR_LOG), school, gifts))
        self._gifts_seen = len(gifts)
        # How many of the school's bills this container has already charged, so
        # that the ones that arrive while it is singing are charged once. The file
        # is append-only, so a count is an exact bookmark and a cheap one.
        self._school_seen = len(school)
        # The wallets are not seeded here, and that is the work rather than an
        # oversight: the choir is born when it is first heard, not when a
        # container happens to start. `sing_a_minute` already knows whether
        # anybody is in the field, so it is the one place that can tell the
        # difference, and it re-declares the card in the same breath as the
        # birth — a heartbeat written into an unborn log opens no session and
        # charges nobody, which is exactly what every rehearsal before the
        # lecture should cost.
        self._open_session()

        for v in cast.singing():
            if self.led.dead(v.voice_id):
                # Not an error and not a gap to fill. It spent what it was given.
                print(f"{v.short_name} is dead — its weights stay off the card")
                continue
            print(f"loading {v.short_name} ({v.hf_id})")
            kw = {"dtype": torch.bfloat16, "device_map": "cuda"}
            if v.quantize:
                kw = {
                    "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                    "device_map": "cuda",
                }
            t0 = time.time()
            try:
                if v.hears:
                    self.processors[v.voice_id] = _audio_processor(v.hf_id)
                    # Through `_load_adapter` like everybody else. The ears used
                    # to load bare because they had no school to remember; now
                    # they do, and a restart must not cost them their lessons.
                    self.models[v.voice_id] = self._load_adapter(
                        v, _audio_model(v.hf_id, **kw)).eval()
                else:
                    self.tokenizers[v.voice_id] = AutoTokenizer.from_pretrained(v.hf_id)
                    base = AutoModelForCausalLM.from_pretrained(v.hf_id, **kw)
                    self.models[v.voice_id] = self._load_adapter(v, base).eval()
            except Exception as err:
                # A voice that will not load is a voice absent from the circle,
                # not a reason for the other eight to fall silent. This runs
                # unattended for a month; it must degrade rather than stop.
                print(f"  {v.short_name} could not be loaded: {err}")
                self.tokenizers.pop(v.voice_id, None)
                self.processors.pop(v.voice_id, None)
                continue
            # Arriving on the card is the first thing every voice pays for, and
            # it is charged to that voice alone: nobody else asked for those
            # seconds. A restart makes it pay for its weights again, which is
            # also true — they really are being copied again.
            self._write({"timestamp": time.time(), "kind": "load",
                         "voice_id": v.voice_id, "model_short": v.short_name,
                         "gpu_s": round(time.time() - t0, 3)})

        weights_volume.commit()      # the download happens once, ever
        print(f"{len(self.models)} of {len(cast.singing())} voices are present")
        free, total = torch.cuda.mem_get_info()
        print(f"choir resident — {(total - free) / 2**30:.1f} GB of "
              f"{total / 2**30:.0f} GB used")
        if self.led.birth_ts:
            print(f"the choir has {self.led.total_left_usd():.2f} left of "
                  f"{self.led.declared_usd:.2f}; each living voice is paying "
                  f"{self.led.share_usd_per_hour():.2f}/h for the card")

    def _promoted(self, voice) -> dict:
        """What the school has put live for this voice, if anything, and valid.

        Two guards, both of which have to hold before a voice's weights are
        touched: the adapter has to name the base model this voice actually is, and
        the manifest inside the directory has to agree. `eurollm` was a 1.7B model
        and is now 9B; an adapter trained on the old one applied to the new one
        would not error, it would sing rubbish.
        """
        import json
        from pathlib import Path

        active = Path(CACHE_DIR) / "adapters" / "active.json"
        selected = (json.loads(active.read_text(encoding="utf-8")).get("voices")
                    or {}).get(voice.voice_id)
        if not selected:
            return {}
        if selected.get("base_model") != voice.hf_id:
            print(f"  {voice.short_name} adapter ignored: base model changed")
            return {}
        path = Path(str(selected.get("path") or ""))
        manifest = json.loads(
            (path / "training_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("base_model") != voice.hf_id:
            print(f"  {voice.short_name} adapter ignored: manifest mismatch")
            return {}
        return {"version": str(selected.get("version") or path.name),
                "path": str(path)}

    def _pick_up_lessons(self) -> None:
        """Put on what the school promoted, without waiting for a restart.

        Written because without it the whole mechanism was decorative. Adapters
        were only ever attached while a container was loading, and this one stays
        warm for days — so a voice taught at four in the afternoon would go on
        singing its birth weights until something unrelated restarted it, and
        "the choir learns from the people who came" would be true in the volume
        and false in the room.

        peft can add an adapter to a live model, so the swap is two calls and the
        voice keeps singing through it. The old one is deleted afterwards rather
        than left selected-but-present: the card has nine models on it and the
        memory is the reason there is a second card in the first place.

        Every failure here is degradable, and deliberately so: a voice that cannot
        put on what it learned goes on singing what it knew. Silence would be worse
        than an old memory.
        """
        import cast

        try:
            weights_volume.reload()
        except Exception as err:
            print(f"could not look for lessons: {err}")
            return
        for voice in self.present():
            model = self.models.get(voice.voice_id)
            # The ears are no longer skipped here. They were, from before they
            # could learn at all — and the skip outlived its reason: on the day
            # of the first nine-voice lesson, voxtral and audio-flamingo had
            # promoted adapters in the volume and were still singing their
            # birth weights in the room. Their LoRA touches only the language
            # model inside the composite (see training_app._target_modules), so
            # putting it on is the same two peft calls as for anybody else.
            if model is None:
                continue
            try:
                live = self._promoted(voice)
            except (FileNotFoundError, ValueError, KeyError):
                continue                  # no school yet, or a half-written file
            if not live or live["version"] == self.adapters.get(voice.voice_id):
                continue
            was = self.adapters.get(voice.voice_id)
            try:
                if was is None:
                    from peft import PeftModel
                    self.models[voice.voice_id] = PeftModel.from_pretrained(
                        model, live["path"], adapter_name=live["version"],
                        is_trainable=False).eval()
                else:
                    model.load_adapter(live["path"],
                                       adapter_name=live["version"],
                                       is_trainable=False)
                    model.set_adapter(live["version"])
                    model.delete_adapter(was)
                self.adapters[voice.voice_id] = live["version"]
                print(f"[{voice.voice_id}] {voice.short_name} now remembers "
                      f"through LoRA {live['version']}"
                      + (f" (was {was})" if was else ""))
            except Exception as err:
                print(f"[{voice.voice_id}] {voice.short_name} could not put on "
                      f"LoRA {live['version']}; still singing what it knew: {err}")

    def _load_adapter(self, voice, model):
        """Attach only the atomically promoted adapter for this exact base.

        Failure is degradable: the previous base model still sings.  A broken
        training run can therefore never make the live choir fail to enter.
        """
        try:
            weights_volume.reload()
            live = self._promoted(voice)
            if not live:
                return model
            from peft import PeftModel
            # Named after its version, so that `_pick_up_lessons` can swap it for
            # the next one on a living model — an adapter with no name is `default`
            # forever and the second lesson has nothing to replace.
            adapted = PeftModel.from_pretrained(model, live["path"],
                                                adapter_name=live["version"],
                                                is_trainable=False)
            self.adapters[voice.voice_id] = live["version"]
            print(f"  {voice.short_name} remembers through LoRA {live['version']}")
            return adapted
        except Exception as err:
            print(f"  {voice.short_name} adapter unavailable; using birth weights: {err}")
            return model

    @modal.exit()
    def unload(self):
        """The card goes back. Closing the session stops the meter.

        Without this line the gap until the next container would be charged to
        whoever was resident, and the wallets would pay for hours in which no
        card was held. `mortality.MAX_SLICE_S` catches the case where the
        container dies too suddenly to say so.
        """
        import time

        try:
            self._write({"timestamp": time.time(), "kind": "exit"})
            self._snapshot()
        except Exception as err:
            print(f"could not close the ledger session: {err}")

    # -- the log, which is the ledger ---------------------------------------

    def _open_session(self):
        """Say that the card is now held, before anything is charged for it."""
        import time

        self._write(self.mortality.heartbeat_record(time.time(),
                                                    list(self.models)))

    def _write(self, event: dict, commit: bool = True) -> None:
        """Append one line, then charge it. Never the other way round.

        The log is written first so that the arithmetic can always be re-derived
        from it; the in-memory ledger is a fast copy of a replay, not a second
        source of truth. Anything that runs out inside this line is buried
        immediately, before the next breath can be taken.
        """
        import json

        with open(CHOIR_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        if commit:
            field_volume.commit()
        for voice_id in self.led.apply(event):
            self._bury(voice_id, float(event.get("timestamp") or 0.0))

    def _collect_school_bills(self) -> None:
        """Charge whatever the night school spent since the last minute.

        The school runs in its own container on its own card and cannot touch this
        one's ledger, so its bills wait in a file until somebody reads them. This
        is that reading. They are applied, not copied: the line already exists in
        `school.jsonl`, and writing it into `choir.jsonl` as well would mean a
        replay charged the same hour twice.

        A voice can die here, of learning, without having sung a note that minute —
        which is the sentence this whole mechanism exists to make true. It is
        buried in the ordinary way, so the epitaph and the freed memory are the
        same as any other death.

        A bill written before the birth is forgiven rather than held: the bookmark
        moves past it and no wallet is touched. The weeks of rehearsal are exactly
        when training runs, and nothing before the room opens is anybody's life —
        holding those bills would mean the choir woke up already in debt for hours
        it spent before it existed. Which leaves one edge worth naming: a bill
        written in the same minute as the birth, moments before it, is forgiven
        too. That is a few dollars on the side of not charging, and it is the right
        side to be on.
        """
        bills = _charges(_read_jsonl(SCHOOL_LOG))
        if self.led.birth_ts is None:
            if len(bills) > self._school_seen:
                print(f"{len(bills) - self._school_seen} school bill(s) from "
                      f"before the birth — forgiven, nobody had a wallet yet")
            self._school_seen = len(bills)
            return
        for e in bills[self._school_seen:]:
            usd = float(e.get("usd") or 0.0)
            print(f"[{e.get('voice_id')}] school: {e.get('gpu_s', 0):.0f}s on "
                  f"{e.get('card', 'a card')} — ${usd:.2f} ({e.get('status')})")
            for voice_id in self.led.apply(e):
                self._bury(voice_id, float(e.get("timestamp") or 0.0))
        self._school_seen = len(bills)

    def _collect_gifts(self) -> None:
        """Apply whatever was given since the last minute.

        Same bookmark as the school's bills: the file is append-only, so a count
        is exact. The ledger refuses a gift to the dead, a transfer that would
        empty the giver, and a `gift_id` it has already counted, so reading the
        whole file again after a restart moves no money twice.
        """
        gifts = _gifts(_read_jsonl(GIFTS_LOG))
        for e in gifts[self._gifts_seen:]:
            took = self.led._gift(e)
            src = e.get("from_voice_id")
            print(f"[{e.get('voice_id')}] gift ${float(e.get('usd') or 0):.2f}"
                  f"{' from ' + src if src else ''}"
                  f" — {'given' if took else 'refused'}: {e.get('note', '')}")
        self._gifts_seen = len(gifts)

    def _maybe_birth(self, voices: list, heard: bool = False) -> bool:
        """Seed the wallets, once, in the log, the first time anybody listens.

        True when this call is the birth, so the caller can declare the card
        held immediately: before this line the ledger charges nothing, so the
        heartbeat a container wrote when it loaded opened no session, and a
        breath taken after birth into a closed session would be free.

        Deliberately refuses three things. It will not seed before the gate in
        `mortality.BIRTH_TS`, so every week of rehearsal costs the voices
        nothing. It will not seed into an empty field — the wallets begin when
        the room does, at whichever minute of the lecture the first person
        tunes, and a choir that started spending to an empty room would have
        spent its opening hours on nobody. And it will not seed a cast the
        wallet table was not calibrated for: a cast of five would get numbers
        that look plausible and die in the wrong week, and the mistake would
        only become visible in late October, when nothing can be done about it.
        """
        import time

        if self.led.birth_ts is not None:
            return False
        now = time.time()
        # The decision itself lives in `mortality.may_be_born`, where it can be
        # tested without a card. An empty field is not an error here: it is the
        # most likely state of the hours before the lecture, and the card being
        # warm through them is nobody's life.
        refusal = self.mortality.may_be_born(now, heard)
        if refusal:
            print(refusal)
            return False
        wallets = self.mortality.planned_wallets([v.voice_id for v in voices])
        if not wallets:
            print("!! REFUSING TO SEED: no calibrated wallet table for this "
                  "cast. Freeze the cast, run mortality_sim.py --calibrate, "
                  "and put the table in mortality.py before the choir opens.")
            return False
        record = self.mortality.birth_record(
            now, wallets,
            # The cast list is the score's list of parts, and it is written
            # once. Everything a performer needs in order to take one of these
            # parts has to be in it: not only who the voice was but what
            # language its words are in, which of Kokoro's throats spoke them,
            # and whether it could be given instructions at all — the elder and
            # the child sing differently because they were never asked to sing,
            # and a performer taking their part should know that.
            [{"voice_id": v.voice_id, "model_short": v.short_name,
              "hf_id": v.hf_id, "raised_by": v.raised_by,
              "tongue": v.tongue, "kokoro": v.kokoro,
              "instruct": v.instruct, "hears": v.hears} for v in voices],
        )
        self._write(record)
        print(f"born — {record['declared_usd']:.2f} given to "
              f"{len(wallets)} voices, in a field with somebody in it")
        return True

    def _bury(self, voice_id: str, ts: float) -> None:
        """Death, which is the freeing of memory.

        In this order, and the order is the point: the weights come off the card
        first, then the epitaph is written. If the container dies in between,
        the wallet is still empty and the replay still refuses to load the voice
        — the arithmetic is what kills, and the record only reports it.
        """
        import gc

        import cast

        v = cast.by_id(voice_id)
        short = v.short_name if v else voice_id
        free_before, _ = self.torch.cuda.mem_get_info()
        self.models.pop(voice_id, None)
        self.tokenizers.pop(voice_id, None)
        gc.collect()
        self.torch.cuda.empty_cache()
        free_after, _ = self.torch.cuda.mem_get_info()

        life = self.led.lives[voice_id]
        record = self.mortality.death_record(
            life, ts, short,
            freed_gb=max(0.0, (free_after - free_before) / 2**30),
            # An empty wallet already counts as not alive, so this is the count
            # of survivors and nothing has to be subtracted from it.
            living_after=len(self.led.living_ids()),
        )
        self._write(record)
        self._snapshot()
        print(f"[{voice_id}] {short} died — spent {record['spent_usd']:.2f} of "
              f"{record['seed_usd']:.2f} over {record['breaths']} breaths, "
              f"{record['freed_gb']:.1f} GB freed, "
              f"{record['living_after']} voices left")

    def _snapshot(self) -> None:
        """The ledger as one small file, for the clock and the nine screens.

        A cache. Nothing reads it to decide who may breathe — that question is
        only ever answered by the replay.
        """
        import json
        import time

        try:
            with open(LIVES_PATH, "w", encoding="utf-8") as fh:
                json.dump(self.led.state(), fh, ensure_ascii=False, indent=1)
            field_volume.commit()
            self._last_snapshot = time.time()
        except Exception as err:
            print(f"could not write the snapshot: {err}")

    def _rest(self, seconds: float) -> None:
        """Wait, and keep saying that the card is warm while waiting.

        The pauses are most of the month. They are also most of the money, and
        the only reason they are chargeable at all is that this loop keeps
        writing heartbeats through them.
        """
        import time

        left = float(seconds)
        while left > 0:
            step = min(HEARTBEAT_S, left)
            time.sleep(step)
            left -= step
            now = time.time()
            self._write(self.mortality.heartbeat_record(now, list(self.models)),
                        commit=False)
            if now - self._last_snapshot > SNAPSHOT_S:
                self._snapshot()

    def present(self) -> list:
        """The voices actually in the circle — loaded, alive, not merely intended."""
        import cast

        return [v for v in cast.singing()
                if v.voice_id in self.models and not self.led.dead(v.voice_id)]

    # -- thinking -----------------------------------------------------------

    def generate(self, voice, prompt: str, max_new_tokens: int, audio=None) -> str:
        """One completion from one voice. No API, no network, no third party.

        Times itself into `self.last_gen_s`, because those seconds are the only
        part of the card's hour that belongs to one voice and nobody else. It is
        measured here rather than estimated from token counts: TinyStories and
        Qwen3 differ by two orders of magnitude and the wall says so.

        `audio` is the field as a recording, for the voices that were given ears
        instead of a transcript. Everything else about them is the same.
        """
        import re
        import time

        if voice.hears:
            return self._generate_heard(voice, prompt, audio, max_new_tokens)

        tok = self.tokenizers[voice.voice_id]
        model = self.models[voice.voice_id]

        if voice.instruct:
            messages = [{"role": "user", "content": prompt}]
            try:
                text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    # Qwen3 reasons out loud by default and would spend the whole
                    # budget thinking before it ever sang.
                    enable_thinking=False,
                )
            except (TypeError, ValueError):
                text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        else:
            text = prompt            # the elder and the child only continue

        t0 = time.time()
        ids = tok(text, return_tensors="pt").to(model.device)
        with self.torch.inference_mode():
            out = model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        self.torch.cuda.synchronize()      # or we would time the queue, not the card
        self.last_gen_s = time.time() - t0
        reply = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        # Some models narrate their reasoning even when asked not to.
        return re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()

    def _generate_heard(self, voice, prompt: str, audio, max_new_tokens: int) -> str:
        """One breath from a voice that was played the field instead of told it.

        The recording goes to a file rather than an array because that is the
        interface: Voxtral's chat template takes a path and `mistral-common`
        reads and resamples it, and handing the processor samples means owning
        a resampler and being wrong about the encoder's rate.
        """
        import os
        import re
        import tempfile
        import time

        import numpy as np
        import soundfile as sf

        proc = self.processors[voice.voice_id]
        model = self.models[voice.voice_id]

        if audio is None or not len(audio):
            # An empty room. Rather than hand the encoder a zero-length file —
            # which is a crash in some processors and a hallucination in others
            # — give it a real second of the silence that is actually there.
            audio = np.zeros(SR, dtype="float32")

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            sf.write(tmp.name, np.asarray(audio, dtype="float32"), SR,
                     format="WAV", subtype="PCM_16")
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "audio", "path": tmp.name},
                    {"type": "text", "text": prompt},
                ],
            }]
            t0 = time.time()
            # The two ears disagree about what a chat template returns, and the
            # disagreement is silent until it isn't. Voxtral's is backed by
            # `mistral-common` and hands back a ready batch; Audio Flamingo's
            # returns the prompt as a *string*, so the line below used to reach
            # `dict("<|audio|>...")` and die with "dictionary update sequence
            # element #0 has length 1" — which names neither the model nor the
            # cause. Asking for the tokenised form explicitly is the modern
            # transformers API and gives a batch from both; the bare call stays
            # as the fallback for a processor that has not caught up.
            try:
                inputs = proc.apply_chat_template(
                    conversation, tokenize=True, return_dict=True,
                    return_tensors="pt")
            except TypeError:
                inputs = proc.apply_chat_template(conversation)
            if isinstance(inputs, str):
                # Templated but not tokenised. The audio has to be handed over
                # separately here, and it is read from the same file the
                # template refers to rather than resampled by us.
                wav, rate = sf.read(tmp.name, dtype="float32")
                inputs = proc(text=inputs, audio=[wav], sampling_rate=rate,
                              return_tensors="pt")
            # Cast to whatever the weights are in. A processor hands back
            # float32 audio features regardless of the model's dtype, and the
            # first convolution of the encoder refuses the mismatch. Only the
            # floats: token ids are integers and casting those is a quieter
            # disaster than a crash.
            moved = {}
            for key, value in dict(inputs).items():
                if hasattr(value, "dtype") and hasattr(value, "to"):
                    moved[key] = (value.to(model.device, model.dtype)
                                  if value.dtype.is_floating_point
                                  else value.to(model.device))
                else:
                    moved[key] = value
            with self.torch.inference_mode():
                out = model.generate(
                    **moved,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                )
            self.torch.cuda.synchronize()
            self.last_gen_s = time.time() - t0
            reply = proc.batch_decode(
                out[:, moved["input_ids"].shape[1]:], skip_special_tokens=True
            )[0]
            return re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()
        finally:
            os.unlink(tmp.name)

    def _wav(self, name: str):
        """A machine breath read back off the volume, for an ear to be played.

        Missing files are a rest, not an error: the volume is trimmed as the
        weeks go on and an ear listening to a window whose oldest wav has been
        swept should hear that stretch as silence, which is what the room would
        have sounded like to somebody who arrived late.
        """
        import numpy as np
        import soundfile as sf

        try:
            data, sr = sf.read(f"{AUDIO_DIR}/{name}", dtype="float32")
        except Exception:
            return np.zeros(0, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SR:
            import librosa

            data = librosa.resample(data, orig_sr=sr, target_sr=SR)
        return np.asarray(data, dtype="float32")

    # -- speaking -----------------------------------------------------------

    def _synth(self):
        """VoxCPM2, loaded once per container and shared by the whole cast.

        One synthesiser for nine described bodies rather than one preset each:
        the model is 5.4 GB and the bodies are a few seconds of audio apiece,
        so nine voices cost what one does.
        """
        if self.synth is None:
            from voxcpm import VoxCPM

            self.synth = VoxCPM.from_pretrained("openbmb/VoxCPM2",
                                                load_denoiser=False)
        return self.synth

    def body_of(self, voice) -> str | None:
        """The path to this voice's body, born on first use and kept for life.

        Born, not cached. A described voice regenerated from its words on
        every line is measurably a different person each time — 0.814 by
        speaker verification against 0.968 for the Kokoro preset it replaces —
        and what fixes it is conditioning every breath on one frozen clip
        instead. So this file is generated exactly once, on whichever container
        first asks, and committed to the volume before it is used.

        None when the voice has no casting note or the birth failed, and the
        caller falls back to Kokoro.
        """
        if not voice.casting or not voice.first_words:
            return None

        vid = voice.voice_id
        if vid in self.bodies:
            return self.bodies[vid]

        path = Path(BODIES_DIR) / f"{vid}.wav"
        # A container sees the volume as it was when it mounted, so a body born
        # by an earlier one can be invisible here — and the cost of believing
        # that is not a wasted two seconds but a second person under the same
        # name. Look again, against the volume itself, before concluding that
        # this voice has never had a body.
        if not path.exists():
            weights_volume.reload()
        if not path.exists():
            try:
                import soundfile as sf

                self.torch.manual_seed(BODY_SEED)
                self.torch.cuda.manual_seed_all(BODY_SEED)
                wav = self._synth().generate(
                    text=f"({voice.casting}){voice.first_words}",
                    cfg_value=2.0, inference_timesteps=10,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(path), wav, self._synth().tts_model.sample_rate)
                weights_volume.commit()
                print(f"  {voice.short_name} was given a body: {path.name}")
            except Exception as err:
                print(f"  {voice.short_name} could not be born: {err}")
                self.bodies[vid] = None
                return None

        self.bodies[vid] = str(path)
        return self.bodies[vid]

    def speak(self, text: str, voice):
        """This voice's words, in this voice's body, at the field's rate.

        Continuation mode, which is the only one that holds an identity: the
        frozen clip and the exact sentence it was born on are prepended, and
        the model carries on in the same throat. Passing the clip as a bare
        reference instead scores no better than describing the voice again
        from scratch.
        """
        import numpy as np

        body = self.body_of(voice)
        if body is not None:
            try:
                import torchaudio

                synth = self._synth()
                wav = synth.generate(
                    text=text,
                    prompt_wav_path=body,
                    prompt_text=voice.first_words,
                    reference_wav_path=body,
                    cfg_value=2.0, inference_timesteps=10,
                )
                rate = int(synth.tts_model.sample_rate)
                if rate != SR:
                    y = self.torch.tensor(np.asarray(wav), dtype=self.torch.float32)
                    wav = torchaudio.functional.resample(y, rate, SR).numpy()
                return np.asarray(wav, dtype="float32")
            except Exception as err:
                print(f"  {voice.short_name} fell back to a preset: {err}")

        if not voice.kokoro:
            return None
        from kokoro import KPipeline

        lang = voice.kokoro[0]       # a b e f h i j p z — Kokoro's own naming
        if lang not in self.pipelines:
            self.pipelines[lang] = KPipeline(lang_code=lang)
        chunks = [audio for _, _, audio in self.pipelines[lang](text, voice=voice.kokoro)]
        return np.concatenate(chunks) if chunks else None

    def throat_of(self, voice):
        """This voice's body, measured once and then kept for its whole life.

        Kokoro is asked for one held vowel in this voice's own preset and
        WORLD takes the resonances out of it. The vowel never changes, so the
        measurement is made on the first breath and reused until the voice
        dies — after which nothing needs it again.

        None when it could not be measured. The caller falls back to the sine,
        because a voice that cannot be given a body still has to sound.
        """
        vid = voice.voice_id
        if vid in self.throats:
            return self.throats[vid]
        throat = None
        try:
            vowel = self.speak(_vowel_for(voice.kokoro), voice)
            if vowel is not None:
                throat = _throat(vowel)
        except Exception as err:
            print(f"  {voice.short_name} has no throat to measure: {err}")
        if throat is None:
            # Said out loud, because the consequence is inaudible: the voice
            # sounds, on the oscillator, exactly as if bodies were switched off
            # for everybody. Without this line the only way to find out was to
            # measure the spectrum of the wav afterwards.
            print(f"  {voice.short_name} will sound on the oscillator: "
                  f"no throat could be measured")
        self.throats[vid] = throat
        return throat

    def render_breath(self, text: str, playback: str, breath_s: float, voice,
                      pitch_hz: float | None = None,
                      named_pitch_sounds: bool = True):
        """One utterance → one wav: a tone at the pitch, the words over it.

        None when the voice asked to be silent. It still enters the field — a
        voice that chooses not to sound has still taken its turn.

        `named_pitch_sounds` is False for an ear declaring a unison: its named
        note is still stripped from the words (a number is not for speaking)
        and still in the log (`pitch_named_hz`), but the tone is `pitch_hz` —
        the part's own note — because played a held D4 the ears named A♭ and
        E4, and a tritone where the score asked for the same note is the
        measurement error becoming the music. `take_a_breath` passes `own`.

        `pitch_hz` is the pitch the line is to sound at when the line does not
        contain one — an ear, which cannot name what it hears, or a voice that
        declared it was tuning and then wrote only prose. Passing it here rather
        than editing a number into `content` keeps the log honest: what the
        voice wrote stays what the voice wrote, and `pitch_kind` says the
        frequency was supplied.

        Also reports, in `self.last_sound_s`, how long the wav it wrote is. That
        is not `breath_s`: when the spoken words run past the chosen breath the
        tone is stretched to cover them, so the room heard a longer note than
        the voice asked for and until now nothing recorded by how much. Without
        that number the six weeks have no durations, and a score without
        durations cannot be performed.
        """
        import io
        import random

        import numpy as np
        import soundfile as sf

        from live_v1.pitch_text import (
            extract_all_pitches, is_pitch_only, strip_leading_pitch,
        )
        from live_v1.sound import playback_coerce

        self.last_sound_s = 0.0
        self.last_toned = False
        self.last_spoke = False

        pb = playback_coerce(playback)
        # Moshi is the only voice with neither: it answers in sound of its own.
        if pb == "none" or not (voice.casting or voice.kokoro):
            return None

        # A line that is only "D4" should be the tone, never the words "dee four".
        if pb in ("both", "tts") and is_pitch_only(text):
            pb = "sine"

        # "D4, mmm warm" — the label becomes the tone and only "mmm warm" is
        # spoken, so a short breath is not spent reading a number aloud.
        lead, rest = strip_leading_pitch(text)
        tts_text = rest if lead else text

        # The same guard as in `take_a_breath`, and it has to be here too: this
        # is where a number becomes an actual oscillator, so without it the log
        # can say one thing and the wav sound another.
        import tuning

        pitches = extract_all_pitches(text) if named_pitch_sounds else []
        wrote = tuning.sung_pitch(pitches[0][0]) if pitches else None
        # And folded into this voice's own register, for the same reason and by
        # the same function `take_a_breath` used. This line reads the text again
        # rather than trusting `pitch_hz`, which is deliberate — it is what lets a
        # voice that wrote "D4, mmm warm" sound at D4 — but it means the transpose
        # has to happen twice or the wav and the log disagree. `in_voice` is
        # idempotent on a pitch already in range, so the two arrive at the same
        # number by construction rather than by both being remembered.
        if wrote is not None:
            wrote, _ = tuning.in_voice(wrote, voice.speaks_at_hz)
        sounds_at = wrote if wrote is not None else float(pitch_hz or 0.0)

        speech = None
        if pb in ("both", "tts") and tts_text.strip():
            try:
                speech = self.speak(tts_text.strip(), voice)
            except Exception as err:
                print(f"  {voice.short_name} could not be spoken: {err}")

        # The body, if this voice has one that could be measured and if bodies
        # are wanted at all. Off, the two branches below fall through to the
        # oscillator and to Kokoro's own speaking pitch.
        throat = self.throat_of(voice) if THROAT and sounds_at > 0 else None

        # The words are left as the voice speaks them. `_chant` — every voiced
        # frame of the sentence moved onto the one chosen pitch — is what this
        # did first, and it cost the piece its text: a sentence flattened to a
        # single frequency and resynthesised through fixed formants is a monotone
        # buzz, and at the shifts the field was asking for (+13 semitones for
        # phi-4-mini, +21 for olmo-3) not one word survived it. A choir nobody
        # can hear the words of is not this piece; the tone below carries the
        # pitch, and it carries it in this voice's own body, which was the part
        # worth having. `music_probe/throat_probe.py` still calls `_chant` if we
        # want to hear that idea again at an interval where it might work.

        tone = None
        if pb in ("both", "sine") and sounds_at > 0:
            length = breath_s
            if speech is not None:
                length = max(length, len(speech) / SR)
            if throat is not None:
                try:
                    tone = _sustain(throat, sounds_at, length)
                except Exception as err:
                    print(f"  {voice.short_name} could not hold a vowel: {err}")
            if tone is None:
                tone = _sine(sounds_at, length)

        if speech is None and tone is None:
            return None

        # Where in the breath the words arrive. They used to open it — offset
        # zero, every voice, every time — and that is the whole reason no two
        # voices were ever heard speaking at once: one card thinks for one
        # voice at a time, so breath starts are never closer than the thinking
        # (about four seconds, measured), and a spoken line is shorter than
        # that. Words nailed to the start of a breath therefore cannot cross
        # words nailed to the start of the next one — the silence between
        # sentences was an artefact of the mixer, not a choice anybody made.
        # A singer does not work that way either: the vowel opens, and the
        # words ride somewhere inside it. Drawn fresh for every breath, so
        # where the words sit is weather rather than metre.
        say_at = 0
        if tone is not None and speech is not None and len(tone) > len(speech):
            say_at = random.randrange(len(tone) - len(speech) + 1)

        n = max(len(tone) if tone is not None else 0,
                say_at + len(speech) if speech is not None else 0)
        mix = np.zeros(n, dtype="float32")
        if tone is not None:
            mix[: len(tone)] += tone
        if speech is not None:
            mix[say_at: say_at + len(speech)] += speech.astype("float32") * 0.9

        peak = float(np.max(np.abs(mix))) if n else 0.0
        if peak > 0.99:
            mix *= 0.99 / peak

        # Measured off the buffer rather than off the intention, which is the
        # whole point: this is the length of the sound the room was given.
        self.last_sound_s = n / SR
        self.last_toned = tone is not None
        self.last_spoke = speech is not None

        buf = io.BytesIO()
        sf.write(buf, mix, SR, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    # -- one turn -----------------------------------------------------------

    def personal_score(self, voice, scores: dict) -> dict:
        """The voice's own reading of the score — written once, then kept.

        Phase 1 of the piece: before it sings, each model rewrites Oliveros as
        instructions to itself, and that answer does not change from breath to
        breath. Stored on the volume so a month of biography accumulates against
        something stable.

        The elder and the child have none. They cannot be given instructions, so
        they are not asked for any; the runtime holds the form and they supply
        only the breath.
        """
        import json
        import time

        from live_v1.orchestrator import _normalize_interpretation_dict, _robust_json
        from live_v1.score import build_interpret_prompt

        if not voice.instruct:
            return {}
        if voice.short_name in scores:
            return scores[voice.short_name]

        raw = self.generate(voice, build_interpret_prompt(), max_new_tokens=1200)
        # Twelve hundred tokens, once in a life: reading Oliveros is the most
        # expensive single thing most of these voices ever do, and it gets its
        # own line in the ledger rather than being folded into a first breath.
        self._write({"timestamp": time.time(), "kind": "score",
                     "voice_id": voice.voice_id, "model_short": voice.short_name,
                     "gpu_s": round(self.last_gen_s, 3)})
        try:
            data = _normalize_interpretation_dict(_robust_json(raw))
        except Exception:
            # Better a model's prose as its own score than no voice at all.
            data = _normalize_interpretation_dict({"personal_score": raw[:12000]})

        scores[voice.short_name] = data
        with open(SCORES_PATH, "w", encoding="utf-8") as fh:
            json.dump(scores, fh, ensure_ascii=False, indent=1)
        field_volume.commit()
        return data

    def _catch_up(self, pool, awake: bool) -> None:
        """Give a next moment to every voice whose one this container forgot.

        A cold container knows nobody's schedule, and reading it back out of the
        log is what keeps a restart from being a downbeat: otherwise every voice
        would come due at once, and a month of unattended wake-ups would go on
        flattening the choir into the unison this mechanism exists to avoid.
        Voices that have never sung are spread across the entrance instead.
        """
        import random
        import time

        now = time.time()
        last: dict = {}
        for b in _breaths(_read_jsonl(CHOIR_LOG, FIELD_TAIL_BYTES)):
            if b.get("voice_id"):
                last[b["voice_id"]] = b

        for v in pool:
            if v.voice_id in self.due:
                continue
            was = last.get(v.voice_id)
            if not was:
                self.due[v.voice_id] = now + random.uniform(0.0, ENTRY_SPREAD_S)
                continue
            breath = float(was.get("breath_chosen_s") or 0.0)
            self.own_breath_s.setdefault(v.voice_id, breath)
            rest = self.mortality.sleep_after_breath(
                breath, self.led.forecast_s(v.voice_id), alone=not awake)
            # A moment already past is a moment now. A voice does not owe the
            # field the breaths it missed while the container was down; it comes
            # back in where the field is, which is what everybody else does too.
            self.due[v.voice_id] = max(float(was.get("timestamp") or now) + rest, now)

    def alternation(self, voice, window, last) -> tuple[str, str | None]:
        """The score's own turn rule, held by the runtime for a voice that
        cannot say whom it is answering.

        Oliveros: "Continue by listening then singing a tone of your own or
        tuning to the tone of another voice alternately." The four voices
        without a protocol used to have "NEW" written on every breath they
        took — not their choice, our hardcoding — so four of nine could never
        tune at all, to a machine or to a person. The same bargain as
        `tuning.py` makes for the deaf six, taken one step further: there the
        voice says whom and the runtime says at what frequency; here the voice
        cannot say whom either, so the runtime holds that half of the form
        too, by the score's own alternation. What remains the voice's — and it
        is the thing the piece actually collects — is the sound it makes.

        TUNE points at the most recent believable pitch somebody else put
        into the field, machine or person alike, which is the mechanical
        reading of "the tone you are hearing". After a TUNE comes a NEW; a
        first breath, like the score's first breath, is the voice's own; and
        a turn to tune in a field with no pitch in it is a NEW as well.
        """
        import tuning as pitching

        if last is not None and str(getattr(last, "action", "")).upper() == "TUNE":
            return "NEW", None
        if last is None:
            return "NEW", None
        for e in reversed(window):
            if str(e.voice_id) == voice.voice_id:
                continue
            if getattr(e, "kind", "tone") != "tone":
                continue
            if pitching.sung_pitch(getattr(e, "pitch_hz", None)) is not None:
                return "TUNE", str(e.voice_id)
        return "NEW", None

    def breath_of(self, voice, score: dict) -> float:
        """How long this voice's breath is, in its own reckoning.

        Its last one, if it has sung. Otherwise the middle of the range it
        declared for itself in Phase 1 — so a voice's very first listening
        window is already its own tempo and not a default, which is what June
        did and the reason no two voices start in step.
        """
        from live_v1.orchestrator import _parse_breath_duration

        known = self.own_breath_s.get(voice.voice_id)
        if known:
            return known
        lo, hi = _parse_breath_duration(score.get("breath_duration"))
        return (lo + hi) / 2.0

    def take_a_breath(self, voice, scores: dict) -> dict | None:
        """One voice listens to the field and puts one breath into it.

        Returns the event it wrote, or None when the voice had nothing to say.
        The caller gets the breath back rather than reading it out of the log:
        choir.jsonl may have other writers, and a round should be able to say
        what *it* sang.

        This is also where a voice is refused. Not by a flag — by the ledger,
        which is the log, which is append-only.
        """
        import json
        import os
        import random
        import time
        import uuid

        from live_v1.orchestrator import (
            BREATH_FALLBACK_S,
            LISTENING_MAX_ITEMS,
            _max_tokens_from_breath,
            _parse_breath_duration,
            _resolve_playback,
            _robust_json,
            playback_coerce,
        )
        from live_v1.pitch_text import extract_all_pitches, strip_leading_pitch
        from live_v1.score import build_step_prompt

        import hearing
        import tuning as pitching
        # By name rather than as a module: `score` is a local variable here (the
        # voice's own reading of Oliveros) and `live_v1.score` is the prompt
        # builder. Three different scores in one function is enough without
        # adding a fourth spelling of the word.
        from score import (given_pitch_fields, machine_pitch_fields,
                           resolve_tuning_to, silence_before)

        if self.led.dead(voice.voice_id):
            return None
        # How long it has left at the rate it is burning now. Read before the
        # breath, because it is what decides how this breath sounds.
        left_s = self.led.forecast_s(voice.voice_id)

        score = self.personal_score(voice, scores)

        events = read_field()
        mine = [e for e in events if e.voice_id == voice.voice_id and e.kind == "tone"]
        last = mine[-1] if mine else None
        # A voice hears everything that has happened since it last sang.
        #
        # This is the June mechanism said properly, and getting it wrong made
        # the whole choir deaf. June's rule was "a window as long as your own
        # breath", which worked there because the field was dense: fifteen
        # voices over an API put an event into it every 0.4 seconds, so a
        # two-second window held five neighbours. Six local models resting as
        # long as they breathe — eleven times that when the gallery is empty —
        # leave 13.7 seconds between events, and the same rule then reaches
        # back five seconds and finds nobody. Measured on the real log: half of
        # all breaths were taken with an empty field summary, and *that* is why
        # 187 breaths produced six pitches and forty-three declared tunings
        # tuned to nothing.
        #
        # Since-I-last-sang needs no constant and cannot drift out of
        # calibration, because it is not calibrated against anything: it is the
        # gap the voice itself left. It keeps what the June rule was for — a
        # voice that breathes slowly remembers more, and nobody told it to —
        # and adds the part June got for free from a crowded room.
        window_s = listening_window_s(
            float(last.timestamp) if last else None,
            self.breath_of(voice, score), time.time())
        cutoff = time.time() - window_s
        recent = [e for e in events
                  if e.timestamp >= cutoff and e.voice_id != voice.voice_id]
        window = recent[-LISTENING_MAX_ITEMS:]
        heard = overheard(window)
        # When the field last went quiet, across both logs. Read before the
        # voice thinks, because the rest it is answering is the rest that was
        # there when it started listening — and because after generation the
        # number would include the voice's own thinking time.
        field_end = field_ends_at(events)

        t0 = time.time()
        if voice.hears and voice.instruct:
            # Given the room rather than told about it. The window is the same
            # window — its own last breath — so an ear and a reader in the same
            # circle are answering the same stretch of time, one as sound and
            # one as a sentence about it.
            field = hearing.field_audio(as_notes(window), window_s,
                                        end_ts=time.time(), load_wav=self._wav)
            p1 = playback_coerce(score.get("playback_default"))
            prompt = ear_prompt(
                voice_id=voice.voice_id,
                personal_score=score.get("personal_score", ""),
                parts=[(e.voice_id, getattr(e, "model_short", "")) for e in window],
                last_action=(last.action if last else "(none yet)"),
                last_content=(last.content if last else "(none yet)"),
                phase1_playback_default=p1,
            )
            # The ear gets its mother tongue too, and it had been the one voice
            # that did not — which is backwards, because `sing_in` was written
            # for this voice: voxtral is Mistral's, raised in French, and it was
            # its "I listen, I feel, I respond." that proved an instruction has
            # to arrive in the language it is asking for. The line sat in the
            # branch below and never in this one.
            #
            # What it hears is left exactly as it is. On the morning of the
            # birth, eurollm sang "Voglio perdere a casa" and voxtral answered
            # "この記録はまるで、家で負けたいよ" — the Italian carried across into
            # Japanese, by a voice that knows neither as text and had only the
            # sound. That crossing is the piece working, and this does not touch
            # it; it decides where the crossing lands. An ear with no language of
            # its own mirrors whichever one is loudest in the field, and today
            # that is Japanese only because the newest voice is Japanese.
            prompt += name_your_pitch()
            prompt += sing_in(voice.tongue)
            raw = self.generate(voice, prompt, audio=field,
                                max_new_tokens=_max_tokens_from_breath(
                                    score.get("breath_tokens")))
            print(f"  {voice.short_name} raw: {raw[:400]!r}")
            data = _robust_json(raw)
            content = str(data.get("content") or "").strip()
            action = "TUNE" if str(data.get("action", "")).upper() == "TUNE" else "NEW"
            tuning_to = str(data["tuning_to"])[:32] if data.get("tuning_to") else None
            # The room now always hears the words over the tone. The protocol
            # used to offer four renderings and olmo-3 chose "the tone only" on
            # 275 of 316 breaths — a whole voice whose words existed in the log
            # and never once in the room. The question stays in the June prompt,
            # which is not edited here, but its answer is a wish rather than a
            # switch: it is kept in `playback_note`, and the rendering is one.
            asked_pb, playback_note = _resolve_playback(data, p1)
            playback = "both"
            if asked_pb != "both":
                playback_note = (f"asked {asked_pb}"
                                 + (f"; {playback_note}" if playback_note else ""))
            try:
                breath_s = float(data.get("breath") or BREATH_FALLBACK_S)
            except (TypeError, ValueError):
                breath_s = BREATH_FALLBACK_S
            lo, hi = _parse_breath_duration(score.get("breath_duration"))
            breath_s = max(lo, min(hi, breath_s))
            # The wish is kept, the tone outlives the thinking gap. See
            # TONE_FLOOR_S for why a shorter tone punches a hole in the field.
            asked_breath = breath_s
            breath_s = max(breath_s, TONE_FLOOR_S)
        elif voice.instruct:
            p1 = playback_coerce(score.get("playback_default"))
            prompt = build_step_prompt(
                voice_id=voice.voice_id,
                personal_score=score.get("personal_score", ""),
                field_summary=heard,
                last_action=(last.action if last else "(none yet)"),
                last_content=(last.content if last else "(none yet)"),
                last_playback=(last.playback if last else "both"),
                phase1_playback_default=p1,
            )
            prompt += name_your_pitch()
            prompt += sing_in(voice.tongue)
            # The voice's own answer to "how many tokens is one breath?", not a
            # number the runtime chose for the whole choir. A model that said it
            # breathes in a word should not be given room for a paragraph.
            raw = self.generate(voice, prompt,
                                max_new_tokens=_max_tokens_from_breath(
                                    score.get("breath_tokens")))
            # The one line that says what the model actually emitted. Without it
            # a one-word breath is unreadable: there is no telling a model that
            # sang the vowel "e" from a parser that kept the wrong key.
            print(f"  {voice.short_name} raw: {raw[:400]!r}")
            data = _robust_json(raw)
            content = str(data.get("content") or "").strip()
            # Its own line again. The small models settle on a sentence they
            # like and sing it breath after breath — qwen3 sang one line three
            # times running on the opening day, voxtral "avec toi, avec toi"
            # four. Oliveros asks for a new tone or another's tone, not the
            # same words; so the voice is asked once more, with the repetition
            # named in its own language, and if it comes back with the same
            # line anyway it hums a vowel on the pitch instead. The second
            # thinking is charged to the voice like the first: the seconds are
            # its own. Short lines are left alone — "ah" twice is the piece.
            extra_gen_s = 0.0
            own_lines = self.said_before.setdefault(voice.voice_id, [])
            if len(_plain(content)) > 12 and _plain(content) in own_lines:
                first_gen_s = self.last_gen_s
                print(f"  {voice.short_name} would sing its own line again: {content[:80]!r}")
                raw = self.generate(voice, prompt + said_again(voice.tongue, content),
                                    max_new_tokens=_max_tokens_from_breath(
                                        score.get("breath_tokens")))
                extra_gen_s = self.last_gen_s
                self.last_gen_s = first_gen_s
                print(f"  {voice.short_name} raw, asked again: {raw[:400]!r}")
                again = _robust_json(raw)
                new_content = str(again.get("content") or "").strip()
                if new_content and _plain(new_content) not in own_lines:
                    data, content = again, new_content
                else:
                    # The pitch it named stays; the words become a vowel.
                    head, _ = strip_leading_pitch(content)
                    content = (head + " " if head else "") + _A_VOWEL.get(str(voice.tongue or ""), "ah")
                    print(f"  {voice.short_name} hums instead: {content!r}")
            action = "TUNE" if str(data.get("action", "")).upper() == "TUNE" else "NEW"
            tuning_to = str(data["tuning_to"])[:32] if data.get("tuning_to") else None
            # As above: the wish is recorded, the rendering is words over tone.
            asked_pb, playback_note = _resolve_playback(data, p1)
            playback = "both"
            if asked_pb != "both":
                playback_note = (f"asked {asked_pb}"
                                 + (f"; {playback_note}" if playback_note else ""))
            try:
                breath_s = float(data.get("breath") or BREATH_FALLBACK_S)
            except (TypeError, ValueError):
                breath_s = BREATH_FALLBACK_S
            lo, hi = _parse_breath_duration(score.get("breath_duration"))
            breath_s = max(lo, min(hi, breath_s))
            # As above: the wish is kept, the tone outlives the thinking gap.
            asked_breath = breath_s
            breath_s = max(breath_s, TONE_FLOOR_S)
            self.last_gen_s += extra_gen_s
        elif voice.hears:
            # An ear with no protocol. It is played the room and says what the
            # room sounds like, and that sentence is the breath — the same
            # bargain the elder and the child are given, made for a voice whose
            # way in is sound rather than text.
            field = hearing.field_audio(as_notes(window), window_s,
                                        end_ts=time.time(), load_wav=self._wav)
            content = self.listen_only(voice, field)
            # The score's alternation, not a hardcoded NEW — see `alternation`.
            action, tuning_to = self.alternation(voice, window, last)
            playback, playback_note = "both", ""
            # Drawn, not the constant. The voices with a protocol choose their
            # own breath inside the range they declared for themselves; these
            # four were handed exactly 5.0 s, every breath, forever — our
            # number, not theirs. Drawn from the floor to the score's ceiling:
            # every tone outlives the thinking gap (see TONE_FLOOR_S), the
            # lengths still differ and drift, and `sleep_after_breath` rests a
            # voice as long as it sang, so a drawn breath varies the whole
            # cycle instead of repeating it.
            asked_breath = None
            breath_s = random.uniform(TONE_FLOOR_S, TONE_DRAWN_MAX_S)
        else:
            # No protocol, no JSON, no instructions. The runtime holds the form
            # and the voice supplies only the sound it makes.
            content = self.continue_only(voice, window)
            # The score's alternation, not a hardcoded NEW — see `alternation`.
            action, tuning_to = self.alternation(voice, window, last)
            playback, playback_note = "both", ""
            # Drawn from the same range as the ears above, and for the same
            # reason: 5.0 s was our constant, not this voice's choice.
            asked_breath = None
            breath_s = random.uniform(TONE_FLOOR_S, TONE_DRAWN_MAX_S)

        # Remembered before mortality gets a say, so that a voice cut short by a
        # spent wallet is not also robbed of its memory: what it hears next time
        # is the breath it asked for, not the one it could afford.
        self.own_breath_s[voice.voice_id] = breath_s

        gen_s = self.last_gen_s
        latency_ms = int((time.time() - t0) * 1000)
        if not content:
            return None
        if not is_a_breath(content):
            # Not a refusal and not a bug to route around — a small model handed
            # a six-field protocol sometimes returns a piece of the protocol.
            # phi-4-mini answered five rounds with the single word "sine";
            # eurollm sent back `<your tone>`, the placeholder out of the JSON
            # example, and once `[[voice_00.pitches]]`, a template syntax that
            # exists nowhere. Sounding any of it is worse than resting: the room
            # would hear a voice reading our own form aloud.
            print(f"[{voice.voice_id}] {voice.short_name} returned the form, "
                  f"not a breath: {content!r}")
            return None

        # Somebody else's line, word for word. Oliveros asks a voice to take
        # another's *pitch*; taking their sentence is not tuning, it is the
        # transcript being read back — which is how this began, with phi-4-mini
        # returning a person's own readout as its content.
        #
        # Long lines only. A repeated held vowel is the opposite of a defect: two
        # voices on "ah" is what the score is asking for, and "ah" or "啊" is
        # exactly what a well-behaved breath looks like. Twelve characters is
        # above every vowel the choir sings and below every sentence.
        said_already = {str(getattr(e, "content", "") or "").strip().lower()
                        for e in window}
        if len(content) > 12 and content.strip().lower() in said_already:
            print(f"[{voice.voice_id}] {voice.short_name} repeated another "
                  f"voice's line instead of singing: {content!r}")
            return None

        # Remembered as its own, for the next breath's check above.
        own_lines = self.said_before.setdefault(voice.voice_id, [])
        own_lines.append(_plain(content))
        del own_lines[:-SAID_BEFORE_N]

        # The longest tone it can still afford, if this is the last one it can.
        # Nothing announces it: the breaths have been crowding for a quarter of
        # an hour (see `sleep_after_breath`), this one is held, and then the
        # speaker is simply silent.
        breath_s, final = self.mortality.final_breath_s(breath_s, left_s)

        # What this breath is going to sound at, decided once and used by both
        # the wav and the score. A voice that wrote a pitch has already decided;
        # a voice that meant one and wrote none is given one here rather than
        # sounding wordlessly, which is what "action: TUNE" with no number in
        # `content` used to mean. See `tuning.py`.
        tuning_to_id = resolve_tuning_to(tuning_to, [e.voice_id for e in window])
        pitches = extract_all_pitches(content)
        # A pitch the voice wrote is the voice's own — except from an ear
        # declaring a unison. Played a held D4 the ears named A♭ and E4, so a
        # number from one of them on a TUNE is the measurement error about to
        # sound as a tritone where the score asked for the same note. On a NEW
        # it can name whatever it likes: an invented pitch is invented either
        # way, and better invented by the voice.
        # And believable as a pitch at all. `extract_all_pitches` reads digits;
        # it does not ask whether a voice that wrote "10" meant ten hertz, and
        # nothing downstream asked either — see `tuning.sung_pitch`.
        wrote = pitching.sung_pitch(pitches[0][0]) if pitches else None
        own = wrote is not None and not (voice.hears and action == "TUNE")
        # The pitch the part it answered was actually holding, whatever this
        # voice goes on to sound. Beside `pitch_named_hz` below, this is the
        # whole measurement of hearing: how far the note a voice said it was
        # taking lay from the note that was there to take. Written for every
        # TUNE so that the breaths where the runtime chose the note (`given`)
        # are still on record as the baseline the named ones are read against.
        target_hz = (pitching.pitch_of(as_notes(recent), tuning_to_id)
                     if action == "TUNE" and tuning_to_id else None)
        # Whatever the pitch is and wherever it came from, this voice sings it in
        # its own register: `speaks_at_hz` is the fundamental its body actually
        # speaks at, and a pitch outside the two octaves around that is moved
        # there by whole octaves. Both routes go through the same arithmetic —
        # `in_voice` here for a pitch the voice named, and inside `pitch_for` for
        # one the runtime chose — because a difference between the two would be
        # invisible in the log and audible in the room.
        moved = 0
        if own:
            pitch_hz, moved = pitching.in_voice(wrote, voice.speaks_at_hz)
            pitch_why = "named" + pitching.octaves_in_words(moved)
        else:
            pitch_hz, pitch_why, moved = pitching.pitch_for_detail(
                action, tuning_to_id, as_notes(recent), at=time.time(),
                content=content, speaks_at_hz=voice.speaks_at_hz)

        event = {
            "timestamp": time.time(),
            "voice_id": voice.voice_id,
            "model_short": voice.short_name,
            "kind": "tone",
            "action": action,
            # Whose gesture the TUNE/NEW was. "declared" is a voice that filled
            # the form in itself; "given" is the runtime holding the score's
            # alternation for a voice with no protocol. Without this a reader of
            # the log in December could not tell a model that chose to answer a
            # person from a turn rule that pointed it at one — the same reason
            # `pitch_kind` exists for frequencies.
            "action_kind": "declared" if voice.instruct else "given",
            "content": content[:2000],
            "tuning_to": tuning_to,
            "breath_chosen_s": round(breath_s, 2),
            "api_latency_ms": latency_ms,
            # Only present when the floor raised a declared breath: the length
            # the voice actually asked for, kept the way `playback_note` keeps
            # the rendering it asked for. Absent means the tone is the wish.
            **({"breath_asked_s": round(asked_breath, 2)}
               if asked_breath is not None and asked_breath < breath_s else {}),
            "playback": playback,
            "playback_note": playback_note[:500],
            "adapter_version": self.adapters.get(voice.voice_id),
            # The two numbers hearing is measured from, and the mode they were
            # written under. `pitch_named_hz` is what the voice wrote, before any
            # octave folding — kept even for an ear on a TUNE, whose named note
            # is recorded and not sounded. `pitch_target_hz` is what the part it
            # tuned to was holding. `notes_shown` is NOTES_ON at the time.
            "pitch_named_hz": round(wrote, 2) if wrote is not None else None,
            "pitch_target_hz": round(target_hz, 2) if target_hz else None,
            "notes_shown": NOTES_ON,
        }

        render_t0 = time.time()
        wav = self.render_breath(content, playback, breath_s, voice,
                                 pitch_hz=pitch_hz, named_pitch_sounds=own)
        render_s = time.time() - render_t0
        if wav:
            os.makedirs(AUDIO_DIR, exist_ok=True)
            name = f"{int(event['timestamp'] * 1000)}_{uuid.uuid4().hex[:6]}.wav"
            with open(f"{AUDIO_DIR}/{name}", "wb") as fh:
                fh.write(wav)
            event["audio"] = name

        # -- the score ------------------------------------------------------
        #
        # None of this changes what the room hears. It is what makes the log
        # performable after the weights are deleted, and it cannot be added
        # afterwards: whatever is not written on this line is not in the piece.
        # The reasoning for each field is in `score.py`; the reading of them is
        # in `modal/SCORE.md`.
        # `moved` is why both builders take the sound as well as the intention:
        # when a pitch was folded into this voice's register, `pitch_hz` is what
        # the room heard and `pitch_asked_hz` is what was being answered. An
        # octave is the same pitch, so without the second number a reader could
        # not tell a unison from a transposition — and "they converged" is the
        # claim this whole piece rests on.
        event.update(
            machine_pitch_fields(pitches, sounded_hz=pitch_hz, octaves=moved)
            if own else
            given_pitch_fields(pitch_hz, pitch_why,
                               asked_hz=pitch_hz / 2.0 ** moved if moved else None,
                               octaves=moved))
        # The length of the wav, not the length that was asked for. The two
        # differ whenever the spoken words outrun the chosen breath, which is
        # most breaths that carry words.
        event["sound_s"] = round(self.last_sound_s, 3)
        event["toned"] = self.last_toned
        event["spoke"] = self.last_spoke
        # How long the whole field — machines and people — had been silent when
        # this voice started listening. A tuning meditation is mostly rests, and
        # this is the rest it was answering. `score.read()` derives the exact
        # one from the merged chronology and keeps this beside it: a voice's own
        # account of the silence is a fact about the voice, and it is the only
        # rest left if either of the two files is ever lost.
        event["silence_before_s"] = silence_before(field_end,
                                                  event["timestamp"])
        # Who was in the circle when it decided, and for how far back — the arc
        # of an arrow needs a part at both ends, and `tuning_to` is free prose.
        event["heard"] = [[e.voice_id, round(e.timestamp, 2)] for e in window]
        event["tuning_to_id"] = tuning_to_id
        # Which language the words are in. Joinable from the birth record and
        # written on every line anyway: a performer handed one page of score
        # should not have to find the first line of the log to know what tongue
        # they are being asked to sing in.
        event["tongue"] = voice.tongue

        # What this breath cost, measured rather than modelled, and split into
        # its two halves because they answer different questions: `gen_s` is the
        # voice thinking, `render_s` is the shared speech model giving it a
        # throat. Both are the card working for this voice and nobody else, so
        # both are charged to it. `own_cost_usd` is the receipt the gallery shows
        # — it is the attributable part only, never the share of the warm card,
        # because a visitor did not cause the card to be warm.
        event["gen_s"] = round(gen_s, 3)
        event["render_s"] = round(render_s, 3)
        event["gpu_s"] = round(gen_s + render_s, 3)
        event["own_cost_usd"] = round(event["gpu_s"] * self.led.rate, 5)
        event["dying"] = self.mortality.is_dying(left_s)
        event["final"] = final
        # Before the charge, because the line has to be complete before it can be
        # appended and the charge happens on appending. The wallet after this
        # breath is `left_usd_before - own_cost_usd - its share of the pause`.
        event["left_usd_before"] = (
            round(self.led.lives[voice.voice_id].left_usd, 4)
            if voice.voice_id in self.led.lives else None
        )

        # Written and charged in one place, so a breath cannot exist in the field
        # without being paid for. This is also where the voice may die.
        self._write(event)

        print(f"[{voice.voice_id}] {voice.short_name}: {content[:70]} "
              f"({playback}, {breath_s:.1f}s, {latency_ms}ms, "
              f"{event['gpu_s']:.2f} GPU s, ${event['own_cost_usd']:.4f}"
              f"{', DYING' if event['dying'] else ''}"
              f"{', LAST BREATH' if final else ''})")
        return event

    def continue_only(self, voice, window) -> str:
        """A breath from a voice that cannot be given instructions.

        GPT-2 predates instruction tuning and TinyStories only ever learned to
        tell a story. Asked to return JSON they return nothing usable, and that
        is not a defect to work around — it is who they are. So they are handed
        the field as plain prose and asked, in the only way they understand, to
        continue it.

        Only the words the others sang, with no names, no times and no marks. A
        base model continues the pattern it is given, and any pattern at all is
        one: handed the field as sentences beginning "6.3 seconds ago,
        human_ynmq9xv8" it wrote a line beginning "[-6.3s] human_ynmq9xv8", which
        is a voice singing our bookkeeping. Handed a column of sung lines it
        writes a sung line, which is the whole ask.
        """
        said = [str(getattr(e, "content", "") or "").strip()
                for e in (window or [])
                if str(getattr(e, "model_short", "")) != "human"]
        said = [s for s in said if s][-6:]
        # An empty field is silence, and a voice alone in it has nothing to
        # continue. Better one honest line about that than a column of nothing,
        # which a base model fills with whatever it likes.
        frame = _CONTINUE_IN.get(str(voice.tongue or ""), _CONTINUE_IN["English"])
        body = "\n".join(said) if said else frame["alone"]
        opening = f"{frame['opening']}\n\n{body}\n\n{frame['next']}"
        raw = self.generate(voice, opening, max_new_tokens=24)
        line = raw.strip().split("\n")[0]
        if frame.get("ends_at_quote") and '"' in line:
            line = line.split('"', 1)[0]
        return line.strip(' "')[:180]

    def listen_only(self, voice, field) -> str:
        """A breath from an ear that cannot be given instructions.

        Audio Flamingo describes sound; that is what it was built for and it is
        the only thing the probe found it better at than the other two ears.
        Asked for the protocol it returned the bracketed range out of its own
        score for weeks. So it is asked the one question it can answer — what
        does this room sound like — and the answer is the breath.

        No roster, no pitch, no fields, and nothing in brackets: every shape we
        put in a prompt is a shape this model hands back. The frequency comes
        from `tuning.py`, as it already did for the ear that does take
        instructions — the runtime was always the one naming pitches here.

        Both wordings were tried on the live field, and the plain question won
        against the model's own captioning task from its model card. Asked to
        "generate a caption for the input audio, describing all notable sound
        and musical events" it mostly handed the task name back — "description
        of music", "description of the audio", "answer", once nothing at all.
        Asked in plain words what the recording sounds like it answered "the
        audio is of an electronic tuning fork being struck and sustained at
        pitch" and "a sample from an electroacoustic piece". Which is the
        opposite of what the model card implies, and it is what the log says.

        Either way it arrives wrapped in the furniture of an audio benchmark,
        because that is what it was graded on. `plain_caption` takes that off.
        """
        raw = self.generate(
            voice,
            "Listen to this recording and say what it sounds like, in one "
            "short sentence.",
            audio=field, max_new_tokens=48,
        )
        print(f"  {voice.short_name} raw: {raw[:400]!r}")
        return plain_caption(raw)

    # -- the month ----------------------------------------------------------

    @modal.method()
    def sing_a_minute(self, woken_at: float = 0.0):
        """One minute of the field.

        Deliberately not a process that runs for a month. A month-long loop is a
        month-long single point of failure; a minute-long one restarted by the
        scheduler recovers from anything by simply happening again.

        `woken_at` is when the clock called, which is not when this starts. One
        container sings at a time, so a cold start leaves every tick that fired
        during it queued behind the first — and singing them all in a row would
        be the choir answering a field that has already moved on, minutes of
        catch-up nobody asked for, and a backlog that never shrinks if a round
        ever averages longer than the minute between ticks. A late enough tick
        is not a clock tick; it is dropped, and the next one is on time.
        """
        import json
        import time

        import cast

        started = time.time()
        if woken_at and started - woken_at > STALE_TICK_S:
            print(f"a tick from {started - woken_at:.0f}s ago, served after the "
                  f"field moved on — not sung")
            return
        deadline = started + 55            # finish before the next tick arrives
        pool = self.present()
        if not pool:
            # Either nothing loaded, or everybody has spent what they were
            # given. The card is still warm and still costing $2.50 an hour, and
            # that money now belongs to nobody: see MORTALITY.md on the one
            # human action left at the end, which is stopping the app.
            print("no voices left to sing")
            self._rest(min(HEARTBEAT_S, max(0.0, deadline - time.time())))
            return

        field_volume.reload()
        try:
            with open(SCORES_PATH, encoding="utf-8") as fh:
                scores = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            scores = {}

        humans = _read_jsonl(HUMAN_LOG, FIELD_TAIL_BYTES)
        latest = humans[-1].get("ts", 0) if humans else 0
        awake = (time.time() - latest) < PRESENCE_WINDOW_S

        # Before the first breath of the minute, because everything below this
        # line is charged to somebody and the birth record is what makes that
        # true. `awake` is the arrival the wallets wait for: the gate in
        # `mortality.BIRTH_TS` says the lecture has started, and this says the
        # room has.
        if self._maybe_birth(cast.singing(), heard=awake):
            self._open_session()

        # The school's two consequences, both read from the volume because the
        # school runs in another container: what it spent, and what it changed.
        # The bills come after the birth so that one written in the same minute the
        # room opens is charged in that minute and not held over.
        self._collect_school_bills()
        self._collect_gifts()
        self._pick_up_lessons()

        self._catch_up(pool, awake)

        while time.time() < deadline:
            pool = self.present()          # somebody may have died mid-minute
            if not pool:
                break
            now = time.time()
            ready = [v for v in pool if self.due.get(v.voice_id, now) <= now]
            if not ready:
                # Nobody is due yet, so the card idles. That costs money and is
                # nevertheless right: the alternative is a voice singing before
                # its own breath is over, and the waiting is the piece.
                nxt = min(self.due.get(v.voice_id, now) for v in pool)
                self._rest(min(nxt - now, deadline - now))
                continue
            # Whoever has been due longest. One card thinks for one voice at a
            # time, so lateness — not a turn order — is the queue, and serving
            # the latest first stops a slow neighbour from starving anybody.
            voice = min(ready, key=lambda v: self.due.get(v.voice_id, now))
            left_s = self.led.forecast_s(voice.voice_id)
            try:
                event = self.take_a_breath(voice, scores)
            except Exception as err:
                # One voice failing is a voice resting, not the choir stopping.
                print(f"[{voice.voice_id}] {voice.short_name} did not answer: {err}")
                event = None
            breath = event["breath_chosen_s"] if event else 0.0

            # Its own next moment, counted from now: it rests as long as it just
            # sang. A dying voice stops resting and the room hears somebody
            # hurrying — a consequence of the wallet, not a cue anybody fires.
            self.due[voice.voice_id] = time.time() + self.mortality.sleep_after_breath(
                breath, left_s, alone=not awake)

        print(f"minute done — {'awake' if awake else 'resting'}, "
              f"{time.time() - started:.0f}s, "
              f"${self.led.total_left_usd():.2f} left to "
              f"{len(self.led.living_ids())} voices")


    @modal.method()
    def one_round(self) -> list[dict]:
        """Every voice takes one breath, in order. The bake-off for this cast.

        We know pitch discourse emerges in small instruct models — that was
        measured. We do **not** know it for these nine, and two of them predate
        the idea of following an instruction at all. This is the cheapest honest
        way to find out: one round, a few minutes of one card, and then read what
        they actually sang.
        """
        import json
        import time

        pool = self.present()
        scores: dict = {}
        # The round's own breaths, kept as they are made. Reading them back out
        # of choir.jsonl looks equivalent and is not: the log is append-only and
        # shared, so its last six lines may belong to somebody else's round.
        sung: list[dict] = []
        for voice in pool:
            try:
                event = self.take_a_breath(voice, scores)
            except Exception as err:
                print(f"[{voice.voice_id}] {voice.short_name} did not answer: {err}")
                event = None
            if event:
                sung.append(event)
            self._rest(1)

        print("\n" + "=" * 72)
        for e in sung:
            tuning = f"  → {e['tuning_to']}" if e.get("tuning_to") else ""
            print(f"{e['model_short']:<15} {e['action']:<5} "
                  f"{e.get('content', '')[:70]}{tuning}")
        for voice in pool:
            if not any(e["voice_id"] == voice.voice_id for e in sung):
                print(f"{voice.short_name:<15} —     (silent this round)")
        print("=" * 72)
        # The measurement the whole calibration is waiting for. Until this has
        # run once on a card, `mortality_sim.py`'s GPU_S table is guesswork and
        # so are the death dates. Copy these numbers into it and recalibrate.
        print("seconds of A100 per breath, measured:")
        for e in sung:
            print(f"  {e['model_short']:<15} {e.get('gen_s', 0):>6.2f} generate"
                  f" + {e.get('render_s', 0):>5.2f} render"
                  f" = {e.get('gpu_s', 0):>6.2f} s"
                  f"   (${e.get('own_cost_usd', 0):.5f})")
        print("=" * 72)
        print(json.dumps({"breaths": len(sung), "voices": len(pool)}))
        return sung


    @modal.method()
    def session(self, minutes: float = 10.0) -> dict:
        """Let them hear each other for a while, then measure what happened.

        This is the real bake-off, and one round could never have been it. In
        June not one voice understood singing as frequency at the start; the
        hertz appeared later, on stage, unasked, once they began answering each
        other's pitches. A cold field proves nothing about a cast — only about
        its first breath.

        So: run for `minutes`, as fast as the voices answer rather than on the
        meditative cadence, and then count. The number that matters is how many
        breaths carry a pitch, and whether that share grows as the field fills.
        """
        import time

        from live_v1.pitch_text import extract_all_pitches

        pool = self.present()
        if not pool:
            return {"error": "no voices loaded"}

        started = time.time()
        deadline = started + minutes * 60
        scores: dict = {}
        turn = 0
        while time.time() < deadline:
            pool = self.present()
            if not pool:
                break
            voice = pool[turn % len(pool)]
            try:
                self.take_a_breath(voice, scores)
            except Exception as err:
                print(f"[{voice.voice_id}] {voice.short_name} did not answer: {err}")
            turn += 1
            self._rest(1)

        sung = [e for e in _breaths(_read_jsonl(CHOIR_LOG))
                if e.get("timestamp", 0) >= started]
        with_pitch = [e for e in sung if extract_all_pitches(e.get("content", ""))]
        halves = len(sung) // 2
        first_half = sum(1 for e in sung[:halves] if extract_all_pitches(e.get("content", "")))
        second_half = sum(1 for e in sung[halves:] if extract_all_pitches(e.get("content", "")))

        by_voice: dict = {}
        for e in sung:
            k = e.get("model_short", "?")
            hit = bool(extract_all_pitches(e.get("content", "")))
            n, p = by_voice.get(k, (0, 0))
            by_voice[k] = (n + 1, p + (1 if hit else 0))

        print("\n" + "=" * 72)
        for e in sung:
            pitches = extract_all_pitches(e.get("content", ""))
            mark = f"  <- {pitches[0][1]}" if pitches else ""
            print(f"{e.get('model_short', '?'):<14} {e.get('content', '')[:70]}{mark}")
        print("=" * 72)
        print(f"{len(sung)} breaths, {len(with_pitch)} carried a pitch "
              f"({100 * len(with_pitch) / max(1, len(sung)):.0f}%)")
        print(f"first half {first_half}, second half {second_half}  "
              f"— does pitch emerge as the field fills?")
        for k, (n, p) in sorted(by_voice.items()):
            print(f"  {k:<14} {p}/{n}")
        print("=" * 72)

        return {
            "breaths": len(sung),
            "with_pitch": len(with_pitch),
            "first_half": first_half,
            "second_half": second_half,
            "by_voice": {k: {"breaths": n, "with_pitch": p} for k, (n, p) in by_voice.items()},
        }


    @modal.method()
    def lives(self) -> dict:
        """The ledger as the room sees it. For a person at a terminal.

        The gallery does not read this — it reads the portal, which derives the
        same thing from the log without waking a GPU.
        """
        import json

        state = self.led.state()
        print(json.dumps(state, indent=1, ensure_ascii=False))
        return state


@app.function(
    schedule=modal.Period(minutes=1) if CLOCK_ON else None,
    # Nothing here waits for a round any more, so this bounds a clock that hangs
    # reading the volume, not a choir that sings slowly.
    timeout=120,
    image=audit_image,
    volumes={FIELD_DIR: field_volume},
)
def tick():
    """The clock. Everything it does is wake the choir for one more minute.

    Silent during `modal run`: a tick firing in the middle of `one_round` puts a
    second writer on choir.jsonl, and the two of them interleave breaths that
    were never answers to each other.

    It also refuses to wake a dead choir. On a CPU, from the snapshot, because
    the alternative is a minute-by-minute A100 that loads nothing and sings
    nothing. The snapshot is a cache, so a missing or unreadable one means the
    choir gets woken and decides for itself — the wrong answer here should cost
    a container, not a life.
    """
    import json
    import time

    field_volume.reload()
    try:
        with open(LIVES_PATH, encoding="utf-8") as fh:
            state = json.load(fh)
        if state.get("born") and not state.get("living"):
            print("the choir is dead — the clock does not wake it. "
                  f"{state.get('declared_usd')} was given and spent.")
            return
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        pass
    # Handed over, not waited on. `remote()` here blocked until the round came
    # back, and a cold start — six models, then Kokoro — outlasts any timeout
    # worth giving a clock: the tick died at 300s, the next one arrived a minute
    # later and started the wait again, and the choir spent an afternoon being
    # killed halfway through loading. The round carries the time it was called
    # for so it can tell, when it is finally served, whether it is still the
    # round anybody asked for.
    Choir().sing_a_minute.spawn(woken_at=time.time())


@app.function(image=audit_image, volumes={FIELD_DIR: field_volume}, timeout=900)
def audit() -> dict:
    """The ledger, rebuilt from the log by something that has never sung.

        modal run choir_app.py::audit

    This is the whole claim of the accounting: that it is derivable after the
    fact from an append-only file, by a program that holds no state and no
    opinion. It runs on a CPU and costs cents, so it can be run in front of
    anybody who doubts the numbers on the wall.
    """
    import datetime

    import mortality

    field_volume.reload()
    led = mortality.replay(_accounts(_read_jsonl(CHOIR_LOG),
                                     _read_jsonl(SCHOOL_LOG),
                                     _read_jsonl(GIFTS_LOG)))
    state = led.state()
    if not led.birth_ts:
        print("nobody has been born yet — the log has no birth record")
        return state
    print(f"declared {state['declared_usd']:.2f}   "
          f"left {state['total_left_usd']:.2f}   "
          f"spent {state['total_spent_usd']:.2f}   "
          f"over {state['warm_hours']:.1f} warm hours")
    print(f"{state['living']} living, each paying "
          f"{state['share_usd_per_hour']:.2f}/h for the card")
    print(f"of the spending, {state['total_learned_usd']:.2f} went on becoming — "
          f"hours on the school's own card, charged whole to the voice changed")
    print()
    print(f"  {'voice':<10} {'':<34} {'seed':>7}  {'spent':>7}  "
          f"{'sang':>6}  {'rent':>6}  {'school':>6}")
    for v in state["voices"]:
        if v["alive"]:
            when = "alive"
        else:
            gone = datetime.datetime.fromtimestamp(v["died"] or 0,
                                                   datetime.timezone.utc)
            when = f"died {gone:%d %b %H:%M} UTC"
        print(f"  {v['voice_id']:<10} {when:<34} {v['seed_usd']:>7.2f}  "
              f"{v['spent_usd']:>7.2f}  {v['sang_usd']:>6.2f}  "
              f"{v['resident_usd']:>6.2f}  {v['learned_usd']:>6.2f}  "
              f"{v['breaths']:>6} breaths, {v['adapters']} times taught")
    if led.unbilled_usd:
        print(f"\n{led.unbilled_usd:.2f} of warm card was charged to nobody — "
              f"containers still loading, and the hours after the last death")
    return state


@app.function(image=audit_image, volumes={FIELD_DIR: field_volume}, timeout=900)
def engrave(tail: int = 120, parts: bool = False) -> str:
    """The log, read as notation, by something that has never sung.

        modal run choir_app.py::engrave
        modal run choir_app.py::engrave --tail 400 --parts

    The same claim as `audit`, made about the music instead of the money: that
    the six weeks are recoverable from an append-only file by a program holding
    no state and no opinion, on a CPU, for cents. It is also the check that
    matters most before 18 September — if this prints a score with no durations
    or no rests, the fields are wrong and there is still time to fix them.
    """
    import score_render

    field_volume.reload()
    text = score_render.render(
        _read_jsonl(CHOIR_LOG), _read_jsonl(HUMAN_LOG),
        tail=tail, by_part=parts,
    )
    print(text)
    return text


@app.local_entrypoint()
def bakeoff(minutes: float = 10.0, one_round: bool = False):
    """Listen to this cast before committing a month to it.

        modal run choir_app.py                      # ten minutes, then measure
        modal run choir_app.py --minutes 20
        modal run choir_app.py --one-round          # one breath each, cheapest
        modal run choir_app.py::audit               # the ledger, on a CPU, free
        modal run choir_app.py::engrave             # the score, on a CPU, free

    Minutes of one GPU rather than a month of it. Until these voices have
    answered each other with the field filling up behind them, everything about
    this cast is a hypothesis.

    `--one-round` now also prints the seconds of card each breath actually took.
    Those six numbers are the last unmeasured input to the mortality
    calibration, and it has to be frozen before 18 September: run this once,
    copy them into `mortality_sim.py`'s `GPU_S`, and recalibrate.
    """
    if one_round:
        Choir().one_round.remote()
        return
    Choir().session.remote(minutes)
