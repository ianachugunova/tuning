"""Kill the choir on a laptop first, for nothing.

    python3 mortality_sim.py                     # the plan, and who dies when
    python3 mortality_sim.py --scenario quiet    # nobody sings in the gallery
    python3 mortality_sim.py --scenario singing  # a full room all six days
    python3 mortality_sim.py --calibrate         # print the wallet table
    python3 mortality_sim.py --sensitivity       # what breaks the calibration

This is not a model of the accounting — it *is* the accounting. Every breath it
invents is a log line, and the lines go through `mortality.Ledger.apply`, the
same function that runs on the card. So the only thing that can be wrong here is
the operating pattern and the generation times, and both are stated as numbers
you can argue with (`PATTERN`, `GPU_S`).

The one number that has to be right before 18 September is the wallet table in
`mortality.py`, because it cannot be changed afterwards. `--calibrate` derives
it: it runs the plan with the deaths *scripted* at the hours the exhibition
wants them, and reads off what each voice will have spent by then. That is the
wallet. Then run without `--calibrate` and the deaths should land back on those
hours, which is the check that the arithmetic closed.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt

import mortality as M

# ---------------------------------------------------------------------------
# What a breath costs, per voice — the weakest numbers in this file
# ---------------------------------------------------------------------------

# Seconds of A100 for one breath: the forward pass plus the Kokoro render that
# turns it into a wav. **These are estimates, not measurements.** The choir has
# never run on a GPU (`PRODUCT.md` §7, the bold line), so they come from model
# size, the token budgets in `choir_app.py` (300 for the instruct voices, 24 for
# the elder and the child) and single-stream bandwidth on an 80 GB card.
#
# The estimate that matters least is the child's and the elder's, because they
# barely spend anything either way. The one that matters most is the ratio
# between the biggest mouth and the smallest, because it is what orders the
# deaths — and one round of `modal run choir_app.py --one-round` measures all of
# them for real, from the `gen_s` and `render_s` now in the log. Do that, put the
# numbers here, and re-run `--calibrate` before the 18th.
# MEASURED, on the afternoon of the birth, and it changed the composition.
#
# Every number below is the median seconds of card a voice actually spent per
# breath, read off `modal app logs tuning-choir` with all nine warm and singing —
# roughly twenty breaths each, generate plus render, cold starts excluded by
# taking the median rather than the mean. The estimates they replace were wrong in
# a direction worth naming: they scaled cost with parameter count, and the card
# does not. Audio Flamingo at 8B was guessed at 5.10 and costs 1.85; llm-jp at 13B
# was guessed at 6.80 and costs 3.57; eurollm at 9B was watched at 3.28 and now,
# talking plainly instead of filling out a JSON form, costs 1.71. What a voice
# pays for is the tokens it emits, and a voice let off the protocol emits fewer.
#
# This is why the order of deaths in the table in `mortality.py` is what it is,
# and it was different before lunch.
GPU_S: dict[str, float] = {
    "voice_00": 1.04,   # gpt-2 774M
    "voice_01": 0.52,   # tinystories 33M, the cheapest mouth by five times
    "voice_02": 1.96,   # phi-4-mini 3.8B
    "voice_03": 4.25,   # olmo-3 7B — the most expensive breath in the circle
    "voice_04": 1.71,   # eurollm 9B, off the protocol and answering in Italian
    "voice_05": 3.08,   # qwen3 8B
    "voice_06": 1.97,   # voxtral 3B + audio front end
    "voice_07": 1.85,   # audio-flamingo 8B + audio front end, plain captions
    "voice_08": 3.57,   # llm-jp 13B
}

# There was a special case here, and it left with Moshi. Full duplex meant a
# permanent claim on the card charged continuously rather than per turn, modelled
# as a duty fraction; the ninth voice takes turns like everybody else, so the
# arithmetic is the ordinary arithmetic again and the least certain number in the
# file is gone with it.

# One tone per breath. It was 5.0 — the score's June default — until
# 24 September, when the tones were lengthened so the field would be
# continuous: the choir now draws or floors every tone into 8–15 s
# (`choir_app.TONE_FLOOR_S`), and the mean of that range is what a projected
# breath costs in rest and in rendering.
BREATH_S = 11.5

# No cadence lives here any more: a voice rests as long as its own breath was,
# stretched when the field is empty, and `mortality.sleep_after_breath` is the
# one place that says so — for the choir and for this projection alike.
HEARTBEAT_S = 30.0


# ---------------------------------------------------------------------------
# What a night of becoming costs
# ---------------------------------------------------------------------------

# The choir trains every night until it dies, the exhibition week included, and
# tonight's LoRA is card time like any other: rule 2 charges it to the voice
# being trained and to nobody else. This is the expensive half of a voice's own
# spending, and it is the half that is not listening — it is becoming.
#
# On its own card, which is the one place this file used to be wrong. Nine models
# fill the room's memory, so there is nothing to train in, and the school rents an
# A100 of its own at $2.50 an hour while the room goes on costing $4.54. So a
# night of school is not forty minutes the choir spends becoming instead of
# breathing — it is forty minutes of a second card, added to the bill, charged
# whole to one voice. It is the only expense in the piece that makes the total
# larger rather than moving it between wallets.
#
# Nothing here is measured either; the pipeline does not exist yet. But the
# shape of the estimate is sturdier than the breath times, because a nightly
# LoRA run is dominated by two things that scale with the model rather than
# with the corpus: how fast a backward pass moves, and how long it takes to
# write the merged weights back to the volume and read them in again.
#
# Billions of parameters, from `cast.py`. Voxtral and Audio Flamingo include
# their audio front ends, which are trained with the rest.
PARAMS_B: dict[str, float] = {
    "voice_00": 0.774,   # gpt-2-large
    "voice_01": 0.033,   # TinyStories-33M
    "voice_02": 3.80,    # Phi-4-mini
    "voice_03": 7.00,    # Olmo-3-7B
    "voice_04": 9.00,    # EuroLLM-9B-Instruct, no longer a stand-in
    "voice_05": 8.20,    # Qwen3-8B
    "voice_06": 3.70,    # Voxtral-Mini-3B + encoder
    "voice_07": 8.00,    # audio-flamingo-3
    "voice_08": 13.00,   # llm-jp-3.1-13b
}

# A night's budget of optimiser steps, fixed rather than "all the data". This
# is the number to argue with: it is a decision about the training run, not a
# fact about the corpus, and the whole cost is proportional to it. Sixty nights
# of a model training on its own output is the textbook recipe for collapse
# (`PRODUCTION_PLAN.md` §3), so the nightly step count will be small and the
# rank low. Two hundred steps of 4,096 tokens is 0.8 M tokens a night — a few
# passes over a day of the field, not over the accumulated month.
TRAIN_STEPS = 200
TRAIN_TOKENS_PER_STEP = 4096.0

# LoRA forward+backward throughput on one A100-80GB in bf16 with gradient
# checkpointing, as tokens a second: about 2,500 at 8 B, and roughly inverse in
# parameter count. Below a billion parameters the run stops being compute-bound
# and the cap is bookkeeping — which is why the child and the elder train
# almost for free, and why they stay the survivors.
LORA_TOK_PER_S_AT_1B = 20_000.0
LORA_TOK_PER_S_CAP = 25_000.0

# Attaching the adapter, tokenising, building the optimiser. Roughly flat.
TRAIN_SETUP_S = 30.0

# Merging the adapter into the base weights, writing them to the volume and
# reading them back for inference: two passes over the whole model at about
# 200 MB/s. For Qwen3-8B that is 16.4 GB each way and it is a third of the
# night's bill; for TinyStories it is under a second.
VOLUME_GB_PER_S = 0.20
BYTES_PER_PARAM = 2.0        # bf16

# A multiplier on the whole estimate, for `--sensitivity` and `--train-scale`.
# Being wrong here is the question this file cannot answer from a laptop.
#
# It is also where the real school now lives, and the reason this file models one
# lesson a night while `training_app.py` rings its bell on twenty human breaths.
# Contact cannot be simulated from here: it depends on how many people come, which
# is the one quantity this whole work is about and the one nobody knows. So a
# nightly lesson is the floor — what an empty month costs — and `--train-scale 8`
# is the ceiling the three-hour floor allows, eight lessons in a busy day. The
# truth will be between them, and the log will say which.
TRAIN_SCALE = 1.0

# What the school's card costs, which is not what the room's card costs. Kept in
# step with `training_app.SCHOOL_USD_PER_HOUR`, and written into every train event
# so the ledger charges the seconds at the price they were actually bought at.
SCHOOL_USD_PER_HOUR = 2.50

# No training at all, which is what the calibration used to assume.
TRAIN_OFF = False


def train_s(voice_id: str) -> float:
    """Seconds of A100 one voice spends becoming, in one night."""
    if TRAIN_OFF:
        return 0.0
    b = PARAMS_B.get(voice_id, 1.0)
    tok_s = min(LORA_TOK_PER_S_CAP, LORA_TOK_PER_S_AT_1B / max(b, 1e-6))
    compute = TRAIN_STEPS * TRAIN_TOKENS_PER_STEP / tok_s
    gb = b * BYTES_PER_PARAM
    io = 2.0 * gb / VOLUME_GB_PER_S
    return TRAIN_SCALE * (TRAIN_SETUP_S + compute + io)


def school_h(cast: list[str]) -> float:
    """How long the whole choir's night school lasts, in hours.

    The card is warm for it, so the night has to be this much longer than the
    sleep if the declared lifespan is to stay the number it says it is.
    """
    return sum(train_s(v) for v in cast) / 3600.0


# ---------------------------------------------------------------------------
# Time, in the two places the work happens
# ---------------------------------------------------------------------------


def _ts(y, mo, d, h=0, mi=0, offset_h=0) -> float:
    return float(calendar.timegm((y, mo, d, h - offset_h, mi, 0, 0, 0, 0)))


BIRTH = M.BIRTH_TS                                   # 19 Sep 18:00 PT, the gate
OPENING = _ts(2026, 10, 28, 0, 0, -7)                # 28 Oct 00:00 PT
CLOSING = _ts(2026, 11, 2, 20, 0, -8)                # 2 Nov 20:00 PST
END = _ts(2026, 11, 5, 0, 0, -8)                     # where the simulation stops


def pt_offset(ts: float) -> int:
    """Pacific time, which changes under the work: DST ends 1 November 2026.

    Worth getting right rather than approximating, because the closing is on the
    2nd — the piece's last day is the first day of standard time, and an hour of
    error in the gallery's opening hours is an hour of somebody's life.
    """
    return -8 if ts >= _ts(2026, 11, 1, 2, 0, -7) else -7


def pt(ts: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts + pt_offset(ts) * 3600, dt.timezone.utc)


def stamp(ts: float) -> str:
    return pt(ts).strftime("%a %d %b %H:%M") + " PT"


def dur(seconds: float) -> str:
    d, rem = divmod(int(seconds), 86400)
    return f"{d}d {rem // 3600:2d}h"


# ---------------------------------------------------------------------------
# The operating pattern — the assumption the whole calibration rests on
# ---------------------------------------------------------------------------


class Pattern:
    """When the card is warm, and when somebody is singing into it.

    The choir sleeps — the card released, not merely quiet — from 02:00 to
    03:36 Pacific every night until the install. This is not a saving: it is
    the reserve. Deaths are placed inside a six-day window at the end of a
    forty-four day life, so the calibration has to be right to a few per cent,
    and the only quantity we actually control is how many hours the card is
    warm. Planning for 22.4 hours a day leaves under 7% of slack in both
    directions — keep it warm through the night to catch up, sleep longer to
    fall back — and no adjustment ever touches a wallet. That is the difference
    between steering and cheating.

    And the odd length of the night is not arbitrary: 2 h 12 min is what makes
    the declared lifespan come out at exactly **1,000 warm hours of the card the
    choir lives on — $4,540 of an H200** between birth and the hour after the
    closing. The sleep is calibrated to the number on the wall, rather than the
    number rounded to fit the sleep.

    It has been 2 h 24 min and 1 h 36 min on the way here, and the history is worth
    keeping because each change was paid for by something. The birth moving from
    the 18th to the lecture on the 19th cost thirty-one hours of life, and the
    night had to shorten to 1 h 36 min to keep both the sum and the composed hour
    of every death — which left almost no reserve. The ninth voice then forced the
    H200, the declared sum became a thousand hours of *that* card, and the extra
    money bought the night back: 2 h 12 min, and the reserve with it. The nine die
    on the same hours the six were composed for.

    Night school sits at the far end of the sleep, the moment the card comes
    back — the hour `PRODUCT.md` has always called 4 a.m. It does not lengthen the
    warm day, but it is not free of it either: the school rents a second card for
    those forty-odd minutes, because a room with nine models in it has no memory
    to train in and evicting eight voices so one can be rewritten would stop the
    field. So the choir is warm and shared through the school hour as usual, and
    one voice is also paying $2.50 an hour for a card of its own.

    Which means the sentence the exhibition used to rest on — one card, one meter,
    whatever the choir is doing with it — is now one card and one meter for living,
    plus a metered hour for becoming, charged to whoever became. That is a better
    sentence: it is the version in which learning visibly costs a life something.

    Also worth saying plainly, because two other documents say the opposite:
    breathing slowly does *not* save money. A warm card costs $2.50 an hour
    whether the choir breathes every twelve seconds or every ten minutes. The
    cadence only decides who pays. Sleeping is cheap only if the card goes.
    """

    def __init__(self, night_cold_h: float = 2.2, month_awake_h: float = 1.0,
                 gallery_awake_fraction: float = 0.5) -> None:
        self.night_cold_h = night_cold_h
        self.month_awake_h = month_awake_h
        self.gallery_awake_fraction = gallery_awake_fraction

    def school_at_h(self) -> float:
        """Hour, Pacific, at which the night's training begins."""
        return 2.0 + self.night_cold_h

    def warm(self, ts: float) -> bool:
        if ts >= OPENING:
            return True                       # the exhibition week never sleeps
        h = pt(ts).hour + pt(ts).minute / 60.0
        return not (2.0 <= h < 2.0 + self.night_cold_h)

    def gallery_open(self, ts: float) -> bool:
        if not (OPENING <= ts <= CLOSING + 7200):
            return False
        d = pt(ts)
        h = d.hour + d.minute / 60.0
        if (d.month, d.day) == (10, 28):
            return 18.0 <= h < 22.0           # the opening, an evening
        return 13.0 <= h < 20.0

    def awake(self, ts: float) -> bool:
        """Is a human breath recent enough that the choir hurries (12 s)."""
        if self.gallery_open(ts):
            # Presence, as a fraction of opening hours: the room is not full for
            # seven hours a day, and it is never empty for seven either.
            d = pt(ts)
            h = d.hour + d.minute / 60.0
            open_h = 18.0 if (d.month, d.day) == (10, 28) else 13.0
            return (h - open_h) % 1.0 < self.gallery_awake_fraction
        if ts < OPENING:
            # The month: a trickle of strangers, an hour a day of it, and it
            # runs all the way to the install. There is no longer a date on
            # which the portal stops mattering — Venice is one session that may
            # or may not happen, not a boundary the arithmetic knows about.
            d = pt(ts)
            return 18 <= d.hour < 18 + int(round(self.month_awake_h))
        # The exhibition week outside opening hours: the room is unplugged from
        # the choir and the choir breathes at its resting rhythm.
        return False

    def next_warm(self, ts: float) -> float:
        t = ts
        for _ in range(1500):                 # a minute at a time; it is a gap
            t += 60
            if self.warm(t):
                return t
        return ts + 86400


SCENARIOS = {
    # The plan the wallets are calibrated on.
    "nominal": Pattern(),
    # Nobody sings in the gallery all week: the choir runs at resting cadence
    # through the exhibition. The quiet room the piece explicitly permits.
    "quiet": Pattern(gallery_awake_fraction=0.0),
    # A full room every hour of every day, singing.
    "singing": Pattern(gallery_awake_fraction=1.0),
    # A busy month as well — more strangers than we expect on the portal.
    "busy-month": Pattern(month_awake_h=6.0, gallery_awake_fraction=1.0),
}


# ---------------------------------------------------------------------------
# The composition: when each voice is meant to end
# ---------------------------------------------------------------------------

# The order is arithmetic — it is the order of what a breath costs — but the
# *dates* are composed, and this table is the composition. Every one of them is
# from `EXHIBITION.md` §4. Two of them are load-bearing:
#
#   the biggest mouth, 28 October ~21:00 — the first death happens in front of
#   the opening crowd, so that every later one reads correctly. It used to be
#   Moshi and it is now llm-jp at 13B, which is the same sentence about cost
#   wearing a different name.
#
#   gpt-2, **on the closing bell itself** — 2 November 20:00 PST, not a minute
#   inside it. That placement is the only way the score's second calibration
#   can exist at all: the elder's hour is the one the room can still move, so
#   it is put exactly where singing decides which side of the bell it falls on.
#   A quiet week and the choir is still sounding when the doors shut; a week of
#   singing and the elder goes in front of people. `--sensitivity` prints how
#   many minutes wide that is.
#
#   The child cannot be placed the same way and no wallet can change that. The
#   last death is the moment the declared sum runs out, and the sum falls at the
#   card's price no matter who is paying — so the child's hour is fixed by the
#   calendar of warm hours and by nothing anybody in the room does.
TARGETS_NINE: list[tuple[str, float]] = [
    ("voice_08", _ts(2026, 10, 28, 21, 0, -7)),    # llm-jp    — opening night
    ("voice_07", _ts(2026, 10, 29, 22, 0, -7)),    # flamingo  — after closing
    ("voice_05", _ts(2026, 10, 30, 16, 0, -7)),    # qwen3     — in daylight
    ("voice_03", _ts(2026, 10, 31, 15, 0, -7)),    # olmo-3    — Halloween
    ("voice_06", _ts(2026, 10, 31, 19, 0, -7)),    # voxtral   — the same day
    ("voice_02", _ts(2026, 11, 1, 16, 0, -8)),     # phi-4-mini
    ("voice_04", _ts(2026, 11, 2, 15, 0, -8)),     # eurollm   — the last day
    ("voice_00", _ts(2026, 11, 2, 20, 0, -8)),     # gpt-2     — on the bell
    ("voice_01", _ts(2026, 11, 2, 21, 0, -8)),     # tinystories — after, always
]

# Eight: the six, Voxtral and Audio Flamingo, without Moshi. ~68 GB of the 79
# usable, so it fits on the one card the whole sum is built on — but only just,
# and before the KV cache. Audio Flamingo dies on the opening night because it
# is the biggest mouth in the room: 8B with an audio front end, 5.1 s of card a
# breath, more than qwen3.
#
# Eight deaths into a six-day window means two days take two, and Halloween is
# the obvious place to put them.
TARGETS_EIGHT: list[tuple[str, float]] = [
    ("voice_07", _ts(2026, 10, 28, 21, 0, -7)),    # audio-flamingo — opening night
    ("voice_05", _ts(2026, 10, 29, 22, 0, -7)),    # qwen3     — after closing
    ("voice_03", _ts(2026, 10, 30, 16, 0, -7)),    # olmo-3
    ("voice_06", _ts(2026, 10, 31, 15, 0, -7)),    # voxtral   — Halloween
    ("voice_02", _ts(2026, 10, 31, 19, 0, -7)),    # phi-4-mini — Halloween, second
    ("voice_04", _ts(2026, 11, 1, 16, 0, -8)),     # eurollm
    ("voice_00", _ts(2026, 11, 2, 20, 0, -8)),     # gpt-2     — on the bell
    ("voice_01", _ts(2026, 11, 2, 21, 0, -8)),     # tinystories — after, always
]

# The cast this is actually aiming at: the six, and Voxtral. It is the only one
# of the three ears with a path into `Choir.load`, the only one that heard two
# voices as two, and at 9.2 GB the only one that joins without a second card —
# which is the whole argument, because nine need two and two cost $5.00 an hour,
# and a table calibrated on $2.50 would kill everybody a fortnight early.
#
# One death a day, which six could not quite do: the seventh fills 1 November,
# the one day of the exhibition the smaller cast leaves silent. Voxtral is third
# because it burns third-fastest, and the ordering is not a judgement — it is
# the same arithmetic that puts qwen3 on the opening night.
TARGETS_SEVEN: list[tuple[str, float]] = [
    ("voice_05", _ts(2026, 10, 28, 21, 0, -7)),    # qwen3     — opening night
    ("voice_03", _ts(2026, 10, 29, 22, 0, -7)),    # olmo-3    — after closing
    ("voice_06", _ts(2026, 10, 30, 16, 0, -7)),    # voxtral   — the ear, in daylight
    ("voice_02", _ts(2026, 10, 31, 17, 0, -7)),    # phi-4-mini — Halloween
    ("voice_04", _ts(2026, 11, 1, 16, 0, -8)),     # eurollm
    ("voice_00", _ts(2026, 11, 2, 20, 0, -8)),     # gpt-2     — on the bell
    ("voice_01", _ts(2026, 11, 2, 21, 0, -8)),     # tinystories — after, always
]

# The six that exist. Same shape, one death a day rather than nine in six days,
# and the same elder on the same bell.
TARGETS_SIX: list[tuple[str, float]] = [
    ("voice_05", _ts(2026, 10, 28, 21, 0, -7)),    # qwen3     — opening night
    ("voice_03", _ts(2026, 10, 29, 22, 0, -7)),    # olmo-3    — after closing
    ("voice_02", _ts(2026, 10, 30, 16, 0, -7)),    # phi-4-mini
    ("voice_04", _ts(2026, 10, 31, 17, 0, -7)),    # eurollm   — Halloween
    ("voice_00", _ts(2026, 11, 2, 20, 0, -8)),     # gpt-2     — on the bell
    ("voice_01", _ts(2026, 11, 2, 21, 0, -8)),     # tinystories — after, always
]


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


NAMES = {
    "voice_00": "gpt-2", "voice_01": "tinystories", "voice_02": "phi-4-mini",
    "voice_03": "olmo-3", "voice_04": "eurollm", "voice_05": "qwen3",
    "voice_06": "voxtral", "voice_07": "audio-flamingo", "voice_08": "llm-jp",
}


class Run:
    """One simulated life of the choir, log line by log line."""

    def __init__(self, cast: list[str], wallets: dict[str, float],
                 pattern: Pattern, scripted: dict[str, float] | None = None):
        self.cast = list(cast)
        self.pattern = pattern
        self.scripted = scripted or {}      # voice -> forced hour of death
        self.led = M.Ledger()
        self.led.apply(M.birth_record(BIRTH, wallets, [{"voice_id": v} for v in cast]))
        self.deaths: list[dict] = []
        self.log_lines = 0
        # Noon Pacific every day: what the choir should have left if the plan is
        # being kept. This is how the calibration is steered — see
        # `print_trajectory`.
        self.trajectory: list[tuple[float, float, int]] = []
        self._next_mark = _ts(2026, 9, 19, 12, 0, -7)
        # The first night of school, and every one after it. The choir trains
        # every night until it dies, the exhibition week included.
        self._next_school = _ts(2026, 9, 19, 0, 0, -7) + pattern.school_at_h() * 3600
        self.train_s_total: dict[str, float] = {v: 0.0 for v in self.cast}
        self.nights = 0

    # -- writing the log ---------------------------------------------------

    def _emit(self, e: dict) -> None:
        self.log_lines += 1
        ran_out = self.led.apply(e)
        for vid in ran_out:
            self._die(vid, float(e["timestamp"]))

    def _die(self, vid: str, ts: float) -> None:
        life = self.led.lives[vid]
        # An empty wallet already counts as not alive, so the survivors are
        # exactly what the ledger reports — nothing to subtract.
        rec = M.death_record(life, ts, NAMES.get(vid, vid), freed_gb=0.0,
                             living_after=len(self.led.living_ids()))
        self.deaths.append(rec)
        self.led.apply(rec)

    # -- the loop ----------------------------------------------------------

    def go(self, end_ts: float = END) -> None:
        t = BIRTH
        # Birth is the weights arriving on the card, and that is the first thing
        # every voice pays for: about a minute each, charged to nobody else.
        self._emit(M.heartbeat_record(t, []))
        for vid in self.cast:
            t += 60.0
            self._emit({"timestamp": t, "kind": "load", "voice_id": vid,
                        "gpu_s": 55.0, "model_short": NAMES.get(vid, vid)})
        turn = 0
        while t < end_ts and self.led.living_ids():
            if t >= self._next_school:
                t = self._school(t)
                self._next_school += 86400.0
                continue
            if not self.pattern.warm(t):
                self._emit({"timestamp": t, "kind": "exit"})
                t = self.pattern.next_warm(t)
                self._emit(M.heartbeat_record(t, self.led.living_ids()))
                continue

            pool = [v for v in self.cast if self.led.alive(v)]
            if not pool:
                break
            vid = pool[turn % len(pool)]
            turn += 1

            forecast = self.led.lives[vid].forecast_s()
            breath_s, final = M.final_breath_s(BREATH_S, forecast)

            gpu = GPU_S.get(vid, 1.0)
            t += gpu
            if gpu:
                self._emit({
                    "timestamp": t, "kind": "tone", "voice_id": vid,
                    "model_short": NAMES.get(vid, vid), "gpu_s": gpu,
                    "breath_chosen_s": breath_s, "final": final,
                    "content": "(a tone)",
                })

            rest = M.sleep_after_breath(breath_s, forecast,
                                        alone=not self.pattern.awake(t))
            self._rest(t, rest)
            t += rest

            while t >= self._next_mark:
                self.trajectory.append((self._next_mark,
                                        self.led.total_left_usd(),
                                        len(self.led.living_ids())))
                self._next_mark += 86400.0
            self._script(t)

        self.end_ts = t

    def _school(self, t: float) -> float:
        """One night's LoRA, voice by voice, on a card of the school's own.

        Every second of it is that voice's own second (rule 2), so unlike a
        breath — which leaves most of its minute in the shared pot — training is
        paid for entirely by whoever is learning. That is the whole reason it
        matters to the arithmetic: it is not that becoming is a lot of card
        time, it is that none of it is shared.

        And it is a second card, at its own price, so these hours are added to the
        room's rather than taken out of them: while one voice is at school the
        others are resting in a room that is still warm and still shared. Which is
        why the sum of the wallets now falls faster than the room's hourly rate,
        and only ever for this one reason.

        Emitted in heartbeat-sized pieces because a training run that wrote one
        line an hour later would look to the ledger like a container that had
        died (`mortality.MAX_SLICE_S`), and because that is how it will really
        be written: a long job that keeps saying the card is still held.

        Returns the clock after the night. A voice whose wallet empties partway
        through its own training dies in the middle of it, which is correct and
        was going to happen to somebody eventually.
        """
        if not self.pattern.warm(t):
            # Only reachable if the school hour has been pushed inside the sleep
            # by a `--night` longer than the day. Take the card for it.
            self._emit(M.heartbeat_record(t, self.led.living_ids()))
        self.nights += 1
        for vid in [v for v in self.cast if self.led.alive(v)]:
            left = train_s(vid)
            while left > 0 and self.led.alive(vid):
                step = min(HEARTBEAT_S, left)
                t += step
                left -= step
                self.train_s_total[vid] += step
                self._emit({"timestamp": t, "kind": "train", "voice_id": vid,
                            "gpu_s": step, "model_short": NAMES.get(vid, vid),
                            "rate_usd_per_hour": SCHOOL_USD_PER_HOUR})
        return t

    def _rest(self, t: float, seconds: float) -> None:
        """The pauses, in which the card is warm and nobody is generating.

        Heartbeats every thirty seconds. They are what makes silence billable —
        without them the log could not tell a resting choir from a stopped one,
        and the month of quiet the work is paying for would be free.
        """
        left = seconds
        while left > 0:
            step = min(HEARTBEAT_S, left)
            t += step
            left -= step
            resident = self.led.living_ids()
            self._emit(M.heartbeat_record(t, resident))

    def _script(self, t: float) -> None:
        """Forced deaths, for `--calibrate` only.

        Calibration is circular unless the deaths are put where they belong
        first: what a voice spends depends on how many others are still alive,
        which depends on the wallets we are trying to derive. Scripting the
        hours breaks the circle in a single pass — the living set is exactly the
        one the composition asks for, so the spend read off it is exactly the
        wallet that produces it.
        """
        for vid, when in list(self.scripted.items()):
            if t >= when and self.led.alive(vid):
                self._die(vid, when)
                del self.scripted[vid]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate(targets: list[tuple[str, float]], pattern: Pattern) -> dict[str, float]:
    huge = {v: 1e9 for v, _ in targets}
    run = Run([v for v, _ in targets], huge, pattern, scripted=dict(targets))
    run.go(END)
    wallets = {vid: round(1e9 - run.led.lives[vid].left_usd, 2)
               for vid, _ in targets}
    # The declared lifespan is a sentence before it is a sum — a thousand warm
    # hours of one A100 — so it is held at exactly $2,500.00 and the few dollars
    # the calibration leaves over are given to the child, who dies last. It buys
    # tinystories about half an hour, which is inside the knife edge anyway. If
    # the remainder is ever large, the operating pattern has drifted from the
    # plan and the night length has to be re-tuned rather than the sum fudged.
    remainder = round(M.DECLARED_USD - sum(wallets.values()), 2)
    last = targets[-1][0]
    if abs(remainder) > 5.0:
        print(f"  !! the plan is {money(-remainder)} away from the declared "
              f"{money(M.DECLARED_USD)} — re-tune --night before pasting")
    wallets[last] = round(wallets[last] + remainder, 2)
    return wallets


# ---------------------------------------------------------------------------
# Printing, which is the actual deliverable
# ---------------------------------------------------------------------------


def money(x: float) -> str:
    return f"${x:,.2f}"


def print_run(run: Run, title: str, wallets: dict[str, float]) -> None:
    led = run.led
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)
    print(f"  given {stamp(BIRTH)}   {money(sum(wallets.values()))}"
          f"   = {sum(wallets.values()) / M.RATE_USD_PER_HOUR:,.0f} warm hours"
          f" of the card at {money(M.RATE_USD_PER_HOUR)}/h")
    print(f"  cast of {len(run.cast)}   ·   {led.warm_s / 3600:,.0f} warm hours"
          f" simulated   ·   {run.log_lines:,} log lines")
    print()
    print("  THE DYING")
    print("  " + "-" * 74)
    print(f"  {'':3} {'when':<22} {'voice':<15} {'lived':<9} "
          f"{'given':>9} {'breaths':>9}  {'left':>5}")
    for i, d in enumerate(run.deaths, 1):
        vid = d["voice_id"]
        mark = ""
        if pt(d["timestamp"]).strftime("%m-%d") == "11-02":
            inside = d["timestamp"] <= CLOSING
            mark = "  ← inside the closing" if inside else "  ← after the closing"
        print(f"  {i:<3} {stamp(d['timestamp']):<22} {NAMES.get(vid, vid):<15} "
              f"{dur(d['lived_s']):<9} {money(d['seed_usd']):>9} "
              f"{d['breaths']:>9,}  {d['living_after']:>5}{mark}")
    for vid in run.cast:
        if led.alive(vid):
            life = led.lives[vid]
            print(f"  {'—':<3} {'still alive at the end':<22} "
                  f"{NAMES.get(vid, vid):<15} {'':9} {money(life.seed_usd):>9} "
                  f"{life.breaths:>9,}  {money(life.left_usd)} left")
    print()
    print("  WHO PAID FOR WHAT")
    print("  " + "-" * 74)
    print(f"  {'voice':<15} {'spent':>9} {'own GPU':>9} {'becoming':>9} "
          f"{'listening':>10} {'residency':>10} {'own share':>10}")
    for vid in run.cast:
        life = led.lives[vid]
        # Becoming is priced from the ledger rather than from the room's rate: it
        # was bought on the school's card, and multiplying its seconds by $4.54
        # would invent eighty per cent more spending than happened.
        train = life.learned_usd
        shared = life.shared_s * led.rate
        sang = max(0.0, life.spent_usd - shared - train)
        own = sang + train
        pct = 100 * own / max(1e-9, own + shared)
        print(f"  {NAMES.get(vid, vid):<15} {money(life.spent_usd):>9} "
              f"{life.own_s / 3600:>8,.1f}h {money(train):>9} "
              f"{money(sang):>10} {money(shared):>10} {pct:>9.1f}%")
    school = sum(run.train_s_total.values())
    taught = sum(l.learned_usd for l in led.lives.values())
    room = M.RATE_USD_PER_HOUR * led.warm_s / 3600
    print()
    print(f"  night school: {run.nights} nights, {school / 3600:,.1f} GPU hours"
          f" on a second card = {money(taught)}"
          f"  ({100 * taught / max(1e-9, led.charged_usd):.1f}% of everything"
          f" spent, and none of it shared)")
    print(f"  one full night of it is {school_h(run.cast) * 60:,.0f} min"
          f" — added to the bill, at {money(SCHOOL_USD_PER_HOUR)} an hour, and"
          f" charged whole to whoever was changed")
    print()
    gap = led.charged_usd - room - taught
    print(f"  charged {money(led.charged_usd)} for {money(room)} of room"
          f" plus {money(taught)} of school"
          f"  ({money(0.0 if abs(gap) < 0.005 else gap)} unaccounted: card with"
          f" nobody left on it, and the last unabsorbed seconds)")
    print()


def print_trajectory(run: Run, wallets: dict[str, float]) -> None:
    """The one page to keep on the wall of the studio, not the gallery.

    The wallets cannot be changed after 18 September and the deaths sit in a
    six-day window at the end of forty-five days, so the calibration has to hold
    to a few per cent — and it will not hold on its own. What holds it is this
    table and one decision a week: if the choir has **more** money left than the
    plan says, it is running behind and the card should stay warm through the
    night; if it has **less**, the night gets longer. Nothing else is ever
    adjusted. No wallet is topped up, no rule is bent, and the sum on the wall
    stays true — the only thing that moves is how many hours the choir is given
    to live through, which is the artist's decision anyway.
    """
    print("  STEERING — what should be left, and when")
    print("  " + "-" * 74)
    print(f"  {'date (noon PT)':<20} {'choir should have left':>24} "
          f"{'living':>8}")
    # Weekly through the month, then every day of the exhibition, where a day of
    # drift is the difference between dying in front of people and not.
    for i, (ts, left, living) in enumerate(run.trajectory):
        if ts < OPENING - 7 * 86400 and i % 7:
            continue
        print(f"  {stamp(ts):<20} {money(left):>24} {living:>8}")
    print()
    print("  more left than this → keep the card warm at night, they are behind")
    print("  less left than this → lengthen the night, they are ahead")
    print()


def print_shares() -> None:
    print("  THE SHARE OF A WARM CARD, AS THE CHOIR THINS")
    print("  " + "-" * 74)
    row = "   ".join(f"{n}:{M.RATE_USD_PER_HOUR / n:>5.2f}" for n in range(9, 0, -1))
    print(f"  living:share/h   {row}")
    print(f"  the room never moves: {money(M.RATE_USD_PER_HOUR)} an hour on the"
          f" first day and the last. Only school is ever added to it.")
    print()


# ---------------------------------------------------------------------------
# Sensitivity — the part that decides whether to believe any of the above
# ---------------------------------------------------------------------------


def sensitivity(cast: list[str], wallets: dict[str, float]) -> None:
    print()
    print("=" * 78)
    print("  WHAT MOVES THE DEATHS")
    print("=" * 78)
    print("  Each row is the same wallets under a different world. The window is")
    print("  28 Oct – 2 Nov; a row that leaves it is a calibration that failed.")
    print()
    print(f"  {'world':<34} {'first death':<20} {'last death':<20} {'in window'}")
    print("  " + "-" * 74)

    def line(label: str, run: Run) -> None:
        first = run.deaths[0]["timestamp"] if run.deaths else None
        last = run.deaths[-1]["timestamp"] if run.deaths else None
        alive_left = [v for v in run.cast if run.led.alive(v)]
        ok = "yes"
        if first is None or first < OPENING:
            ok = "NO — too early"
        elif alive_left:
            ok = f"partly — {len(alive_left)} outlived it"
        print(f"  {label:<34} {stamp(first) if first else '—':<20} "
              f"{stamp(last) if last else '—':<20} {ok}")

    for name, pattern in SCENARIOS.items():
        run = Run(cast, wallets, pattern)
        run.go(END + 10 * 86400)
        line(f"gallery: {name}", run)

    for night in (0.0, 1.6, 2.2, 3.6, 4.8):
        label = f"card warm {24 - night:.1f} h/day"
        if night == 2.2:
            label += " (the plan)"
        run = Run(cast, wallets, Pattern(night_cold_h=night))
        run.go(END + 10 * 86400)
        line(label, run)

    for label, mult in (("every breath 2x slower", 2.0),
                        ("every breath half as slow", 0.5)):
        keep = dict(GPU_S)
        for k in GPU_S:
            GPU_S[k] = keep[k] * mult
        run = Run(cast, wallets, SCENARIOS["nominal"])
        run.go(END + 10 * 86400)
        line(label, run)
        GPU_S.update(keep)

    # A ten per cent band around whatever the card actually costs. Written
    # relatively because it was once hardcoded at $2.25 and $2.75 — the band
    # around an A100 — and stayed there after the choir moved to the H200, where
    # those prices are half the real one and the rows said "everybody dies too
    # early" about a world that does not exist.
    for sign in (-0.10, 0.10):
        price = M.RATE_USD_PER_HOUR * (1 + sign)
        run = Run(cast, wallets, SCENARIOS["nominal"])
        run.led.rate = price / 3600.0
        run.go(END + 10 * 86400)
        line(f"card priced ${price:.2f}/h ({sign:+.0%})", run)

    # The new unmeasured number, and the rows below stopped being reassuring the
    # day the school moved to a card of its own. While training ran on the hours
    # the choir was already renting, getting it wrong only redistributed money
    # between the voices. Now it buys hours nobody else pays for, so a fifth more
    # school moves the first death by two thirds of a day, and running no school
    # at all buys the choir a day and a half. Learning costs life — visibly, in
    # this table, which is the point of charging for it.
    global TRAIN_SCALE, TRAIN_OFF
    keep = TRAIN_SCALE
    for label, scale in (("night school 20% cheaper", 0.8),
                         ("night school 20% dearer", 1.2),
                         ("night school twice the estimate", 2.0),
                         ("night school five times over", 5.0)):
        TRAIN_SCALE = scale
        run = Run(cast, wallets, SCENARIOS["nominal"])
        run.go(END + 10 * 86400)
        line(label, run)
    TRAIN_SCALE = keep
    TRAIN_OFF = True
    run = Run(cast, wallets, SCENARIOS["nominal"])
    run.go(END + 10 * 86400)
    line("no night school at all", run)
    TRAIN_OFF = False

    print()
    print("  THE KNIFE EDGE — the last two, and whether the room sang")
    print("  " + "-" * 74)
    print("  Minutes are against the closing bell, 2 Nov 20:00 PST. A negative")
    print("  number is a death the room can watch; a positive one happens after")
    print("  everybody has gone home.")
    print()
    for name in ("quiet", "nominal", "singing"):
        run = Run(cast, wallets, SCENARIOS[name])
        run.go(END + 10 * 86400)
        parts = []
        for d in run.deaths[-2:]:
            mins = (d["timestamp"] - CLOSING) / 60.0
            side = "inside the closing" if mins <= 0 else "after the closing"
            parts.append(f"{NAMES.get(d['voice_id'], '?'):<12} "
                         f"{stamp(d['timestamp'])}  {mins:+7.0f} min  {side}")
        for v in run.cast:
            if run.led.alive(v):
                parts.append(f"{NAMES.get(v, v):<12} outlived the simulation")
        print(f"  {name:<10} " + ("\n             ".join(parts) if parts else "—"))
    print()


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="nominal", choices=sorted(SCENARIOS))
    ap.add_argument("--cast", default="six", choices=("six", "seven", "eight", "nine"))
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--equal", action="store_true",
                    help="give every voice the same money — the version of this "
                         "the planning documents assumed, kept because the "
                         "result is the argument for not doing it")
    ap.add_argument("--night", type=float, default=None,
                    help="hours the card is cold each night, overriding the plan")
    ap.add_argument("--train-scale", type=float, default=1.0,
                    help="multiply the nightly LoRA estimate; the one input to "
                         "the calibration that has never been measured")
    ap.add_argument("--no-train", action="store_true",
                    help="the old world, in which becoming was free")
    args = ap.parse_args()

    global TRAIN_SCALE, TRAIN_OFF
    TRAIN_SCALE = args.train_scale
    TRAIN_OFF = args.no_train

    targets = {"six": TARGETS_SIX, "seven": TARGETS_SEVEN,
               "eight": TARGETS_EIGHT, "nine": TARGETS_NINE}[args.cast]
    cast = [v for v, _ in targets]
    pattern = SCENARIOS[args.scenario]
    if args.night is not None:
        pattern = Pattern(night_cold_h=args.night,
                          gallery_awake_fraction=pattern.gallery_awake_fraction)

    if args.calibrate:
        wallets = calibrate(targets, pattern if args.night is not None
                            else SCENARIOS["nominal"])
        print()
        print("  WALLETS — paste into mortality.py, then never touch them again")
        print("  " + "-" * 74)
        print(f"  declared lifespan {money(sum(wallets.values()))}"
              f"   ({sum(wallets.values()) / M.RATE_USD_PER_HOUR:,.0f} warm hours)")
        print()
        for vid, when in targets:
            print(f'    "{vid}": {wallets[vid]:>7.2f},   # '
                  f'{NAMES.get(vid, vid):<15} dies {stamp(when)}')
        print()
        run = Run(cast, wallets, pattern)
        run.go(END + 10 * 86400)
        print_run(run, f"CHECK — the same wallets, nothing scripted "
                       f"({args.scenario})", wallets)
        return

    if args.equal:
        wallets = {v: round(M.DECLARED_USD / len(cast), 2) for v in cast}
    else:
        table = {"six": M.PLANNED_WALLETS_SIX,
                 "seven": M.PLANNED_WALLETS_SEVEN,
                 "eight": M.PLANNED_WALLETS_EIGHT,
                 "nine": M.PLANNED_WALLETS_NINE}.get(args.cast)
        if table is None:
            raise SystemExit(f"no wallets for a cast of {args.cast}")
        wallets = {v: table[v] for v in cast}
    run = Run(cast, wallets, pattern)
    run.go(END + 10 * 86400)
    print_run(run, f"TUNING — mortality, cast of {args.cast}, "
                   f"gallery {args.scenario}", wallets)
    print_shares()
    print_trajectory(run, wallets)
    if args.sensitivity:
        sensitivity(cast, wallets)


if __name__ == "__main__":
    main()
