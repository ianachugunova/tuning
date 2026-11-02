"""The ledger — what each voice has spent, and what it has left.

Nine voices live on one H200. The card costs $4.54 an hour whether anybody
is singing or not, so "what did this voice cost" is a question with an answer we
have to *choose*. This file is that choice, written once, in one place, so that
the number on the wall and the number in the code cannot drift apart.

**The rule, in three lines.**

1. Every second the card is held is billed once, to somebody. The total spend is
   the card's price times the seconds it was warm — nothing more, nothing less,
   no second meter.
2. **Attributable seconds** — the card inside one voice's forward pass, and the
   card loading that voice's weights — are charged to that voice alone.
3. **The rest** — a warm card with nobody generating on it — is split **equally
   among the voices resident on it and still alive**.

Three consequences, all of them wanted.

*The total does not accelerate; the lives do.* The card burns $2.50 an hour on
the first day and on the last. But rule 3 divides a fixed sum among a shrinking
number, so each survivor's share climbs — $0.28 an hour at nine, $1.25 at two,
$2.50 at one. Nothing is staged: dying speeds up dying, and the big number in
the room still falls at a perfectly even rate. Both sentences are true and they
are not the same sentence.

*Singing does not add to the bill; it decides who pays it.* A visitor cannot
make the choir cost more — rule 1 forbids it. What a visitor does is wake the
choir, and a woken choir spends more of its hour inside somebody's forward
pass, which moves money out of the shared pot (rule 3) and onto whoever answers
at length (rule 2). So singing shortens the lives of the voices that talk and
lengthens the lives of the two that answer in a quarter of a second. See
`MORTALITY.md` for the measured size of that effect; it is real, and it is
smaller than the exhibition text currently implies.

*Becoming is the same rule, and it is the sharpest case of it.* The choir trains
every night until it dies, the exhibition week included, and tonight's LoRA
(`kind == "train"`) is card time like any other: rule 2 charges every second of
it to the voice being trained. That is what makes learning different from
listening — not that it is a great deal of card time, but that **none of it is
shared**. A breath leaves most of its minute in the common pot; a training run
leaves nothing. So the choir's night school is where a big voice's size finally
costs it something in its own name, and where the child pays almost nothing.

The school is the one thing that is not on the card the choir lives on, and this
paragraph used to claim otherwise. It cannot be: nine models fill the room's
memory, so there is nothing to train in, and the alternative — evicting eight
voices for an hour so one can be rewritten — would stop the field. So the night
school rents its own card, and the event it writes carries that card's price with
it ($2.50 an hour, against the room's $4.54), charged whole to the one voice
being changed. Learning is therefore the only expense that makes the project's
total spend larger rather than only redistributing it, and that is the truthful
version of the sentence: a voice pays for its own becoming, in the same wallet it
breathes out of, and the money is really gone. Every wallet holds a night or two
of school; a voice that is taught every night dies measurably sooner than one
left as it was born, and `/lives` shows both numbers so that the difference is
something a visitor can see rather than something the wall text asserts.

*Equal shares are what make the small ones mortal at all.* TinyStories-33M
spends about a fifth of a second per breath. Charged only for what it generates
it would need a century to spend a hundred dollars, and the piece would have no
ending. Charged an equal share of a warm card it dies in six weeks, like
everyone else, only last. This is also the honest answer to "resident weights
occupy memory whether the voice sings or not": they do, and rule 3 is the rent.

**Why not weight the shared cost by memory footprint.** It is the obvious
alternative — Qwen3-8B holds 16 GB and TinyStories 0.13 GB, so charge them in
proportion. Rejected for three reasons. The card is indivisible: we rent all
80 GB, and we would rent all 80 GB for the child alone, so a resident voice's
marginal memory cost is genuinely zero. Size is already paid for in rule 2,
where a big model's slow forward pass charges it directly; weighting the
residual by size charges the same fact twice. And equal shares say the thing
the work says — inside the circle each living voice is one voice, and the piece
is about tuning without merging.

**Death is derived, not stored.** There is no `alive` flag anywhere that a
future edit could flip. A voice is dead if `choir.jsonl` — the append-only log
that is also the only evidence the field has been sounding since 18 September —
contains a death record for it. The state machine here has no edge back to
alive: appending anything at all cannot revive a voice, because no event type
can. Reviving means deleting a line from the log that proves the work, and then
the log proves nothing. That is the whole refusal, and it is structural.

**The wallet is a historical fact, not a constant.** `PLANNED_WALLETS` below is
used exactly once — to write the birth record into the log on the evening of
19 September, during the lecture, at whichever minute the first person in the
room sings. From then on the ledger reads the seeds *out of the log*, so editing
this file cannot give a living voice another dollar — only a `gift` line in
`gifts.jsonl` can, in the open, and never to the dead. And nothing is seeded
before that: until `BIRTH_TS` has passed *and* somebody is audibly in the field,
the ledger charges nothing, which is what lets the choir rehearse for weeks
without quietly starting the clock early.

Stdlib only, on purpose: the simulation in `mortality_sim.py` runs the same
arithmetic as the card does, on a laptop, for free.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# The declared lifespan
# ---------------------------------------------------------------------------

# Modal's price for the card the choir lives on. If this ever changes, the
# calibration below is wrong and has to be recomputed — it is the one number
# everything else is measured in.
#
# It changed. The ninth voice is 13B and the nine no longer fit on an A100-80GB,
# so the choir lives on an H200 — 141 GB at $4.54 an hour, which is also cheaper
# than the two A100s the alternative needed. See the note on `gpu=` in
# `choir_app.py` for why one bigger card and not two.
#
# Nothing here is metered before birth, so no rehearsal was charged at the wrong
# price. What this does break is the sentence the declared sum came from: a
# thousand warm hours cost $2500 on an A100 and cost about $4500 here. That
# sentence is on a wall label and is not a number to quietly double — see
# `DECLARED_USD` directly below.
RATE_USD_PER_HOUR = 4.54
RATE_USD_PER_S = RATE_USD_PER_HOUR / 3600.0

# What was given, and it is a sentence rather than a budget line: **a thousand
# warm hours of the one card the choir lives on.** Not the whole $20,000 grant —
# a grant does not run out by itself, and a lifespan that does not end is not a
# lifespan. This number goes on the wall next to the date.
#
# The sentence is unchanged and the sum is not, because the card changed. A
# thousand warm hours cost $2,500 on the A100 the work was composed for; the nine
# voices need an H200, and a thousand hours of it cost $4,540. Keeping $2,500
# would have kept a number and thrown away what it meant — 550 hours, the whole
# choir dead in early October, three weeks before anybody reaches the gallery. The
# hours are the unit the piece is written in; the dollars are what Nvidia happens
# to charge for them this year.
#
# Of the three ways out this comment used to list, this is the first and the
# smallest: the composed calendar is kept intact — one death a day across the
# exhibition, the elder on the closing bell — and $4,540 of a $20,000 grant is
# spent, which leaves what to do with the rest a question still open rather than
# one answered by arithmetic on the day of the birth. What it is not is a claim
# that the choir dies when the money runs out; it dies when *this* money runs out,
# and the difference belongs on the wall label rather than in a comment.
DECLARED_USD = 4540.00

# The earliest the choir may be born: 19 September 2026, 18:00 in San Francisco
# (PDT, UTC-7), when the lecture starts. Recorded in UTC because a log is read
# years later by people in other time zones.
#
# It is a gate and not the birthday. Nothing is seeded until somebody is audibly
# in the field, so the wallets start when the choir is first heard rather than
# when a clock passes a number — during the lecture, at whatever minute the room
# begins to tune. `choir_app.Choir._maybe_birth` is where both halves are
# enforced; the birthday itself is only ever readable out of the log.
#
# Moving this line moves every death by the same amount, so it cannot be changed
# without re-running `mortality_sim.py --calibrate`: the wallets below are the
# money each voice is predicted to spend between birth and the hour it is meant
# to end, and those hours are pinned to the exhibition rather than to birth.
BIRTH_TS = float(calendar.timegm((2026, 9, 20, 1, 0, 0, 0, 0, 0)))

# The wallets, in dollars, as they will be written into the log at birth.
#
# They are not equal, and the reason is arithmetic rather than taste. With equal
# wallets the order of deaths is right — cheap voices last — but the *dates* are
# absurd: Moshi, which cannot stop listening, would die in the first fortnight
# and TinyStories would outlive the gallery by years. The exhibition needs the
# deaths inside six days at the end of a forty-five day life, so each wallet is
# the money that voice is predicted to spend between Endzeit and the hour it is
# meant to end — its breathing, its share of the silence, and its nightly
# training, all three. See `mortality_sim.py --calibrate`, which prints this
# table, and `MORTALITY.md` for what that does and does not leave to chance.
#
# Read the pairing carefully, because the obvious reading of it is false and it
# is the kind of false that ends up on a wall label. The wallets do not decide
# who dies first. Give every voice the same money — `mortality_sim.py --equal`,
# on any of the three casts — and they die in exactly the same order they die in
# below: the biggest mouth first, the child last, every time. That order is not
# a decision anybody made. It is what these models cost to run, and no sum of
# money rearranges it.
#
# What the unequal wallets buy is the spacing. Equal, the six-day window gets
# nothing for its first three days and then four deaths in the last two, two of
# them after the closing bell — the whole ending crushed into one afternoon
# nobody would be in the room for. Calibrated, it is one death a day.
#
# So a wallet is not a lifespan handed out. It is the money that voice is
# predicted to *spend* by the hour it is meant to end, which is mostly its share
# of the warm card — and a voice meant to live longer is therefore given more.
#
# None of the three tables below is live now. Eight was the live one and it was
# calibrated for an A100 at $2.50; the choir has since grown a ninth voice and
# moved to an H200 at $4.54, so every number in all three is against a price the
# card no longer costs. They are kept as the record of a shape that worked — one
# death a day, the elder on the bell, the child always after — and not as numbers
# to read. Recalibrating is one run of `mortality_sim.py --calibrate`, and what it
# is waiting for is the decision under `DECLARED_USD`, not an afternoon.
#
# CAST OF SIX — stale: calibrated for birth on 18 September.
PLANNED_WALLETS_SIX: dict[str, float] = {
    "voice_05": 391.56,   # qwen3       — 8B; dies Wed 28 Oct 21:00, opening night
    "voice_03": 394.26,   # olmo-3      — 7B; Thu 29 Oct 22:00, after closing time
    "voice_02": 391.26,   # phi-4-mini  — 3.8B; Fri 30 Oct 16:00
    "voice_04": 404.68,   # eurollm     — 1.7B stand-in; Sat 31 Oct 17:00
    "voice_00": 459.14,   # gpt-2       — the elder; Mon 2 Nov 20:00, on the bell
    "voice_01": 459.10,   # tinystories — the child; Mon 2 Nov 21:36, always after
}

# CAST OF SEVEN — the six and Voxtral, which is what this is aiming at. It is
# the ear with a loader, the ear that heard two voices as two, and at 9.2 GB the
# ear that fits beside the others on the one card the whole sum is built on.
#
# One death a day, which the six could not quite manage: the seventh fills
# 1 November, the one day of the exhibition the smaller cast leaves silent.
PLANNED_WALLETS_SEVEN: dict[str, float] = {
    "voice_05": 337.00,   # qwen3       — 8B; dies Wed 28 Oct 21:00, opening night
    "voice_03": 338.32,   # olmo-3      — 7B; Thu 29 Oct 22:00, after closing time
    "voice_06": 337.46,   # voxtral     — the ear; Fri 30 Oct 16:00, in daylight
    "voice_02": 350.08,   # phi-4-mini  — 3.8B; Sat 31 Oct 17:00, Halloween
    "voice_04": 362.66,   # eurollm     — 1.7B stand-in; Sun 1 Nov 16:00
    "voice_00": 387.15,   # gpt-2       — the elder; Mon 2 Nov 20:00, on the bell
    "voice_01": 387.33,   # tinystories — the child; Mon 2 Nov 21:36, always after
}

# CAST OF EIGHT — the six, Voxtral and Audio Flamingo, without Moshi. About
# 68 GB of the 79 usable, so it still lives on the one card the whole sum is
# built on, but with less room than any other option.
#
# Audio Flamingo dies on the opening night, and it is the only table where the
# first to go is an ear. That is not a decision either: at 8B with an audio
# front end it is the biggest mouth in the room, more expensive per breath than
# qwen3. Eight deaths into a six-day window means two fall on Halloween.
PLANNED_WALLETS_EIGHT: dict[str, float] = {
    "voice_07": 295.60,   # audio-flamingo — the ear for rooms; Wed 28 Oct 21:00
    "voice_05": 302.15,   # qwen3       — 8B; Thu 29 Oct 22:00, after closing time
    "voice_03": 301.08,   # olmo-3      — 7B; Fri 30 Oct 16:00
    "voice_06": 303.31,   # voxtral     — the ear for tuning; Sat 31 Oct 15:00
    "voice_02": 302.73,   # phi-4-mini  — 3.8B; Sat 31 Oct 19:00, Halloween, second
    "voice_04": 314.13,   # eurollm     — 1.7B stand-in; Sun 1 Nov 16:00
    "voice_00": 339.98,   # gpt-2       — the elder; Mon 2 Nov 20:00, on the bell
    "voice_01": 341.02,   # tinystories — the child; Mon 2 Nov 21:48, always after
}

# CAST OF NINE — the cast that sings, and the one that will be born.
#
# Derived on 19 September, hours before the birth, from
#
#     python3 mortality_sim.py --calibrate --cast nine --night 2.2
#
# against $4,540 — a thousand warm hours of the H200 — and checked by replaying
# the same wallets with nothing scripted. The per-breath cost underneath it is
# measured rather than estimated: the nine were left singing and the medians were
# read off the log the same afternoon, which moved several of these numbers by
# thirty or forty dollars and is the difference between a death date and a guess. The check reproduces the composed
# calendar to the minute: llm-jp on the opening night, one death a day through the
# show, eurollm inside the closing afternoon, gpt-2 on the closing bell at 20:03,
# and the child an hour and a half after everyone, as it must be.
#
# The night is 2 h 12 min of cold card, up from 1 h 36 min, and that is what makes
# the sum come out at a thousand hours exactly rather than 1,022. It also gives
# back the operating slack the later birth had taken away, which is the one place
# in this file where the arithmetic and the sleep were both improved by the same
# decision.
#
# There was an earlier table here, calibrated for Moshi at $2.50 an hour against a
# birth on the 18th. It was deleted rather than annotated, because kept as numbers
# it would have matched this cast on the set of ids and seeded silently — the one
# failure this file exists to prevent. These numbers replace it because all four
# facts underneath it are now settled: the cast is these nine, the card is the
# H200, the birth is the evening of the 19th, and the declared sum is a thousand
# hours of the card the choir actually lives on.
PLANNED_WALLETS_NINE: dict[str, float] = {
    "voice_08": 467.33,   # llm-jp      — 13B, the newest voice; Wed 28 Oct 21:00
    "voice_07": 456.83,   # audio-flamingo — the ear for rooms; Thu 29 Oct 22:00
    "voice_05": 481.96,   # qwen3       — 8B; Fri 30 Oct 16:00
    "voice_03": 511.11,   # olmo-3      — 7B, the priciest breath; Sat 31 Oct 15:00
    "voice_06": 483.82,   # voxtral     — the ear for tuning; Sat 31 Oct 19:00
    "voice_02": 509.52,   # phi-4-mini  — 3.8B; Sun 1 Nov 16:00
    "voice_04": 552.20,   # eurollm     — 9B; Mon 2 Nov 15:00, inside the closing
    "voice_00": 539.28,   # gpt-2       — the elder; Mon 2 Nov 20:00, on the bell
    "voice_01": 537.95,   # tinystories — the child; Mon 2 Nov 21:30, always after
}


def planned_wallets(cast_ids: list[str]) -> dict[str, float]:
    """The wallets for the cast that actually loaded, or nothing at all.

    Deliberately refuses to improvise. A cast this table was not calibrated for
    would get plausible-looking numbers and wrong death dates, and the error
    would only become visible in late October when it is unfixable. Better to
    have no birth record and a loud complaint in the logs on 18 September, while
    somebody is still watching.
    """
    ids = set(cast_ids)
    for table in (PLANNED_WALLETS_NINE, PLANNED_WALLETS_EIGHT,
                  PLANNED_WALLETS_SEVEN, PLANNED_WALLETS_SIX):
        if ids == set(table):
            return {k: table[k] for k in cast_ids}
    return {}


# ---------------------------------------------------------------------------
# How much of a warm card a slice of the log accounts for
# ---------------------------------------------------------------------------

# The longest gap between two log lines that is still assumed to be a warm card.
# Longer than this and we assume the container was down and charge only this
# much. The choir writes a heartbeat every 30 seconds, so in normal running the
# clamp never binds; it exists so that a restart, an outage or a redeploy cannot
# invent hours nobody paid for. It errs in the voices' favour, which is the
# right direction: the log must never overstate what was spent.
MAX_SLICE_S = 180.0

# When a voice's forecast drops below this, it is dying and the room should be
# able to hear it: its breaths crowd together (see `sleep_after_breath`).
DYING_S = 900.0

# And below this, the next breath is the last one it can afford, so it takes the
# longest tone the score allows instead of the one it chose.
LAST_BREATH_S = 75.0

# The score's own ceiling on a breath (live_v1 clamps 1–15 s).
BREATH_MAX_S = 15.0


# ---------------------------------------------------------------------------
# One life
# ---------------------------------------------------------------------------


@dataclass
class Life:
    """One voice's account. Everything here is derived from the log."""

    voice_id: str
    seed_usd: float
    born_ts: float
    spent_usd: float = 0.0
    own_s: float = 0.0            # card seconds charged to it alone (rule 2)
    shared_s: float = 0.0         # card seconds charged as its equal share (rule 3)
    # Of `own_s`, the part spent becoming rather than breathing, and what it cost.
    # Kept separately because it is the one expense a visitor can be told about in
    # a sentence — this voice has spent this much of its life learning from you —
    # and because it is bought on a different card at a different price, so it
    # cannot be recovered later by multiplying seconds by the rate on the wall.
    learn_s: float = 0.0
    learned_usd: float = 0.0
    adapters: int = 0             # how many times it has been changed by learning
    breaths: int = 0
    answered_human_usd: float = 0.0   # spent on breaths that tuned to a person
    last_words: str = ""
    last_breath_ts: float = 0.0
    died_ts: float | None = None
    # Money that reached this wallet after birth, net: gifts in, minus what it
    # handed to another voice. Kept apart from `seed_usd` so the birth record stays
    # what it was and every later change of a lifespan is a line anybody can read.
    given_usd: float = 0.0
    # A smoothed dollars-per-second, for the screens. The instantaneous rate is
    # useless to look at — it is either a share of an idle card or a burst of
    # generation — and a forecast that jumps is a forecast nobody believes.
    burn_usd_per_s: float = 0.0

    @property
    def alive(self) -> bool:
        # An empty wallet is death whether or not the record got written. If the
        # container dies between the last charge and the epitaph, the voice is
        # still gone: the stop belongs to the arithmetic, not to the bookkeeping.
        return self.died_ts is None and self.wallet_usd - self.spent_usd > 0.0

    @property
    def wallet_usd(self) -> float:
        return self.seed_usd + self.given_usd

    @property
    def left_usd(self) -> float:
        return max(0.0, self.wallet_usd - self.spent_usd)

    @property
    def gpu_s(self) -> float:
        return self.own_s

    def forecast_s(self) -> float:
        """Seconds of life left at the rate it is burning now. inf if unknown."""
        if not self.alive:
            return 0.0
        if self.burn_usd_per_s <= 0:
            return float("inf")
        return self.left_usd / self.burn_usd_per_s


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

# Half-life of the burn-rate smoothing, in seconds of warm card.
_BURN_HALFLIFE_S = 600.0


class Ledger:
    """The log, replayed. Feed it every line of `choir.jsonl` in order.

    The same `apply()` runs live inside the choir's container as it writes each
    line and offline when anybody replays the file, so there is exactly one
    implementation of the arithmetic and the audit cannot disagree with the
    performance. `mortality_sim.py` feeds it invented logs; the portal feeds it
    the real one, incrementally, as it grows.

    One invariant holds at every line, and it is the whole audit:

        charged_usd == rate * (warm_s + _owed)

    that is, the money taken out of the wallets is exactly the card that was
    held, plus the handful of attributed seconds that arrived just after the
    slice they belonged to and will be taken out of the next one. Nothing is
    billed twice and nothing is free. `test_mortality.py` asserts it.
    """

    def __init__(self, rate_usd_per_s: float = RATE_USD_PER_S) -> None:
        self.rate = rate_usd_per_s
        self.lives: dict[str, Life] = {}
        self.declared_usd = 0.0
        self.birth_ts: float | None = None
        self.resident: set[str] = set()
        self.warm_s = 0.0             # warm card seconds the log accounts for
        self._owed = 0.0              # attributed seconds not yet taken out of a slice
        self.charged_usd = 0.0
        self.unbilled_usd = 0.0       # warm card with nobody resident to charge
        self.clock: float | None = None   # end of the last slice settled
        self.session = False          # is a container currently holding the card
        self.deaths: list[dict] = []
        self.gifted_usd = 0.0         # money from outside the declared sum
        self.gift_ids: set[str] = set()
        # The wallets, once an hour, since birth: what the screen in the room
        # draws as nine falling lines. Taken inside `_settle`, so a replay of
        # the log rebuilds the whole curve and nothing has to be stored.
        self.history: list[tuple[float, dict[str, float]]] = []
        self._next_sample = 0.0

    # -- reading the log ----------------------------------------------------

    def apply(self, e: dict) -> list[str]:
        """One log line. Returns the voice_ids that ran out inside it.

        Nothing at all happens before the birth record: the choir was tested for
        weeks before Endzeit and none of that is anybody's life.
        """
        kind = str(e.get("kind") or "")
        ts = float(e.get("timestamp") or 0.0)

        if kind == "birth":
            self._birth(e, ts)
            return []
        if self.birth_ts is None:
            return []

        if kind == "gift":
            self._gift(e)
            return []

        if kind == "card":
            # A heartbeat from the container that holds the card. It carries the
            # authoritative list of what is resident on it, so a restart that
            # loaded fewer voices needs no reconciliation logic here.
            # An explicit empty list means "nothing is on the card yet", which is
            # what a container says while it is still loading. Only a missing
            # field leaves the previous list standing.
            resident = e.get("resident")
            if not self.session:
                self.session, self.clock = True, ts
                if resident is not None:
                    self.resident = set(resident)
                return []
            if resident is not None:
                self.resident = set(resident)
            return self._settle(ts)

        if kind == "exit":
            # The container let the card go. Charge up to here, then stop the
            # clock: the gap until the next container is nobody's money.
            died = self._settle(ts) if self.session else []
            self.session, self.clock, self.resident = False, None, set()
            return died

        if kind == "death":
            # Authoritative when replaying a log written by a previous run.
            vid = str(e.get("voice_id") or "")
            died = self._settle(ts) if self.session else []
            life = self.lives.get(vid)
            if life and life.alive:
                life.died_ts = ts
                self.deaths.append(e)
            self.resident.discard(vid)
            return [d for d in died if d != vid]

        # Everything else that names a voice and used the card: a breath
        # (`kind == "tone"`), a voice's weights arriving (`load`), a stream of
        # full-duplex listening (`stream`), tonight's LoRA (`train`).
        vid = str(e.get("voice_id") or "")
        if kind == "load" and vid:
            self.resident.add(vid)
        if not self.session:
            return []
        own = max(0.0, float(e.get("gpu_s") or 0.0))
        # A night of learning is bought on its own card, away from the room: the
        # eight resident voices cannot be asked to step aside for hours while one
        # of them is rewritten, and a card with nine models on it has no memory
        # left to train in. So the school's hours are extra hours, at the school
        # card's own price, and the event carries that price with it — seconds
        # bought at $2.50 must not be re-valued at the room's $4.54 later.
        #
        # This is the one place where the arithmetic on the wall stops being a
        # single card's hourly rate. It is also the only way the sentence is true:
        # learning costs a voice its life, and now it does, in the same wallet it
        # breathes out of. What it buys is not a share of anything — a shared
        # charge would mean the others paid for one voice's schooling — and it is
        # not carved out of the room's warm slice either, because the room stayed
        # warm and shared all night whether or not anybody was at school.
        rate = float(e.get("rate_usd_per_hour") or 0.0) / 3600.0
        elsewhere = kind == "train" and rate > 0.0
        died = self._settle(ts, vid, own, rate if elsewhere else None)
        life = self.lives.get(vid)
        if life is not None and kind == "train":
            life.learn_s += own
            life.learned_usd += own * (rate or self.rate)
            if e.get("version"):
                life.adapters += 1
        if life is not None and kind == "tone":
            life.breaths += 1
            life.last_words = str(e.get("content") or "")[:2000]
            life.last_breath_ts = ts
            if str(e.get("tuning_to") or "").startswith("human"):
                life.answered_human_usd += own * self.rate
        return died

    def _birth(self, e: dict, ts: float) -> None:
        if self.birth_ts is not None:
            return                     # one birth, ever; later ones are noise
        self.birth_ts = ts
        self.declared_usd = float(e.get("declared_usd") or 0.0)
        rate_h = e.get("rate_usd_per_hour")
        if rate_h:
            self.rate = float(rate_h) / 3600.0
        for vid, seed in (e.get("wallets") or {}).items():
            self.lives[vid] = Life(voice_id=vid, seed_usd=float(seed), born_ts=ts)

    def _gift(self, e: dict) -> bool:
        """Money into a living wallet: from outside, or handed over by another voice.

        The only way a lifespan changes after birth, and it changes in the open.
        A gift reaches no one dead — the refusal to revive is the same refusal as
        everywhere else in this file — and a voice can hand over only what it
        could live without: a transfer that would empty the giver is refused
        whole rather than turned into a death nobody's breathing caused. It uses no
        card, so it settles no slice. Each `gift_id` counts once, so the file of
        gifts can be read again from the top by anybody who replays it.
        """
        gid = str(e.get("gift_id") or "")
        usd = float(e.get("usd") or 0.0)
        to = self.lives.get(str(e.get("voice_id") or ""))
        src_id = str(e.get("from_voice_id") or "")
        src = self.lives.get(src_id) if src_id else None
        if not gid or gid in self.gift_ids or usd <= 0.0:
            return False
        if to is None or not to.alive:
            return False
        if src_id and (src is None or not src.alive or src.left_usd <= usd
                       or src is to):
            return False
        self.gift_ids.add(gid)
        to.given_usd += usd
        if src is not None:
            src.given_usd -= usd
        else:
            self.gifted_usd += usd
        # A gift is a step on the line, and the screen should draw the step.
        self._sample(max(float(e.get("timestamp") or 0.0), self.clock or 0.0))
        return True

    # -- the arithmetic ----------------------------------------------------

    def _settle(self, ts: float, own_voice: str = "", own_s: float = 0.0,
                elsewhere_rate: float | None = None) -> list[str]:
        """Charge the slice of warm card that ends at `ts`.

        Two charges, and between them they are the whole slice, which is what
        makes the sum of the wallets fall at exactly the card's price and by no
        other rule.

        `own_s` seconds were spent inside `own_voice` and are charged to it at
        once — not conditionally on the clock, because two log lines can share a
        timestamp (a heartbeat and a full-duplex voice's stream, for instance)
        and a slice of zero length must not swallow real seconds. Those seconds
        are then remembered in `_owed` and subtracted from the warm slices they
        happened inside, so nobody is billed for them twice.

        The rest of the slice is split equally between the living residents.

        `elsewhere_rate` says those seconds were bought on another card, at that
        price: they are charged to the one voice exactly the same way, and that is
        the point — but they are not remembered in `_owed`, because they were
        never inside a slice of this room's warm card to begin with.
        """
        own_s = max(0.0, own_s)
        charges: dict[str, float] = {}

        if own_voice and own_s > 0:
            life = self.lives.get(own_voice)
            if life is not None and life.alive:
                life.own_s += own_s
                charges[own_voice] = own_s * (elsewhere_rate or self.rate)
            if elsewhere_rate is None:
                self._owed += own_s

        if self.clock is None:
            self.clock = ts
            return []
        dt = ts - self.clock
        late = dt < 0
        if dt > 0:
            # The clock only advances. A line from the past — the portal reads
            # the school's bills after the choir's log, each stamped minutes or
            # weeks earlier — charges what it names and nothing else; it does
            # not wind the card back so that the next line pays for a slice
            # that was already paid for. 26 October: that rewind, over some two
            # hundred bills, had the portal's wallets $54 below the choir's.
            self.clock = ts
            dt = min(dt, MAX_SLICE_S)   # a gap this long was a container restart
            self.warm_s += dt
            spoken_for = min(self._owed, dt)
            self._owed -= spoken_for
            pool = dt - spoken_for
            living = [v for v in self.resident
                      if v in self.lives and self.lives[v].alive]
            if living:
                per = pool / len(living)
                for vid in living:
                    self.lives[vid].shared_s += per
                    charges[vid] = charges.get(vid, 0.0) + per * self.rate
            else:
                # A warm card with nobody resident on it: a container still
                # loading, or the hours after the last death. Chargeable to
                # nobody either way, and in both cases that is right — the
                # loading itself is already charged by rule 2, and what the card
                # costs after the last voice is gone is not part of any life.
                self.unbilled_usd += pool * self.rate
        else:
            dt = 0.0

        over = dt if dt > 0 else max(own_s, 1.0)
        died: list[str] = []
        for vid, usd in charges.items():
            life = self.lives[vid]
            # A line from the past is money, not weather: it says nothing
            # about how fast the voice is spending now, so it leaves the
            # forecast alone.
            self._charge(life, usd, over, smooth=not late)
            if life.left_usd <= 0.0:
                died.append(vid)
        if self.clock >= self._next_sample or died:
            self._sample(self.clock)
        return died

    HISTORY_STEP_S = 3600.0

    def _sample(self, ts: float) -> None:
        """One point per voice on the falling lines; also at every death."""
        self.history.append((round(ts, 1), {vid: round(max(0.0, life.left_usd), 2)
                                            for vid, life in self.lives.items()}))
        self._next_sample = ts - ts % self.HISTORY_STEP_S + self.HISTORY_STEP_S

    def _charge(self, life: Life, usd: float, over_s: float, smooth: bool = True) -> None:
        life.spent_usd += usd
        self.charged_usd += usd
        if not smooth:
            return
        # Exponential smoothing with a half-life in warm seconds, so a slow hour
        # and a busy hour are weighted by how long they lasted.
        w = 1.0 - 0.5 ** (over_s / _BURN_HALFLIFE_S)
        life.burn_usd_per_s += w * ((usd / over_s) - life.burn_usd_per_s)

    # -- what the room asks it --------------------------------------------

    def alive(self, voice_id: str) -> bool:
        life = self.lives.get(voice_id)
        return bool(life and life.alive)

    def dead(self, voice_id: str) -> bool:
        """The hard stop, and the only question the loader asks.

        Note that it is not `not alive`. Before the birth record exists nobody
        has a wallet, and a voice without a wallet is unborn rather than dead —
        which is what lets the choir be tested in August without either dying
        or quietly starting to spend.
        """
        life = self.lives.get(voice_id)
        return bool(life and not life.alive)

    def forecast_s(self, voice_id: str) -> float:
        life = self.lives.get(voice_id)
        return life.forecast_s() if life else float("inf")

    def living_ids(self) -> list[str]:
        return [v for v, l in self.lives.items() if l.alive]

    def total_left_usd(self) -> float:
        """The big number in the room: what is left to the whole choir."""
        return sum(l.left_usd for l in self.lives.values())

    def share_usd_per_hour(self) -> float:
        """Each living voice's share of an idle card, right now."""
        n = len(self.living_ids())
        return RATE_USD_PER_HOUR / n if n else 0.0

    def state(self, now: float | None = None) -> dict:
        """Everything the nine screens and the big number need, as plain data."""
        return {
            "declared_usd": round(self.declared_usd, 2),
            "gifted_usd": round(self.gifted_usd, 2),
            "born": self.birth_ts,
            "total_left_usd": round(self.total_left_usd(), 2),
            "total_spent_usd": round(self.charged_usd, 2),
            "warm_hours": round(self.warm_s / 3600.0, 2),
            "total_learned_usd": round(sum(l.learned_usd
                                           for l in self.lives.values()), 2),
            "living": len(self.living_ids()),
            "share_usd_per_hour": round(self.share_usd_per_hour(), 4),
            "rate_usd_per_hour": round(self.rate * 3600.0, 4),
            "voices": [self.voice_state(v) for v in self.lives],
        }

    def voice_state(self, voice_id: str) -> dict:
        life = self.lives[voice_id]
        forecast = life.forecast_s()
        # The three things a wallet goes on, and they are three different kinds of
        # spending, not one number sliced for the screens: singing is what it did,
        # residency is what it cost to be in the room at all, and learning is what
        # it paid to stop being what it was. Residency is derived from seconds and
        # singing from the remainder, so the three always add up to `spent_usd`
        # even though learning was bought at another card's price.
        resident_usd = life.shared_s * self.rate
        return {
            "voice_id": voice_id,
            "alive": life.alive,
            "dying": life.alive and forecast <= DYING_S,
            "seed_usd": round(life.seed_usd, 2),
            "given_usd": round(life.given_usd, 2),
            "spent_usd": round(life.spent_usd, 4),
            "left_usd": round(life.left_usd, 4),
            "hours_left": (None if forecast == float("inf")
                           else round(forecast / 3600.0, 2)),
            "verdict": verdict(life),
            "born": life.born_ts,
            "died": life.died_ts,
            "breaths": life.breaths,
            "gpu_s": round(life.own_s, 2),
            "shared_s": round(life.shared_s, 2),
            "sang_usd": round(max(0.0, life.spent_usd - resident_usd
                                  - life.learned_usd), 4),
            "resident_usd": round(resident_usd, 4),
            "learned_usd": round(life.learned_usd, 4),
            "learn_s": round(life.learn_s, 1),
            "adapters": life.adapters,
            "answered_human_usd": round(life.answered_human_usd, 4),
            "last_words": life.last_words,
            "last_breath": life.last_breath_ts,
            "burn_usd_per_hour": round(life.burn_usd_per_s * 3600.0, 4),
        }


def verdict(life: Life) -> str:
    """The forecast in words, because a ticking clock turns dying into a game.

    Coarse on purpose (`EXHIBITION.md` §6): the screen of a voice with hours to
    live says nothing at all, and only a dying one speaks.
    """
    if not life.alive:
        return "dead"
    left = life.forecast_s()
    if left == float("inf"):
        return ""
    if left <= 600:
        return "dies within the hour"
    if left <= 2 * 3600:
        return "dies today"
    if left <= 8 * 3600:
        return "hours left"
    return ""


# ---------------------------------------------------------------------------
# Records written into the log
# ---------------------------------------------------------------------------


def may_be_born(now: float, heard: bool) -> str:
    """Whether the wallets may be seeded yet, and if not, why not.

    Empty string means yes. Anything else is the reason, said the way the
    container should print it, and the two reasons are not the same kind of
    thing: before the gate the work has not opened, after it with an empty
    field the work has opened and nobody has arrived.

    Here rather than in `choir_app` because it is the one decision in the piece
    that happens exactly once and cannot be taken back, and this file can be
    tested on a laptop with no card, no volume and no models.
    """
    if now < BIRTH_TS:
        return "before the gate — no wallets, no charging, nobody is dying"
    if not heard:
        return ("the gate has passed and the field is empty — "
                "waiting to be heard before anybody is born")
    return ""


def birth_record(ts: float, wallets: dict[str, float], cast: list[dict]) -> dict:
    """The one line that starts the clock. Written once, ever.

    It carries the wallets themselves rather than a reference to this file, so
    that from the moment it is written the seeds are a historical fact. Editing
    `PLANNED_WALLETS_*` afterwards changes nothing for anybody already alive.
    """
    return {
        "timestamp": ts,
        "kind": "birth",
        "declared_usd": round(sum(wallets.values()), 2),
        "rate_usd_per_hour": RATE_USD_PER_HOUR,
        "wallets": {k: round(v, 2) for k, v in wallets.items()},
        "cast": cast,
        "note": ("given the first minute a person was heard in the field, on "
                 "the evening of 19 September 2026 in San Francisco — a "
                 "thousand warm hours of one H200, split among the nine voices"),
    }


def death_record(life: Life, ts: float, short_name: str, freed_gb: float,
                 living_after: int) -> dict:
    """An epitaph, not an alarm.

    It is written the moment the wallet empties, which is the moment after the
    long last breath rather than at the end of the silence that follows. The
    room needs it then: the screen becomes a headstone while the speaker is
    still ringing, and nothing announces anything.
    """
    return {
        "timestamp": ts,
        "kind": "death",
        "voice_id": life.voice_id,
        "model_short": short_name,
        "born": life.born_ts,
        "lived_s": round(ts - life.born_ts, 1),
        "seed_usd": round(life.seed_usd, 2),
        "given_usd": round(life.given_usd, 2),
        "spent_usd": round(life.spent_usd, 4),
        "breaths": life.breaths,
        "gpu_s": round(life.own_s, 2),
        "shared_s": round(life.shared_s, 2),
        "answered_human_usd": round(life.answered_human_usd, 4),
        "last_words": life.last_words,
        "freed_gb": round(freed_gb, 2),
        "living_after": living_after,
    }


def gift_record(ts: float, gift_id: str, voice_id: str, usd: float,
                from_voice_id: str | None = None, note: str = "") -> dict:
    """One line of `gifts.jsonl`: who received, how much, and from whom.

    `from_voice_id` empty means the money came from outside the declared sum.
    """
    return {
        "timestamp": ts,
        "kind": "gift",
        "gift_id": gift_id,
        "voice_id": voice_id,
        "usd": round(usd, 2),
        "from_voice_id": from_voice_id or None,
        "note": note,
    }


def heartbeat_record(ts: float, resident: list[str]) -> dict:
    """Proof that the card was warm, and of who was on it.

    Without this the ledger cannot tell a quiet choir from a stopped one, and
    the difference is the whole month of silence that the work is paying for.
    """
    return {"timestamp": ts, "kind": "card", "resident": sorted(resident)}


# ---------------------------------------------------------------------------
# The policy the room hears — shared by the choir and the simulation
# ---------------------------------------------------------------------------


def is_dying(forecast_s: float) -> bool:
    return forecast_s <= DYING_S


# How much longer a voice rests when the field is empty. It was eleven — the
# number that reproduced the old empty-field minute, and the pace the wallets
# were calibrated at. Softened to five on 24 September, the day the tones were
# lengthened to 8–15 s (`choir_app.TONE_FLOOR_S`), and the two changes were
# made together because they cancel: a longer breath means a longer rest and
# fewer breaths an hour, a softer stretch means more, and at five the composed
# calendar of deaths holds within hours on the seeded wallets — `_gift_check.py`
# is the arithmetic. What the pair buys is a field that is continuous even
# with nobody in it, at no cost to the total: the card burns at its price
# whether the choir holds its tones or clips them, and the stretch only ever
# decided who pays.
#
# Exhibition week, 28 October – 1 November: a gallery is full of people who
# listen without singing, and a choir that thins out ten minutes after the
# last human note would be thin for most of the day. For the week the stretch
# is 1 — the choir breathes the same whether anybody sang or not.
ALONE_STRETCH = 1.0


def sleep_after_breath(breath_s: float, forecast_s: float,
                       alone: bool = False) -> float:
    """How long a voice rests after singing.

    As long as its own breath was. This is the rule the June concert ran on and
    it is the reason the choir has no metre: a voice that sings for nine seconds
    rests for nine and one that sings for two rests for two, so their periods
    differ, drift apart, and cross. Nothing here divides a shared bar between
    them, and a longer breath lengthens a voice's whole cycle instead of eating
    into a slot it was allotted.

    Alone in an empty field, every voice rests `ALONE_STRETCH` times longer.
    That is money and not music — a card at $2.50 an hour singing to nobody
    spends a month of wallets in days — and stretching each voice by its own
    breath keeps the ratios between them that a common cadence would flatten.

    A voice with minutes to live stops resting altogether. Its pauses close up
    and its breaths crowd together, which is the only warning the room gets and
    the only one it needs.
    """
    if is_dying(forecast_s):
        return 1.0
    rest = max(breath_s, 1.0)
    return rest * (ALONE_STRETCH if alone else 1.0)


def final_breath_s(breath_s: float, forecast_s: float) -> tuple[float, bool]:
    """The longest tone it can still afford, if this is the last one.

    Returns the breath length and whether it is final. The player uses the flag
    to hold the note; nothing else in the system changes.
    """
    if forecast_s <= LAST_BREATH_S:
        return BREATH_MAX_S, True
    return breath_s, False


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay(events: list[dict], rate_usd_per_s: float = RATE_USD_PER_S) -> Ledger:
    """The audit: the whole ledger rebuilt from the log and nothing else.

    In file order, deliberately not sorted by timestamp. The log is append-only,
    so the order it was written in *is* the chronology, and a clock that steps
    backwards on a container restart should not be able to move a birth after a
    heartbeat or a death before the breath that caused it. Timestamps are used
    for the length of each slice, and `_settle` ignores the negative ones.
    """
    led = Ledger(rate_usd_per_s)
    for e in events:
        led.apply(e)
    return led
