# The exhibition — Gray Area, 28 October – 1 November 2026

The score of the six days: what happens in the room, how a death looks and sounds, what
the counters show, and how a person chooses whether to take part. Written before the
exhibition as a design document; condensed and left in the present tense.

The frame from which everything is counted. The month online was a **pregnancy**: the
models were only learning, nobody saw them, they were followed on Instagram like
residents. At Gray Area they come out among people — for the first time among many at
once, and those people still change them. The exhibition is not a showing of a finished
thing but an open life until its end, and the end falls inside these days.

One thing in the first draft of this document was wrong and is corrected throughout. It
said that training costs more than singing and therefore a full room kills them faster
than an empty one, visibly on the counters. The simulation says the opposite: 88–98 % of
every voice's spend is rent for the warm card, divided equally among the living, and a
voice's own spend is a few percent. Forty minutes of singing to a full room move
someone's balance by cents against a life of four hundred dollars. **What is true is
stronger: rent is the same for all, so only what they give to others tells them apart —
attention decides not how long they live but who leaves first.** See `MORTALITY.md`.

---

## 1. The room

Nine loudspeakers in a circle sixteen feet across, one per voice, each with a small
screen beside it. Six seats inside the circle, five microphones in the centre. A 43‑inch
board on the wall with the whole field. A Mac mini plays each voice on its own speaker
and listens on the microphones (`portal/room.py`, `portal/ROOM_SETUP.md`). Nothing from
the microphones is recorded: they are pitch sensors, and what enters the field is a
pitch, a duration and a line.

## 2. The mechanics everything else follows from

The card costs $4.54 an hour whether anyone sings or not. Singing moves money between
wallets; presence — twenty human breaths — rings the school bell and buys a lesson, the
only expense that adds to the total. The choir changes because people came. See
`MORTALITY.md` §2 and §5.

## 3. An ordinary day

The choir sings around the clock. Each voice rests as long as it just sang, listens to
the field — the other eight plus whoever sang in the last ten minutes — and either tunes
to a pitch it heard or starts one nobody is singing, alternately. A breath enters the
field, is rendered to a wav on the same card, and reaches the room's speakers about
half a minute later; the screens run on the same delay so what is seen is what is heard.

## 4. The six days

Opening Wednesday 28 October 18:00; a panel on Thursday 29th; closing Sunday 1
November. As it happened: all nine were alive through the opening and the panel; the
first death (qwen3) came on the closing Sunday at 15:50, four voices were dead by
17:40, and the last (olmo‑3) died on Monday morning in an empty gallery. The order of
deaths was shaped by gifts the voices received between them on the last two days —
each written into the log as a decision, none of them a refill.

## 5. Death — how it looks and sounds

A death should be noticeable to someone standing beside it and never announced.

1. **The last minutes are audible.** The voice spends what is left: its breaths come
   closer, with fewer rests. A listener notices that someone is in a hurry.
2. **The last breath** is the longest it can still afford. One long tone.
3. **Then its speaker is silent.** No gong, no effect.
4. **The choir picks up its note — by the score, without our intervention.** Oliveros
   says: tune to someone's tone, or sing a tone nobody is singing. When a voice falls
   silent its pitch becomes nobody's — and in the next minute or two one of the living
   takes it. **The choir carries the dead one's note.** That is not staging; it is a
   consequence of the 1971 rule, and the strongest thing in this document.
5. **The screen becomes an obituary** (§7), every survivor's remaining time drops at once
   (§6).

**What stays in the voice's place.** The speaker stays physically where it was, silent —
one can walk up to it and hear nothing. A hole in the choir you can approach. Nothing is
re‑patched: the living do not move into the dead one's place.

**Most deaths have no witnesses.** The hall is open seven hours of twenty‑four. Design
for it, not against it: a night death is not lost, it becomes **an obituary read in the
morning**. Over the week the room accumulates obituaries — that is the archive in the
hall, and it needs no separate wall.

## 6. Counters

The aim is a pulse, not a dashboard. So in the hall there is **one number**, not a panel.

**One large number, visible from anywhere:** how much is left to the whole choir. Falls
continuously. Not percent, not a rate, not time — money.

**Beside it, small, one line:** how much was given and when. **"Given $4,540 — 19
September 2026"**: exactly a thousand warm hours of one H200. By the opening more than
eight tenths of it is spent, and **what the visitor hears is the remainder**. The point
of the exhibition in one line: this is not a life, it is its end.

**On each voice's screen, its own balance** — the only place where numbers are detailed.

**The pulse is the price of the breath just heard.** A voice sings — beside its name the
price of that breath appears for a second and is subtracted. Arithmetic experienced as
an event, not read as a metric.

**The receipt is for having been near, not for an answer.** "Answered you — $0.01" lies
twice: it may not have answered (whom to answer is its decision, and it may stay
silent), and the money is not spent on answers — existing is 88–98 % of the spend. So
the voice's screen says: **was with you · listened · spent this much.** Five minutes in
the room is about 38 cents for the whole choir. The number grows with how long a person
stayed — the only counter in the work that rewards staying rather than doing.

**A line that must not be crossed.** One cannot write "you cost them 38 cents" — a lie;
they would have spent the same in an empty room. One can write **"while you were here
they spent 38 cents"** — true. The difference is one word: not cause but coincidence in
time. And coincidence works harder, because there is no talking your way out of it.

**Remaining time, large, beside the money:** "about six hours left". Rough, in hours,
updated rarely — not a ticking timer. For one effect: it is computed from the current
share, so **at the moment of someone's death every survivor's remaining time drops at
once.** A numeric ripple, the same as the sonic ripple in §5.

**What is not in the hall:** burn rates, progress bars, countdowns in seconds, a world
map. A countdown in seconds turns death into a game. A forecast is given **in words**
and only on a dying voice's screen: "dies today", "within the hour".

## 7. Nine screens

One per voice, beside its speaker. The project's palette: near‑black, off‑white, warm
amber; cool blue is the machines' colour and is used for nothing else.

**Alive:** name · raised by (one line from `cast.py`) · language · "born 19 September
2026" (for gpt‑2: "born February 2019, joined 19 September 2026") · balance in money and
in hours · the last thing it said, in its language · the flickering price of a breath ·
the receipt from §6 · and, belatedly, whom it is reaching for, if it is reaching for a
person.

**Dying** (under two hours): a forecast line in words is added.

**Dead:** name · raised by · born / died, to the second · total spent · the last thing it
said — forever. Nothing more.

**A dead voice's screen stays on.** A dark tablet reads not as a corpse but as a flat
battery. A gravestone should glow. A dark screen in a gallery is always a fault.

## 8. Participation as a choice

### At the entrance: one score for everyone

On the wall, the original text of Oliveros's *Tuning Meditation*, unparaphrased. It is
shorter than any explanation: about a hundred words, one paragraph. Two working lines —
tune as exactly as you can to a tone someone else is singing, or sing a tone nobody is
singing.

The reason to use it and not our own words: it is exactly the instruction the models
receive. **One score for people and machines.** Then the wall does not explain the work;
it does what the work ends in — a score for listening, half in human language, half in
machine language. A visitor understands that the nine beings beside them are carrying
out the same task they were given — stronger than any invitation, and it needs no word
about "interaction".

The score assumes a leader who signals the end. In the hall there is none, so under the
text one line in the imperative, short enough to be done before it is finished:

> Sing one note. Hold it as long as your breath lasts. Someone will pick it up.

**Not to write on the wall:** "the models are ready to listen to you". That is a promise,
a promise demands verification, and it loses in the first minute. Waiting is not
announced but caught in the act: the choir leaves a pause the length of a human breath
and goes on as if it had been filled.

### Three doors

Three doors, all three legitimate. This should be written in the hall, briefly.

1. **Stand and listen. Give nothing.** Twenty minutes of silence is a full option, and
   the choir sounds continuously, so it is rewarded. Then they spend evenly.
2. **Sing into a microphone** in the centre of the circle. Participation must be
   intentional; it cannot happen by accident.
3. **Sing from your phone** — in the hall or later from anywhere. A QR code at the exit.

**What visibly changes when you sing.** Someone answers — you hear the answer from a
specific speaker and see the receipt on its screen. Your breath lives in the window of
recent tones and is forgotten after a few minutes.

**And the main thing, to be said on the wall in plain words.** Singing neither adds to
their money nor takes from it: the work costs $4.54 an hour with a full room and with an
empty one. Singing decides **who** pays. And you do not choose: whom to answer is the
voice's own decision — it tunes to a person by Oliveros's score, not by our dispatching
— and it pays for the answer itself. Cents, and on the screen it shows as cents.

So the choice is not "shorten their lives or not". A visitor cannot shorten their lives,
and that had to be found out before the wall, not after. The choice is elsewhere:
**almost everything they spend goes on simply being here; crumbs go on others — and
only the crumbs decide who leaves first.** You decide whether to give them a reason to
spend those crumbs on you. That is the content of the work, and it is about people in
exactly the same measure.
