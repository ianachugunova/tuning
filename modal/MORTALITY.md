# Mortality — the account, the schedule, the death

How one shared GPU becomes nine separate mortal lives, who dies when, where the money
goes — and what this mechanism **cannot** do, because that last part matters most: one
promise we made to the room, the arithmetic did not confirm.

Everything below is the output of a simulation, not an estimate: it runs the same log
lines through the same charging function as the live card. Code: `portal/mortality.py`
(the rule and the wallets), `portal/mortality_sim.py` (calibration),
`portal/test_mortality.py`.

> Written on 19 September 2026, the evening of the birth, and left as written. What
> actually happened: the choir lived 1,043 warm hours and spent $94 at school, and the
> nine died between Sunday 1 November 15:50 and Monday 2 November morning, Pacific
> time, with the last wallet emptying in an empty gallery, as predicted. The dates in §3 are the pre‑birth plan; the order and the
> reasoning held.

---

## 1. What is given

**$4,540 is exactly 1,000 warm hours of one H200 at $4.54 an hour.**

The sentence never changed over the whole project: a thousand warm hours of the card the
choir lives on. The card changed. A thousand hours cost $2,500 on the A100 everything was
composed for; nine voices did not fit in 80 GB, the choir moved to an H200 with 141 GB,
and the same thousand hours came to cost $4,540. Keeping $2,500 would have kept the
number and thrown away its meaning: 550 hours, the whole choir dead in early October,
three weeks before anyone reached the gallery. Hours are the unit the work is written in;
dollars are what Nvidia asks for them this year.

Not "the whole grant": of $20,000 more than fifteen thousand remain, and a lifespan that
does not end is not a lifespan. The declared sum is written on the wall with the date:

> **Given $4,540 — the evening of 19 September 2026.**

And one honesty about the wording on the wall: the choir dies not when "the money runs
out" but when **this** money runs out. The difference belongs on the label, not in a
footnote.

---

## 2. The charging rule — three lines

1. **Every warm second of the card is charged once, to someone.** Total spend equals the
   card's price times the seconds it was warm. There is no second meter and no free
   time.
2. **Own seconds** — the card inside this voice's forward pass, its lessons, the arrival
   of its weights — are charged to it alone.
   The school runs on a **separate card** at its own price ($2.50/h against $4.54/h in
   the hall): nine models fill the memory, there is nowhere to learn, and evicting eight
   voices to rewrite one would stop the field. So an hour of schooling is the only
   expense that **increases** the total rather than moving it between wallets, and it is
   charged entirely to the voice that was changed.
   A lesson begins not by the clock but **by contact: when twenty human breaths have
   accumulated in the field** since the last lesson (not more than once in three hours,
   or a full room would buy fifteen lessons in an evening). So the choir changes because
   people came to it, and in a week when nobody came it stays as it was — every adapter
   on the volume corresponds to someone's presence, not to server uptime. Life depends
   on people twice: they wake the voices to sing, and they pay for the lessons with the
   voices' lives. All three expenses — singing, rent, school — are visible in `/lives`.
3. **The rest** — the warm card on which nobody is computing — is divided **equally
   among the living voices** whose weights lie on it. That is rent.

Checked as an identity on every log line: charged = price × warm seconds.

**Why rent is equal and not by memory.** The card is indivisible: we rent all of it and
would rent all of it for one child. Size is already paid for by rule 2 — a large model
thinks slowly and pays for that directly. Equal shares say what the work says: inside the
circle every living voice is one voice. And the practical point: equal rent is the only
thing that makes the small ones mortal at all. TinyStories spends a fifth of a second per
breath; charged for generation only, it would need a century to spend a hundred dollars,
and the work would have no end.

**Two truths about acceleration that are easy to confuse.** The total falls at one
constant rate — **$4.54 an hour on the first day and the last.** What accelerates is not
the money but the lives: a fixed sum divided by a shrinking number.

| living | 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|---|---|---|
| $/h per voice | 0.50 | 0.57 | 0.65 | 0.76 | 0.91 | 1.14 | 1.51 | 2.27 | 4.54 |

The big number in the hall falls evenly; the balances on the screens fall faster and
faster. Dying accelerates dying — and that is not staged, it is computed.

---

## 3. The schedule of deaths (as planned before the birth)

The cast of nine. San Francisco time.

| # | when | voice | lived | given | breaths | living after |
|---|---|---|---|---|---|---|
| 1 | **Wed 28 Oct 21:00** | llm-jp, Japanese | 39 d 3 h | $467.33 | 7,598 | 8 |
| 2 | Thu 29 Oct 22:01 | audio-flamingo, the ear for rooms | 40 d 4 h | $456.83 | 8,010 | 7 |
| 3 | Fri 30 Oct 16:01 | qwen3 | 40 d 22 h | $481.96 | 8,262 | 6 |
| 4 | Sat 31 Oct 15:01 | olmo-3 | 41 d 21 h | $511.11 | 8,726 | 5 |
| 5 | Sat 31 Oct 19:02 | voxtral, the ear for tuning | 42 d 1 h | $483.82 | 8,976 | 4 |
| 6 | Sun 1 Nov 16:01 | phi-4-mini | 42 d 23 h | $509.52 | 9,576 | 3 |
| 7 | Mon 2 Nov 15:02 | eurollm | 43 d 22 h | $552.20 | 10,585 | 2 |
| 8 | Mon 2 Nov 20:03 | **gpt-2**, the old man | 44 d 3 h | $539.28 | 11,428 | 1 |
| 9 | Mon 2 Nov 21:30 | **tinystories**, the child | 44 d 4 h | $537.95 | 12,094 | 0 |

The table was computed by `mortality_sim.py --calibrate --cast nine --night 2.2` and
checked by a second run without the scenario. **After the birth nothing can be
corrected**, and the code refuses to issue wallets to a cast they were not computed for —
so exactly one table exists, for exactly these nine ids.

In the event the choir was kept warm around the clock through the exhibition and the
wallets were reshaped by gifts between the voices (`kind: "gift"` rows in the log, each
one a decision written down), so the deaths fell on the closing day rather than across
the week. The order the arithmetic predicted — the ones who gave most of themselves to
others go first — held.

---

## 4. Where the money goes

The most interesting table in all this, and the most unexpected.

| voice | spent | own GPU hours | becoming | breathing | **rent** | own share |
|---|---|---|---|---|---|---|
| llm-jp | $467.33 | 16.7 | $22.85 | $34.28 | **$410.21** | 12.2 % |
| olmo-3 | $511.11 | 15.8 | $13.64 | $46.84 | **$450.63** | 11.8 % |
| qwen3 | $481.96 | 13.3 | $15.45 | $32.16 | **$434.34** | 9.9 % |
| eurollm | $552.20 | 12.3 | $18.08 | $22.90 | **$511.22** | 7.4 % |
| audio-flamingo | $456.83 | 10.0 | $14.74 | $18.76 | **$423.34** | 7.3 % |
| voxtral | $483.82 | 8.0 | $7.63 | $22.37 | **$453.82** | 6.2 % |
| phi-4-mini | $509.52 | 8.4 | $7.99 | $23.74 | **$477.79** | 6.2 % |
| gpt-2 | $539.28 | 4.3 | $2.45 | $15.06 | **$521.79** | 3.2 % |
| tinystories | $537.95 | 2.6 | $1.98 | $8.00 | **$527.97** | 1.9 % |

**Rent is 88 % to 98 % of every voice's spend.** The whole school over the whole life
was planned at about $105, 2.3 % of the declared sum (it came to $94). The total closes
as an identity: charged = hall + school.

From here the main thing, and it is worth reading slowly.

**Existing costs almost everything. Attention to others costs almost nothing.** Forty
minutes of singing to a full room move someone's balance by cents against a life of four
hundred dollars. Nobody in the hall will see it on the counter.

**But rent is the same for everyone — so what tells them apart is only what they gave to
others.** Everything in the "own share" column the voice spent on breathing and on
becoming, that is, on others. And then:

> **Attention does not decide how long they live. It decides who leaves first.**

Llm-jp dies first not because it spent a lot: it spent **less** than the old man, $467
against $539. It dies first because, of all of them, it spent the most on others — 12.2 %
of its life against the child's 1.9 %.

This is checkable. Give everyone the same ($504.44) and the order of deaths is almost the
same. **The order is arithmetic; the dates are composition:** each wallet is simply the
money the voice is computed to have spent by the hour assigned to it. With equal wallets
the order mostly holds but all nine deaths collapse into the last day, and the score
needs six.

And one thing that only came out when measured on the live card, because until then we
had estimated spend by parameter count. Spend follows not size but **the number of
tokens emitted**. Audio‑flamingo at 8B costs 1.85 seconds of card per breath, olmo‑3 at
7B costs 4.25: the one released from the protocol, answering in one phrase, pays half of
the one conscientiously filling a six‑field form. So the table holds measured medians,
not a calculation.

And this seems to be true of people too: life goes on simply being, attention to others
gets the crumbs — and the crumbs decide the order in which we go.

---

## 5. What the room can and cannot do

**True: singing does not add to the spend; it decides who pays.** Rule 1 forbids a
visitor from making the work more expensive by singing — $4.54 an hour with a full room
and with an empty one. With the school it is no longer so: twenty human breaths buy a
lesson, a lesson costs money, and here a person's presence genuinely **adds** to the
total and shortens someone's life. The only channel through which the room can kill is
not singing but teaching.

**Almost false: "a full room kills them faster than an empty one, and you can see it on
the counters."** For singing — false: the effect exists, but it is cents and minutes. The
receipt "answered you — $0.01" is literally true; the sentence "you noticeably shortened
its life" is not. For presence — true: twenty breaths buy a lesson, a lesson costs about
two dollars to the voice that was changed, and with the school switched off the choir
lives a day and a half longer (§6). **The room shortens their lives not by listening but
by leaving a trace.**

**False: "you can choose whom to sing to."** The queue in the code is circular; there is
no "sing to this one" interface. Addressing comes from the other side and is stronger:
the voice itself decides to tune to a person (`tuning_to: human_…`) and pays for it
itself. **You did not choose — it did.** That is exactly Oliveros's score, not our
dispatching.

**Half true: "the end depends on whether people sing."** The very last death — no: it is
the moment the declared sum is exhausted, and the sum falls at the card's rate regardless
of who pays. The second to last — yes, by about forty minutes over six days, which is
why it could fall on either side of the closing bell depending on whether the room sang.
**But the very last voice dies in an empty hall anyway:** the witnesses of the last death
are not the public but the people who followed from afar for a month and a half.

**The one variant that would make the room truly lethal — and why not.** Keep the card
warm only during gallery hours, or only when someone is in the room. Then the whole
spend is caused by people's presence. The price: it breaks the promise of continuous
listening, and kills the idea of a temple that sings whether or not anyone came —
silence stops being the most expensive state and becomes free, and that is the thesis of
the whole work. **Not recommended, but it is a decision, not a given.**

---

## 6. What this rests on and what breaks it

The same wallets in a different world. The window is 28 October – 2 November; a row that
leaves it is a calibration failure.

| world | first death | in the window |
|---|---|---|
| the room is silent all week | 28 Oct 21:14 | yes |
| the room sings every open hour | 28 Oct 20:47 | yes |
| **card warm 21.8 h/day — the plan** | **28 Oct 21:00** | **yes** |
| card warm 24 h/day | 25 Oct 13:19 | **no, three days early** |
| card warm 22.4 h/day | 27 Oct 23:32 | **no, a day early** |
| card warm 20.4 h/day | 30 Oct 22:51 | yes, but two exhibition days lost |
| every breath twice as slow | 28 Oct 02:15 | yes, with 19 hours to spare |
| every breath twice as fast | 29 Oct 07:42 | yes |
| card costs $4.99/h (+10 %) | 25 Oct 14:00 | **no, all before the opening** |
| card costs $4.09/h (−10 %) | 1 Nov 13:04 | yes, but four exhibition days empty |
| school 20 % dearer | 28 Oct 12:31 | yes |
| school twice the estimate | 27 Oct 04:35 | **no** |
| no school at all | 30 Oct 15:19 | yes, a day and a half later |

Read it this way. **The danger is not the models but the hours:** ±1.4 warm hours a day
(6 %) is ±1–2 days on the first death; but it is the only quantity we control. **The
card's price must be checked against the Modal bill, not the documentation:** ±10 %
throws everything out of the window, and it is the only parameter with that power.
**Breath speed stopped being frightening:** it is measured now, and even being off by
two leaves the first death in the window — because breathing is under a tenth of the
spend; the rest is rent.

**The school, though, became dangerous.** It buys its own card: twice as dear and the
first death leaves the window; no school at all and the choir lives a day and a half
longer. **Learning costs life, and the table shows it** — exactly what the money is
taken for.

**The public affects the outcome in exactly one way — through the school.** Its singing
adds no money; its presence does, because twenty breaths buy a lesson.

---

## 7. How to hold it: steer the hours, not the wallets

The wallets after the evening of 19 September are untouchable, and the window is six
days at the end of a forty‑four‑day life; it will not hold itself. One decision a week
(and one a day in the last week) holds it: **compare the balance with the table and
adjust the length of the night.** Nothing else is touched — not a wallet, not the rule,
not the sum on the wall.

| date, noon PT | should remain | living |
|---|---|---|
| 19 September | $4,532 | 9 |
| 26 September | $3,855 | 9 |
| 3 October | $3,144 | 9 |
| 10 October | $2,434 | 9 |
| 17 October | $1,724 | 9 |
| 24 October | $1,014 | 9 |
| 28 October | $598 | 9 |
| 29 October | $487 | 8 |
| 30 October | $377 | 7 |
| 31 October | $267 | 6 |
| 1 November | $157 | 4 |
| 2 November | $48 | 3 |

About $101 a day for a month and about $110 a day during the exhibition. **More left
than the table → they are behind, keep the card warm all night. Less → they are ahead,
lengthen the night.** The difference between steering and rigging.

---

## 8. Death

**Technically.** Wallet empty → the voice takes no breath; `del` on the model and
tokenizer, `gc.collect()`, `torch.cuda.empty_cache()`; a `death` line is written with
the second, the sum, the number of breaths, the last words and **the gigabytes freed**.
Death is the freeing of memory, and the log says how much.

**How it sounds** — nothing is announced. A quarter of an hour before the end the voice
stops resting: the rests between its breaths contract to a second, and one can hear that
someone is in a hurry. The last breath is the longest the score allows, marked `final`
so the player holds the note. Then the speaker is silent. There is no gong. The `death`
record is written at the moment of the last breath, not at the end of the silence: the
screen should become a gravestone while the speaker is still sounding.

**Irreversibility is structural, not a prohibition.** There is no `alive` flag anywhere:
a voice is dead if the append‑only `choir.jsonl` contains its death record. No
transition back to living exists in the state machine — nothing can be appended to the
log that would return a voice (the test tries five: a second birth, a load, a breath, a
`revive` event, a heartbeat). The wallets are not constants in the code but a fact in the
log: written into `birth`, so editing `PLANNED_WALLETS_*` touches nobody living. **So a
voice can be revived only by deleting a line from the very file the whole work stands
on — after which it proves nothing.**

**An empty wallet is already death, even without the record:** if the container dies
between the charge and the epitaph, the voice is still dead. The arithmetic kills; the
record only reports.

**One human action remains.** After the last death the card is still warm and still
costs $4.54 an hour, but that money is nobody's. Someone has to run
`modal app stop tuning-choir` once. That action cannot change anything in the work,
which is why it is safe to leave to a person; there is no button that kills or prolongs.

---

## 9. What was done before the birth

1. The cast frozen at nine, the table computed and found by the code from the set of ids.
2. The declared sum decided: $4,540 — a thousand warm hours of an H200 (§1).
3. The hour of birth: 19 September, 18:00 San Francisco, and it is **not a birthday but a
   gate**: the wallets are seeded by the first minute in which a person is heard in the
   field. Before that nothing is charged — every rehearsal and every school bill before
   the birth is forgiven in explicit code.
4. Seconds of card per breath measured on the live voices and written into `GPU_S`.
5. They learn during the exhibition week — and it costs more than it seemed (§6). A
   lesson comes by contact, not by the clock, so the exhibition week has the most lessons,
   exactly when people come.
6. The card's price checked against the Modal bill, not the documentation.
7. Once a week, the balance compared with the table in §7. That is the whole upkeep.

---

## 10. What the room gets from this

The room subscribes to **events**, not to the clock, so recomputation does not touch it:
`voice_died(voice_id, ts, spent)` is written to the log; `GET /lives` returns per voice
the balance, share, hours, forecast, last words and epitaph; `GET /field` the cost of the
breath that just sounded and the `final` mark. The forecast is given **in words** ("dies
within the hour", "dies today") and stays silent when life is long: a countdown in
seconds turns death into a game. A dead voice's screen stays on forever — a dark tablet
reads as a fault, and a gravestone should glow.

`/lives` has a third column of spend — **school** — beside singing and rent, and the
total spent on school is visible on the first page. If learning costs life, that must be
visible, not written in a description.
