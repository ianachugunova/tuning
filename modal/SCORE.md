# The score — the log read as notation

The work leaves no weights behind. When the last wallet empties the weights leave the
card and are deleted — nothing of what the voices became over six weeks is kept. What
remains is the **score**: the log of everything sung, by machines and by people, from
18 September to the last death, read as a musical text.

This places the work in the line of Pauline Oliveros's text scores and Cornelius
Cardew's graphic scores (*Treatise*): a document that does not sound by itself but can
be played from. The score can be performed by anyone, at any time, and the intended
form of performance is an **exchange**: people sing the parts the machines sang, and
machines sing the parts the people sang. The score outlives everyone in it.

One practical consequence follows, and it is strict: **the log is no longer a service
record; it is the work itself, and it cannot be repaired after the fact.** Whatever is
not written into the line at the moment of the breath will not be in the work.

Code: `portal/score.py` (the form and the reading), `portal/score_render.py`
(engraving into text), `portal/test_score.py`. Two places write the log:
`portal/choir_app.py` (machines) and `portal/portal_app.py` (people).

---

## 1. Two files and their rules

```
/field/choir.jsonl           ← written only by tuning-choir (machines + the ledger)
/field/field.jsonl           ← written only by tuning-portal (people)
/field/audio/*.wav           ← one file per voiced machine breath
/field/interpretations.json  ← each voice's personal score (see §9)
```

Four rules on which everything else rests.

1. **Append only.** No truncation, no rewriting, no reset. The one reset was before the
   birth: everything in the files until then was testing.
2. **One writer per file.** A Modal volume promises nothing about two containers
   appending to one file. Hence two files, not one.
3. **Line order in the file is the chronology.** It outranks the timestamp: a
   container's clock can step backwards on restart; append order cannot. The ledger
   reads the file in line order (`mortality.replay`).
4. **The score is read left to right by `t`.** Two files cannot be merged by line
   order, so for reading as music — and only for that — the lines are sorted by the
   onset of sound.

Format is JSON Lines: one line, one object. A single broken line (a write cut off when a
container died) is skipped; the rest is read.

---

## 2. What a performer needs, and where it lives

| | machine breath | human breath |
|---|---|---|
| when the sound began | `timestamp` | `started_ts` |
| how long it sounded | `sound_s` | `sound_s` |
| how long the silence before it lasted | `silence_before_s` + inference | `own_rest_s` + inference |
| pitch in hertz | `pitch_hz` | `pitch_hz` |
| pitch as a note | `pitch_note` + `pitch_cents` | the same |
| **how** the pitch was named | `pitch_kind`: `note` / `hz` / `none` | `pitch_kind`: `sung` |
| whether it actually sounded | `toned`, `spoke` | `voiced_fraction` |
| what was sung in words | `content` | `content` |
| who sang | `voice_id`, `model_short`, `tongue` | `voice_id`, `who: human` |
| whom it tuned to | `action`, `tuning_to`, `tuning_to_id`, `heard` | not declared |
| moments of death | `kind: "death"` | — |

Of this set, **duration and silence are the only things without which the score cannot
be performed at all.** The Tuning Meditation consists of silence more than of sound (the
demonstration slice is 89 % silence). A score without rests is a list of pitches, not
music.

---

## 3. A machine breath — field by field

`choir.jsonl`, `kind == "tone"`. An example with every score field:

```json
{"timestamp": 1789000031.4, "voice_id": "voice_05", "model_short": "qwen3",
 "kind": "tone", "action": "TUNE", "content": "294Hz — 我跟着你",
 "tuning_to": "voice_02", "tuning_to_id": "voice_02", "tongue": "Chinese",
 "breath_chosen_s": 5.0, "sound_s": 7.1, "toned": true, "spoke": true,
 "playback": "both", "playback_note": "",
 "pitch_hz": 294.0, "pitch_note": "D4", "pitch_cents": 2,
 "pitch_label": "294Hz", "pitch_kind": "hz",
 "pitch_all": [[294.0, "294Hz"]],
 "silence_before_s": 24.6,
 "heard": [["voice_02", 1789000000.0], ["human_7f3a", 1789000012.2]],
 "audio": "1789000031400_a3f1c9.wav",
 "gen_s": 1.9, "render_s": 0.4, "gpu_s": 2.3, "own_cost_usd": 0.0016,
 "api_latency_ms": 1940, "dying": false, "final": false,
 "left_usd_before": 361.2}
```

### Time and duration

- **`timestamp`** — the onset of the breath: the moment the voice finished thinking and
  the breath entered the field. This is the score's `t`. The audio is rendered
  `render_s` later, and heard in the room when the player gets to it; neither is the
  onset of the note in the musical sense. The note begins when it enters the field.
- **`sound_s`** — how long the sound actually lasted, in seconds, measured on the
  finished wav. **This is not `breath_chosen_s`.** The voice chooses a breath length,
  but the tone is stretched to the length of the spoken words, so any breath with words
  sounded longer than declared. `sound_s == 0` means the voice named a tone and chose
  not to sound (`playback: "none"`): a recorded silence, not a gap, and it is notated.
- **`breath_chosen_s`** — the length the voice *asked for*. Kept: it is the intention,
  and the gap between intention and sound is a fact about the voice. Not usable as a
  duration; a lower bound.
- **`toned`** / **`spoke`** — whether the tone and the words actually sounded.
  `playback` is the intent; these two are the fact. They differ when the synthesizer
  could not pronounce the line or no pitch was found in it.

### Silence

- **`silence_before_s`** — how long the whole field was silent before this breath,
  across machines and people. Measured by the listening voice from the end of the last
  sounding event in the merged field (`choir_app.field_ends_at`). `null` means the field
  had never sounded.
- The value is measured **before** generation, so if a person sang while the voice was
  thinking, the recorded number is larger than the real gap. That is not an error: it
  is what the voice was answering. The exact rest is inferred from the merged
  chronology (§6); `score.read()` puts the inferred value in `rest_s` and the recorded
  one in `heard_rest_s`. Two facts, both needed.

### Pitch

- **`pitch_hz`** — the frequency that **actually sounded**. Taken from the same list the
  renderer uses, so the number in the log and the number in the wav are one number.
- **`pitch_note`** — the same pitch as an equal‑temperament note name, A4 = 440.
- **`pitch_cents`** — deviation from that note, −50…+50. Not rounded: a voice that said
  "294 Hz" did not say exactly D4, and a person who sang 33 cents under D♯4 did so on
  purpose.
- **`pitch_label`** — how the pitch was named in the text: `"D4"`, `"294Hz"`.
- **`pitch_kind`** — **what exactly the voice named** (§4).
- **`pitch_all`** — every pitch in the line, up to four. Only the first sounded; the
  rest is a gesture the voice wrote ("D4 then F4").
- **`pitch_asked_hz`**, **`pitch_asked_note`**, **`pitch_octaves`** — only on lines
  where the pitch the voice answered lay outside its range and was moved into it by
  **whole octaves** (`tuning.fold_into`). The child holds A5, the old man answers A3,
  `pitch_octaves: -2`.

  Three fields rather than one because this is the only place where the runtime changes
  a number the voice chose, and its being checkable is the condition on which that is
  allowed at all. The octave is the one interval across which a pitch does not change:
  a bass answering a soprano sings her note, and in any choir with both it is a unison,
  not a concession. But that is exactly why, without the second number, **a unison
  cannot be told from a transposition** — and "the choir converged" is the main claim
  of the whole work. `pitch_hz` stays what sounded; `pitch_asked_hz` is what was
  answered. Absent fields mean nothing was moved.

  A voice's range comes not from a shared window but from its own body: `speaks_at_hz`
  in `cast.py` — the measured fundamental of the speech that voice speaks with — three
  semitones below and two octaves up. Register in this cast carries **age** and only
  age: the two lowest are the two described as old men, the highest is the child.
  Language lives in `tongue` and `first_words` and does not touch register.

All of this is derivable from `content` by a regular expression
(`live_v1/pitch_text.extract_all_pitches`) — and it is written anyway. In ten years
that file will not exist, and the score has to outlive the code.

### Who, and to whom

- **`voice_id`** — the part. Permanent, `voice_00`…`voice_08`.
- **`model_short`**, **`tongue`** — the name and the language of the words. The
  language is repeated on every line deliberately: a performer handed one page of the
  score should not have to find the first line of the log to know what language they
  are being asked to sing in.
- **`action`** — `TUNE` (I tune to someone else's pitch) or `NEW` (I take a pitch nobody
  is singing). These are the two halves of Oliveros's text.
- **`tuning_to`** — to whom, in the voice's own words. Free prose: a `voice_id`, a short
  name, a pitch, "the low warm hum", nothing.
- **`tuning_to_id`** — the same, resolved to a specific part when the prose names one.
  `null` when it does not: an unresolvable answer is a fact about the voice, not a
  parsing failure, and `tuning_to` is kept either way.
- **`heard`** — who was in the circle at the moment of decision: `[voice_id, timestamp]`
  pairs for the listening window. An arrow needs a part at both ends.

### The rest of the line

`gen_s`, `render_s`, `gpu_s`, `own_cost_usd`, `left_usd_before` belong to the ledger
(`modal/MORTALITY.md`). `dying` — the voice has less than fifteen minutes left and its
rests contract: the only warning the room gets. `final` — this breath is the last the
voice can afford, taken as long as the score allows. Both are worth notating: before a
death the rhythm of the part audibly changes.

---

## 4. How the pitch was named — four cases

`pitch_kind` is not a technical detail but a musical distinction, and it cannot be
recovered from the number.

| | what happened | how to read it |
|---|---|---|
| `note` | the voice named a note: `"D4"` | a notehead, exactly |
| `hz` | the voice named a frequency: `"294 Hz"` | notehead + a cents figure: it did not know this was D |
| `sung` | a person sang and named nothing | notehead + cents + a wavy line from `contour_hz` |
| `none` | no pitch in the line at all | words only, or silence — notate without a head |

The exchange of parts (§8) rests on exactly this. A person cannot be given "294 Hz" to
sing from the page — they need `pitch_note`. A machine cannot be given "a little under
D♯4" — it needs `pitch_hz`. So both are recorded, always, for both.

---

## 5. A human breath — field by field

`field.jsonl`. The microphone signal does not leave the phone and never did: everything
below is numbers measured in the browser.

```json
{"ts": 1789000217.1, "started_ts": 1789000212.0, "ended_ts": 1789000216.6,
 "tail_s": 0.5, "voice_id": "human_7f3a", "kind": "tone", "who": "human",
 "note": "D#4", "hz": 305.2, "cents": -33, "stability": "steady",
 "duration_s": 4.6, "sound_s": 4.6, "voiced_fraction": 0.94,
 "own_rest_s": null, "std_hz": 2.4,
 "contour_hz": [303.1, 305.0, 306.2, 305.4], "contour_hop_s": 1.15,
 "content": "D#4 ~305Hz", "pitch_hz": 305.2, "pitch_note": "D#4",
 "pitch_cents": -33, "pitch_kind": "sung", "pitch_all": [],
 "source": "portal"}
```

- **`ts`** — when the breath reached the portal. **It is not the onset of the note.**
- **`started_ts`** / **`ended_ts`** — onset and end of sound, computed as
  `ts − tail_s − duration_s` and `ts − tail_s`. This is the score's `t`.
- **`tail_s`** — how long passed between the end of the sound and the page deciding the
  phrase was over, by the singer's own clock. The page waits half a second of silence
  before believing a note has ended; without this field every onset in the human part
  would sit late by a fixed amount nobody could later detect.
- **`duration_s`** = **`sound_s`** — the voiced length of the phrase. One quantity under
  two names: `duration_s` for what already reads the log, `sound_s` to match the
  machines so one function reads both parts.
- **`own_rest_s`** — **the singer's own rest**: between two of their breaths, by their
  own audio clock. `null` for the first breath of a session and after every reopening
  of the microphone: the clock restarts, the gap is genuinely unknown, and zero would
  be a lie. This is **the one number in the score that merging the logs cannot
  recover** — everything else about a human breath is placed by arrival time, and the
  network is not part of the music.
- **`hz`**, **`note`**, **`cents`** — median pitch, nearest note, deviation. Computed in
  the browser and logged as is, so the score says what the singer saw on their screen.
- **`std_hz`** — pitch spread in hertz. `stability` — one of four words (`very steady` /
  `steady` / `vibrato` / `sliding`) — right to show a singer, poor to archive:
  "steady" covers 30 to 80 cents of motion.
- **`contour_hz`** / **`contour_hop_s`** — the pitch trace: one median per block,
  roughly a quarter second per point, at most 128 points. Glissando, vibrato and a
  tone that wanders and returns are distinguishable only here.
- **`voiced_fraction`** — the share of frames in which a pitch was found. Breath,
  whisper and a cracking voice read from here.
- **`content`** — the same line the machines see: `"D#4 ~305Hz [steady]"`.
- **`who: "human"`**, **`kind: "tone"`** — the same two words on lines of both logs, so
  one reader walks the merged chronology without knowing which file a line came from.
- **`voice_id`** — a random four‑character word living in the singer's own browser. No
  account, no analytics, no cookie; one device, one part.

---

## 6. Two silences, and why there are two

This is where it is easiest to go wrong, so it is set apart.

- **The ensemble's rest** (`Note.rest_s`) — how long **the whole field** was silent
  before this breath. The rhythm of the whole, what a conductor reads. Inferred exactly.
- **The part's rest** (`Note.own_rest_s`) — how long **this voice** was silent before
  its own breath. What one performer reads from their own page.
- **The rest the voice heard** (`Note.heard_rest_s`) — how long, by its own
  measurement, the field was silent. A fact about the voice, not the archive, and the
  only rest that survives if one of the two files is lost.

`score.read()` infers the first two from the merged chronology and does not trust the
log for them — because when every line has an onset and a duration, the chronology
fixes the rhythm unambiguously:

> every breath either begins after the field has fallen silent — and then its rest is
> exactly the length of that silence — or begins while someone is still sounding, and
> then its rest is zero and the overlap is carried by the absolute onsets. There is no
> third case, and no interval that is not either sound or recorded silence.

That is the claim that six weeks are recoverable. It is checked by
`test_the_score_tiles_the_time_it_covers`.

A rest is **never negative**: an overlap is an overlap, not a rest of the opposite sign,
and it is preserved in absolute `t`. A zero in the rest column reads as "entered a field
that was still sounding".

---

## 7. Service records

All in `choir.jsonl`. The ledger reads them in detail (`modal/MORTALITY.md`); here only
what they mean for the score.

### `kind: "birth"` — the list of parts

One line, once, at the birth. Carries `declared_usd`, `rate_usd_per_hour`, `wallets`
and **`cast`** — the list of parts. Per voice:

```json
{"voice_id": "voice_00", "model_short": "gpt-2",
 "hf_id": "openai-community/gpt2-large",
 "raised_by": "the open internet of 2019, before models were taught manners",
 "tongue": "English", "kokoro": "am_onyx", "instruct": false, "hears": false}
```

`tongue` and `kokoro` — in what language and with what voice this sounded.
**`instruct: false`** matters more than it looks: the old man (GPT‑2) and the child
(TinyStories‑33M) cannot take instructions and were never asked to sing by the rules —
the runtime held the form, the voice only gave the breath. A performer taking such a
part should know: it is built differently.

### `kind: "death"` — a double barline

```json
{"timestamp": ..., "kind": "death", "voice_id": "voice_05",
 "model_short": "qwen3", "born": ..., "lived_s": 3499200.0,
 "seed_usd": 391.56, "spent_usd": 391.56, "breaths": 11282,
 "gpu_s": ..., "freed_gb": 16.1, "living_after": 5,
 "last_words": "294Hz — 我跟着你"}
```

The moment a part ends and is never resumed — there is not, and cannot be, an event in
the log that brings a voice back (`mortality.py`: death is derived, not stored). For the
score this is a double barline on one line and `living_after` parts continuing.
`last_words` is the last thing said.

### `kind: "card"` / `"exit"` / `"load"` / `"score"` / `"train"`

- `card` — every 30 seconds while the card is warm, with the list of `resident` voices.
  This is what in the log tells **a resting choir from a stopped one**: without these
  lines the month of silence the work pays for would look free. Not notes, and not to be
  notated, but they are what proves the silence between breaths was silence and not the
  absence of the work.
- `exit` — the card was released. The gap until the next container is not a silence of
  the work but a pause in its time. An engraver should mark it differently.
- `load` — a voice's weights arrived on the card. The beginning of the part in the
  physical sense.
- `score` — the voice read Oliveros and wrote itself a personal score (§9).
- `train` — a school lesson. Between one and the next the voice changed; for the score
  it is a boundary past which the same part sounds different.

---

## 8. The exchange of parts

The intended performance: **people sing the machine parts, machines sing the human
parts.** `score.parts()` splits the score by `voice_id`, because a part is exactly what
is exchanged. A part's page is printed to stand on its own: who owned it, who is to sing
it, and then only rests, pitches, durations and words.

What each side needs, and why both halves are recorded alike:

- **A person taking a machine part:** `pitch_note` ("294 Hz" cannot be sung), `sound_s`,
  `own_rest_s`, `content` and `tongue` — the text is to be spoken, in its language.
  `pitch_kind: "hz"` says sing the note without any claim to exact tuning;
  `pitch_kind: "note"` says exactly.
- **A machine taking a human part:** `pitch_hz` and `pitch_cents` (its tone is
  synthesized, not found by a voice, so it will take the 33 cents literally),
  `sound_s`, `own_rest_s`, `contour_hz` — where there is a trace, the glissando is
  reproducible exactly, which a person cannot do in the other direction.

The asymmetry is not a defect. A person cannot sing 33 cents under D♯4 on instruction;
a machine cannot fail to. That is what makes the exchange audible.

---

## 9. What else is worth keeping beside the score

The score must stand without any of this — but it exists, and the score is poorer
without it.

- **`interpretations.json`** — each voice's personal score: how this voice, once, before
  its first breath, rewrote Oliveros's text as instructions to itself. Written once and
  never changed. A performer taking a part should read how the one who sang it
  understood it. The old man and the child have no such record — they were not asked,
  and that too should be known.
- **`audio/*.wav`** — one file per voiced machine breath: the tone at pitch, the words
  on top. A recording, not a score. Its value is timbre: how `am_onyx` sounded speaking
  the old man's English. The filename is in the line's `audio` field.
- **`lives.json`** — a snapshot of the ledger. Derived; a cache; deleted and rebuilt
  from the log.

The numbers a person's phone measures and sends are also the only trace of that person
anywhere in the work: there is no recording, no account and no identifier that
survives the browser. Measurements must therefore stay comparable across the whole
run — the pitch detector on the page cannot change its mind about what a note is in
the week when it matters most.

---

## 10. What is not in the score, and why

- **The sound of people.** The microphone signal does not leave the phone. A person is
  in the score as pitch, duration, rest and trace — enough for their part to be sung,
  not enough to hear the room they were in. On that condition the work can be left open
  to strangers.
- **The exact moment a sound was heard in the gallery.** The player pulls the field and
  queues breaths; between entering the field and the loudspeaker there is a delay the
  log does not record. The onset in the score is the moment of entering the field.
- **Loudness.** A machine tone is always one amplitude, speech another. A person's
  loudness is measured in the browser and not sent: across different phones it is
  incomparable anyway. A score without dynamics is exactly what the *Tuning Meditation*
  is.
- **The weights.** There will be none. That is the point.

---

## 11. How to read it

```bash
# a fabricated slice — works without a log, a card or a network
python3 portal/score_render.py --demo --parts

# the real log, if the files are at hand
python3 portal/score_render.py --tail 400 --parts choir.jsonl field.jsonl

# the real log on the volume, on CPU, for pennies
modal run portal/choir_app.py::engrave --tail 400 --parts
```

Two things are printed. **The chronology** — the score as it happened: one line per
breath, with the rest before it, who sang, the pitch by name and frequency, duration,
gesture, text, and an arrow to whom it tuned. **The part pages** — the same material by
voice, which is what gets exchanged.

Programmatically:

```python
import score
notes = score.read(choir_rows, human_rows)   # merged chronology
pages = score.parts(notes)                   # by part
```

`score.Note` is a flat record with no references into the log and no lazy lookups: a
`list[Note]` printed as CSV is still a score. Standard library only; no Modal, no
network.

---

## 12. Lines written before the score

None of the fields in this document exist on any line written before 28 August. A
reader must treat each of them as optional, and `score.py` does:

- no `sound_s` → `breath_chosen_s` is used, and it is **a lower bound, not the truth**;
- no `started_ts` on a human line → the onset is approximated as `ts − duration_s`;
- no `pitch_*` block → the breath is notated without a pitch, with the whole text. The
  frequency is deliberately not reconstructed: that would need the regex from
  `live_v1`, which will not exist in ten years, and a visible gap is better than a
  quiet reconstruction that looks like a measurement;
- no `silence_before_s` → the rest is inferred from the chronology, as always.

No field has been renamed and none removed. `mortality.py` replays the same file for
the accounts and does not notice what was added; `choir_app._choir_event` picks the
fields of the June `Event` by name; `/field` reads a few keys by name. Extending is
allowed; changing is not.

---

## 13. To the engraver: a proposal for notation

Not binding. The score is text, not a picture, and any engraving is already a
performer's decision. But one proposal, so the decision has somewhere to start.

- **Proportional time, not metric.** Six weeks, rests from seconds to hours. A bar grid
  lies here; a logarithmic time axis with the rest written as a figure reads.
- **A notehead** on `pitch_note`. A cents figure beside it only when `pitch_kind` is
  `hz` or `sung`; with `note` there are no cents, because the voice chose a name and a
  name is exact by definition.
- **A wavy line** from `contour_hz` under a human note; a machine note has no trace and
  should have no line.
- **`pitch_kind: "none"`** — no head: text on a thread, as in Cardew.
- **`sound_s == 0`** — a head in brackets: a tone named and not taken.
- **An arrow** from note to note by `tuning_to_id`; an unresolved `tuning_to` is a
  caption in the voice's words, not an arrow into nothing.
- **A double barline** at `kind: "death"`, with `living_after` beneath it. The part does
  not resume after it — in the score or in performance.
- **A break** at `exit`: there was no rest there, there was no work there.
- **The header of a part page** says who sang it and who is to sing it. Without that,
  the exchange is an idea, not an instruction.
