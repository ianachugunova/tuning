# Hearing — what the "ears" actually heard

A report on real runs of `portal/music_probe/hearing_probe.py` on Modal, September
2026. Below is what the models actually answered, not a retelling of expectations.

Five audio‑language models were tried as ears for the choir, with the same bench and the
same questions: **Voxtral** (`mistralai/Voxtral-Mini-3B-2507`), **Audio Flamingo 3**
(`nvidia/audio-flamingo-3-hf`), **Moshi**, **Music Flamingo**
(`nvidia/music-flamingo-2601-hf`) and **Qwen2.5‑Omni** (`Qwen/Qwen2.5-Omni-7B`).

## What was asked

All of them the same thing, otherwise it is five anecdotes rather than a comparison.
Four pieces of material, each of which the piece needs for a reason:

| material | what it is | why |
|---|---|---|
| `silence` | room noise, almost no signal | will it mistake silence for a scrap of speech |
| `breath_D4` | a vowel tone at D4 = 293.66 Hz with vibrato | does it hear a tone as a tone, and name the note |
| `two_voices` | D4 and A4 together, a perfect fifth | the score asks to tune to someone else's pitch |
| `human_glide` | 220 → 233 Hz over 3 s, exactly a semitone up | a glissando is what tells a voice from a tuning fork |

The last exists for a separate reason. There is no human audio anywhere in this work:
the phone sends pitch, duration and the line the pitch travelled, and the promise "the
recording goes nowhere" is what `/breath` is written around. So the only way a machine
can hear a person here is a tone rebuilt from those numbers, exactly as the listener's
browser already rebuilds it. `human_glide` is such a tone.

## What they answered

| | Voxtral | Audio Flamingo | Moshi | Music Flamingo | Qwen2.5‑Omni |
|---|---|---|---|---|---|
| memory | **9.2 GB** | **16.0 GB** | 17.1 GB | 16.0 GB | 22.7 GB |
| load | 58 s | 101 s | 94 s | 92 s | 113 s |
| silence | "faint, distant humming sound" | "I hear a sound" | "Good day. What's going on?" | "a clean digital tone with a sharp attack" — invented a sound | **"I don't actually hear anything"** — the only one that did not lie |
| D4 | **A♭** — off by a tritone | **E4** — off by a tone | "Hey, how is it going?" | **A3** — off by a fourth | "I'd probably sing back a C" — off by a tone |
| two voices | **"2, C"** — count right | "three voices" — count wrong | "Hi there, how can I help you?" | **"2 voices, C♯4 and C♯5"** — count right, octave instead of fifth | "One voice, C5" — count wrong |
| glissando | **"moves from C to C#"** — a semitone up, the interval exact | "from a low pitch to a high pitch" | "Hey, how can I help you?" | **"It holds still"** — heard no motion at all | **"It moves. From A3 to A4"** — start exact (220 Hz is A3), end off by an octave |
| a field recording | transcribed the speech | "Glints of conversation woven through empty spaces" | — | "experimental ambient… for a sci‑fi game" | — |

## The main thing, and it is unpleasant

**No ear names absolute pitch.** Voxtral missed by a tritone, Flamingo by a whole tone.
No ear can be relied on for "what note is this".

**But Voxtral hears relations, and hears them exactly.** It alone counted two voices as
two. And on the glissando it answered "from C to C#" — the absolute notes wrong, the
interval right to the cent: it really is exactly a semitone up (220 → 233 Hz is a ratio
of 1.0595). It does not know *what* is sounding, but it knows *how far* it moved and
*how many* there are.

This matters more than it looks, because Oliveros's score does not ask anyone to name
notes. It asks you to tune to a pitch someone else is holding. Naming is a demand of our
machinery, which needs hertz in the log, not a demand of the score. Hence the working
arrangement: **the ear reports a relation and the runtime supplies the number.** The
other voices' pitches are already in the log; if the ear says "there are two, I go to
the lower one", the system knows how many hertz that is.

**Audio Flamingo hears texture, not relations.** It came closest on the single note but
counted two voices as three and said only "low to high" of the glissando. On a real
field recording, though, it answered not with a transcription but with a description of
the room: "glints of conversation woven through empty spaces, ripples, a silence calling
for colour". That is a different genre of listening — describing a space rather than
tuning. In the cast it does exactly that: it is given the room's audio and asked what
the room sounds like.

**Moshi does not hear a tone as a tone at all.** To every material — including four
seconds of clean D4 — it replied with a line of conversation. Its output pitch had no
relation to what it heard (heard 307 Hz, sang 271; heard 224, sang 500). Spoken to
directly, in seven attempts it never produced a number: "I'm a pitch of sound, a pitch
of tune, a pitch of joy, a pitch of glee". It does not measure; it banters. Dropped.

**Music Flamingo** counted two voices as two but heard the semitone glide as "it holds
still" — for a piece whose one obligatory question is whether someone else's pitch is
moving and where, that is disqualifying. Tuning differs from coincidence only by motion.

**Qwen2.5‑Omni** was the only one not to invent a sound in silence and the only one to
name an absolute pitch correctly (A3), but counted two voices as one, and when asked to
sing a long vowel answered: *"I'm sorry, I can't physically sing or hold a long steady
vowel on a single pitch. I'm not a singer."* The most capable model of the five politely
declined to perform the score — a refusal that is not technical but educational.

**Conclusion on ears: Voxtral for tuning, Audio Flamingo for the room.** Nothing beat
Voxtral on the one thing the score asks — how many, and which way the pitch moves — and
everything else cost more memory.

## Memory

Measured, not estimated: 9.2 + 16.0 GB for the two ears over about 43 GB for the six
text voices. Nine voices did not fit an 80 GB A100 and the choir moved to an H200
(141 GB).

## How an ear hears people when there is no human audio

`portal/hearing.py` is the bridge. A machine breath is a wav that was rendered when the
voice sang and lies on the volume; a human breath has no recording and never will, so
`human_tone` builds a tone from pitch, duration and contour — exactly what `/breath`
accepts, and exactly the tone the listener's browser already builds (`soundHuman` in
`pitch.html`). `field_audio` lays each breath at its real offset inside the window,
silences included: what overlapped overlaps, and a rest stays a rest. Without this the
ear would hear the machines and be deaf to the people — the opposite of the intent.

One consequence worth keeping apart from the code: **there is a microphone in the hall,
but it is not a source for the ear.** The microphone is a pitch sensor
(`live_v1/human.py`, YIN); its audio is never stored, and numbers enter the field. The
ear listens to a room assembled from the log, not a recorded one. The promise "there is
no recording" is kept not by an access policy but by the recording not existing.
