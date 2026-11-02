# The mouth — a voice that is described, not picked from a list

A report on real runs of `portal/music_probe/voice_probe.py`, `voice_stability_probe.py`,
`voice_cost_probe.py` and `bodies_probe.py` on Modal, 10–11 September 2026. Measurements,
not intentions.

## Where it started

With a complaint about hearing: the voice of `tinystories` — a model raised on stories
told in a four‑year‑old's words — sounded like a woman of thirty. That turned out to be
not a badly chosen setting but **a hole in the set**: Kokoro has 54 voices and not one of
them is a child.

A second question closed at the same time — the sung vowel. The throat measured with
WORLD for each voice (`portal/music_probe/throat_probe.py`, accuracy −0.1…−2.6 cents)
was switched off (`THROAT = False`) after listening: nine humming bodies tuning to one
another is a choir of people, which is exactly the substitution this work must not be.
An oscillator claims nothing about a body. The difference between the voices stayed
where it belongs — in what they say and when they decide to sound — and, from 11
September, also in **whose voice** they say it.

## What replaced it

**VoxCPM2** (`openbmb/VoxCPM2`), 2B, Apache‑2.0, 30 languages, 48 kHz. It has no list of
voices at all: a voice is set by a descriptive phrase, which the model's own guide calls
"casting notes". So the cast stops being nine choices from a menu and becomes nine
descriptions — in the same register as the `raised_by` line each voice already had.

The child came out, and is measurably a child rather than a pitched‑up adult: 317 Hz
against 236 for the Italian woman, 194 for the French woman and 87 for the man's voice
for gpt‑2, spectral centroid 2500–2800 Hz. Of four descriptions the quietest was chosen
by ear — "very quietly, right at the microphone, almost a whisper, with long pauses".

## The main problem and its solution

A voice with a new body on every breath is not a character but a stranger with the same
name. Measured with a speaker‑verification model (`microsoft/wavlm-base-plus-sv`),
cosine similarity between five different phrases; the bar is not 1.0 but the constancy
of the Kokoro the work already lives with:

| mode | mean cosine | worst pair |
|---|---|---|
| Kokoro `af_sky` — **the bar** | **0.968** | **0.963** |
| description generated on every phrase | 0.814 | 0.675 |
| a frozen excerpt as reference | 0.902 | 0.812 |
| **frozen excerpt + continuation mode** | **0.965** | **0.946** |

Three findings, the first not obvious.

**Generation is deterministic.** Two files made with one description, one phrase and one
seed in different runs matched byte for byte. So the spread is not randomness but the
influence of the text: the description sets the type of voice, but the body adapts to
the particular phrase. Hence 0.814 and "different people".

**Only continuation mode holds.** An excerpt given as a plain reference gives 0.902 —
the same as no freezing at all. What works is the pair of `prompt_wav_path` plus its
exact transcript, with the same file also given as reference: the model continues in the
same throat.

**The child's low figures are a property of the ruler, not the voice.** The child scored
0.902–0.917, which looked like a failure until a control was added: the same test with
an adult description gave 0.965. WavLM is trained on adult speech. The child has to be
judged by ear; there is no graduated instrument for it.

## Cost

Eight breaths of realistic length on an A100‑40GB:

| | per breath | real‑time factor |
|---|---|---|
| Kokoro | 0.12 s | 0.047× |
| VoxCPM2, description every time | 0.75 s | 0.328× |
| **VoxCPM2, continuation mode** | **0.76 s** | **0.337×** |

Conditioning is free: continuation costs the same as plain generation. Identity costs
**+0.64 s per breath per voice** against Kokoro. Memory 5.4 GB against 0.6.

**This does not affect the money, nor the composition of deaths.** A warm card costs the
same whether the choir breathes every twelve seconds or every ten minutes; rent is almost
all of a voice's spend, and an extra half second moves pennies between "own" and
"rent". `mortality_sim` with `GPU_S + 0.64`: the order of deaths is unchanged, dates
move by at most 70 minutes, recalibration puts them back to the minute.

The price is **memory**: with EuroLLM 9B and Voxtral, 72.6 GB of 79 available on the
A100 — one of the reasons the choir moved to an H200.

## The birth protocol — what cannot be done twice

Each voice's body is a file on the `tuning-weights` volume, in `/weights/bodies/`, one
wav per `voice_id`. It is **born**, not cached: a cache can be deleted and rebuilt;
regeneration gives a different person.

How it works in `choir_app.py`:

* `casting` and `first_words` in `cast.py` — the description of the body and the phrase
  it is born on, in its own language. The description is always in English: it is a stage
  direction, not speech.
* `BODY_SEED = 42` — the seed with which four children were drawn, one of them chosen by
  ear. Another seed on the same description gives a plausible child, but not that one.
* `body_of(voice)` births the body on first use and commits the volume **before** using
  it. `weights_volume.reload()` is called first: a container sees the volume as it was
  mounted, and the price of trusting a stale snapshot is not two lost seconds but a
  second person under the same name. That bug happened: two consecutive runs wrote
  `BORN` instead of `found`.
* `speak(text, voice)` synthesizes in continuation mode and resamples 48 kHz to the
  field's 24 kHz. Kokoro remains underneath as a fallback — a breath has no right not to
  sound.

Checked by `bodies_probe.py`: eight bodies born, a second run found all eight (birth
0.0 s), peak memory 5.5 GB.

## What stayed open

**A description does not reliably set sex.** `olmo-3`, `eurollm` and `voxtral` are
described as "an adult woman", and the medians came out 168, 134 and 171 Hz — a male or
borderline range; with the previous seed the same three gave 260, 183 and 224. So the
seed is part of the body, not a technical detail. Accepted by ear as is; the measured
fundamentals became the voices' `speaks_at_hz`, from which each voice's singing range is
derived (`cast.py`).

**`audio-flamingo` is brighter than described**: 79.6 Hz with a spectral centroid of
4505 Hz, where all the others are 1900–2800. Described as a dark, low voice.
