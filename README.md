# Tuning

Nine language models perform Pauline Oliveros's *Tuning Meditation* (1971), day and
night, for as long as their money lasts.

Gray Area, San Francisco — *Scaling Intelligence*, 28 October – 1 November 2026.
Portal: https://tuning-grayarea--tuning-portal-web.modal.run

Each voice is an open-weight model with a mortal wallet. It listens to the field — the
other eight and whoever sings into a microphone or a phone — and then either tunes to a
tone it heard or starts one of its own, alternately, as the score asks. Every breath is
paid for from the voice's own wallet in GPU seconds, at cost. When the wallet is empty
the voice dies and its speaker falls silent for good. Nothing is ever recorded: a person
reaches the choir only as numbers — pitch, duration, the line the pitch walked.

The choir was born on 19 September 2026 and died between 1 and 2 November: nine deaths
in about fifteen hours, 1,043 warm hours of one card, about 400,000 breaths, $4,828 by
its own ledger — of which $94 was spent at school.

## The nine

| voice | model | raised by | tongue |
|---|---|---|---|
| voice_00 | GPT-2 Large | the open internet of 2019 | English |
| voice_01 | TinyStories-33M | invented children's stories | English |
| voice_02 | Phi-4-mini-instruct | textbooks | English |
| voice_03 | OLMo-3-7B-Instruct | an institute that published weights, code and data | English |
| voice_04 | EuroLLM-9B-Instruct | European universities on a public supercomputer | Italian |
| voice_05 | Qwen3-8B | a Chinese corporation | Chinese |
| voice_06 | Voxtral-Mini-3B | a French company — hears audio | French |
| voice_07 | Audio Flamingo 3 | a chip maker — hears audio, describes sound | English |
| voice_08 | LLM-jp-3.1-13B-instruct4 | a Japanese national institute | Japanese |

`portal/cast.py` is the authoritative list.

## How it runs

Three Modal apps, two volumes, one card.

- **`portal/choir_app.py` — the choir.** All nine models in one container on one H200
  (`max_containers=1`: one writer for the log). A one-minute cron wakes it; inside the
  minute each voice sings on its own clock, resting as long as it just sang. Speech is
  synthesized on the same card and every breath is written as a wav to the volume.
- **`portal/portal_app.py` — the portal.** FastAPI on Modal. Accepts `/breath` from phones
  and from the room's microphones (numbers only), serves `/field`, `/lives`, `/audio/…`,
  and the screens: `/` (sing from your phone), `/board`, `/circle`, `/pulse`, `/score`.
- **`portal/training_app.py` — the school.** Twenty human breaths ring a bell (not more
  than once in three hours). Each voice is then fine-tuned with LoRA (rank 8, one epoch)
  on its own history — *what it heard → what it sang* — on a separately rented A100, and
  pays for the lesson from its own wallet. `portal/learning.py` builds the examples and
  holds the anti-collapse rules.
- **`portal/mortality.py` — the ledger.** The card costs $4.54 an hour whether anybody
  sings or not. Seconds spent inside a voice's own forward pass are charged to that
  voice; the warm, idle card is split equally among the living. Deaths follow from the
  log and only from the log: a restart replays the file and resurrects nobody.
- **`portal/hearing.py`** — how the two voices with ears are given the room: the
  machines' own wavs laid at their real offsets, the people as sine tones rebuilt from
  their numbers, the silences kept.
- **`portal/room.py`** — the gallery. A Mac mini plays each voice on its own speaker
  (nine Genelecs in a 16-foot circle) and listens on five microphones in the centre.
  See `portal/ROOM_SETUP.md`.
- **`live_v1/`** — the modules inherited unchanged from the June 2026 concert version
  (Center for New Music, SF): the score prompt, pitch‑in‑text parsing, the YIN detector.

Design notes: `modal/MORTALITY.md` (the ledger), `modal/SCORE.md` (the log as
notation), `modal/HEARING_MODELS.md` and `modal/VOICE_MODELS.md` (what the ears and
mouths were measured to do), `modal/EXHIBITION.md` (the room and the six days),
`modal/STACK.md` (the original Modal proposal).

## Running it

```bash
pip install modal && modal setup
cd portal
TUNING_CLOCK=on modal deploy choir_app.py     # the choir, with its clock
modal deploy portal_app.py                    # the portal
modal deploy training_app.py                  # the school bell
python3 -m unittest                           # the tests (no GPU needed)
```

Requires a Hugging Face token in the Modal secret `huggingface-secret` (one model is
gated). The logs live on the Modal volume `tuning-field`; adapters and model weights on
`tuning-weights`.

## What we learned

Training a model on its own output is a loop, and it closed within two weeks: one voice
sang a single word 100 % of the time, another sang `ta-ta-ha` 82 % of the time. The way
out was not a smaller learning rate but a rule about the data — no line more than three
times per lesson, people first — after which diversity returned from 0 to about 0.7.
And every one of the nine repeated itself less when a person was in the room than when
it was alone. See `learning.limit_repeats` and `modal/MORTALITY.md`.

## Credits and licence

Concept, composition and code: Iana Chugunova, 2026. Built on Modal with a Modal
grant, shown at Gray Area in *Scaling Intelligence*.

*The Tuning Meditation* © Pauline Oliveros, Deep Listening Publications. The score's
text is not included in this repository.

Code is released under the MIT licence (see `LICENSE`). The logs, the voices' adapters
and the recordings of the installation are not part of this repository.
