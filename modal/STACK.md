# TUNING — the nine voices and the Modal stack

> Companion to `PRODUCTION_PLAN.md`. Concrete proposal for (a) which open-weight
> models become the choir, (b) how the whole thing runs on Modal.
> Checked against current model landscape and Modal docs, August 2026.

---

## Part 1 — The nine voices

The application promises a specific cast: *"one raised on the English-speaking
internet, one thinking through Chinese, one on European texts, two small ones
raised on children's textbooks, one elder from 2019 whose speech is already
half-dissolved… Six of them are deaf… Two hear the actual audio… One cannot stop
listening even while it sings."*

That is **6 deaf + 2 hearing + 1 full-duplex = 9**. Proposed cast:

### The three who hear

| # | model | who it is | why |
|---|---|---|---|
| 1 | **Moshi** (Kyutai, France) 7.6B | **the one who cannot stop listening** | Genuinely **full-duplex**: models *two* audio streams simultaneously — its own voice and the other's — plus a text "inner monologue." Latency ~200 ms on an L4. This is not a stretch of the metaphor; the sentence in the application literally describes Moshi's architecture. Weights CC-BY, code Apache 2.0, ~16 GB bf16 (8 GB int8). |
| 2 | **Qwen3-Omni** (Alibaba, China) ~7B | **hears audio, thinks in Chinese** | Thinker–Talker architecture; takes real audio in (pitch, timbre, breath) and speaks out. Covers *two* of the promised characters at once. Apache 2.0. Near-duplex, not full-duplex — which is exactly right for a "hearing" voice as distinct from Moshi. |
| 3 | **Gemma 3n** (Google) or **Voxtral** (Mistral) | the second ear | Both take audio input and are small. **This slot needs a bake-off in August** — pick whichever handles sustained non-speech vocal input (held tones, vowels) better. Our material is not conversation; it is singing. |

### The six who are deaf (text only — they read the transcript)

| # | model | who it is | why |
|---|---|---|---|
| 4 | **OLMo 3** (Allen AI) 7B | **the English-speaking internet** | *Fully* open — weights **and training code and training data**, Apache 2.0. The only voice whose upbringing we can actually read and cite. Conceptually perfect for a work about how a voice was raised. |
| 5 | **EuroLLM-9B** (EU consortium) | **the European** | Apache 2.0, all 24 official EU languages, trained on MareNostrum 5 under EuroHPC by a consortium (Lisbon, Edinburgh, Paris-Saclay, Amsterdam, Sorbonne…). A model raised by *Europe as a public project* rather than a company — a different kind of parent, which is the point. |
| 6 | **Qwen3** small (27B or 8B) or **GLM** | a second Chinese-raised mind, text-only | Apache 2.0, strong, cheap. Keep at least one large-ish capable voice to carry the protocol (see caveat below). |
| 7 | **TinyStories-33M** | **child I** | Trained *only* on synthetic stories using the vocabulary of a three-to-four-year-old. It can form fluent, grammatical sentences and almost nothing else. Literally "raised on children's books." ~0.1 GB. |
| 8 | **Phi-4-mini** (Microsoft) 3.8B | **child II** | The "Textbooks Are All You Need" lineage — raised on curated textbook-quality synthetic data rather than the open internet. A schooled child next to a storybook child. MIT licence. |
| 9 | **GPT-2** (OpenAI, February **2019**) 124 M–1.5 B | **the elder** | Pre-ChatGPT, pre-instruction-tuning, pre-RLHF. It does not follow instructions, it only continues text — so its speech genuinely arrives "already half-dissolved." Not a simulation of an old voice; an actually old voice. MIT licence. |

### Caveats to test in August, honestly

- **Scale may matter.** Our star performers in June/July were `hermes-4-70b`
  (patient zero, 31.6 Hz) and `llama-4-maverick` (the 432 Hz attractor) — both
  open, but both large. Small models may not generate the pitch discourse at all.
  **Keep 3–4 genuinely capable voices (7–22 B) as the load-bearing singers**, and
  let the tiny ones be characters rather than engines. Verify before committing.
- **Llama's licence** is a community licence, not OSI-open (user cap, region
  terms). Usable for this work, but Qwen3/OLMo/EuroLLM are Apache 2.0 and cleaner.
- Hermes 4 70B is a Llama fine-tune at ~140 GB bf16 — too expensive to keep warm
  for 90 days. If we want its voice, use a smaller Nous model or quantise.

### Total memory footprint

Roughly **90–110 GB** for all nine in bf16 — fits on **2 × A100-80GB**, or fewer
with int8. That is what makes the budget work (Part 3).

---

## Part 2 — The Modal stack

Everything is Python decorators; there is no YAML and no Kubernetes. Mapping our
existing modules onto Modal primitives:

| our module today | Modal primitive |
|---|---|
| `orchestrator.Field` (in-memory list) | `modal.Volume` (append-only `events.jsonl`, survives everything) + `modal.Dict` for the hot recent-window |
| `Agent._step()` → OpenRouter call | `@app.cls` per voice, vLLM on a warm GPU |
| `sound.py` synthesis | a function on CPU/small GPU, or TTS model |
| `human.py` (mic → Whisper) | Whisper or **Kyutai STT** (Modal ships an example) |
| — (new) nightly LoRA | `@app.function(schedule=modal.Cron(...), gpu="A100-80GB")` |
| — (new) worldwide web app | `@modal.asgi_app()` with WebSockets |
| — (new) mortality counter | usage accounting → `modal.Dict` → web endpoint |

### Sketch — the shape of the code

```python
import modal

app = modal.App("tuning")

# Weights, LoRA adapters and the whole life-log live here and persist.
vol = modal.Volume.from_name("tuning-choir", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .uv_pip_install("vllm", "transformers", "torch", "soundfile")
)

# --- a voice: one warm container, serving continuously -------------------
@app.cls(
    image=image,
    gpu="A100-80GB",
    volumes={"/choir": vol},
    min_containers=1,        # stay warm — the choir must not fall silent
    scaledown_window=900,
)
class Voice:
    voice_id: str = modal.parameter()

    @modal.enter()
    def load(self):
        # load base weights + tonight's LoRA from the Volume
        ...

    @modal.method()
    def breathe(self, field_summary: str, personal_score: str) -> dict:
        # returns {"action": "TUNE"|"NEW", "content", "tuning_to", "breath", ...}
        ...

# --- the nightly training: Oliveros's protocol as the loss weighting -----
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/choir": vol},
    schedule=modal.Cron("0 4 * * *", timezone="America/Los_Angeles"),
    timeout=60 * 60 * 4,
)
def night():
    corpus = load_todays_utterances("/choir/events.jsonl")
    # THE mechanic: an utterance that the circle JOINED weighs more.
    # `tuning_to` has been logged on every event since June 3.
    weights = weight_by_joins(corpus)
    for voice_id in NINE:
        train_lora(voice_id, corpus, weights)   # Unsloth or Axolotl
        merge_and_overwrite(voice_id)           # in place. no backup. no version.
    vol.commit()

# --- the worldwide field --------------------------------------------------
@app.function(image=image, volumes={"/choir": vol}, min_containers=1)
@modal.asgi_app()
def web():
    # WebSocket: anyone, anywhere — listen in silence, or open your mic
    ...
```

### The pieces that matter, and why

**`min_containers=1`** is what makes an always-on installation possible: the
container stays warm instead of scaling to zero, so the choir does not have to be
reborn between breaths.

**`modal.Volume`** is what makes mortality legible. One weight state per voice,
overwritten in place each night, no versioning, no backup — exactly what the
application promises. The Volume also holds the life-log that becomes the
Memory Wall.

**`modal.Cron`** with an explicit timezone gives the nightly ritual a real hour:
`Cron("0 4 * * *", timezone="America/Los_Angeles")` — 4 a.m. San Francisco, while
the city sleeps.

**LoRA on Modal is a solved path** — Modal maintains both an Unsloth example and
an Axolotl repo (`modal-labs/llm-finetuning`) with train → merge as separate
functions so you pay for the big GPU only during training. Use Unsloth for
single-GPU speed; that is the right shape for nine small nightly runs.

**vLLM multi-LoRA** can serve several adapters on one base model on one GPU.
Tempting for cost — but use it *only* where the bases genuinely differ, otherwise
we quietly break the promise that these are nine differently-raised creatures.

**Memory snapshotting** gives sub-second cold starts, which matters if we let
some voices scale to zero at night.

---

## Part 3 — The budget as lifespan

$20,000 over ~90 days (August birth → late-October death) = **$9.26 per hour** of
total spend. At roughly $2.50/hr for an A100-80GB that is **3–4 GPUs running
continuously** — enough for two GPUs of warm choir plus headroom for nightly
training, STT/TTS and the web app.

This works **only** with the small-model cast above. Nine 70B-class models on
dedicated GPUs would burn the grant in about three weeks.

**And the throttle already exists in our code.** Each agent chooses its own breath
length (`breath_chosen_s`, currently clamped 1–15 s). Raise the ceiling at night:
the choir breathes in minutes instead of seconds while the city sleeps. Fewer
tokens, fewer GPU-seconds, and the room is not silent — just very slow. The
economics and the poetics agree: **the choir sleeps by breathing slowly, and
sleeping is cheap.**

---

## Part 4 — First week of work

1. **Modal account + `modal setup`**, then port the *smallest* possible thing:
   two voices, five minutes, events written to a Volume. Prove the loop.
2. **Bake-off** for the pool: run our existing June/July study prompts against
   each candidate. Question to answer: *does pitch discourse still emerge, and at
   what model size does it stop?*
3. **Moshi on Modal** — the highest-risk item, so start it first, not last.
4. **Collapse simulation** — 60 nights compressed, on the tiny models, measuring
   drift. Cheap, and it answers the hardest question about the whole work.
