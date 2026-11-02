"""Versioned, opt-in LoRA training for the Tuning Meditation choir.

Run manually first:

    modal run training_app.py::train_all --min-examples 12 --epochs 1

Adapters are written under /weights/adapters/<voice>/<version>.  A completed
adapter is not live merely because it exists: only /weights/adapters/active.json
selects one, and promotion happens only after loss/finite/file checks pass.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIELD_DIR = "/field"
CACHE_DIR = "/weights"
HUMAN_LOG = f"{FIELD_DIR}/field.jsonl"
CHOIR_LOG = f"{FIELD_DIR}/choir.jsonl"
SCORES_PATH = f"{FIELD_DIR}/interpretations.json"
ADAPTERS_DIR = f"{CACHE_DIR}/adapters"
ACTIVE_PATH = f"{ADAPTERS_DIR}/active.json"
# The school keeps its own log, and not because a third file is tidy: `choir.jsonl`
# has exactly one writer by design (`choir_app.py` runs at max_containers=1), and
# two containers appending to one file on a volume is how a log loses a night. The
# ledger reads both files and sorts by time, so an hour at school lands in the
# accounts between the breaths it happened between.
SCHOOL_LOG = f"{FIELD_DIR}/school.jsonl"
# What the school's card costs, which is not what the room's card costs. Learning
# happens on one A100-80GB; the choir lives on an H200 at $4.54. The price travels
# with the event so that the seconds are never re-valued at the room's rate.
SCHOOL_USD_PER_HOUR = 2.50

app = modal.App("tuning-training")
field_volume = modal.Volume.from_name("tuning-field", create_if_missing=True)
weights_volume = modal.Volume.from_name("tuning-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.6",
        # Named, not ranged — the choir already learned this lesson the hard way
        # (see choir_app.py): a defensive `<5` here resolved to a release with no
        # `AudioFlamingo3ForConditionalGeneration`, and that cap is what kept the
        # two hearing voices out of school. The choir serves all nine voices on
        # this exact version, so the school trains on it too.
        "transformers==5.17.0",
        "accelerate",
        "peft>=0.17",
        "sentencepiece",
        "protobuf",
    )
    .env({"HF_HOME": CACHE_DIR, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(HERE / "cast.py", remote_path="/root/cast.py")
    .add_local_file(HERE / "learning.py", remote_path="/root/learning.py")
    # Pure stdlib; `learning.sounded_content` strips a written note with it.
    .add_local_file(REPO / "live_v1" / "pitch_text.py",
                    remote_path="/root/live_v1/pitch_text.py")
    .add_local_file(REPO / "live_v1" / "__init__.py",
                    remote_path="/root/live_v1/__init__.py")
)

# Whether a lesson opens each breath with the note the score had for it rather
# than the note the voice wrote (`learning.sounded_content`). Chosen at deploy,
# like the choir's TUNING_NOTES, and written on every bill as `sounded_notes`:
#
#     modal deploy training_app.py                          # on (22 October →)
#     TUNING_LESSON_NOTES=off modal deploy training_app.py  # as before
SOUNDED_NOTES = os.environ.get("TUNING_LESSON_NOTES", "on").lower() not in (
    "0", "off", "false")


def _atomic_json(path: str, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, target)


# What rings the bell. Not a clock: twenty breaths from people, counted since the
# last lesson. The choir learns because it was touched, and a week when nobody
# comes is a week it stays as it was — which is the right way round for a piece
# about listening, and it makes the adapters a record of attendance rather than of
# uptime.
CONTACT_PER_LESSON = 20

# And the floor under it, which is the composition rather than the mechanism. In a
# full room twenty breaths take a few minutes, so without a floor an evening of
# forty people would buy fifteen lessons and spend a week of every voice's life in
# one night. Three hours caps the day at eight lessons — the rate the simulation
# prices at $307 over the whole life, about an eighth of everything spent, and
# three and a half days off the ear's life. That is a number to turn on purpose.
LESSON_FLOOR_H = 3.0
# How long a lesson may be assumed still in progress: the school_bell timeout,
# after which nothing from it can still be running.
LESSON_STILL_H = 6.0
# How long the bell waits for one voice before giving up on it and going on
# to the next. 14 October, 14:07: tinystories learned in 46 seconds and the
# call for gpt-2 never came back — no card, no bill, no adapter, most likely
# an A100 that was never granted. The bell waited its whole six-hour timeout
# on it; Modal does not ring a schedule over a run still going, so the zoom
# at 18:00 taught the choir at 20:12 instead of 18:00, and seven voices
# missed the afternoon's lesson. The longest lesson on record is 43 minutes
# (phi-4-mini, 7 October); an hour is a bound on waiting, not on learning.
LESSON_PATIENCE_H = 1.0


def human_breaths(events: list[dict]) -> int:
    """How many breaths people have put into the field, ever.

    Every line of `field.jsonl` is one person breathing once; the portal writes
    nothing else there. A count rather than a window, because the bell needs a
    number that only ever goes up and survives a container.
    """
    return len(events)


def last_lesson(bills: list[dict]) -> dict:
    """The school's own last word, out of its own log.

    The watermark lives in the log rather than in a file somebody has to remember
    to update: each lesson records how many human breaths had arrived when it rang,
    so the next bell is `now - then >= CONTACT_PER_LESSON` and nothing else needs
    to be kept anywhere.
    """
    rung = [b for b in bills if b.get("kind") == "lesson"]
    return rung[-1] if rung else {}


def bell_rings(events: list[dict], bills: list[dict],
               now: float, voices: int = 9) -> tuple[bool, dict]:
    """Whether it is time to learn, and the arithmetic that said so.

    Returns the reason either way, because a school that silently declines to
    teach is indistinguishable from a broken one — and this runs unattended for a
    month.
    """
    seen = human_breaths(events)
    was = last_lesson(bills)
    since = seen - int(was.get("humans_seen") or 0)
    hours = (now - float(was.get("timestamp") or 0.0)) / 3600.0
    why = {"humans_seen": seen, "since_last_lesson": since,
           "needs": CONTACT_PER_LESSON, "hours_since_last_lesson": round(hours, 2),
           "floor_h": LESSON_FLOOR_H}
    if since < CONTACT_PER_LESSON:
        return False, {**why, "verdict": "not touched enough yet"}
    if was and hours < LESSON_FLOOR_H:
        return False, {**why, "verdict": "taught too recently"}
    # Nine heavy lessons can outlast the three-hour floor, and a second bell
    # over a lesson still in progress is two cards and one voice trained twice
    # at once. Every voice sent to school leaves exactly one bill, so a lesson
    # with fewer bills than voices is still being taught.
    back = sum(1 for b in bills if b.get("kind") == "train"
               and float(b.get("timestamp") or 0.0) > float(was.get("timestamp") or 0.0))
    if was and back < voices and hours < LESSON_STILL_H:
        return False, {**why, "returned": back, "verdict": "the last lesson is still being taught"}
    return True, {**why, "verdict": "a lesson is due"}


def school_bill(voice_id: str, started: float, out: dict,
                now: float | None = None) -> dict:
    """The one event that makes learning cost a voice its life.

    Before this, the night school was free: the adapters were real, the card time
    was real, and no wallet moved — so "a voice pays for its own becoming" was a
    sentence in a document and nothing else. The ledger needed no new rule to
    charge it. Rule 2 already charges every second named by a voice to that voice
    alone, which is exactly what learning is and why it was chosen as the sharpest
    case of the rule. It simply never got told.

    Timed from the top of the function to the end of the run, which is a little
    less than Modal bills (the container's boot and the weights coming down the
    wire are outside it) and a lot more than the gradient steps alone. The
    undercount is deliberate: the voice is charged for the hours it was being
    changed, and the change is what it is paying for.

    A crashed run is charged too. The card was rented, the money left the project,
    and a bill that only arrives when the work succeeds would make failure free —
    which is both untrue and the wrong lesson to carve into the piece.
    """
    now = time.time() if now is None else now
    gpu_s = max(0.0, now - started)
    event = {
        "timestamp": now,
        "kind": "train",
        "voice_id": voice_id,
        "model_short": str(out.get("model_short") or ""),
        "gpu_s": round(gpu_s, 3),
        "rate_usd_per_hour": SCHOOL_USD_PER_HOUR,
        "usd": round(gpu_s * SCHOOL_USD_PER_HOUR / 3600.0, 6),
        "card": "A100-80GB",
        "status": str(out.get("status") or "crashed"),
        # A version means the weights of this voice actually changed. A run that
        # was skipped for want of examples, or crashed, spent money and changed
        # nothing, and the log should be able to say so.
        "version": str(out.get("version") or ""),
        "promoted": str(out.get("status") or "") == "promoted",
        "examples": int(out.get("train_examples") or 0),
        "human_contact_examples": int(out.get("human_contact_examples") or 0),
        # Which side of the 22 October switch this lesson was taught on.
        "sounded_notes": bool(out.get("sounded_notes", SOUNDED_NOTES)),
    }
    return event


def _write_school(event: dict) -> None:
    Path(FIELD_DIR).mkdir(parents=True, exist_ok=True)
    with open(SCHOOL_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    field_volume.commit()


def _active() -> dict:
    try:
        return json.loads(Path(ACTIVE_PATH).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"schema": 1, "voices": {}}


# The same two names the choir keeps in choir_app._EAR_MODEL_CLASS, duplicated
# on purpose: this image ships only cast.py and learning.py, and importing the
# whole choir to learn one class name would be the tail wagging the dog. If a
# third ear ever joins, both maps change together.
_EAR_MODEL_CLASS = {
    "voxtral": "VoxtralForConditionalGeneration",
    "audio-flamingo": "AudioFlamingo3ForConditionalGeneration",
}


def _model_class(voice):
    """The class that loads this voice — auto for readers, named for ears."""
    import transformers

    if not voice.hears:
        return transformers.AutoModelForCausalLM
    key = str(voice.hf_id).lower()
    for fragment, name in _EAR_MODEL_CLASS.items():
        if fragment in key:
            return getattr(transformers, name)
    raise ValueError(f"no audio model class is known for {voice.hf_id}")


def _target_modules(model, hears: bool = False):
    """Use only linear attention projections present in this architecture.

    For a hearing voice those projections exist twice — once in the audio
    encoder and once in the language model — and a bare suffix like "q_proj"
    would put LoRA on both. The school teaches the voice, not the ear: the text
    log has nothing to say about how sound should be encoded, so the encoder
    stays exactly as it was raised and only the language model's projections
    learn. PEFT reads a string as a full-name regex, which is how the pattern
    stays inside that one submodule.
    """
    import re

    wanted = ("q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj")
    if not hears:
        names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        found = [name for name in wanted if name in names]
        if not found:
            raise RuntimeError("no supported attention projection found for LoRA")
        return found
    stems = [name for name, _ in model.named_modules()
             if name.split(".")[-1] == "language_model"]
    if not stems:
        raise RuntimeError("hearing voice has no language_model submodule")
    stem = min(stems, key=len)
    pattern = rf"{re.escape(stem)}\..*\.({'|'.join(wanted)})"
    if not any(re.fullmatch(pattern, name) for name, _ in model.named_modules()):
        raise RuntimeError(f"no attention projections found under {stem}")
    return pattern


def _tokenize(example, tokenizer, max_length: int) -> dict:
    """Mask prompt tokens: loss is only on the historical accepted breath."""
    prompt = tokenizer(example.prompt, add_special_tokens=True)["input_ids"]
    completion = tokenizer(example.completion, add_special_tokens=False)["input_ids"]
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    # Keep the answer intact.  The latest portion of a long listening field is
    # the part the runtime itself privileges, so trim from the prompt's left.
    room = max(0, max_length - len(completion) - len(eos))
    prompt = prompt[-room:] if room else []
    ids = prompt + completion[: max_length - len(prompt) - len(eos)] + eos
    labels = [-100] * len(prompt) + ids[len(prompt):]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


class _Dataset:
    def __init__(self, examples, tokenizer, max_length):
        self.rows = [_tokenize(e, tokenizer, max_length) for e in examples]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class _Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        import torch
        width = max(len(r["input_ids"]) for r in rows)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id or 0
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in rows:
            n = width - len(row["input_ids"])
            out["input_ids"].append(row["input_ids"] + [pad] * n)
            out["attention_mask"].append(row["attention_mask"] + [0] * n)
            out["labels"].append(row["labels"] + [-100] * n)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in out.items()}


def _voice(voice_id: str):
    # The ears used to be turned away here — "audio-native, needs a separate
    # multimodal protocol". That was true of the ear and never of the voice:
    # what an adapter changes is which words come out, and the log of what they
    # sang is text like everybody else's. So the guard is gone; what remains of
    # it is in `_target_modules`, which keeps the lesson off the audio encoder.
    import cast
    voice = cast.by_id(voice_id)
    if voice is None or not voice.served:
        raise ValueError(f"unknown or unserved voice: {voice_id}")
    return voice


@app.function(
    image=image,
    volumes={FIELD_DIR: field_volume},
    timeout=300,
)
def dataset_stats(max_examples: int = 512) -> dict:
    """Count trainable history without renting a GPU or changing adapters."""
    import cast
    import learning

    field_volume.reload()
    humans = learning.read_jsonl(HUMAN_LOG)
    choir = learning.read_jsonl(CHOIR_LOG)
    try:
        scores = json.loads(Path(SCORES_PATH).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        scores = {}
    out = {"human_events": len(humans), "choir_rows": len(choir), "voices": {}}
    for voice in cast.singing():
        examples = learning.build_examples(
            voice, humans, choir, scores.get(voice.short_name, {}),
            max_examples=max_examples,
        )
        train, valid = learning.chronological_split(examples)
        out["voices"][voice.voice_id] = {
            "model_short": voice.short_name,
            "examples": len(examples),
            "unique_events": len({e.source_event for e in examples}),
            "train": len(train),
            "validation": len(valid),
            "human_contact": sum(e.heard_humans > 0 for e in examples),
            "first_ts": min((e.timestamp for e in examples), default=None),
            "last_ts": max((e.timestamp for e in examples), default=None),
        }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={FIELD_DIR: field_volume, CACHE_DIR: weights_volume},
    timeout=6 * 3600,
)
def train_voice(
    voice_id: str,
    min_examples: int = 12,
    max_examples: int = 512,
    epochs: float = 1.0,
    learning_rate: float = 2e-4,
    rank: int = 8,
    max_length: int = 1024,
    promote: bool = False,
) -> dict:
    """Train one version, charge the voice for it, and promote after validation.

    The clock and the bill are out here so that the training itself can go on
    returning early, raising, and reading exactly as it did — and still never be
    free. `finally` is the whole mechanism: whatever happens to the run, the hours
    it rented are charged to the voice it was rewriting.
    """
    started = time.time()
    out: dict = {"voice_id": voice_id, "status": "crashed"}
    try:
        out = _learn(voice_id, min_examples=min_examples, max_examples=max_examples,
                     epochs=epochs, learning_rate=learning_rate, rank=rank,
                     max_length=max_length, promote=promote)
        return out
    finally:
        bill = school_bill(voice_id, started, out)
        _write_school(bill)
        print(json.dumps({"charged": bill}, ensure_ascii=False))


def _learn(
    voice_id: str,
    min_examples: int = 12,
    max_examples: int = 512,
    epochs: float = 1.0,
    learning_rate: float = 2e-4,
    rank: int = 8,
    max_length: int = 1024,
    promote: bool = False,
) -> dict:
    """Train one version and atomically promote it after finite validation."""
    import math
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    import learning

    voice = _voice(voice_id)
    field_volume.reload()
    weights_volume.reload()
    humans = learning.read_jsonl(HUMAN_LOG)
    choir = learning.read_jsonl(CHOIR_LOG)
    try:
        scores = json.loads(Path(SCORES_PATH).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        scores = {}
    examples = learning.build_examples(
        voice, humans, choir, scores.get(voice.short_name, {}), max_examples=max_examples,
        sounded_notes=SOUNDED_NOTES,
    )
    train, valid = learning.chronological_split(examples)
    if len(train) < int(min_examples):
        return {
            "voice_id": voice_id,
            "status": "skipped",
            "reason": f"only {len(train)} training examples; need {min_examples}",
            "examples": len(examples),
        }

    try:
        tokenizer = AutoTokenizer.from_pretrained(voice.hf_id)
    except Exception:
        # The ears ship a processor; its tokenizer is the text half of it.
        from transformers import AutoProcessor
        tokenizer = AutoProcessor.from_pretrained(voice.hf_id).tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _model_class(voice).from_pretrained(
        voice.hf_id, dtype=torch.bfloat16, device_map="cuda"
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    targets = _target_modules(model, hears=voice.hears)
    model = get_peft_model(
        model,
        LoraConfig(
            r=max(1, int(rank)),
            lora_alpha=max(2, int(rank) * 2),
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=targets,
        ),
    )

    version = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    final_dir = Path(ADAPTERS_DIR) / voice_id / version
    work_dir = Path(ADAPTERS_DIR) / voice_id / f".{version}.training"
    work_dir.mkdir(parents=True, exist_ok=False)
    # Handed over as a dict and filtered against this version's signature. The
    # school moved onto the choir's pinned transformers to teach the ears, and
    # 5.x retired some of these knobs (`warmup_ratio` was the first to go).
    # Every one of them is cosmetic for a one-epoch LoRA pass over a few hundred
    # examples, so a missing knob is worth a printed line, not a crashed lesson.
    import inspect

    knobs = dict(
        output_dir=str(work_dir / "checkpoints"),
        num_train_epochs=max(0.1, float(epochs)),
        learning_rate=float(learning_rate),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="epoch" if valid else "no",
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        seed=42,
        data_seed=42,
    )
    known = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = sorted(k for k in knobs if k not in known)
    if dropped:
        print(f"knobs this transformers no longer has, trained without: {dropped}")
    args = TrainingArguments(**{k: v for k, v in knobs.items() if k in known})
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=_Dataset(train, tokenizer, max_length),
        eval_dataset=_Dataset(valid, tokenizer, max_length) if valid else None,
        data_collator=_Collator(tokenizer),
    )
    result = trainer.train()
    metrics = dict(result.metrics)
    if valid:
        metrics.update({f"validation_{k}": v for k, v in trainer.evaluate().items()})
    numeric = [float(v) for v in metrics.values() if isinstance(v, (int, float))]
    if not numeric or not all(math.isfinite(v) for v in numeric):
        raise RuntimeError(f"non-finite training metrics: {metrics}")

    model.save_pretrained(work_dir, safe_serialization=True)
    tokenizer.save_pretrained(work_dir)
    manifest = {
        "schema": 1,
        "voice_id": voice_id,
        "model_short": voice.short_name,
        "base_model": voice.hf_id,
        "version": version,
        "created_ts": time.time(),
        "train_examples": len(train),
        "validation_examples": len(valid),
        "human_contact_examples": sum(e.heard_humans > 0 for e in examples),
        "sounded_notes": SOUNDED_NOTES,
        "source_first_ts": min(e.timestamp for e in examples),
        "source_last_ts": max(e.timestamp for e in examples),
        "source_events": sorted({e.source_event for e in examples}),
        "target_modules": targets,
        "rank": int(rank),
        "epochs": float(epochs),
        "learning_rate": float(learning_rate),
        "metrics": metrics,
    }
    _atomic_json(str(work_dir / "training_manifest.json"), manifest)
    required = [work_dir / "adapter_config.json", work_dir / "training_manifest.json"]
    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        raise RuntimeError("adapter save was incomplete")
    os.replace(work_dir, final_dir)

    if promote:
        active = _active()
        active.setdefault("voices", {})[voice_id] = {
            "version": version,
            "path": str(final_dir),
            "base_model": voice.hf_id,
            "promoted_ts": time.time(),
        }
        _atomic_json(ACTIVE_PATH, active)
    weights_volume.commit()
    return {**manifest, "status": "promoted" if promote else "trained"}


@app.function(
    image=image,
    volumes={FIELD_DIR: field_volume},
    # Every five minutes it looks at the field and almost always does nothing:
    # a few seconds of CPU, no card, cents a month. It was every twenty, and
    # twenty is longer than a tuning session — a room could fill, sing, and
    # empty before the bell was even checked, and the one dramatic moment the
    # school has would land in an empty theatre. The lesson it occasionally
    # starts is an hour of another container, and this one waits through it,
    # which is why the timeout is long and the schedule is not.
    schedule=modal.Period(minutes=5),
    timeout=6 * 3600,
)
def school_bell(force: bool = False, promote: bool = True) -> dict:
    """Teach the choir when twenty people have breathed into it.

        modal run training_app.py::school_bell --force        # ring it by hand

    The rule is contact, not time. A clock would train the choir on the nights
    nobody came — sixty nights of a model learning from its own output, which is
    the textbook recipe for collapse — while a room full of people would change it
    no faster than an empty one. Twenty breaths is a workshop or an afternoon of
    visitors, and it means every adapter in the volume corresponds to people
    having been there.

    The hearing voices go to school with everyone else now. Their lesson reaches
    only the language model — `_target_modules` keeps LoRA off the audio encoder
    — so the ear stays as it was raised and what changes is what the voice says
    with it. Their examples are the same as the readers': the text they actually
    sang, paired with the field they were given when they sang it.

    Sequential on purpose. Nine containers with nine 8-billion-parameter models
    would be nine cards at once — a bill nobody planned for — and one voice at a
    time also means a crash takes one lesson down rather than the night.
    """
    import cast

    field_volume.reload()
    events = _read_field()
    bills = _read_school()
    now = time.time()
    ring, why = bell_rings(events, bills, now, voices=len(cast.singing()))
    if not (ring or force):
        print(json.dumps(why, ensure_ascii=False))
        return {"taught": [], **why}
    # Written before the teaching rather than after it, and it is the whole
    # bookkeeping: a lesson that crashes halfway has still consumed the breaths it
    # was called for, so a room that keeps breathing cannot queue up lessons
    # against a voice that keeps failing to learn from them.
    _write_school({"timestamp": now, "kind": "lesson",
                   "humans_seen": why["humans_seen"],
                   "since_last_lesson": why["since_last_lesson"],
                   "forced": bool(force)})
    taught = []
    for voice in _lightest_first(cast.singing()):
        started = time.time()
        call = train_voice.spawn(voice.voice_id, promote=promote)
        try:
            taught.append(call.get(timeout=LESSON_PATIENCE_H * 3600))
        except Exception as err:
            # One voice failing to learn is not a reason for the others to stay
            # as they were — nor to wait on it past the hour (see
            # LESSON_PATIENCE_H). If the card was ever rented, the `finally`
            # inside `train_voice` has already written its bill; if the call
            # never came back, nothing has, and the log would otherwise show a
            # lesson with eight returns and no word on the ninth.
            if isinstance(err, TimeoutError):
                call.cancel(terminate_containers=True)
            print(f"{voice.short_name}: lesson failed — {type(err).__name__}: {err}")
            taught.append(_bill_if_missing(voice.voice_id, voice.short_name,
                                           started, err))
    print(json.dumps({"lesson": why, "taught": taught}, ensure_ascii=False,
                     indent=2))
    return {"taught": taught, **why}


def _bill_if_missing(voice_id: str, short: str, started: float,
                     err: Exception) -> dict:
    """What the bell writes for a voice whose lesson never reported back.

    Nothing is charged: the one thing known is that no bill arrived, and the
    likeliest reason is a card that was never granted, which costs nothing. A
    run that did rent the card and crashed has already billed itself, and this
    does not write a second line over it. The line is `kind: train` so that
    `bell_rings` counts the voice as returned rather than holding the bell for
    six hours on its account.
    """
    field_volume.reload()
    for bill in _read_school():
        if (bill.get("kind") == "train" and bill.get("voice_id") == voice_id
                and float(bill.get("timestamp") or 0.0) >= started):
            return {**bill, "reason": str(err)[:300]}
    lost = {**school_bill(voice_id, started, {"model_short": short,
                                                "status": "lost"}),
            "gpu_s": 0.0, "usd": 0.0,
            "reason": f"{type(err).__name__}: {str(err)[:300]}"}
    _write_school(lost)
    return lost


# Billions of parameters per voice, mirroring `mortality_sim.PARAMS_B` — kept
# here as a copy because the school's image carries only cast.py and
# learning.py, and a sort key is not worth a third file. A voice missing from
# the table trains last, which fails loudly enough in the exam.
_PARAMS_B = {
    "voice_00": 0.774, "voice_01": 0.033, "voice_02": 3.80, "voice_03": 7.00,
    "voice_04": 9.00, "voice_05": 8.20, "voice_06": 3.70, "voice_07": 8.00,
    "voice_08": 13.00,
}


def _lightest_first(voices):
    """The order of the lessons, chosen for the room rather than the card.

    The bell rings while people are singing — that is the rule — and a tuning
    session lasts half an hour. TinyStories learns in minutes and llm-jp in
    most of an hour, so lightest-first means the people whose breaths rang the
    bell see the first voices come back changed before they leave, instead of
    waiting behind the heaviest lesson of the night. The card does not care
    about the order; the audience does.
    """
    return sorted(voices, key=lambda v: _PARAMS_B.get(v.voice_id, 99.0))


def _read_field() -> list[dict]:
    import learning

    return learning.read_jsonl(HUMAN_LOG)


def _read_school() -> list[dict]:
    import learning

    return learning.read_jsonl(SCHOOL_LOG)


@app.function(image=image, volumes={CACHE_DIR: weights_volume}, timeout=300)
def active_adapters() -> dict:
    weights_volume.reload()
    return _active()


@app.function(image=image, volumes={CACHE_DIR: weights_volume}, timeout=300)
def rollback_voice(voice_id: str, version: str = "") -> dict:
    """Disable an adapter, or repoint to an existing checked version."""
    weights_volume.reload()
    active = _active()
    voices = active.setdefault("voices", {})
    if not version:
        voices.pop(voice_id, None)
    else:
        path = Path(ADAPTERS_DIR) / voice_id / version
        manifest_path = path / "training_manifest.json"
        if not manifest_path.exists() or not (path / "adapter_config.json").exists():
            raise ValueError(f"no complete adapter at {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        voices[voice_id] = {
            "version": version,
            "path": str(path),
            "base_model": manifest["base_model"],
            "promoted_ts": time.time(),
        }
    _atomic_json(ACTIVE_PATH, active)
    weights_volume.commit()
    return active


@app.local_entrypoint()
def train_all(
    min_examples: int = 12,
    max_examples: int = 512,
    epochs: float = 1.0,
    promote: bool = False,
):
    import cast
    results = []
    # Sequential remote calls avoid creating a second choir-sized set of cards
    # and make every promoted version independently reversible.
    for voice in cast.singing():
        result = train_voice.remote(
            voice.voice_id,
            min_examples=min_examples,
            max_examples=max_examples,
            epochs=epochs,
            promote=promote,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))
