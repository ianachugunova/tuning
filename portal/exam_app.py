"""The exam: the same questions, every voice, with its lessons and without.

    modal run exam_app.py                      # all nine, report to modal/exam_reports/
    modal run exam_app.py --voices voice_08    # one voice
    modal run exam_app.py --refreeze           # new questions (only on purpose)

The live log cannot say whether a voice learned, because every generation of
adapters sang in a different room: people one day, nobody the next, and a fresh
roll of the dice on every breath. The exam holds the room still. See
`exam_metrics.py` for what is measured and `modal/EXAM.md` for why.

The questions are frozen once into `/field/exam/probes.json` and reused: an exam
whose questions change is not an exam. Only the "fresh" set is rebuilt each run,
because it is by definition what came after the newest lesson.

It runs in its own app, so the choir is not restarted, and it is paid for by the
room rather than the voices: nothing here is written to `school.jsonl` or
`choir.jsonl`, and no wallet moves. Results are appended to
`/field/exam/exams.jsonl`.
"""
from __future__ import annotations

import json
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
EXAM_DIR = f"{FIELD_DIR}/exam"
PROBES_PATH = f"{EXAM_DIR}/probes.json"
EXAMS_LOG = f"{EXAM_DIR}/exams.jsonl"

# Questions per set. Enough that one odd breath does not move the average,
# few enough that the whole exam is minutes rather than an hour.
PROBES_PER_SET = 48
FRESH_PER_VOICE = 32
# The Arsenale session, 14 October, 14:12 in Venice: four phones in the field.
ARSENALE_TS = 1791979920.0

app = modal.App("tuning-exam")
field_volume = modal.Volume.from_name("tuning-field", create_if_missing=True)
weights_volume = modal.Volume.from_name("tuning-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.6",
        # The version the choir sings on and the school teaches on.
        "transformers==5.17.0",
        "accelerate",
        "peft>=0.17",
        "sentencepiece",
        "protobuf",
        "mistral-common",
        "librosa",
        "soundfile",
    )
    .env({"HF_HOME": CACHE_DIR, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(HERE / "cast.py", remote_path="/root/cast.py")
    .add_local_file(HERE / "learning.py", remote_path="/root/learning.py")
    .add_local_file(HERE / "exam_metrics.py", remote_path="/root/exam_metrics.py")
    # Pure stdlib; the one piece of live_v1 the metrics read pitches with.
    .add_local_file(REPO / "live_v1" / "pitch_text.py",
                    remote_path="/root/live_v1/pitch_text.py")
    .add_local_file(REPO / "live_v1" / "__init__.py",
                    remote_path="/root/live_v1/__init__.py")
)

# Mirrors training_app._EAR_MODEL_CLASS; this image ships neither app.
_EAR_MODEL_CLASS = {
    "voxtral": "VoxtralForConditionalGeneration",
    "audio-flamingo": "AudioFlamingo3ForConditionalGeneration",
}


def _model_class(voice):
    import transformers

    if not voice.hears:
        return transformers.AutoModelForCausalLM
    key = str(voice.hf_id).lower()
    for fragment, name in _EAR_MODEL_CLASS.items():
        if fragment in key:
            return getattr(transformers, name)
    raise ValueError(f"no audio model class is known for {voice.hf_id}")


def _json(path: str, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _versions(voice_id: str) -> list[dict]:
    """Every finished adapter for a voice, oldest first, with its manifest."""
    root = Path(ADAPTERS_DIR) / voice_id
    out = []
    for path in sorted(root.glob("*")) if root.exists() else []:
        if path.name.startswith(".") or not path.is_dir():
            continue
        manifest = _json(str(path / "training_manifest.json"), None)
        if manifest:
            out.append({"version": path.name, "path": str(path),
                        "source_last_ts": manifest.get("source_last_ts"),
                        "base_model": manifest.get("base_model")})
    return out


def _example(e) -> dict:
    return {"prompt": e.prompt, "completion": e.completion,
            "ts": e.timestamp, "source": e.source_event,
            "heard_humans": e.heard_humans}


def _tickets(humans: list[dict], now: float) -> dict:
    """Three moments: the busiest room, the Arsenale, and an empty room."""
    import learning

    times = sorted(learning.event_time(r) for r in humans)
    busiest, best = None, -1
    j = 0
    for i, t in enumerate(times):
        while times[j] < t - 900:
            j += 1
        if i - j + 1 > best:
            best, busiest = i - j + 1, t + 1.0
    # The latest moment with nobody in the field for the three hours before it.
    empty = now - 3600
    for t in reversed(times):
        if t > empty:
            continue
        if t <= empty - 3 * 3600:
            break
        empty = t - 1.0
    return {"busiest_room": busiest, "arsenale": ARSENALE_TS, "empty_room": empty}


@app.function(image=image, volumes={FIELD_DIR: field_volume, CACHE_DIR: weights_volume},
              timeout=1800, memory=16384)
def prepare(voices: list[str], refreeze: bool = False) -> dict:
    """Freeze the questions once; rebuild only the fresh set; list adapters."""
    import cast
    import exam_metrics as xm
    import learning

    field_volume.reload()
    weights_volume.reload()
    humans = learning.read_jsonl(HUMAN_LOG)
    choir = learning.read_jsonl(CHOIR_LOG)
    scores = _json(SCORES_PATH, {})
    chronology = sorted((r for r in [*humans, *choir] if learning.is_tone(r)),
                        key=learning.event_time)
    times = [learning.event_time(r) for r in chronology]

    probes = None if refreeze else _json(PROBES_PATH, None)
    if probes is None:
        now = time.time()
        moments = _tickets(humans, now)
        probes = {"frozen_ts": now, "moments": moments, "voices": {}}
        for voice in cast.singing():
            score = scores.get(voice.short_name, {})
            built = learning.build_examples(voice, humans, choir, score,
                                            max_examples=10 ** 6, human_oversample=1)
            seen, unique = set(), []
            for e in built:
                if e.source_event not in seen:
                    seen.add(e.source_event)
                    unique.append(e)
            own = [r for r in choir if r.get("voice_id") == voice.voice_id
                   and learning.audible_machine_breath(r)]
            own.sort(key=learning.event_time)
            probes["voices"][voice.voice_id] = {
                "model_short": voice.short_name,
                "with_people": [_example(e) for e in xm.spread(
                    [e for e in unique if e.heard_humans], PROBES_PER_SET)],
                "alone": [_example(e) for e in xm.spread(
                    [e for e in unique if not e.heard_humans], PROBES_PER_SET)],
                "tickets": {name: xm.ticket_prompt(voice, score, chronology, times,
                                                   own, at)
                            for name, at in moments.items() if at},
                # What each part was holding in that window, for the hearing
                # reading of the TUNE takes.
                "ticket_pitches": {name: xm.ticket_pitches(
                    xm.ticket_heard(voice, chronology, times, at))
                    for name, at in moments.items() if at},
            }
        Path(EXAM_DIR).mkdir(parents=True, exist_ok=True)
        Path(PROBES_PATH).write_text(json.dumps(probes, ensure_ascii=False),
                                     encoding="utf-8")
        field_volume.commit()

    active = (_json(ACTIVE_PATH, {}).get("voices") or {})
    out = {"frozen_ts": probes["frozen_ts"], "moments": probes["moments"],
           "voices": {}}
    for voice in cast.singing():
        if voices and voice.voice_id not in voices:
            continue
        frozen = probes["voices"].get(voice.voice_id)
        if not frozen:
            continue
        versions = [v for v in _versions(voice.voice_id)
                    if v["base_model"] == voice.hf_id]
        newest = max((v["source_last_ts"] or 0 for v in versions), default=0)
        built = learning.build_examples(voice, humans, choir,
                                        scores.get(voice.short_name, {}),
                                        max_examples=10 ** 6, human_oversample=1)
        after = {e.source_event: e for e in built if e.timestamp > newest}
        lines = {" ".join(str(r.get("content") or "").split()).casefold()
                 for r in choir if r.get("voice_id") == voice.voice_id
                 and learning.audible_machine_breath(r)}
        out["voices"][voice.voice_id] = {
            **frozen,
            "fresh": [_example(e) for e in xm.spread(
                sorted(after.values(), key=lambda e: e.timestamp), FRESH_PER_VOICE)],
            "versions": versions,
            "active": (active.get(voice.voice_id) or {}).get("version"),
            "known_lines": sorted(k for k in lines if k),
        }
    return out


def _tokenize(prompt: str, completion: str, tokenizer, max_length: int = 1024):
    """Exactly the school's masking: loss only on the breath, prompt trimmed left."""
    p = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    c = tokenizer(completion, add_special_tokens=False)["input_ids"]
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    room = max(0, max_length - len(c) - len(eos))
    p = p[-room:] if room else []
    ids = p + c[: max_length - len(p) - len(eos)] + eos
    return ids, [-100] * len(p) + ids[len(p):]


@app.function(image=image, gpu="A100-80GB",
              volumes={CACHE_DIR: weights_volume},
              secrets=[modal.Secret.from_name("huggingface-secret")],
              timeout=3 * 3600)
def examine(voice_id: str, payload: dict, generations: int = 6,
            samples: int = 12) -> dict:
    """One voice, its base weights and a spread of its lessons, on frozen questions."""
    from contextlib import nullcontext

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    import cast
    import exam_metrics as xm

    started = time.time()
    voice = cast.by_id(voice_id)
    try:
        tok = AutoTokenizer.from_pretrained(voice.hf_id)
    except Exception:
        from transformers import AutoProcessor
        tok = AutoProcessor.from_pretrained(voice.hf_id).tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = _model_class(voice).from_pretrained(voice.hf_id, dtype=torch.bfloat16,
                                               device_map="cuda").eval()

    versions = payload["versions"]
    chosen = xm.spread(versions, generations)
    if payload.get("active") and all(v["version"] != payload["active"] for v in chosen):
        chosen += [v for v in versions if v["version"] == payload["active"]]
    model = None
    for v in chosen:
        if model is None:
            model = PeftModel.from_pretrained(base, v["path"], adapter_name=v["version"],
                                              is_trainable=False).eval()
        else:
            model.load_adapter(v["path"], adapter_name=v["version"], is_trainable=False)
    model = model or base

    def state(name):
        if name == "base":
            return model.disable_adapter() if chosen else nullcontext()
        model.set_adapter(name)
        return nullcontext()

    def mean_loss(rows: list[dict], batch: int = 8) -> float | None:
        total, count = 0.0, 0
        for i in range(0, len(rows), batch):
            enc = [_tokenize(r["prompt"], r["completion"], tok) for r in rows[i:i + batch]]
            width = max(len(ids) for ids, _ in enc)
            pad = tok.pad_token_id
            ids = torch.tensor([a + [pad] * (width - len(a)) for a, _ in enc]).cuda()
            lab = torch.tensor([b + [-100] * (width - len(b)) for _, b in enc]).cuda()
            att = (torch.arange(width)[None, :] <
                   torch.tensor([len(a) for a, _ in enc])[:, None]).long().cuda()
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=att).logits.float()
            shift, target = logits[:, :-1], lab[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                shift.reshape(-1, shift.size(-1)), target.reshape(-1),
                ignore_index=-100, reduction="sum")
            total += float(loss)
            count += int((target != -100).sum())
        return total / count if count else None

    def takes(prompt: str) -> list[str]:
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
        ids = enc["input_ids"][:, -1024:].cuda()
        with torch.inference_mode():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=160 if voice.instruct else 48,
                                 do_sample=True, temperature=1.0, top_p=0.95,
                                 num_return_sequences=samples,
                                 pad_token_id=tok.pad_token_id)
        return [tok.decode(o[ids.shape[1]:], skip_special_tokens=True) for o in out]

    known = set(payload["known_lines"])
    report = {"voice_id": voice_id, "model_short": voice.short_name,
              "states": [], "tickets": {}}
    for name in ["base"] + [v["version"] for v in chosen]:
        row = {"state": name}
        with state(name):
            for key in ("with_people", "alone", "fresh"):
                loss = mean_loss(payload.get(key) or [])
                row[key] = {"loss": loss, "choices": xm.choices_per_word(loss),
                            "n": len(payload.get(key) or [])}
        report["states"].append(row)
        print(voice.short_name, name,
              {k: round(row[k]["choices"] or 0, 2) for k in ("with_people", "alone", "fresh")})

    live = payload.get("active") or (chosen[-1]["version"] if chosen else None)
    for ticket, prompt in payload["tickets"].items():
        report["tickets"][ticket] = {}
        for name in ["base"] + ([live] if live else []):
            try:
                with state(name):
                    got = takes(prompt)
                report["tickets"][ticket][name] = xm.describe_takes(
                    got, known, (payload.get("ticket_pitches") or {}).get(ticket))
            except Exception as err:
                report["tickets"][ticket][name] = {"error": str(err)[:300]}
    report["gpu_s"] = round(time.time() - started, 1)
    report["live"] = live
    return report


@app.function(image=image, volumes={FIELD_DIR: field_volume}, timeout=300)
def record(rows: list[dict]) -> None:
    """One writer for the exam log."""
    Path(EXAM_DIR).mkdir(parents=True, exist_ok=True)
    with open(EXAMS_LOG, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    field_volume.commit()


def _when(version: str) -> str:
    try:
        t = time.strptime(version, "%Y%m%dT%H%M%SZ")
        return time.strftime("%d.%m", t)
    except (TypeError, ValueError):
        return version


def _fmt(x) -> str:
    return "—" if x is None else (f"{x:.1f}" if x < 100 else f"{x:.0f}")


def _pct(x) -> str:
    return "—" if x is None else f"{round(100 * x)}%"


def _hearing_lines(b: dict | None, a: dict | None) -> list[str]:
    """Hearing on a ticket, base → live, or nothing when nobody tuned."""
    b, a = b or {}, a or {}
    if not (b.get("tunes") or a.get("tunes")):
        return []
    def cents(x):
        return "—" if x.get("cents_median") is None else f"{x['cents_median']:.0f}¢"
    return [f"- слух, из {b.get('tunes', 0)} → {a.get('tunes', 0)} настроек назвал ноту: "
            f"{_pct(b.get('named'))} → {_pct(a.get('named'))}; промах: "
            f"{cents(b)} → {cents(a)}; в четверть тона: "
            f"{_pct(b.get('hit'))} → {_pct(a.get('hit'))}"]


def write_report(rows: list[dict], moments: dict, path: Path) -> None:
    """A page Iana can read: choices per word over lessons, then the tickets."""
    lines = [f"# Экзамен · {time.strftime('%Y-%m-%d %H:%M')}", "",
             "**Выбор на слово** — между сколькими словами голос в среднем колеблется, "
             "отвечая так, как он на самом деле ответил. Меньше — значит, урок сделал этот "
             "ответ для него своим. «С людьми» — его настоящие ответы, когда в поле были "
             "люди; «один» — когда людей не было; «новое» — то, что он спел после "
             "последнего урока, и чего ни один урок не видел.", "",
             "Билеты — замороженное поле в эти моменты (UTC): " + ", ".join(
                 f"{k} {time.strftime('%d.%m %H:%M', time.gmtime(v))}"
                 for k, v in moments.items() if v) + ".", ""]
    for r in rows:
        if "error" in r:
            lines += [f"## {r.get('model_short', r.get('voice_id'))}", "",
                      f"не сдал: {r['error']}", ""]
            continue
        lines += [f"## {r['model_short']}", "",
                  "| | без уроков | " + " | ".join(_when(s["state"]) for s in r["states"][1:]) + " |",
                  "|---|---|" + "---|" * (len(r["states"]) - 1)]
        for key, label in (("with_people", "с людьми"), ("alone", "один"),
                           ("fresh", "новое")):
            lines.append(f"| {label} | " + " | ".join(
                _fmt(s[key]["choices"]) for s in r["states"]) + " |")
        lines.append("")
        for ticket, by_state in r["tickets"].items():
            b, a = by_state.get("base", {}), by_state.get(r.get("live"), {})
            if "error" in b or "error" in a:
                lines.append(f"*{ticket}*: {b.get('error') or a.get('error')}")
                continue
            lines += [f"**Билет «{ticket}»**, по {b.get('takes', 0)} вдохов: без уроков → с уроками",
                      f"- разных ответов: {_pct(b.get('different'))} → {_pct(a.get('different'))}",
                      f"- самая частая фраза: {_pct(b.get('top_line'))} → {_pct(a.get('top_line'))}"
                      f" («{a.get('top_line_text', '')[:80]}»)",
                      f"- дословно из его прошлого: {_pct(b.get('from_its_past'))} → {_pct(a.get('from_its_past'))}",
                      f"- тянется к человеку: {_pct(b.get('to_a_person'))} → {_pct(a.get('to_a_person'))}",
                      *_hearing_lines(b.get("hearing"), a.get("hearing")),
                      f"- без уроков: «{(b.get('samples') or [''])[0][:100]}»",
                      f"- с уроками: «{(a.get('samples') or [''])[0][:100]}»", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@app.local_entrypoint()
def main(voices: str = "", generations: int = 6, samples: int = 20,
         refreeze: bool = False):
    wanted = [v for v in voices.split(",") if v]
    prepared = prepare.remote(wanted, refreeze)
    ids = list(prepared["voices"])
    results = list(examine.map(
        ids, [prepared["voices"][v] for v in ids],
        kwargs={"generations": generations, "samples": samples},
        return_exceptions=True))
    rows = []
    for vid, got in zip(ids, results):
        if isinstance(got, Exception):
            got = {"voice_id": vid,
                   "model_short": prepared["voices"][vid]["model_short"],
                   "error": str(got)[:300]}
        rows.append(got)
    stamp = time.time()
    record.remote([{"timestamp": stamp, "kind": "exam",
                    "frozen_ts": prepared["frozen_ts"], **r} for r in rows])
    out = REPO / "modal" / "exam_reports" / time.strftime("%Y-%m-%d_%H%M.md")
    write_report(rows, prepared["moments"], out)
    raw = out.with_suffix(".json")
    raw.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"report: {out}")
