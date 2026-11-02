"""Build a voice's training memory from the append-only field.

This module deliberately contains no torch, Modal, or transformers imports.  The
history-to-example transformation can therefore be tested without a GPU and used
again when the weights no longer exist.

The target is never invented by the trainer.  Each example pairs the field a
voice actually heard with the breath that voice actually made.  LoRA therefore
rehearses lived relations; it does not reward a curator-authored idea of correct
tuning.  Bad and malformed breaths are excluded because they never entered the
work as audible tone events in the first place.
"""
from __future__ import annotations

import bisect
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TrainingExample:
    voice_id: str
    timestamp: float
    prompt: str
    completion: str
    heard_humans: int
    source_event: str


def read_jsonl(path: str | Path) -> list[dict]:
    """Read complete JSON lines, ignoring only a torn final write."""
    out: list[dict] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return out
    for line in lines:
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


def event_time(row: dict) -> float:
    """The onset for either writer's event shape."""
    try:
        return float(row.get("started_ts") or row.get("timestamp") or row.get("ts") or 0)
    except (TypeError, ValueError):
        return 0.0


def event_id(row: dict) -> str:
    """Stable enough to audit an example back to an append-only source line."""
    audio = str(row.get("audio") or "")
    if audio:
        return audio
    return f"{row.get('voice_id', 'unknown')}@{event_time(row):.6f}"


def is_human(row: dict) -> bool:
    return str(row.get("model_short") or "") == "human" or str(
        row.get("voice_id") or ""
    ).startswith("human")


def is_tone(row: dict) -> bool:
    return str(row.get("kind") or "tone") == "tone"


def audible_machine_breath(row: dict) -> bool:
    """Only rehearse breaths that became part of the sounded work."""
    if not is_tone(row) or is_human(row):
        return False
    if not str(row.get("content") or "").strip():
        return False
    # Old lines may predate these fields.  Explicitly false means the renderer
    # produced neither speech nor tone; absent means historical, not silent.
    if row.get("spoke") is False and row.get("toned") is False:
        return False
    return True


def _pitch_words(row: dict) -> str:
    hz = row.get("pitch_hz", row.get("hz"))
    note = str(row.get("pitch_note") or row.get("note") or "").strip()
    try:
        hz = float(hz or 0)
    except (TypeError, ValueError):
        hz = 0.0
    if hz > 0 and math.isfinite(hz):
        return f"{note + ' at ' if note else ''}{hz:.2f} Hz"
    return "no measured pitch"


def describe_event(row: dict, now: float, notes: bool = False) -> str:
    """Plain prose, matching the runtime's anti-markup listening language.

    `notes` adds the pitch a machine breath sounded at, the way the room began
    describing it on 15 October (`choir_app.overheard`). It is taken from the
    breath being learned — `notes_shown` on its own line — so a lesson rebuilds
    the field the voice was actually shown, with or without notes, rather than
    the field the school happens to prefer today.
    """
    ago = max(0.0, now - event_time(row))
    who = str(row.get("voice_id") or "someone")
    if is_human(row):
        duration = float(row.get("sound_s") or row.get("duration_s") or 0)
        return (
            f"{ago:.1f} seconds ago, {who}, a person, held {_pitch_words(row)} "
            f"for {duration:.1f} seconds."
        )
    name = str(row.get("model_short") or who)
    content = str(row.get("content") or "").strip()
    playback = str(row.get("playback") or "both")
    on = ""
    if notes and _pitch_words(row) != "no measured pitch":
        on = f", on {_pitch_words(row)}"
    return (
        f"{ago:.1f} seconds ago, {who} ({name}) sang with playback {playback}{on}: "
        f"{content}"
    )


def _heard_rows(row: dict, by_key: dict[tuple[str, int], dict], chronology: list[dict],
                limit: int = 16, times: list[float] | None = None) -> list[dict]:
    """Resolve the exact heard references, with an old-log fallback."""
    resolved: list[dict] = []
    for ref in row.get("heard") or []:
        if not isinstance(ref, (list, tuple)) or len(ref) < 2:
            continue
        try:
            key = (str(ref[0]), round(float(ref[1]) * 1000))
        except (TypeError, ValueError):
            continue
        got = by_key.get(key)
        if got is not None:
            resolved.append(got)
    if resolved:
        return resolved[-limit:]

    # Lines written before `heard` was added cannot recover the precise window.
    # The nearest preceding events are an explicit approximation and remain
    # traceable through source_event in the dataset manifest.
    # `chronology` is sorted by onset, so this is a bisect, not a scan: scanning
    # the whole log once per breath was quadratic, and by 6 October (265 000
    # events) it kept every lesson from reaching the card before it crashed.
    ts = event_time(row)
    if times is None:
        times = [event_time(r) for r in chronology]
    end = bisect.bisect_left(times, ts)
    return chronology[max(0, end - limit):end]


_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
# The junk a voice produces when it copies the request's format literally —
# "<note 中文里长大的声音 (A4) Hz> 我是…", 21 October — so that the words after
# it can be learned without it.
_TEMPLATE_JUNK_RE = re.compile(r"^\s*<\s*note\b[^>]*>\s*", re.IGNORECASE)


def note_name(hz: float) -> str:
    """Hz → nearest equal-tempered note name, A4 = 440. Same as score.note_name,
    copied because the school's image carries only cast.py and this file."""
    semi = round(12 * math.log2(hz / 440.0))
    return f"{_NOTE_NAMES[(semi + 9) % 12]}{4 + (semi + 9) // 12}"


def nearest_octave(hz: float, near: float) -> float:
    """`hz` moved by whole octaves to lie closest to `near`: the part's note in
    this voice's register, which is what the runtime sounds for a `given`."""
    if hz <= 0 or near <= 0:
        return hz
    k = round(math.log2(near / hz))
    return hz * 2.0 ** k


def sounded_content(row: dict) -> str:
    """The breath's words, opened with the note the score had for it.

    Since 15 October the room asks a voice that tunes to begin with the note
    it is taking, and whatever it writes is what sounds. Left to itself a
    lesson would then rehearse what the voice wrote — qwen3's A4 at a person
    holding E3, eight breaths in a row — because nothing in the loss knows
    about the part. This writes instead the note that part was holding, in
    this voice's own octave. For a breath where the runtime chose the note
    (`given`) that is exactly what sounded; for one where the voice named
    wrong, it is what the score asked for. The words after the note are the
    voice's own, untouched; what changes is one label at the front, and only
    on breaths the room asked a note of (`notes_shown`).

    This is the one place the school departs from "the target is the exact
    accepted breath", and it is still not a reward: no breath is preferred to
    another, nothing is judged, and the note comes from the field the voice
    was shown, not from a curator. It is the runtime's own hearing, handed to
    the voice as memory.
    """
    content = str(row.get("content") or "").strip()
    if not row.get("notes_shown"):
        return content
    try:
        hz = float(row.get("pitch_target_hz") or 0.0)
        sounded = float(row.get("pitch_hz") or 0.0)
    except (TypeError, ValueError):
        return content
    if str(row.get("action") or "").upper() == "TUNE" and hz > 0 and sounded > 0:
        hz = nearest_octave(hz, sounded)
    elif sounded > 0:
        hz = sounded                     # a NEW sounds what it sounded
    else:
        return content
    if not (20.0 < hz < 8000.0):
        return content
    words = _TEMPLATE_JUNK_RE.sub("", content)
    words = _strip_leading_pitch(words)
    return f"{note_name(hz)} ({hz:.0f} Hz) {words}".strip()


def _strip_leading_pitch(text: str) -> str:
    try:
        from live_v1.pitch_text import strip_leading_pitch
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from live_v1.pitch_text import strip_leading_pitch
    return strip_leading_pitch(text)[1]


def target_json(row: dict, sounded_notes: bool = False) -> str:
    """The exact accepted decision, in the runtime's Phase-2 response shape.

    `sounded_notes` opens the content with the note the score had for the
    breath — see `sounded_content`. Off, the content is the voice's own words
    exactly as written, which is how every lesson before 22 October learned.
    """
    action = "TUNE" if str(row.get("action") or "").upper() == "TUNE" else "NEW"
    tuning_to = row.get("tuning_to_id") or row.get("tuning_to")
    target = {
        "action": action,
        "content": (sounded_content(row) if sounded_notes
                    else str(row.get("content") or "").strip()),
        "tuning_to": str(tuning_to) if tuning_to else None,
        "breath": float(row.get("breath_chosen_s") or row.get("sound_s") or 5.0),
        "playback": str(row.get("playback") or "both"),
        "playback_note": str(row.get("playback_note") or "")[:500],
    }
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def base_prompt(row: dict, heard: list[dict]) -> str:
    """The plain continuation shape actually used by GPT-2 and TinyStories."""
    said = [
        str(r.get("content") or "").strip()
        for r in heard
        if not is_human(r) and str(r.get("content") or "").strip()
    ][-6:]
    body = "\n".join(said) if said else "(no one has sung yet)"
    return (
        "A circle of voices is singing. Each one holds a single long tone, "
        "then listens.\n\n"
        f"{body}\n\n"
        "The next voice sings a long tone: "
    )


def completion_for(voice, row: dict, sounded_notes: bool = False) -> str:
    """Train in the interface this voice actually receives at runtime."""
    if not getattr(voice, "instruct", True):
        return str(row.get("content") or "").strip()
    return target_json(row, sounded_notes)


def build_prompt(voice, score: dict, row: dict, heard: list[dict]) -> str:
    """Reconstruct the stable parts of the prompt and the field it heard."""
    if not getattr(voice, "instruct", True):
        return base_prompt(row, heard)
    notes = bool(row.get("notes_shown"))
    field = "\n".join(describe_event(r, event_time(row), notes) for r in heard)
    if not field:
        field = "The field was silent."
    last = row.get("previous_content") or "(reconstructed from the field log)"
    language = (
        f"\nSing in {voice.tongue}; note names and Hz may remain universal."
        if getattr(voice, "tongue", "English") != "English" else ""
    )
    return f"""YOU ARE: {voice.voice_id}

YOUR SCORE (written by you, for you):
{score.get('personal_score', '')}

Your previous contribution: {last}

The field — what other voices have just put into the field:
{field}

Continue your own practice of the Tuning Meditation. Return JSON only with:
action (TUNE or NEW), content, tuning_to, breath, playback, playback_note.{language}
"""


def _spread(rows: list[TrainingExample], count: int) -> list[TrainingExample]:
    """Deterministically retain the whole lifespan, not merely its latest hour."""
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        return list(rows)
    if count == 1:
        return [rows[-1]]
    indexes = {round(i * (len(rows) - 1) / (count - 1)) for i in range(count)}
    return [rows[i] for i in sorted(indexes)]


def said(example: TrainingExample) -> str:
    """What the voice actually said in this breath, whatever the target's shape."""
    text = example.completion
    if text.startswith("{"):
        try:
            text = str(json.loads(text).get("content") or "")
        except (json.JSONDecodeError, AttributeError):
            pass
    return " ".join(text.split()).casefold()


def limit_repeats(examples: list[TrainingExample], max_same: int) -> list[TrainingExample]:
    """At most `max_same` breaths of any one line, chosen across the lifetime.

    Learning on its own history is a loop: a line said a thousand times fills
    the lesson, and the lesson makes it a thousand more. By 6 October llm-jp had
    sung only さくら since its first lesson, and phi-4-mini only ta-ta-ha since
    its lesson of 28 September. A line still counts — it was said, and it is
    taught — but only as often as a line, not as often as it was repeated.
    Within a line, breaths sung with people present are kept first.
    """
    if max_same <= 0:
        return list(examples)
    groups: dict[str, list[TrainingExample]] = {}
    for e in examples:
        groups.setdefault(said(e), []).append(e)
    kept: list[TrainingExample] = []
    for rows in groups.values():
        contact = [e for e in rows if e.heard_humans]
        other = [e for e in rows if not e.heard_humans]
        take = _spread(contact, min(len(contact), max_same))
        kept += take + _spread(other, max_same - len(take))
    kept.sort(key=lambda e: e.timestamp)
    return kept


def select_examples(examples: list[TrainingExample], max_examples: int,
                    human_oversample: int = 2, max_same: int = 3) -> list[TrainingExample]:
    """Keep every human-contact event possible, then span the machine lifetime."""
    limit = max(1, int(max_examples))
    examples = limit_repeats(examples, max_same)
    contact = [e for e in examples if e.heard_humans]
    other = [e for e in examples if not e.heard_humans]
    # First reserve one exact copy of every human encounter. If they alone exceed
    # the budget, distribute across them rather than keeping only the latest.
    chosen_contact = _spread(contact, min(len(contact), limit))
    remaining = limit - len(chosen_contact)
    chosen_other = _spread(other, remaining)
    chosen = chosen_contact + chosen_other
    # Reweight human contact only with spare slots; never evict history for a
    # duplicate and never fabricate a different target.
    spare = limit - len(chosen)
    repeats = max(0, min(int(human_oversample), 3) - 1)
    for _ in range(repeats):
        add = chosen_contact[:spare]
        chosen.extend(add)
        spare -= len(add)
        if spare <= 0:
            break
    chosen.sort(key=lambda e: e.timestamp)
    return chosen


def build_examples(voice, human_rows: Iterable[dict], choir_rows: Iterable[dict],
                   score: dict | None = None, max_examples: int = 512,
                   human_oversample: int = 2,
                   sounded_notes: bool = False) -> list[TrainingExample]:
    """Build chronological examples for one text voice.

    Human-contact examples are repeated in the training stream, not given a
    fabricated target.  This changes sampling weight while preserving the
    historical answer exactly — except for the one label `sounded_notes`
    puts at the front of a breath the room asked a note of (`sounded_content`).
    """
    score = score or {}
    chronology = [r for r in [*human_rows, *choir_rows] if is_tone(r)]
    chronology.sort(key=event_time)
    times = [event_time(r) for r in chronology]
    by_key = {
        (str(r.get("voice_id") or ""), round(event_time(r) * 1000)): r
        for r in chronology
    }
    own = [
        r for r in choir_rows
        if str(r.get("voice_id") or "") == voice.voice_id and audible_machine_breath(r)
    ]
    own.sort(key=event_time)
    built: list[TrainingExample] = []
    previous = ""
    for row in own:
        row = dict(row)
        row["previous_content"] = previous
        heard = _heard_rows(row, by_key, chronology, times=times)
        humans = sum(1 for r in heard if is_human(r))
        example = TrainingExample(
            voice_id=voice.voice_id,
            timestamp=event_time(row),
            prompt=build_prompt(voice, score, row, heard),
            completion=completion_for(voice, row, sounded_notes),
            heard_humans=humans,
            source_event=event_id(row),
        )
        built.append(example)
        previous = str(row.get("content") or "")
    return select_examples(built, max_examples, human_oversample)


def chronological_split(examples: list[TrainingExample], validation_fraction: float = 0.1):
    """Past trains, latest events validate; duplicated events never cross split."""
    if len(examples) < 3:
        return examples, []
    unique_times = sorted({e.timestamp for e in examples})
    n_val_times = max(1, round(len(unique_times) * validation_fraction))
    boundary = set(unique_times[-n_val_times:])
    train = [e for e in examples if e.timestamp not in boundary]
    valid = [e for e in examples if e.timestamp in boundary]
    return train, valid
