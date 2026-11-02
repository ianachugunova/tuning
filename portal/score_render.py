"""The log, printed as a score. The proof that the data suffices.

    python3 score_render.py --demo                 # a fabricated slice
    python3 score_render.py choir.jsonl field.jsonl
    python3 score_render.py --tail 400 --parts choir.jsonl field.jsonl
    modal run choir_app.py::engrave                # the real log, on a CPU

An argument that the log is performable is worth less than a page of it read as
notation, so this is that page. Plain text on purpose: the engraved version is a
design job for later, and the thing that has to be true now is that nothing a
performer needs is missing. If a column here is empty, a field is missing, and
until 18 September that is still fixable.

Two views, and they are the two ways the piece is used.

**The chronology** is the score as it happened: one line per breath, in time,
with the rest before it, who sang, the pitch as a name and as a frequency, how
long it sounded, what was said, and an arrow to whatever it was tuning toward.
This is what an engraver sets and what a conductor reads.

**The parts** are the same material split by voice, which is what the intended
performance needs: people take the machine parts and machines take the people's,
so each part has to stand alone as something one performer can sing from — a
sequence of rests and pitches and durations with nothing else required.

Stdlib only. It runs on the log and needs no card, no Modal, no network, which
is the point of a score.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys

import score as S


# ---------------------------------------------------------------------------
# Reading the files
# ---------------------------------------------------------------------------


def read_jsonl(path: str) -> list[dict]:
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue          # a torn last line is not worth dying for
    except FileNotFoundError:
        pass
    return rows


# ---------------------------------------------------------------------------
# Notating one event
# ---------------------------------------------------------------------------

# How a pitch was arrived at, in one character, because it is a real musical
# distinction and it should be visible without reading a column of words.
#   =  the voice named a note        ("D4")
#   ~  the voice named a frequency   ("294 Hz") — a note plus a cents figure
#   °  a person sang and named nothing
#   ·  no pitch: words only, or a chosen silence
MARK = {S.PITCH_NOTE: "=", S.PITCH_HZ: "~", S.PITCH_SUNG: "°", S.PITCH_NONE: "·"}


def clock(t: float, origin: float) -> str:
    """Time from the first breath, as h:mm:ss.t — the score's own clock."""
    d = max(0.0, t - origin)
    h, rem = divmod(d, 3600.0)
    m, s = divmod(rem, 60.0)
    return f"{int(h):>3}:{int(m):02d}:{s:04.1f}"


def rest(seconds: float | None) -> str:
    """A silence, as a musician would want to see it: seconds, minutes, hours.

    Six weeks of a resting choir means rests of hours, and a human evening means
    rests of seconds. One column has to hold both, so the unit travels with the
    number rather than being declared in the header.
    """
    if seconds is None:
        return "     —"
    if seconds >= 3600:
        return f"{seconds / 3600:>5.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:>5.1f}m"
    return f"{seconds:>5.1f}s"


def pitch(n: S.Note) -> str:
    """The pitch, as a note name and — when it matters — how far off it is.

    A note name alone would lie about two thirds of this score: a voice that
    said "294 Hz" did not say D4, and a person who sang 305.2 Hz was 22 cents
    under D#4 and meant it. So the name is the head and the cents are the arrow.
    """
    if not n.pitch_hz:
        return " " * 15 + MARK[S.PITCH_NONE]
    name = n.pitch_note or "?"
    if n.pitch_kind == S.PITCH_NOTE:
        # A named note has no cents to show: the voice chose the name, so the
        # name is exact by definition and the frequency is what we made of it.
        return f"{name:<4}    {n.pitch_hz:>6.1f} {MARK[n.pitch_kind]}"
    sign = "+" if n.pitch_cents >= 0 else "\u2212"
    return (f"{name:<4}{sign}{abs(n.pitch_cents):<3}{n.pitch_hz:>6.1f} "
            f"{MARK.get(n.pitch_kind, '?')}")


def gesture(n: S.Note) -> str:
    """What the pitch did while it was held, when anything was recorded.

    Only human breaths have a trace; a machine breath is one steady tone by
    construction. Three characters of it are enough to show a slide.
    """
    if len(n.contour_hz) < 3:
        return ""
    lo, hi = min(n.contour_hz), max(n.contour_hz)
    if not lo or hi / lo < 1.012:            # under ~20 cents: a held tone
        return "\u2014"                       # an em dash: flat
    first, last = n.contour_hz[0], n.contour_hz[-1]
    if last > first * 1.012:
        return "\u2197"
    if last < first / 1.012:
        return "\u2198"
    return "\u223c"                           # went somewhere and came back


def arrow(n: S.Note) -> str:
    if n.tuning_to_id:
        return f"\u2192 {n.tuning_to_id}"
    if n.tuning_to:
        return f"\u2192 {n.tuning_to[:18]!r}"
    return ""


def breaths_ru(n: int) -> str:
    """"1 вдох", "3 вдоха", "12 вдохов" — the score is read in Russian."""
    tail = n % 100
    if 11 <= tail <= 14:
        return f"{n} вдохов"
    last = n % 10
    if last == 1:
        return f"{n} вдох"
    if 2 <= last <= 4:
        return f"{n} вдоха"
    return f"{n} вдохов"


def one_line(text: str, width: int) -> str:
    """Text as a single line of at most `width`, newlines folded into spaces."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[: width - 1] + "\u2026"


# ---------------------------------------------------------------------------
# The score
# ---------------------------------------------------------------------------

HEAD = ("      время   пауза  голос           высота             длит.  "
        "жест  что спето")
RULE = "\u2500" * 108


def chronology(notes: list[S.Note], origin: float) -> list[str]:
    out = [HEAD, RULE]
    for n in notes:
        if n.kind == "birth":
            out.append(f"{clock(n.t, origin)}  \u2016 РОЖДЕНИЕ  "
                       f"{one_line(n.text, 60)}")
            out.append(RULE)
            continue
        if n.kind == "death":
            out.append(RULE)
            out.append(f"{clock(n.t, origin)}  \u2016 СМЕРТЬ  {n.name} "
                       f"\u2014 {one_line(n.note, 58)}")
            if n.text:
                out.append(f"{'':>12}    последнее сказанное: "
                           f"{one_line(n.text, 60)}")
            out.append(RULE)
            continue
        who = "\u25cb" if n.who == "human" else "\u25cf"
        out.append(
            f"{clock(n.t, origin)} {rest(n.rest_s)}  "
            f"{who} {n.name or n.voice_id:<13.13} "
            f"{pitch(n):<18} {n.sound_s:>5.1f}s  {gesture(n):^4}  "
            f"{one_line(n.text, 40):<40} {arrow(n)}"
        )
    return out


def part_pages(notes: list[S.Note], origin: float) -> list[str]:
    """The same score, one page per part. This is what gets swapped.

    Each page is deliberately printable on its own: the header says who the part
    belonged to and who is meant to sing it in a performance, and the body is
    rests, pitches and durations. Nothing on the page refers to the rest of the
    log, which is what makes it a part rather than an index into an archive.
    """
    out: list[str] = []
    by_part = S.parts(notes)
    for vid, part in sorted(by_part.items(),
                            key=lambda kv: (-len(kv[1]), kv[0])):
        kind = part[0].who
        swap = "поёт человек" if kind == "machine" else "поёт машина"
        name = part[0].name if kind == "machine" else vid
        sounded = sum(n.sound_s for n in part)
        held = sum(n.own_rest_s or 0.0 for n in part)
        out.append("")
        out.append(f"\u2550\u2550 ПАРТИЯ: {name}  ({vid}, "
                   f"{'машина' if kind == 'machine' else 'человек'}"
                   f"{', ' + part[0].tongue if part[0].tongue else ''})"
                   f"  \u2014 в исполнении: {swap}")
        out.append(f"   {breaths_ru(len(part))}, {sounded:.0f} с звука, "
                   f"{held:.0f} с собственной тишины между ними")
        out.append("")
        for n in part:
            # `own_rest_s`, not `rest_s`: a performer with one part in front of
            # them counts their own silence, not the ensemble's.
            out.append(f"   {clock(n.t, origin)} {rest(n.own_rest_s)}  "
                       f"{pitch(n):<18} {n.sound_s:>5.1f}s  {gesture(n):^4}  "
                       f"{one_line(n.text, 46)}")
    return out


def summary(notes: list[S.Note]) -> list[str]:
    tones = [n for n in notes if n.kind == "tone"]
    if not tones:
        return ["ни одного вдоха в этом срезе"]
    with_pitch = [n for n in tones if n.pitch_hz]
    with_dur = [n for n in tones if n.sound_s > 0]
    with_rest = [n for n in tones if n.rest_s is not None]
    sounded = sum(n.sound_s for n in tones)
    span = max(n.end for n in tones) - min(n.t for n in tones)
    kinds: dict[str, int] = {}
    for n in tones:
        kinds[n.pitch_kind] = kinds.get(n.pitch_kind, 0) + 1
    return [
        f"{breaths_ru(len(tones))}  "
        f"(машин: {sum(1 for n in tones if n.who == 'machine')}, "
        f"людей: {sum(1 for n in tones if n.who == 'human')})",
        f"с высотой: {len(with_pitch)}/{len(tones)}   "
        f"с длительностью: {len(with_dur)}/{len(tones)}   "
        f"с паузой перед: {len(with_rest)}/{len(tones)}",
        "как названа высота: "
        + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())),
        f"звук {sounded / 60:.1f} мин из {span / 60:.1f} мин — "
        f"тишина {100 * (1 - sounded / span) if span > 0 else 0:.0f}%",
        f"смертей в срезе: {sum(1 for n in notes if n.kind == 'death')}",
    ]


def render(choir_rows: list[dict], human_rows: list[dict],
           tail: int = 120, by_part: bool = False) -> str:
    notes = S.read(choir_rows, human_rows)
    tones = [n for n in notes if n.kind == "tone"]
    if not tones:
        return "лог пуст: ни одного тона"
    if tail and len(notes) > tail:
        notes = notes[-tail:]
    origin = min(n.t for n in tones)
    first = datetime.datetime.fromtimestamp(origin, datetime.timezone.utc)

    lines = [
        "TUNING MEDITATION \u2014 ПАРТИТУРА",
        f"ноль партитуры: {first:%d %b %Y %H:%M:%S} UTC  ({origin:.0f})",
        "",
        *summary(notes),
        "",
    ]
    lines += chronology(notes, origin)
    if by_part:
        lines += part_pages(notes, origin)
    lines += ["", "\u25cf машина   \u25cb человек   "
              "= названа нота   ~ названа частота   \u00b0 спето без имени   "
              "\u00b7 без высоты"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A fabricated slice, for when there is no log yet
# ---------------------------------------------------------------------------


def demo_logs() -> tuple[list[dict], list[dict]]:
    """Twelve minutes of a field that never happened, in the real shape.

    Built with `score.py`'s own field builders — the same calls the choir and
    the portal make — so this is a test of the writing as well as the reading.
    It exists because the score has to be provably renderable *before* 18
    September, and before 18 September there is nothing in the log but tests.
    """
    t0 = 1_789_000_000.0
    cast = [
        {"voice_id": "voice_00", "model_short": "gpt-2", "tongue": "English",
         "hf_id": "openai-community/gpt2-large", "instruct": False},
        {"voice_id": "voice_02", "model_short": "phi-4-mini", "tongue": "English",
         "hf_id": "microsoft/Phi-4-mini-instruct", "instruct": True},
        {"voice_id": "voice_05", "model_short": "qwen3", "tongue": "Chinese",
         "hf_id": "Qwen/Qwen3-8B", "instruct": True},
    ]
    choir: list[dict] = [{
        "timestamp": t0 - 60, "kind": "birth", "declared_usd": 2500.0,
        "rate_usd_per_hour": 2.5, "cast": cast,
        "wallets": {c["voice_id"]: 833.33 for c in cast},
        "note": "given at Endzeit, 18 September 2026",
    }]
    human: list[dict] = []

    # One chronology, written the way the two apps write it: the choir measures
    # the rest it answered against the whole field, and each phone measures the
    # rest between its own two breaths. Interleaved here rather than written as
    # two independent sequences, because that is the mistake that makes a
    # merged score disagree with itself, and it is the mistake this demo exists
    # to be checked against.
    #
    # (offset, machine, voice, text, pitches, breath, sound, tuning, playback)
    script = [
        (0.0, "voice_02", "D4, a low steady hum", [(293.66, "D4")], 6.0, 6.4,
         None, "both"),
        (31.0, "voice_05", "294Hz \u2014 \u6211\u8ddf\u7740\u4f60", [(294.0, "294Hz")],
         5.0, 7.1, "voice_02", "both"),
        (74.0, "voice_00", "a long tone that does not end", [], 4.0, 5.2,
         None, "both"),
        (140.0, "voice_02", "G3", [(196.0, "G3")], 9.0, 9.0, None, "sine"),
        # A voice that names a tone and chooses not to sound it. Still its turn.
        (302.0, "voice_05", "(silence)", [(392.0, "G4")], 12.0, 0.0, None,
         "none"),
        (455.0, "voice_02", "F#4, following the voice that just sang",
         [(369.99, "F#4")], 8.0, 8.9, "human_7f3a", "both"),
    ]
    # (offset, voice, hz, duration, contour) — two people, one of them sliding.
    sung = [
        (208.0, "human_7f3a", 305.2, 4.6, [303.1, 305.0, 306.2, 305.4]),
        (399.0, "human_7f3a", 306.1, 7.2, [299.0, 303.4, 309.8, 316.2]),
        (402.0, "human_c1d9", 220.4, 3.1, [220.0, 220.6, 220.1]),
    ]

    timeline = ([(s[0], True, s) for s in script]
                + [(h[0], False, h) for h in sung])
    field_end: float | None = None
    own_end: dict[str, float] = {}

    for off, is_machine, item in sorted(timeline, key=lambda x: x[0]):
        ts = t0 + off
        if is_machine:
            _, vid, content, pitches, breath, sound, tuning, playback = item
            row = {
                "timestamp": ts, "voice_id": vid, "kind": "tone",
                "model_short": next(c["model_short"] for c in cast
                                    if c["voice_id"] == vid),
                "action": "TUNE" if tuning else "NEW",
                "content": content, "tuning_to": tuning,
                "breath_chosen_s": breath, "api_latency_ms": 900,
                "playback": playback, "playback_note": "",
                "gpu_s": 2.0, "own_cost_usd": 0.0014,
                "sound_s": sound,
                "toned": bool(pitches) and playback != "none",
                "spoke": playback in ("both", "tts") and sound > 0,
                "silence_before_s": S.silence_before(field_end, ts),
                "heard": [], "tuning_to_id": tuning,
                "tongue": next(c["tongue"] for c in cast
                               if c["voice_id"] == vid),
            }
            row.update(S.machine_pitch_fields(pitches))
            choir.append(row)
        else:
            _, vid, hz, sound, cont = item
            row = {
                "ts": ts + sound + 0.5, "voice_id": vid, "kind": "tone",
                "who": "human", "source": "portal",
                "note": S.note_name(hz), "hz": hz, "cents": S.cents_off(hz),
                "stability": "vibrato" if len(cont) > 3 else "steady",
                "duration_s": sound, "voiced_fraction": 0.94,
                "started_ts": ts, "ended_ts": ts + sound, "tail_s": 0.5,
                "sound_s": sound,
                "own_rest_s": S.silence_before(own_end.get(vid), ts),
                "std_hz": 2.4, "contour_hz": cont,
                "contour_hop_s": round(sound / len(cont), 3),
                "content": f"{S.note_name(hz)} ~{hz:.0f}Hz",
            }
            row.update(S.human_pitch_fields(hz, S.note_name(hz),
                                            S.cents_off(hz)))
            human.append(row)
        if sound > 0:
            field_end = max(field_end or 0.0, ts + sound)
            own_end[vid] = ts + sound

    choir.append({
        "timestamp": t0 + 640, "kind": "death", "voice_id": "voice_05",
        "model_short": "qwen3", "born": t0 - 60, "lived_s": 700.0,
        "seed_usd": 833.33, "spent_usd": 833.33, "breaths": 2,
        "gpu_s": 4.0, "shared_s": 200.0, "freed_gb": 16.1, "living_after": 2,
        "last_words": "294Hz \u2014 \u6211\u8ddf\u7740\u4f60",
    })
    return choir, human


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("choir", nargs="?", help="choir.jsonl")
    ap.add_argument("field", nargs="?", help="field.jsonl")
    ap.add_argument("--tail", type=int, default=120,
                    help="how many events to print, from the end (0 = all)")
    ap.add_argument("--parts", action="store_true",
                    help="also print one page per part, as it is performed")
    ap.add_argument("--demo", action="store_true",
                    help="render a fabricated slice, for when the log is empty")
    args = ap.parse_args(argv)

    if args.demo or not args.choir:
        choir, human = demo_logs()
        if not args.demo:
            print("нет файлов лога \u2014 показан демонстрационный срез "
                  "(--demo)\n", file=sys.stderr)
    else:
        choir = read_jsonl(args.choir)
        human = read_jsonl(args.field) if args.field else []

    print(render(choir, human, tail=args.tail, by_part=args.parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
