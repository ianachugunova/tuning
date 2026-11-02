"""How well the voices hear, read off the live log.

    modal run hearing_report.py              # since the notes were switched on
    modal run hearing_report.py --days 3     # the last three days only

For every TUNE breath since 15 October the log carries the note the voice named
(`pitch_named_hz`) and the note the part it answered was holding
(`pitch_target_hz`). This prints, per voice and per generation of lessons, how
often it named a note at all, how far the named note lay from the target in
cents with octaves set aside, and how often it was within a quarter-tone. The
arithmetic is `exam_metrics.describe_hearing`; this only reads the file where it
is, because the file does not come down.

Costs nothing but a CPU: no card, no wallet, no line written anywhere.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIELD_DIR = "/field"
CHOIR_LOG = f"{FIELD_DIR}/choir.jsonl"

app = modal.App("tuning-hearing")
field_volume = modal.Volume.from_name("tuning-field")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file(HERE / "learning.py", remote_path="/root/learning.py")
    .add_local_file(HERE / "exam_metrics.py", remote_path="/root/exam_metrics.py")
    .add_local_file(REPO / "live_v1" / "pitch_text.py",
                    remote_path="/root/live_v1/pitch_text.py")
    .add_local_file(REPO / "live_v1" / "__init__.py",
                    remote_path="/root/live_v1/__init__.py")
)


@app.function(image=image, volumes={FIELD_DIR: field_volume}, timeout=1200,
              memory=4096)
def read(since: float) -> dict:
    import exam_metrics as xm

    field_volume.reload()
    groups: dict[tuple, list] = {}
    with open(CHOIR_LOG, encoding="utf-8") as fh:
        for line in fh:
            if '"TUNE"' not in line or '"pitch_target_hz"' not in line:
                continue
            e = json.loads(line)
            if e.get("kind") != "tone" or float(e.get("timestamp") or 0) < since:
                continue
            if not e.get("pitch_target_hz"):
                continue
            key = (e.get("model_short"), e.get("adapter_version") or "base",
                   bool(e.get("notes_shown")))
            groups.setdefault(key, []).append(
                (e.get("pitch_named_hz"), e.get("pitch_target_hz")))
    return {"|".join(map(str, k)): xm.describe_hearing(v) for k, v in groups.items()}


@app.local_entrypoint()
def main(days: float = 0.0):
    since = time.time() - days * 86400 if days else 1792400000.0   # 15 Oct 2026
    got = read.remote(since)
    rows = sorted(got.items())
    print(f"{'voice':<15}{'lessons of':<17}{'notes':<7}{'tunes':>6}{'named':>7}"
          f"{'cents':>7}{'hit':>6}")
    for key, r in rows:
        voice, gen, notes = key.split("|")
        gen = gen[6:8] + "." + gen[4:6] if gen != "base" else "base"
        print(f"{voice:<15}{gen:<17}{'on' if notes == 'True' else 'off':<7}"
              f"{r['tunes']:>6}{r['named']:>7.0%}"
              f"{(r['cents_median'] if r['cents_median'] is not None else float('nan')):>7.0f}"
              f"{(r['hit'] if r['hit'] is not None else float('nan')):>6.0%}")
