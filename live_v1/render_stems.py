#!/usr/bin/env python3
"""
live_v1/render_stems.py — OFFLINE "render-to-stems" mode.

Reproduces a past performance as separate per-voice mono WAV files ("stems"),
all aligned to one common timeline so they line up when imported together into
a DAW (e.g. Reaper for spatialization).

For every ``kind == "tone"`` event of an LLM voice (voice_00..voice_09) in the
run's ``events.jsonl``, we render that breath with EXACTLY the same code path
the live engine uses (``live_v1.sound._render_breath``), then place the audio
into a per-voice buffer at offset ``event.timestamp - started_at_unix``. The
result is one mono float32 WAV per voice, all the same length (the full run
duration), so the breaths sit at their real time offsets.

Notes / faithfulness:
  * ``playback == "none"`` breaths were SILENT in the live performance, so we
    render nothing for them (the span stays silent).
  * Human voices (``human_*``) CANNOT be reproduced — only their pitch/text was
    logged, not their recorded audio — so they are skipped entirely.
  * Rendering uses macOS ``say`` per breath (offline, no network/API keys).
    With ~1100 events this can take several minutes.

Usage:
    python3 -m live_v1.render_stems --run run_2026-06-07_03-11-12
    python3 -m live_v1.render_stems --run <id> --out-dir live_v1/out
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

# Reuse the live engine's rendering + helpers so stems sound identical to the
# live output. Do NOT reinvent TTS / sine synthesis here.
from live_v1.sound import (
    RUNS_PARENT,
    _render_breath,
    resolve_run_dir,
    voice_for,
)

SR = 48000  # match live engine (sound.SR_TARGET) so rendered audio lines up
PEAK_TARGET = 0.95
LLM_VOICE_IDS = [f"voice_{i:02d}" for i in range(10)]


def _load_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _read_tone_events(events_path: Path) -> list[dict]:
    """Return all ``kind == "tone"`` events with non-empty content."""
    events: list[dict] = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("kind") != "tone":
                continue
            events.append(ev)
    return events


def render_stems(run_dir: Path) -> dict:
    import numpy as np

    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        raise SystemExit(f"[render_stems] events.jsonl not found: {events_path}")

    meta = _load_meta(run_dir)
    started = meta.get("started_at_unix")
    wallclock = float(meta.get("wallclock_duration_s") or 0.0)

    all_events = _read_tone_events(events_path)

    # Group tone events by voice.
    by_voice: dict[str, list[dict]] = {}
    human_breaths = 0
    for ev in all_events:
        vid = str(ev.get("voice_id", ""))
        if vid.startswith("human_") or ev.get("action") == "HUMAN":
            human_breaths += 1
            continue
        by_voice.setdefault(vid, []).append(ev)

    # Establish the common timeline origin. Prefer meta started_at_unix; fall
    # back to the earliest tone-event timestamp.
    if started is None:
        first_ts = min(
            (float(ev["timestamp"]) for ev in all_events if ev.get("timestamp") is not None),
            default=0.0,
        )
        started = first_ts

    # Total length = furthest breath end, but at least the wallclock duration so
    # every stem is the same length and aligns on import.
    max_end_s = wallclock
    for ev in all_events:
        ts = ev.get("timestamp")
        if ts is None:
            continue
        vid = str(ev.get("voice_id", ""))
        if vid.startswith("human_") or ev.get("action") == "HUMAN":
            continue
        offset = float(ts) - float(started)
        breath = float(ev.get("breath_chosen_s") or 0.0)
        max_end_s = max(max_end_s, offset + breath)

    total_s = max(0.1, max_end_s)
    total_n = int(SR * total_s) + 1
    print(f"[render_stems] run        = {run_dir.name}")
    print(f"[render_stems] origin     = {started} (unix)")
    print(f"[render_stems] timeline   = {total_s:.2f}s  ({total_n} samples @ {SR} Hz)")
    print(f"[render_stems] tone events= {len(all_events)} total, "
          f"{human_breaths} human breaths skipped (cannot reproduce)")
    print()

    stems_dir = run_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)

    # voice_id -> model short name (for nice filenames)
    short_by_voice: dict[str, str] = {}
    for agent in meta.get("agents", []) or []:
        vid = agent.get("voice_id")
        if vid:
            short_by_voice[vid] = agent.get("model_short") or ""

    report: list[dict] = []

    for vid in LLM_VOICE_IDS:
        events = by_voice.get(vid, [])
        voice = voice_for(vid)
        short = short_by_voice.get(vid, "")
        fname = f"{vid}_{short}.wav" if short else f"{vid}.wav"
        out_path = stems_dir / fname

        buf = np.zeros(total_n, dtype=np.float32)
        n_rendered = 0
        n_silent = 0
        n_total = len(events)
        print(f"[{vid}] {short or voice} — {n_total} tone breaths -> {fname}")

        for i, ev in enumerate(events, 1):
            ts = ev.get("timestamp")
            content = str(ev.get("content", "")).strip()
            breath_s = float(ev.get("breath_chosen_s") or 5.0)
            playback = str(ev.get("playback") or "both")

            if not content or ts is None:
                continue

            # Faithful to the live performance: a "none" breath was silent.
            if playback == "none":
                n_silent += 1
                continue

            audio, sr = _render_breath(content, voice, breath_s, playback)
            if audio is None or sr <= 0 or len(audio) == 0:
                continue

            start_sample = int(round((float(ts) - float(started)) * SR))
            if start_sample < 0:
                start_sample = 0
            end_sample = start_sample + len(audio)
            if start_sample >= total_n:
                continue
            if end_sample > total_n:
                audio = audio[: total_n - start_sample]
                end_sample = total_n
            # Mix (add) in case of overlap.
            buf[start_sample:end_sample] += np.asarray(audio, dtype=np.float32)
            n_rendered += 1

            if i % 25 == 0 or i == n_total:
                print(f"    {vid}: {i}/{n_total} breaths "
                      f"({n_rendered} rendered, {n_silent} silent)")

        # Peak-normalize per file to avoid clipping.
        peak = float(np.abs(buf).max()) if buf.size else 0.0
        if peak > PEAK_TARGET:
            buf = buf * (PEAK_TARGET / peak)
            final_peak = PEAK_TARGET
        else:
            final_peak = peak

        _write_wav_float32(out_path, buf, SR)
        duration_s = len(buf) / SR
        report.append({
            "voice_id": vid,
            "model_short": short,
            "filename": fname,
            "path": str(out_path),
            "duration_s": duration_s,
            "peak": final_peak,
            "n_rendered": n_rendered,
            "n_silent": n_silent,
            "n_total": n_total,
        })
        print(f"    -> wrote {fname}  dur={duration_s:.2f}s  peak={final_peak:.4f}")
        print()

    _print_summary(report, stems_dir, human_breaths)
    return {"report": report, "stems_dir": str(stems_dir), "human_breaths": human_breaths}


def _write_wav_float32(path: Path, audio, sr: int) -> None:
    """Write a mono float32 WAV (subtype FLOAT) at ``sr`` Hz."""
    import numpy as np
    import soundfile as sf

    data = np.asarray(audio, dtype=np.float32).ravel()
    sf.write(str(path), data, int(sr), subtype="FLOAT")


def _print_summary(report: list[dict], stems_dir: Path, human_breaths: int) -> None:
    print("=" * 78)
    print("PER-VOICE STEM SUMMARY")
    print("=" * 78)
    header = f"{'file':28s} {'dur(s)':>8s} {'peak':>8s} {'rendered':>9s} {'silent':>7s}"
    print(header)
    print("-" * 78)
    silent_files = []
    for r in report:
        print(f"{r['filename']:28s} {r['duration_s']:8.2f} {r['peak']:8.4f} "
              f"{r['n_rendered']:9d} {r['n_silent']:7d}")
        if r["peak"] <= 0.0:
            silent_files.append(r["filename"])
    print("-" * 78)
    print(f"output folder: {stems_dir}")
    print(f"human voices skipped: {human_breaths} breaths "
          f"(only pitch/text was logged, not recorded audio — cannot reproduce)")
    if silent_files:
        print(f"SILENT stems (peak == 0): {', '.join(silent_files)}")
    else:
        print("no entirely-silent stems")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render a past run to per-voice mono WAV stems (offline).")
    ap.add_argument("--run", default=None,
                    help="Run name or path (default: latest under --out-dir)")
    ap.add_argument("--out-dir", default=str(RUNS_PARENT))
    args = ap.parse_args()

    runs_dir = Path(args.out_dir).resolve()
    run_dir = resolve_run_dir(args.run, runs_dir)

    t0 = time.time()
    render_stems(run_dir)
    print(f"[render_stems] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
