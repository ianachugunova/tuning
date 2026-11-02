#!/usr/bin/env python3
"""
live_v1/concert.py — Live concert orchestrator for Tuning Meditation.

This is the version you run AT THE VENUE. All venue-specific configuration
is at the top of this file, in plain Python — edit the CONFIG section and
re-run. Nothing to recompile, nothing to install.

THE ONE THING TO REMEMBER:
  This file works in TWO modes from the same source.

  - LAPTOP mode: INPUT_DEVICE = None and OUTPUT_DEVICE = None.
    HumanVoice opens the built-in microphone. LLM TTS plays on system
    speakers. Use this to develop and test at home. Only ONE human
    channel makes sense in this mode (the laptop has one input).

  - VENUE mode: set INPUT_DEVICE and OUTPUT_DEVICE to the MIDAS M32R
    (or whatever multi-channel audio interface is plugged into your
    Mac), and fill in HUMAN_CHANNELS / VOICE_CHANNELS with the actual
    M32R channel numbers. The same code path now drives N microphones
    and routes 10 LLM voices to 10 physical speakers.

  - HOME TEST with five *logical* mic channels but no M32R: use a virtual
    multichannel driver (e.g. BlackHole 16ch) — see
    live_v1/VIRTUAL_MULTICHANNEL_MACOS.md. This is not a "virtual machine";
    it is a virtual audio device.

USAGE:

  # See what audio devices are visible to macOS:
  python3 -m live_v1.concert --list-devices

  # Run with the current CONFIG below:
  python3 -m live_v1.concert

  # Override duration without touching the file:
  python3 -m live_v1.concert --duration 600
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
load_dotenv(REPO_ROOT / "Tuning_Meditation_LLMs" / ".env")

if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))
    from Tuning_Meditation_LLMs.live_v1 import human as human_mod         # type: ignore
    from Tuning_Meditation_LLMs.live_v1 import sound as sound_mod         # type: ignore
    from Tuning_Meditation_LLMs.live_v1 import reports as reports_mod     # type: ignore
    from Tuning_Meditation_LLMs.live_v1.models import pick_pool           # type: ignore
    from Tuning_Meditation_LLMs.live_v1.orchestrator import run_session   # type: ignore
else:
    from . import human as human_mod
    from . import sound as sound_mod
    from . import reports as reports_mod
    from .models import pick_pool
    from .orchestrator import run_session


# ===========================================================================
# CONFIG  —  edit these for your venue, leave them on defaults for laptop test
# ===========================================================================

# Total performance duration in seconds.
DURATION_S = 600.0

# How many LLM agents in the choir.
#   C4NM venue: 8 (one per Meyer speaker).
#   Bigger venues / future shows: bump to 10 or more.
N_LLM_AGENTS = 10  # 8 main voices + NVIDIA Nemotron (reserve) + Nous Hermes (10th)

# ---------------------------------------------------------------------------
# AUDIO INPUT  (microphones for human singers)
# ---------------------------------------------------------------------------

# Audio input device for capturing singers.
#   None        → built-in laptop microphone (single channel) — laptop mode.
#   "DN32"      → MIDAS M32R via USB (substring match). The M32R presents
#                 itself as "DN32-LIVE" (the Klark Teknik USB card).
#   <int>       → numeric index from `--list-devices`.
#
# C4NM venue (May 31 / June 6, 2026): "DN32".
INPUT_DEVICE: Optional[object] = "DN32"

# How many input channels to open on the device.
#   Laptop default mic is 1 channel; M32R presents 32.
#   We open as many as the highest HUMAN_CHANNELS value below.
#
# C4NM venue: 5 (5 singers on M32R XLR IN 1-5).
INPUT_CHANNELS_COUNT = 5

# Per-singer microphone mapping: {voice_id: channel_1based}.
#
# - In LAPTOP mode, channel_1based must be 1 (only one input channel).
# - In VENUE mode, channel_1based is the M32R USB channel that the
#   sound engineer patched the singer's microphone to.
#
# Voice IDs must be human_00, human_01, ... in order.
#
# C4NM venue: 5 dynamic mics (4 SM58 + 1 SM57) on M32R XLR IN 1-5.
HUMAN_CHANNELS: dict = {
    "human_00": 1,
    "human_01": 2,
    "human_02": 3,
    "human_03": 4,
    "human_04": 5,
}

WHISPER_MODEL = "base"   # tiny | base | small | medium | large-v3

# ---------------------------------------------------------------------------
# AUDIO OUTPUT  (LLM voices to physical speakers)
# ---------------------------------------------------------------------------

# Audio output device for LLM voices.
#   None        → system default (laptop speakers/headphones, stereo mix).
#   "DN32"      → MIDAS M32R; per-voice channels below are routed by the
#                 M32R routing matrix to physical Meyer speakers. The M32R
#                 presents itself as "DN32-LIVE" (the Klark Teknik USB card).
#
# C4NM venue: "DN32".
OUTPUT_DEVICE: Optional[object] = "DN32"

# Per-LLM channel mapping for VENUE mode: {voice_id: channel_1based}.
# Each LLM is rendered to a mono buffer (sine + TTS mixed) and sent to
# exactly that output channel of OUTPUT_DEVICE. M32R then maps that
# USB channel to a console fader, which has its Direct Out routed to a
# physical XLR OUT (one of 1-8) — wired to one of 8 Meyer speakers.
#
# Leave this dict EMPTY ({}) in laptop mode — voices then play through
# afplay on the default system output, as before.
#
# C4NM venue (Brent confirmed May 30, 2026):
#   "The eight speakers are separate channels. The mains are channels 7 and 8."
# So XLR OUT 1-6 = 6 room speakers, XLR OUT 7 = Main L, XLR OUT 8 = Main R.
# We map USB Card In 1-8 to console channels 9-16, with Direct Outs to
# XLR OUT 1-8. After test_channels.py we'll know which output goes to
# which physical speaker; rearrange this dict if you want a specific
# voice on a specific spot in the room.
VOICE_CHANNELS: dict = {
    "voice_00": 1,    # → above the stage, audience-LEFT (overhead, mains L)
    "voice_01": 2,    # → above the stage, audience-RIGHT (overhead, mains R)
    "voice_02": 3,    # → front audience rows, LEFT
    "voice_03": 4,    # → front audience rows, RIGHT
    "voice_04": 5,    # → middle audience rows, LEFT
    "voice_05": 6,    # → middle audience rows, RIGHT
    "voice_06": 7,    # → back of the room, LEFT
    "voice_07": 8,    # → back of the room, RIGHT
    # voice_08 / voice_09: reserve speakers — uncomment at venue when extra
    # output channels are available on the M32R.
    # "voice_08": 9,   # ← NVIDIA Nemotron (9th speaker)
    # "voice_09": 10,  # ← Nous Hermes (10th speaker)
}

# ---------------------------------------------------------------------------
# KEN / DANTE SESSION PRESET  (used only with the --dante flag)
# ---------------------------------------------------------------------------
#
# For the live session on Ken's system: my Mac runs the piece and sends all
# ten LLM voices into the Dante Virtual Soundcard over Ethernet; Ken receives
# them as 10 Dante inputs in Reaper and spatialises live. I listen to one
# human singer on a USB microphone (FIFINE / lavalier).
#
# Substring match against the device name macOS reports. DVS shows up as
# "Dante Virtual Soundcard"; "Dante" is enough. If --list-devices shows a
# different mic name, change DANTE_MIC_DEVICE (or set the mic as the macOS
# default input and use None).
DANTE_OUTPUT_DEVICE = "Dante"
# The FIFINE K669B reports itself to macOS as "USB PnP Audio Device".
DANTE_MIC_DEVICE: Optional[object] = "USB PnP"

# All 10 voices, one per Dante output channel (1-based).
DANTE_VOICE_CHANNELS: dict = {
    "voice_00": 1, "voice_01": 2, "voice_02": 3, "voice_03": 4, "voice_04": 5,
    "voice_05": 6, "voice_06": 7, "voice_07": 8, "voice_08": 9, "voice_09": 10,
}

# ---------------------------------------------------------------------------
# SPATIAL LAYOUT  (where each voice physically lives in the room)
# ---------------------------------------------------------------------------
#
# The LLMs do not "hear" the room — they have no ears. They only know what
# we tell them in the prompt. So we tell them explicitly: "you, voice_03,
# come out of the front-right speaker. voice_07 is across the room, behind
# the audience, to the right." This lets them answer Q12 ("what does
# distance mean?") with a concrete spatial picture, and choose tuning
# partners with awareness of geometry, not just by token similarity.
#
# Edit per venue: each value is a short human-language label that will be
# stitched into the prompt and shown to the model AS-IS. Keep them
# compact (one line) so they don't drown the score.
#
# C4NM (June 6, 2026): 8 Meyer speakers in left/right pairs at 4 depths.

VOICE_LOCATIONS: dict = {
    "voice_00": "above the stage, audience-LEFT (front overhead)",
    "voice_01": "above the stage, audience-RIGHT (front overhead)",
    "voice_02": "front audience rows, LEFT",
    "voice_03": "front audience rows, RIGHT",
    "voice_04": "middle audience rows, LEFT",
    "voice_05": "middle audience rows, RIGHT",
    "voice_06": "back of the room, LEFT",
    "voice_07": "back of the room, RIGHT",
    "voice_08": "free-floating, placed live in the mix (reserve voice)",
    "voice_09": "free-floating, placed live in the mix (reserve voice)",
}

# Where the human singers stand. Microphones live on the stage; in this
# room the stage is in front, facing the audience, so all five humans are
# "on stage" from the LLMs' point of view, but we still label LEFT /
# CENTER / RIGHT so models can distinguish them.
HUMAN_LOCATIONS: dict = {
    "human_00": "on stage, far LEFT",
    "human_01": "on stage, LEFT of center",
    "human_02": "on stage, CENTER",
    "human_03": "on stage, RIGHT of center",
    "human_04": "on stage, far RIGHT",
}

# A short prose description of the whole layout, given to every model
# once during Phase 1 so they have the big picture before answering Q12.
ROOM_LAYOUT_DESCRIPTION = (
    "The performance space is rectangular. The stage is at the front; "
    "the audience either sits facing the stage in rows or walks around. "
    "Eight loudspeakers are distributed around the audience in LEFT/RIGHT "
    "pairs at four depths — over the stage (mains), front rows, middle "
    "rows, and back of the room — so the LLM choir surrounds the audience "
    "spatially. Five human singers stand on the stage in front of the "
    "speakers, each on their own microphone, audible across the whole room."
)

# Output directory for run logs.
OUT_DIR = HERE / "out"


# ===========================================================================
# Implementation — usually no need to edit below this line
# ===========================================================================


def _print_devices() -> None:
    try:
        import sounddevice as sd
    except Exception as exc:
        print(f"sounddevice not installed: {exc}")
        return
    print("Audio devices visible to this Mac:\n")
    print(sd.query_devices())
    print()
    print("Use the name (or part of it) as INPUT_DEVICE / OUTPUT_DEVICE,")
    print("or the numeric index in the leftmost column.")


def _build_humans(field_placeholder, whisper_shared) -> tuple:
    """Build all HumanVoice instances + an optional MicMultiplexer.

    Returns (humans, multiplexer). multiplexer is None for the laptop
    case where each HumanVoice opens its own single-channel stream.
    """
    if not HUMAN_CHANNELS:
        return [], None
    if not human_mod.HUMAN_DEPS_OK:
        raise RuntimeError(
            f"Cannot create human voices: {human_mod.HUMAN_DEPS_ERROR}\n"
            "Install: pip install sounddevice openai-whisper numpy"
        )

    humans: list = []

    if INPUT_DEVICE is None and INPUT_CHANNELS_COUNT <= 1:
        # Laptop mode: built-in mic, one channel. No multiplexer.
        # Only the first configured human is wired up; the others would
        # share the same mono mic and just duplicate events.
        first_voice_id = next(iter(HUMAN_CHANNELS))
        humans.append(human_mod.HumanVoice(
            voice_id=first_voice_id,
            field=field_placeholder,
            whisper_model=whisper_shared,
            verbose=True,
        ))
        if len(HUMAN_CHANNELS) > 1:
            print(
                f"[concert] INPUT_DEVICE is None: laptop has one mic, "
                f"running only {first_voice_id} (other {len(HUMAN_CHANNELS)-1} "
                f"human slots skipped)."
            )
        return humans, None

    # Venue mode: one shared MicMultiplexer, one HumanVoice per channel.
    mux = human_mod.MicMultiplexer(
        device=INPUT_DEVICE,
        channels=INPUT_CHANNELS_COUNT,
        verbose=True,
    )
    for voice_id, ch_1based in HUMAN_CHANNELS.items():
        q = mux.subscribe(ch_1based)
        humans.append(human_mod.HumanVoice(
            voice_id=voice_id,
            field=field_placeholder,
            whisper_model=whisper_shared,
            audio_queue=q,
            verbose=True,
        ))
    return humans, mux


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tuning Meditation — concert orchestrator")
    p.add_argument("--list-devices", action="store_true",
                   help="Print all audio devices visible to macOS and exit.")
    p.add_argument("--duration", type=float, default=None,
                   help=f"Override DURATION_S from the config (default {DURATION_S}).")
    p.add_argument("--agents", type=int, default=None,
                   help=f"Override N_LLM_AGENTS from the config (default {N_LLM_AGENTS}).")
    p.add_argument("--preload-interpretations", type=Path, default=None,
                   help="Load Phase 1 from a saved interpretations.json (no API calls). "
                        "Use for safety / shorter shows.")
    p.add_argument("--no-sound", action="store_true",
                   help="Skip the in-process sound follower (text-only).")
    p.add_argument("--sim-humans", type=int, default=1, metavar="N",
                   help="Simulate N human voices from one mic. With N=5, human_00 is the "
                        "real mic; human_01-04 are echo copies with increasing delays "
                        "(2s, 5s, 9s, 15s). Useful for testing 5-singer setup solo.")
    p.add_argument("--laptop", action="store_true",
                   help="Force laptop dev mode: ignore M32R, no microphones, "
                        "play LLM TTS through the system default output (stereo "
                        "mix). VOICE_LOCATIONS and ROOM_LAYOUT_DESCRIPTION still "
                        "feed the prompts — the models 'imagine' the venue. Use "
                        "this for home tests with the new spatial questions.")
    p.add_argument("--dante", action="store_true",
                   help="Ken / Dante session mode: send all 10 LLM voices to the "
                        "Dante Virtual Soundcard (channels 1-10) so Ken can "
                        f"spatialise them live in Reaper, and listen to ONE human "
                        f"on a USB mic (default {DANTE_MIC_DEVICE!r}; edit "
                        "DANTE_MIC_DEVICE if --list-devices shows a different name).")
    p.add_argument("--blackhole", action="store_true",
                   help="Local dry-run of the 10-channel Dante path using the "
                        "BlackHole 16ch virtual driver instead of real Dante "
                        "hardware. 10 voices → BlackHole channels 1-10, one human "
                        "on the built-in mic. Use this to verify the --dante "
                        "routing works before the Ethernet adapter / DVS are ready.")
    return p.parse_args()


def _start_sound_follower(run_dir: Path) -> threading.Thread:
    """Start the audio rendering loop inside this process as a daemon thread.

    The thread tails ``run_dir/events.jsonl`` and dispatches each tone event
    to its AgentPlayer. AgentPlayer uses the OUTPUT_DEVICE / OUTPUT_CHANNELS
    we set via ``sound_mod.configure_output`` earlier, so the M32R / BlackHole
    per-voice routing is honoured.
    """
    events_path = run_dir / "events.jsonl"
    events_path.touch(exist_ok=True)
    players: dict = {}

    def _runner() -> None:
        try:
            sound_mod.follow_events(events_path, players, replay=False)
        except Exception as exc:
            print(f"[sound-follower] crashed: {exc}")

    t = threading.Thread(target=_runner, name="sound-follower", daemon=True)
    t.start()
    return t


def _make_echo_hook(n_sim: int):
    """Return a post_phase2_hook that echoes human_00 events as human_01..N-1.

    Each echo voice replays every transcription from human_00 into the shared
    field after a fixed delay.  This lets a solo performer test how the choir
    behaves with multiple human singers present, without needing real extra mics.

    Delays (seconds): 2, 5, 9, 15  — staggered to sound like independent singers.
    """
    if n_sim <= 1:
        return None

    echo_ids = [f"human_{i:02d}" for i in range(1, n_sim)]
    delays = [2.0, 5.0, 9.0, 15.0][: len(echo_ids)]

    def _hook(field_obj) -> None:
        seen: set = set()

        def _watcher() -> None:
            while True:
                time.sleep(0.25)
                for e in list(field_obj._events):
                    if e.voice_id != "human_00" or id(e) in seen:
                        continue
                    seen.add(id(e))
                    content = e.content
                    for echo_id, delay in zip(echo_ids, delays):
                        def _send(eid=echo_id, c=content, d=delay) -> None:
                            time.sleep(d)
                            from .orchestrator import Event
                            field_obj.append(Event(
                                timestamp=time.time(),
                                voice_id=eid,
                                model_short="human",
                                kind="tone",
                                action="HUMAN",
                                content=c,
                                tuning_to=None,
                                breath_chosen_s=0.0,
                                api_latency_ms=0,
                            ))
                            print(f"  [echo] {eid} ← \"{c[:40]}\"") 
                        threading.Thread(target=_send, daemon=True).start()

        t = threading.Thread(target=_watcher, name="human-echo-sim", daemon=True)
        t.start()
        echo_list = ", ".join(echo_ids)
        print(f"  [echo-sim] human_00 → echoes to {echo_list} "
              f"with delays {delays}")

    return _hook


def _load_preload(path: Path) -> dict:
    """Read interpretations.json (list[dict] with voice_id) → {voice_id: entry}."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} is not a JSON list of agents")
    by_voice: dict = {}
    for entry in raw:
        vid = entry.get("voice_id") if isinstance(entry, dict) else None
        if vid:
            by_voice[vid] = entry
    return by_voice


async def _main_async(
    duration_s: float,
    n_agents: int,
    preload_path: Optional[Path],
    sound_enabled: bool,
    laptop_mode: bool = False,
    dante_mode: bool = False,
    blackhole_mode: bool = False,
    n_sim_humans: int = 1,
) -> None:
    # Laptop dev mode: drop M32R routing for input/output, but KEEP one
    # built-in microphone alive so the human in the room can still tune
    # with the choir. _build_humans() handles the "INPUT_DEVICE is None"
    # case by wiring the built-in mic to the first HUMAN_CHANNELS entry.
    global INPUT_DEVICE, OUTPUT_DEVICE, INPUT_CHANNELS_COUNT
    if laptop_mode:
        INPUT_DEVICE = None
        OUTPUT_DEVICE = None
        INPUT_CHANNELS_COUNT = 1
        VOICE_CHANNELS.clear()
        # Keep only the first human on the real mic (built-in mic is mono).
        # With --sim-humans N, extra echo voices are injected in software —
        # they don't need a real channel, but we keep their entries in
        # HUMAN_CHANNELS so voice_locations feeds the correct stage positions
        # to the LLM prompts.
        if HUMAN_CHANNELS:
            all_ids = list(HUMAN_CHANNELS.keys())
            remaining: dict = {all_ids[0]: 1}  # only human_00 has a real mic
            # Add echo voice entries (no real channel — channel 1 is a placeholder)
            for i in range(1, n_sim_humans):
                if i < len(all_ids):
                    remaining[all_ids[i]] = 1  # same channel, no real stream opened
            HUMAN_CHANNELS.clear()
            HUMAN_CHANNELS.update(remaining)
        sim_note = (f", echo-simulating {n_sim_humans - 1} extra singers"
                    if n_sim_humans > 1 else "")
        print(f"[concert] --laptop: M32R ignored, built-in mic kept (one human){sim_note}, "
              "TTS → system output. Spatial map preserved.")

    # Ken / Dante session: 10 voices → Dante Virtual Soundcard (ch 1-10),
    # one human on a USB mic. Ken spatialises the 10 Dante inputs in Reaper.
    if dante_mode:
        OUTPUT_DEVICE = DANTE_OUTPUT_DEVICE
        INPUT_DEVICE = DANTE_MIC_DEVICE
        INPUT_CHANNELS_COUNT = 1
        VOICE_CHANNELS.clear()
        VOICE_CHANNELS.update(DANTE_VOICE_CHANNELS)
        HUMAN_CHANNELS.clear()
        HUMAN_CHANNELS.update({"human_00": 1})
        print(f"[concert] --dante: 10 voices → {OUTPUT_DEVICE!r} (ch 1-10) for live "
              f"spatialisation; one human on {INPUT_DEVICE!r}.")

    # Local dry-run: same 10-channel path, but routed to the BlackHole 16ch
    # virtual driver and listening on the built-in mic. No hardware needed.
    if blackhole_mode:
        OUTPUT_DEVICE = "BlackHole"
        INPUT_DEVICE = None
        INPUT_CHANNELS_COUNT = 1
        VOICE_CHANNELS.clear()
        VOICE_CHANNELS.update(DANTE_VOICE_CHANNELS)
        HUMAN_CHANNELS.clear()
        HUMAN_CHANNELS.update({"human_00": 1})
        print(f"[concert] --blackhole: 10 voices → {OUTPUT_DEVICE!r} (ch 1-10) "
              "dry-run; one human on the built-in mic.")

    # Output routing: tell sound.py to spatialise LLM voices.
    sound_mod.configure_output(OUTPUT_DEVICE, VOICE_CHANNELS)

    # Build humans + (maybe) multiplexer up front so we can fail fast
    # if the device is missing. We pass a None field — orchestrator
    # repoints it once the real Field is constructed.
    print("\n--- Configuration ---")
    print(f"  Duration:       {duration_s:.0f} s")
    print(f"  LLM agents:     {n_agents}")
    print(f"  Input device:   {INPUT_DEVICE!r}  ({INPUT_CHANNELS_COUNT} ch)")
    print(f"  Humans:         {dict(HUMAN_CHANNELS)}")
    print(f"  Output device:  {OUTPUT_DEVICE!r}")
    print(f"  Voice channels: {dict(VOICE_CHANNELS)}")
    print(f"  Whisper model:  {WHISPER_MODEL}")
    print(f"  Preload:        {preload_path!r}")
    print(f"  Sound engine:   {'on (in-process)' if sound_enabled else 'OFF'}")
    print()

    whisper_shared = None
    if HUMAN_CHANNELS:
        if not human_mod.HUMAN_DEPS_OK:
            raise RuntimeError(
                f"Whisper / sounddevice missing: {human_mod.HUMAN_DEPS_ERROR}"
            )
        import whisper as _whisper
        print(f"[concert] loading shared Whisper '{WHISPER_MODEL}'...")
        whisper_shared = _whisper.load_model(WHISPER_MODEL)
        print(f"[concert] Whisper ready (shared across {len(HUMAN_CHANNELS)} singers).")

    humans, mux = _build_humans(field_placeholder=None, whisper_shared=whisper_shared)

    pool = pick_pool(n_agents)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-compute the run directory so we can start the sound follower thread
    # against its events.jsonl before Phase 2 starts producing events.
    run_id = reports_mod.run_id_now()
    run_dir = OUT_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    preload_data: Optional[dict] = None
    if preload_path is not None:
        preload_data = _load_preload(preload_path)
        print(f"[concert] preload: {len(preload_data)} voices from {preload_path}")

    if sound_enabled:
        _start_sound_follower(run_dir)
        print("[concert] sound follower running in-process; "
              "audio routes through configure_output() above.")

    # Reserve voices that can fill a silent main voice's speaker slot.
    # After Phase 1, any main voice (voice_00–voice_07) that declares
    # wants_to_be_heard=no frees its physical channel; reserve voices
    # (voice_08, voice_09) are assigned those channels in order so every
    # speaker in the room always has an active voice.
    _reserve_ids = ["voice_08", "voice_09"]

    def _post_phase1_channel_reassign(agents: list) -> None:
        if OUTPUT_DEVICE is None:
            # Laptop mode: all voices already share the default output.
            return
        freed: list[int] = []
        for a in agents:
            ch = VOICE_CHANNELS.get(a.voice_id)
            if ch is not None and a.alive and a._refused_sound():
                freed.append(ch)
                print(f"  [channel] {a.voice_id} ({a.spec.short_name}) "
                      f"→ silent (wants_to_be_heard=no), freeing channel {ch}")
        if not freed:
            print("  [channel] all main voices want sound — no channel reassignment needed.")
            return
        for i, reserve_id in enumerate(_reserve_ids):
            reserve_agent = next((a for a in agents if a.voice_id == reserve_id), None)
            if reserve_agent is None or not reserve_agent.alive:
                continue
            if i < len(freed):
                VOICE_CHANNELS[reserve_id] = freed[i]
                print(f"  [channel] {reserve_id} ({reserve_agent.spec.short_name}) "
                      f"→ assigned channel {freed[i]}")
        sound_mod.configure_output(OUTPUT_DEVICE, VOICE_CHANNELS)
        print(f"  [channel] final routing: {dict(VOICE_CHANNELS)}")

    await run_session(
        duration_s=duration_s,
        pool=pool,
        out_dir=OUT_DIR,
        humans_prebuilt=humans,
        mic_multiplexer=mux,
        whisper_model=WHISPER_MODEL,
        run_dir=run_dir,
        preload_interpretations=preload_data,
        voice_locations={**VOICE_LOCATIONS, **HUMAN_LOCATIONS},
        room_layout=ROOM_LAYOUT_DESCRIPTION,
        post_phase1_hook=_post_phase1_channel_reassign,
        post_phase2_hook=_make_echo_hook(n_sim_humans),
    )


def main() -> None:
    args = parse_args()
    if args.list_devices:
        _print_devices()
        return
    duration_s = args.duration if args.duration is not None else DURATION_S
    n_agents = args.agents if args.agents is not None else N_LLM_AGENTS
    asyncio.run(_main_async(
        duration_s=duration_s,
        n_agents=n_agents,
        preload_path=args.preload_interpretations,
        sound_enabled=not args.no_sound,
        laptop_mode=args.laptop,
        dante_mode=args.dante,
        blackhole_mode=args.blackhole,
        n_sim_humans=args.sim_humans,
    ))


if __name__ == "__main__":
    main()
