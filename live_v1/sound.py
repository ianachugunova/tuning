#!/usr/bin/env python3
"""
live_v1/sound.py — Real-time sound engine for Tuning Meditation.

For each tone event in events.jsonl:
  1. Take the agent's text content
  2. Sine layer (if any pitch in text): exactly ``breath_s`` seconds
  3. TTS (`say`) at a **natural** speaking rate (default WPM), then **loop** that
     clip back-to-back until it fills ``breath_s`` — no time-stretch, so pitch
     and timbre stay normal (short "a a a" becomes several identical segments).
  4. Play through the agent's assigned macOS voice

Usage (live — follow events as they arrive):
    python3 -m live_v1.sound

    Or start the orchestrator with audio in one step (macOS):
    python3 -m live_v1.main --duration 120 --agents 10 --sound

Usage (replay from a past run):
    python3 -m live_v1.sound --replay
    python3 -m live_v1.sound --run run_2026-05-12_19-38-24 --replay
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

# Reading a tone out of a line of text is not a macOS concern, and the portal
# renders the same voices on Linux from the same rules.
from live_v1.pitch_text import (
    AUDIBLE_HI,
    AUDIBLE_LO,
    extract_all_pitches,
    extract_frequencies,
    extract_notes,
    fold_to_audible,
    is_pitch_only as _is_pitch_only,
    note_to_freq,
    strip_leading_pitch as _strip_leading_pitch,
)

# ---------------------------------------------------------------------------
# Voice assignment — distinct macOS `say` voices.  Ten entries for the
# default 10-agent choir; `voice_for()` uses index modulo len(VOICES).
VOICES = [
    "Samantha", "Daniel", "Karen", "Fred", "Moira",
    "Zarvox", "Cellos", "Albert", "Organ", "Whisper",
]

# ---------------------------------------------------------------------------
# Output routing — for concert with MIDAS M32R or similar multichannel device
# ---------------------------------------------------------------------------
#
# When `OUTPUT_DEVICE` is None, each agent uses macOS `afplay` to the default
# system output (laptop speakers / headphones). This is the laptop test mode.
#
# When `OUTPUT_DEVICE` is set (e.g. "M32R") AND `OUTPUT_CHANNELS` maps a
# voice_id to a 1-based channel index, that agent's audio (sine + TTS mixed
# into one mono buffer) is sent to ONLY that channel of the device. The
# device's matrix routing then sends each channel to a physical speaker.
#
# These are set by concert.py before the player loop starts.

OUTPUT_DEVICE: Optional[object] = None
OUTPUT_CHANNELS: dict = {}
# Number of channels every per-voice OutputStream opens. ALL concurrent
# streams on one CoreAudio device must share the SAME channel count, or the
# device throws "Invalid Property Value" (-10851) / PortAudio -9986 when a
# second stream tries to open it with a different width. (The Midas/DN32
# driver tolerated mismatched widths; Dante Virtual Soundcard does not.) We
# therefore open every stream at the WIDEST configured channel and just
# write silence on the channels a given voice doesn't use.
OUTPUT_DEVICE_CHANNELS: int = 0

# Master synthesis / output sample rate (Hz). Set to 48000 to match the Dante
# network (Ken's AV system runs at 48 kHz), so audio is native end-to-end with
# no CoreAudio resampling. Change back to 44100 only if the whole Dante network
# is switched to 44.1 kHz.
SR_TARGET: int = 48000


def configure_output(device: Optional[object], channels: dict) -> None:
    """Set the multi-channel output device and per-voice channel mapping.

    device: passed to sounddevice (substring of device name or int index).
    channels: {voice_id: 1-based channel index on the device}.
    Voices not in `channels` fall back to the default audio device.
    """
    global OUTPUT_DEVICE, OUTPUT_CHANNELS, OUTPUT_DEVICE_CHANNELS
    OUTPUT_DEVICE = device
    OUTPUT_CHANNELS = dict(channels) if channels else {}
    OUTPUT_DEVICE_CHANNELS = max(OUTPUT_CHANNELS.values()) if OUTPUT_CHANNELS else 0

WPM_DEFAULT = 175
# `say` rate for tiled TTS — natural rhythm; breath length is matched by looping.
TTS_SAY_WPM = WPM_DEFAULT

# How a tone event is rendered for the listener (from the model each step).
_PLAYBACK_ALIASES: dict[str, frozenset[str]] = {
    "both": frozenset({"both", "all", "full", "default", ""}),
    "tts": frozenset({"tts", "speak", "voice", "speech", "say", "read", "words"}),
    "sine": frozenset({
        "sine", "tone", "hz", "pitch", "synth", "synthesize", "instrument",
        "buzz",
    }),
    "none": frozenset({"none", "silent", "silence", "mute", "off", "skip", "no"}),
}


def playback_coerce(raw: Optional[str]) -> str:
    """Normalise model output to one of: both | tts | sine | none."""
    if raw is None:
        return "both"
    s = str(raw).lower().strip()
    for mode, aliases in _PLAYBACK_ALIASES.items():
        if s in aliases:
            return mode
    return "both"


RUNS_PARENT = Path(__file__).parent / "out"

# ---------------------------------------------------------------------------
# Synthesis. Reading pitch out of text lives in live_v1/pitch_text.py.
# ---------------------------------------------------------------------------


def synthesize_tone(freq: float, duration_s: float, output_path: str,
                    sample_rate: int = SR_TARGET, amplitude: float = 0.6) -> None:
    """Write a sine wave at `freq` Hz for `duration_s` seconds to a WAV file."""
    n_samples = int(sample_rate * duration_s)
    with wave.open(output_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)
        # Fade in/out over 10 ms to avoid clicks
        fade = int(sample_rate * 0.01)
        samples = []
        for i in range(n_samples):
            t = i / sample_rate
            s = amplitude * math.sin(2 * math.pi * freq * t)
            # apply fade envelope
            if i < fade:
                s *= i / fade
            elif i > n_samples - fade:
                s *= (n_samples - i) / fade
            samples.append(int(s * 32767))
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))


def _play_breath_laptop(text: str, voice: str, breath_s: float, playback: str) -> None:
    """`afplay` path: honour ``playback`` (both / tts / sine / none)."""
    pb = playback_coerce(playback)
    if pb == "none":
        time.sleep(breath_s)
        return
    # If the entire content is a pitch label (D4, 440Hz…), never read it aloud —
    # play only the synthesized tone regardless of the requested mode.
    if pb in ("both", "tts") and _is_pitch_only(text):
        pb = "sine"
    pitches = extract_all_pitches(text)
    # A leading pitch ("D4, mmm…") becomes a tone but is never spoken: TTS reads
    # only the prose after it, so a short breath isn't wasted saying the number.
    lead_label, tts_rest = _strip_leading_pitch(text)
    tts_text = tts_rest if lead_label else text
    want_sine = pb in ("both", "sine") and bool(pitches)
    want_tts = pb in ("both", "tts") and bool(tts_text.strip())
    # A leading pitch always sounds as a tone, even in tts-only mode.
    if lead_label and pb in ("both", "tts") and pitches:
        want_sine = True
    # sine requested but no pitch found → fall back to TTS so we're not silent
    if pb == "sine" and not want_sine and tts_text.strip():
        want_tts = True
    if not want_sine and not want_tts:
        time.sleep(breath_s)
        return

    tmp_sine: Optional[str] = None
    tmp_tts: Optional[str] = None
    errors: list[str] = []

    if want_sine:
        freq = pitches[0][0]
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_sine = f.name
            synthesize_tone(freq, breath_s, tmp_sine)
        except Exception as exc:
            errors.append(f"synth: {exc}")
            tmp_sine = None

    if want_tts:
        tmp_tts = _render_tts_to_breath_wav(tts_text, voice, breath_s)
        if not tmp_tts:
            try:
                with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
                    fb = f.name
                subprocess.run(
                    [
                        "say",
                        "-v",
                        voice,
                        "-r",
                        str(TTS_SAY_WPM),
                        "-o",
                        fb,
                        "--",
                        tts_text,
                    ],
                    check=True,
                )
                tmp_tts = fb
            except Exception as exc:
                errors.append(f"tts: {exc}")
                tmp_tts = None

    procs: list = []
    if tmp_sine:
        procs.append(subprocess.Popen(["afplay", tmp_sine]))
    if tmp_tts:
        procs.append(subprocess.Popen(["afplay", tmp_tts]))
    for p in procs:
        p.wait()

    for err in errors:
        print(f"  [sound error] {err}")

    for tmp in (tmp_sine, tmp_tts):
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def speak_hybrid(text: str, voice: str, breath_s: float) -> None:
    """Backward-compatible: full hybrid (sine+TTS when applicable)."""
    _play_breath_laptop(text, voice, breath_s, "both")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def voice_for(voice_id: str) -> str:
    try:
        idx = int(voice_id.split("_")[-1]) % len(VOICES)
    except (ValueError, IndexError):
        idx = 0
    return VOICES[idx]


def _tile_audio_to_breath(audio, sr: int, breath_s: float):
    """Repeat ``audio`` end-to-end until it lasts ``breath_s`` (no tempo change).

    If the utterance is longer than ``breath_s``, it is truncated.
    Returns float32 mono of length ``int(sr * breath_s)``, or ``None`` if
    numpy is unavailable.
    """
    try:
        import numpy as np
    except Exception:
        return None
    breath_s = float(max(0.05, breath_s))
    n_target = max(1, int(sr * breath_s))
    a = np.asarray(audio, dtype=np.float32).ravel()
    if a.size == 0:
        return np.zeros(n_target, dtype=np.float32)
    if len(a) >= n_target:
        return a[:n_target].astype(np.float32)
    n_rep = int(np.ceil(n_target / len(a)))
    tiled = np.tile(a, n_rep)[:n_target]
    return tiled.astype(np.float32)


def _write_wav_mono_i16(path: str, audio, sr: int) -> None:
    import numpy as np

    sig = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (sig * 32767.0).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())


def _play_and_unlink(path: str) -> None:
    try:
        subprocess.run(["afplay", path], check=True)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _render_tts_to_breath_wav(text: str, voice: str, breath_s: float, sr: int = SR_TARGET) -> Optional[str]:
    """Run `say` at natural WPM, loop clip to ``breath_s``, write temp WAV."""
    try:
        import numpy as np
    except Exception:
        return None
    if not text.strip():
        return None
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
        tmp_ai = f.name
    try:
        subprocess.run(
            [
                "say",
                "-v",
                voice,
                "-r",
                str(TTS_SAY_WPM),
                "-o",
                tmp_ai,
                "--",
                text,
            ],
            check=True,
        )
        v, sr_v = _read_audio_file(tmp_ai)
        # Guard against `say` producing an empty WAV (happens on very short
        # text fragments and on punctuation-only inputs). Without this the
        # resample math below indexes into an empty array → IndexError on
        # the thread, and that breath of TTS is lost. With the early-out,
        # the agent is silent for this one breath and we move on.
        if v is None or sr_v <= 0 or len(v) == 0:
            return None
        if sr_v != sr:
            ratio = sr / sr_v
            new_n = max(1, int(len(v) * ratio))
            idx = (np.arange(new_n, dtype=np.float64) / ratio).astype(np.int32)
            idx = np.clip(idx, 0, len(v) - 1)
            v = v[idx].astype(np.float32)
        tiled = _tile_audio_to_breath(v, sr, breath_s)
        if tiled is None:
            return None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        _write_wav_mono_i16(out, tiled, sr)
        return out
    except (subprocess.CalledProcessError, OSError, ValueError):
        return None
    finally:
        try:
            os.unlink(tmp_ai)
        except OSError:
            pass


def speak(text: str, voice: str, breath_s: float) -> None:
    """TTS for ``breath_s``: natural `say` rate, then loop to fill."""
    if not text.strip():
        return
    path = _render_tts_to_breath_wav(text, voice, breath_s)
    if path:
        _play_and_unlink(path)
        return
    # Fallback: single `say` play (duration ≠ breath_s)
    try:
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
            tmp = f.name
        subprocess.run(
            [
                "say",
                "-v",
                voice,
                "-r",
                str(TTS_SAY_WPM),
                "-o",
                tmp,
                "--",
                text,
            ],
            check=True,
        )
        subprocess.run(["afplay", tmp], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"  [sound error] {exc}")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Per-agent player (dedicated thread + depth-1 queue)
# ---------------------------------------------------------------------------


def _read_audio_file(path: str):
    """Read WAV/AIFF into a (float32 mono, samplerate) pair.

    Uses soundfile (already a transitive dep of openai-whisper). If
    soundfile is unavailable for some reason, returns (None, 0) and
    callers must fall back to afplay.
    """
    try:
        import soundfile as sf
    except Exception:
        return None, 0
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if hasattr(audio, "ndim") and audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


def _render_breath(
    text: str, voice: str, breath_s: float, playback: str = "both"
) -> "tuple":
    """Mono buffer of length ``breath_s``: layers selected by ``playback``."""
    try:
        import numpy as np
    except Exception:
        return None, 0

    pb = playback_coerce(playback)
    if pb in ("both", "tts") and _is_pitch_only(text):
        pb = "sine"
    pitches = extract_all_pitches(text)
    # A leading pitch ("D4, mmm…") becomes a tone but is never spoken: TTS reads
    # only the prose after it, so a short breath isn't wasted saying the number.
    lead_label, tts_rest = _strip_leading_pitch(text)
    tts_text = tts_rest if lead_label else text
    sr_target = SR_TARGET
    n_target = max(1, int(sr_target * breath_s))

    if pb == "none":
        return np.zeros(n_target, dtype=np.float32), sr_target

    want_sine = pb in ("both", "sine") and bool(pitches)
    want_tts = pb in ("both", "tts") and bool(tts_text.strip())
    # A leading pitch always sounds as a tone, even in tts-only mode.
    if lead_label and pb in ("both", "tts") and pitches:
        want_sine = True
    if pb == "sine" and not want_sine and tts_text.strip():
        want_tts = True

    if not want_sine and not want_tts:
        return np.zeros(max(1, n_target), dtype=np.float32), sr_target

    sine = np.zeros(n_target, dtype=np.float32)
    if want_sine:
        freq = pitches[0][0]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_sine = f.name
        try:
            synthesize_tone(freq, breath_s, tmp_sine, sample_rate=sr_target)
            s, _ = _read_audio_file(tmp_sine)
            if s is not None:
                m = min(len(s), n_target)
                sine[:m] = s[:m]
        finally:
            try:
                os.unlink(tmp_sine)
            except OSError:
                pass

    voice_layer = np.zeros(n_target, dtype=np.float32)
    if want_tts:
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
            tmp_tts = f.name
        try:
            subprocess.run(
                [
                    "say",
                    "-v",
                    voice,
                    "-r",
                    str(TTS_SAY_WPM),
                    "-o",
                    tmp_tts,
                    "--",
                    tts_text,
                ],
                check=True,
                # `say` occasionally wedges (speech-synthesis server stall).
                # A breath is at most ~30s, and generating its clip is far
                # faster than realtime, so a 30s ceiling never trims normal
                # output but prevents one stuck call from hanging forever.
                timeout=30,
            )
            v, sr_v = _read_audio_file(tmp_tts)
            # Guard against `say` producing an empty WAV (very short
            # text, punctuation-only). Without `len(v) > 0` the resample
            # below indexes an empty array and raises IndexError.
            if v is not None and sr_v > 0 and len(v) > 0:
                if sr_v != sr_target:
                    ratio = sr_target / sr_v
                    new_n = max(1, int(len(v) * ratio))
                    idx = (np.arange(new_n, dtype=np.float64) / ratio).astype(np.int32)
                    idx = np.clip(idx, 0, len(v) - 1)
                    v = v[idx].astype(np.float32)
                tiled = _tile_audio_to_breath(v, sr_target, breath_s)
                if tiled is not None:
                    voice_layer = tiled
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(f"  [sound error] say: {exc}")
        finally:
            try:
                os.unlink(tmp_tts)
            except OSError:
                pass

    if want_sine and want_tts:
        mix = sine * 0.4 + voice_layer * 0.8
    elif want_sine:
        mix = sine
    else:
        mix = voice_layer
    peak = float(abs(mix).max()) if len(mix) else 0.0
    if peak > 0.95:
        mix = mix * (0.95 / peak)
    return mix.astype("float32"), sr_target


# Opening and closing PortAudio streams is NOT thread-safe. `sd.play()` makes
# it worse: it drives a single MODULE-GLOBAL stream, so calling it from all ~10
# voice threads corrupts that shared object and PortAudio's allocator, crashing
# the process ("double free" inside CloseStream). We therefore (1) give every
# breath its OWN OutputStream instead of the global one, and (2) serialise only
# the open/close with this lock. The blocking write stays OUTSIDE the lock, so
# all voices still sound simultaneously — polyphony is preserved.
_PORTAUDIO_LOCK = threading.Lock()


def _play_to_channel(audio, sr: int, channel_1based: int) -> None:
    """Play a mono float32 array on ONE channel of OUTPUT_DEVICE."""
    try:
        import sounddevice as sd
        import numpy as np
    except Exception as exc:
        print(f"  [sound error] sounddevice unavailable: {exc}")
        return

    # Open EVERY stream at the same (widest) channel count so concurrent
    # streams share one device format — Dante rejects mismatched widths.
    # Fall back to this voice's own channel if the width wasn't configured.
    n_ch = max(int(channel_1based), int(OUTPUT_DEVICE_CHANNELS))
    mono = np.asarray(audio, dtype="float32").ravel()
    buf = np.zeros((len(mono), n_ch), dtype="float32")
    buf[:, int(channel_1based) - 1] = mono

    stream = None
    try:
        # Serialise stream creation/start — the unsafe part in PortAudio.
        with _PORTAUDIO_LOCK:
            stream = sd.OutputStream(
                samplerate=sr,
                device=OUTPUT_DEVICE,
                channels=n_ch,
                dtype="float32",
            )
            stream.start()
        # Blocking write happens OUTSIDE the lock so voices overlap.
        stream.write(buf)
    except Exception as exc:
        print(f"  [sound error] play ch{channel_1based}: {type(exc).__name__}: {exc}")
    finally:
        if stream is not None:
            # Serialise stop/close — the other unsafe part in PortAudio.
            with _PORTAUDIO_LOCK:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass




class AgentPlayer:
    """Each agent speaks in its own thread. If a new breath arrives while
    the previous one is still being spoken, it is queued. If two are already
    pending, the older one is dropped — the agent moves on."""

    def __init__(self, voice_id: str) -> None:
        self.voice_id = voice_id
        self.voice = voice_for(voice_id)
        self._queue: Queue = Queue(maxsize=2)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        # CRITICAL: each iteration is wrapped in try/except. Without it,
        # one bad TTS render (e.g. `say` emitting an empty WAV on a short
        # punctuation-only text) raises an uncaught exception that kills
        # this agent's thread for the rest of the run — and the agent
        # goes silent forever even though new events keep arriving.
        # Better behavior: log the failure, skip this breath, keep playing.
        while True:
            try:
                text, breath_s, playback, _note = self._queue.get(timeout=1.0)
            except Empty:
                continue
            try:
                channel = OUTPUT_CHANNELS.get(self.voice_id)
                if OUTPUT_DEVICE is not None and channel is not None:
                    audio, sr = _render_breath(text, self.voice, breath_s, playback)
                    if audio is not None and sr > 0:
                        _play_to_channel(audio, sr, channel)
                        continue
                _play_breath_laptop(text, self.voice, breath_s, playback)
            except Exception as exc:  # noqa: BLE001 — keep the thread alive
                print(f"  [sound error] {self.voice_id}: {type(exc).__name__}: {exc}")
                continue

    def enqueue(
        self,
        text: str,
        breath_s: float,
        playback: str = "both",
        playback_note: str = "",
    ) -> None:
        pb = playback_coerce(playback)
        preview = text[:70].replace("\n", " ")
        pitches = extract_all_pitches(text)
        if pitches:
            note_names = " ".join(src for _, src in pitches[:3])
            mode = f"hybrid [{note_names}]"
        else:
            mode = "tts"
        nb = f' note="{playback_note[:40]}"' if playback_note else ""
        print(
            f"  ♪ {self.voice_id:10s}  {mode:12s}  play={pb:5s}{nb}  "
            f"breath={breath_s:.1f}s  \"{preview}\""
        )
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except Empty:
                pass
        try:
            self._queue.put_nowait((text, breath_s, pb, playback_note))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Event follower
# ---------------------------------------------------------------------------


def follow_events(
    events_path: Path,
    players: dict,
    replay: bool,
    replay_speed: float = 1.0,
) -> None:
    """Read events.jsonl and dispatch tone events to AgentPlayers.

    live mode  (replay=False): seek to end, follow new lines only.
    replay mode (replay=True): read from start with original timing.
    """
    last_ts: Optional[float] = None

    with open(events_path, "r", encoding="utf-8") as f:
        if not replay:
            f.seek(0, 2)  # tail -f: jump to end

        while True:
            line = f.readline()
            if not line:
                time.sleep(0.04)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if event.get("kind") != "tone":
                continue

            # Skip events from human performers — they sang it themselves,
            # we must not echo their own voice back through TTS (and during
            # a live run that would also create an instant feedback loop:
            # mic → whisper → TTS → mic).
            if event.get("action") == "HUMAN" or str(event.get("voice_id", "")).startswith("human_"):
                continue

            if replay and replay_speed > 0:
                ts = event.get("timestamp")
                if ts is not None:
                    if last_ts is not None:
                        gap = (ts - last_ts) / replay_speed
                        if 0 < gap < 30:
                            time.sleep(gap)
                    last_ts = ts

            voice_id = event.get("voice_id", "voice_00")
            content = str(event.get("content", "")).strip()
            breath_s = float(event.get("breath_chosen_s") or 5.0)

            if not content:
                continue

            playback = playback_coerce(event.get("playback"))
            playback_note = str(event.get("playback_note") or "").strip()

            if voice_id not in players:
                players[voice_id] = AgentPlayer(voice_id)

            players[voice_id].enqueue(
                content, breath_s, playback=playback, playback_note=playback_note
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def find_latest_run(runs_dir: Path) -> Path:
    runs = sorted(runs_dir.glob("run_*/"), reverse=True)
    if not runs:
        raise SystemExit(f"[sound] No runs found in {runs_dir}")
    return runs[0]


def resolve_run_dir(run_arg: Optional[str], runs_dir: Path) -> Path:
    if run_arg is None:
        return find_latest_run(runs_dir)
    p = Path(run_arg)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    candidate = runs_dir / p
    if candidate.exists():
        return candidate.resolve()
    raise SystemExit(f"[sound] Run directory not found: {run_arg!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sound engine for Tuning Meditation runs")
    ap.add_argument("--run", default=None,
                    help="Run name or path (default: latest under --out-dir)")
    ap.add_argument("--out-dir", default=str(RUNS_PARENT))
    ap.add_argument("--replay", action="store_true",
                    help="Replay all past events with original timing")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Replay speed multiplier (default 1.0)")
    args = ap.parse_args()

    runs_dir = Path(args.out_dir).resolve()
    run_dir = resolve_run_dir(args.run, runs_dir)
    events_path = run_dir / "events.jsonl"

    if not events_path.exists():
        raise SystemExit(f"[sound] events.jsonl not found: {events_path}")

    print(f"[sound] run      = {run_dir.name}")
    print(f"[sound] replay   = {args.replay}  speed={args.speed}x")
    print(f"[sound] routing  = default output (each voice_N uses VOICES[N % {len(VOICES)}])")
    print()

    players: dict = {}
    try:
        follow_events(
            events_path,
            players,
            replay=args.replay,
            replay_speed=args.speed,
        )
    except KeyboardInterrupt:
        print("\n[sound] stopped.")


if __name__ == "__main__":
    main()
