#!/usr/bin/env python3
"""The room: nine speakers and five microphones, on the Mac mini at Gray Area.

The choir lives on Modal and never stops. This is its body in the hall — the
one process that touches hardware. It does two things and nothing else:

  play    polls the portal for breaths, downloads each one, and sounds it on
          the one output channel that belongs to that voice. Nine voices, nine
          channels, no mixing. A dead voice sends no breaths; its channel is
          silent and its speaker stays where it is.
  listen  opens the five microphone channels, finds held notes in them with
          the same gate and the same YIN the phones use (`pitch-core.js`,
          `live_v1/human.py`), and posts numbers — pitch, length, contour — to
          the portal. No audio is kept or sent. Each microphone is its own
          person in the field.

Setup on the Mac mini (once, at install):

    pip3 install sounddevice numpy
    python3 room.py devices                 # find the interface's name
    cp room.example.json room.json          # then edit device names / channels
    python3 room.py tone --channel 1        # 2 s sine on output 1; walk the ring
    python3 room.py tone --all              # 1..9 in turn, naming each voice
    python3 room.py listen --print          # sing into each mic, watch the gate
    python3 room.py run                     # both, for the week

Autostart: `room.plist.example` → ~/Library/LaunchAgents/com.tuning.room.plist,
`launchctl load` it. The process keeps nothing on disk and reconnects on its
own; a power cycle from the LPC-3 is a complete repair.

Channel numbers are 1-based, as on the back of the interface. The room's
interface is a Behringer UMC1820 (class-compliant, no driver): CoreAudio
numbers its analog outputs 1–10 (1–2 are the MAIN OUT pair under the big
knob, 3–10 are fixed-level line outs) and its analog inputs 1–8 (the eight
combo mic/line jacks, phantom in two banks of four). So the nine speakers
are outputs 1–9 and the five microphones are inputs 1–5 — but check with
`devices` and `listen --print`; the numbering is the one thing this file
cannot know in advance.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import queue
import random
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))        # live_v1.human, for YIN

PORTAL = "https://tuning-grayarea--tuning-portal-web.modal.run"

# --- the page's timing, kept (field.html) ----------------------------------
STALE_S = 15.0        # how far back the room reaches when it starts hearing
LATE_S = 300.0        # how late a breath may arrive and still be sung
GAP_MAX_S = 8.0       # a held-up batch keeps the choir's gaps, capped
POLL_S = 2.0
# How long the room holds a breath before sounding it, counted from the
# choir's own timestamp. A breath is stamped when the voice has decided what
# to sing and reaches the portal 10–19 s later (the speech is synthesized and
# the file travels in between), and that spread is different for every breath.
# Played on arrival, the choir's even 12–13 s rhythm comes out of the speakers
# as 5 s, 28 s, 20 s — holes where the hall hears nothing. Held until
# `ts + HOLD_S`, every breath sounds exactly as far behind the choir as every
# other, and the rhythm the choir keeps is the rhythm the room hears. A
# breath that arrives later than this is played at once. The screens take the
# same number as `?lag=` so a speaker lights while it sounds. 0 = on arrival.
HOLD_S = 30.0

# --- the gate, mirrored from pitch-core.js (TUNING, GALLERY) ----------------
TARGET_RATE = 16000
FMIN_HZ, FMAX_HZ = 70.0, 1100.0
FRAME_S, HOP_S = 0.05, 0.025
FRAME_SILENCE_AMP = 0.01
SILENCE_TAIL_S = 0.5
MAX_UTTERANCE_S = 30.0
BLOCK_S = 0.1
CONTOUR_HOP_S, CONTOUR_MAX_POINTS = 0.25, 128
GALLERY = {
    "GATE_FLOOR": 0.02,
    "GATE_CEIL": 0.10,
    "GATE_OVER_NOISE": 6.0,
    "MIN_UTTERANCE_S": 1.0,
    "MIN_VOICED_FRACTION": 0.6,
    "SESSION_GAP_S": 90.0,
    "MAX_BREATHS_PER_MIN": 20,
}
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


# =============================================================================
# arithmetic, all of it testable without a sound card
# =============================================================================

def hz_to_note(hz: float) -> str:
    if hz <= 0:
        return ""
    semi = round(12 * math.log2(hz / 440.0))
    return f"{NOTE_NAMES[(semi + 9) % 12]}{4 + (semi + 9) // 12}"


def cents_off(hz: float) -> int:
    if hz <= 0:
        return 0
    exact = 12 * math.log2(hz / 440.0)
    return int(round((exact - round(exact)) * 100))


def stability_label(std_hz: float, median_hz: float) -> str:
    if median_hz <= 0:
        return ""
    cents = 1200 * math.log2((median_hz + std_hz) / median_hz) if std_hz > 0 else 0
    if cents < 30:
        return "very steady"
    if cents < 80:
        return "steady"
    if cents < 200:
        return "vibrato"
    return "sliding"


def contour_of(pitches: list[float], duration_s: float) -> list[float]:
    if not pitches or duration_s <= 0:
        return []
    n = max(1, min(CONTOUR_MAX_POINTS, round(duration_s / CONTOUR_HOP_S)))
    out = []
    for i in range(n):
        a = (i * len(pitches)) // n
        b = max(a + 1, ((i + 1) * len(pitches)) // n)
        out.append(round(float(np.median(pitches[a:b])), 1))
    return out


def gallery_admits(duration_s: float, median_hz: float, std_hz: float,
                   voiced: float, peak_rms: float) -> str:
    """"" when a person sang into the room's microphone; otherwise why not."""
    if median_hz <= 0:
        return "no pitch in it"
    if duration_s < GALLERY["MIN_UTTERANCE_S"]:
        return "too short for a sung note"
    if voiced < GALLERY["MIN_VOICED_FRACTION"]:
        return "too broken to be a voice"
    if stability_label(std_hz, median_hz) == "sliding":
        return "wandering — the room, not a voice"
    if peak_rms < GALLERY["GATE_FLOOR"]:
        return "too quiet for an open microphone"
    return ""


def track_pitch(audio: np.ndarray, sr: int) -> tuple[list[float], int]:
    """Per-frame YIN over an utterance: (pitches of voiced frames, frames)."""
    from live_v1.human import _yin_frame

    audio = audio.astype(np.float32)
    frame_n, hop_n = int(sr * FRAME_S), int(sr * HOP_S)
    if len(audio) < frame_n:
        return [], 0
    tau_min, tau_max = int(sr / FMAX_HZ), int(sr / FMIN_HZ)
    n_frames = (len(audio) - frame_n) // hop_n + 1
    pitches = []
    for i in range(n_frames):
        frame = audio[i * hop_n:i * hop_n + frame_n]
        if float(np.max(np.abs(frame))) < FRAME_SILENCE_AMP:
            continue
        f0 = _yin_frame(frame, sr, tau_min, tau_max)
        if f0 > 0:
            pitches.append(float(f0))
    return pitches, n_frames


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Linear; a sine and a spoken voice do not need better."""
    if sr_from == sr_to or len(x) == 0:
        return x.astype(np.float32)
    n_to = int(round(len(x) * sr_to / sr_from))
    src = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    dst = np.linspace(0.0, 1.0, num=n_to, endpoint=False)
    return np.interp(dst, src, x).astype(np.float32)


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Bytes of a wav → (mono float32 in -1..1, sample rate)."""
    with wave.open(io.BytesIO(data)) as wf:
        sr, n_ch, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width {width}")
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    return x, sr


def sine(hz: float, seconds: float, sr: int, contour: list[float] | None = None,
         peak: float = 0.18) -> np.ndarray:
    """A person, sounded from numbers — the page's `soundHuman`, in numpy."""
    n = int(seconds * sr)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    pts = [p for p in (contour or []) if p > 0]
    if len(pts) > 1:
        freq = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(pts)), pts)
    else:
        freq = np.full(n, hz, dtype=np.float32)
    phase = np.cumsum(2 * np.pi * freq / sr)
    x = np.sin(phase).astype(np.float32)
    ramp = max(1, int(0.08 * sr))
    env = np.ones(n, dtype=np.float32)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.minimum(env[-ramp:], np.linspace(1, 0, ramp))
    return x * env * peak


def parse_channels(spec: str) -> dict[str, int]:
    """'voice_00=1,voice_01=2' → {'voice_00': 1, ...}."""
    out = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        vid, ch = part.split("=")
        out[vid.strip()] = int(ch)
    return out


def default_channels() -> dict[str, int]:
    """Cast order onto outputs 1–9. Overridden by room.json at install."""
    try:
        import cast
        return {v.voice_id: i + 1 for i, v in enumerate(cast.CHOIR)}
    except Exception:
        return {f"voice_0{i}": i + 1 for i in range(9)}


def voice_names() -> dict[str, str]:
    try:
        import cast
        return {v.voice_id: v.short_name for v in cast.CHOIR}
    except Exception:
        return {}


# =============================================================================
# the mixer: one output stream, breaths summed into their own channels
# =============================================================================

class Mixer:
    """What is sounding right now, by channel. The audio callback reads it.

    One stream for all nine, because every concurrent CoreAudio stream on a
    device has to agree on its channel count (live_v1/sound.py learned this),
    and because a tuning meditation is sustained tones overlapping: the choir
    left a third of its breaths overlapping in the field and they must overlap
    in the hall.
    """

    def __init__(self, n_channels: int):
        self.n_channels = n_channels
        self._lock = threading.Lock()
        self._voices: list[list] = []     # [buf, channel0, pos, gain]
        self.underruns = 0

    def add(self, buf: np.ndarray, channel: int, gain: float = 1.0) -> None:
        if not (1 <= channel <= self.n_channels):
            raise ValueError(f"channel {channel} not on a {self.n_channels}-channel stream")
        with self._lock:
            self._voices.append([buf.astype(np.float32), channel - 1, 0, float(gain)])

    def sounding(self) -> int:
        with self._lock:
            return len(self._voices)

    def render(self, frames: int) -> np.ndarray:
        out = np.zeros((frames, self.n_channels), dtype=np.float32)
        with self._lock:
            keep = []
            for v in self._voices:
                buf, ch, pos, gain = v
                chunk = buf[pos:pos + frames]
                out[:len(chunk), ch] += chunk * gain
                v[2] = pos + frames
                if v[2] < len(buf):
                    keep.append(v)
            self._voices = keep
        np.clip(out, -1.0, 1.0, out=out)
        return out

    def callback(self, outdata, frames, _time, status) -> None:
        if status:
            self.underruns += 1
        outdata[:] = self.render(frames)


# =============================================================================
# the player
# =============================================================================

class Player(threading.Thread):
    def __init__(self, mixer: Mixer, sr_out: int, channels: dict[str, int],
                 gains: dict[str, float] | None = None, portal: str = PORTAL,
                 humans_channel: int | None = None, verbose: bool = True,
                 hold_s: float = HOLD_S):
        super().__init__(daemon=True, name="player")
        self.mixer, self.sr_out = mixer, sr_out
        self.channels, self.gains = channels, gains or {}
        self.portal, self.humans_channel, self.verbose = portal.rstrip("/"), humans_channel, verbose
        self.hold_s = max(0.0, float(hold_s))
        self.names = voice_names()
        self.since = 0.0
        self.heard_since = time.time() - STALE_S
        self.seen: set[str] = set()
        self.stop_flag = threading.Event()
        self.played = 0

    def _get(self, path: str, timeout: float = 20.0) -> bytes:
        with urllib.request.urlopen(self.portal + path, timeout=timeout) as r:
            return r.read()

    def poll(self) -> list[dict]:
        q = f"/field?since={self.since}" if self.since else "/field?limit=12"
        data = json.loads(self._get(q))
        fresh = data.get("recent") or []
        if fresh:
            self.since = max(self.since, max(float(e.get("ts") or 0) for e in fresh))
        return fresh

    def choose(self, events: list[dict], now: float) -> list[dict]:
        """The page's `playNext`: sung while hearing, not older than LATE_S."""
        out = []
        for e in events:
            ts = float(e.get("ts") or 0)
            if ts <= self.heard_since or now - ts > LATE_S:
                continue
            key = e.get("audio") or f"{e.get('voice_id')}@{ts}"
            if key in self.seen:
                continue
            if e.get("who") == "model" and not e.get("audio"):
                continue
            if e.get("who") == "human" and (self.humans_channel is None
                                            or e.get("name") == "a voice in the room"):
                continue                 # people in the room are heard directly
            self.seen.add(key)
            out.append(e)
        if len(self.seen) > 2000:
            self.seen = set(list(self.seen)[-500:])
        return sorted(out, key=lambda e: float(e["ts"]))[-6:]

    def schedule(self, batch: list[dict]) -> None:
        if not batch:
            return
        if self.hold_s > 0:
            # Each breath at its own moment, `hold_s` behind the choir's clock.
            wall, mono = time.time(), time.monotonic()
            for e in batch:
                at = mono + max(0.0, float(e["ts"]) + self.hold_s - wall)
                threading.Thread(target=self._sound, args=(e, at), daemon=True).start()
            return
        at, prev = time.monotonic(), float(batch[0]["ts"])
        for e in batch:
            at += min(max(float(e["ts"]) - prev, 0.0), GAP_MAX_S)
            prev = float(e["ts"])
            threading.Thread(target=self._sound, args=(e, at), daemon=True).start()

    def _sound(self, e: dict, at: float) -> None:
        try:
            if e.get("who") == "human":
                dur = min(12.0, max(0.25, float(e.get("dur") or 0.6)))
                buf = sine(float(e["hz"]), dur, self.sr_out, e.get("contour"))
                channel, label = self.humans_channel, "a voice"
            else:
                x, sr = decode_wav(self._get("/audio/" + e["audio"]))
                buf = resample(x, sr, self.sr_out)
                vid = str(e.get("voice_id") or "")
                channel = self.channels.get(vid)
                label = self.names.get(vid, vid)
                if channel is None:
                    return
            wait = at - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.mixer.add(buf, channel, self.gains.get(e.get("voice_id", ""), 1.0))
            self.played += 1
            if self.verbose:
                print(f"{time.strftime('%H:%M:%S')}  ch{channel:<2} {label:<15} "
                      f"{len(buf) / self.sr_out:4.1f}s  {str(e.get('text') or '')[:50]}",
                      flush=True)
        except Exception as err:
            print(f"breath lost: {type(err).__name__}: {err}", file=sys.stderr, flush=True)

    def run(self) -> None:
        while not self.stop_flag.is_set():
            try:
                self.schedule(self.choose(self.poll(), time.time()))
            except Exception as err:
                print(f"poll failed: {type(err).__name__}: {err}", file=sys.stderr, flush=True)
                self.stop_flag.wait(5.0)
            self.stop_flag.wait(POLL_S)


# =============================================================================
# the listener
# =============================================================================

class MicGate:
    """One microphone's state: a noise floor, a gate above it, an utterance.

    Fed 100 ms blocks; returns an utterance when one ends. The gate follows
    the room upward — a busy hour raises the bar — and never below the floor,
    so the hall's own noise is not notated as a person (pitch-core GALLERY).
    """

    def __init__(self, sr: int):
        self.sr = sr
        self.floor = GALLERY["GATE_FLOOR"] / GALLERY["GATE_OVER_NOISE"]
        self.blocks: list[np.ndarray] = []
        self.onset = 0.0
        self.quiet_s = 0.0
        self.peak = 0.0

    @property
    def gate(self) -> float:
        return min(GALLERY["GATE_CEIL"],
                   max(GALLERY["GATE_FLOOR"], self.floor * GALLERY["GATE_OVER_NOISE"]))

    @property
    def open(self) -> bool:
        return bool(self.blocks)

    def feed(self, block: np.ndarray, now: float) -> dict | None:
        rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2))) if len(block) else 0.0
        loud = rms > self.gate
        if not loud:
            self.floor = 0.95 * self.floor + 0.05 * rms
        block_s = len(block) / self.sr
        if loud:
            if not self.blocks:
                self.onset, self.peak, self.quiet_s = now - block_s, 0.0, 0.0
            self.blocks.append(block)
            self.peak = max(self.peak, rms)
            self.quiet_s = 0.0
            if now - self.onset >= MAX_UTTERANCE_S:
                return self._close(now)
            return None
        if self.blocks:
            self.blocks.append(block)
            self.quiet_s += block_s
            if self.quiet_s >= SILENCE_TAIL_S:
                return self._close(now)
        return None

    def _close(self, now: float) -> dict:
        audio = np.concatenate(self.blocks)
        tail = self.quiet_s
        u = {"audio": audio[:max(1, len(audio) - int(tail * self.sr))],
             "onset": self.onset, "ended": now - tail, "tail_s": tail,
             "peak_rms": self.peak}
        self.blocks, self.quiet_s = [], 0.0
        return u


def measure(u: dict, sr: int) -> dict:
    """The numbers the page would post for this utterance, and the verdict."""
    x = resample(u["audio"], sr, TARGET_RATE)
    pitches, n_frames = track_pitch(x, TARGET_RATE)
    duration_s = len(x) / TARGET_RATE
    med = float(np.median(pitches)) if pitches else 0.0
    sd = float(np.std(pitches)) if pitches else 0.0
    voiced = len(pitches) / max(1, n_frames)
    return {
        "duration_s": round(duration_s, 2), "hz": round(med, 2), "std_hz": round(sd, 2),
        "voiced_fraction": round(voiced, 3), "peak_rms": round(u["peak_rms"], 4),
        "note": hz_to_note(med), "cents": cents_off(med),
        "stability": stability_label(sd, med),
        "contour_hz": contour_of(pitches, duration_s),
        "why_not": gallery_admits(duration_s, med, sd, voiced, u["peak_rms"]),
    }


class Listener(threading.Thread):
    def __init__(self, device, mics: list[int], sr: int, portal: str = PORTAL,
                 room: str = "gallery", verbose: bool = True, post: bool = True):
        super().__init__(daemon=True, name="listener")
        self.device, self.mics, self.sr = device, mics, sr
        self.portal, self.room, self.verbose, self.post = portal.rstrip("/"), room, verbose, post
        self.gates = {m: MicGate(sr) for m in mics}
        self.turn: dict[int, tuple[str, float]] = {}      # mic → (voice_id, last end)
        self.accepted: dict[int, list[float]] = {m: [] for m in mics}
        self.work: queue.Queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.posted = 0

    def singer_at(self, mic: int, onset: float) -> tuple[str, float | None]:
        """A turn at the microphone, not a person — pitch-core `singerAt`."""
        vid, ended = self.turn.get(mic, ("", 0.0))
        if not vid or (ended and onset - ended > GALLERY["SESSION_GAP_S"]):
            vid = f"human_{self.room}{mic}_{random.randbytes(2).hex()}"
            self.turn[mic] = (vid, 0.0)
            return vid, None
        return vid, round(onset - ended, 3) if ended else None

    def flooding(self, mic: int, now: float) -> bool:
        acc = self.accepted[mic]
        while acc and now - acc[0] > 60:
            acc.pop(0)
        return len(acc) >= GALLERY["MAX_BREATHS_PER_MIN"]

    def payload(self, mic: int, u: dict, m: dict, now: float) -> dict:
        vid, rest = self.singer_at(mic, u["onset"])
        return {
            "voice_id": vid, "hz": m["hz"], "note": m["note"], "cents": m["cents"],
            "stability": m["stability"], "duration_s": m["duration_s"],
            "voiced_fraction": m["voiced_fraction"], "std_hz": m["std_hz"],
            "contour_hz": m["contour_hz"], "source": "gallery",
            "tail_s": round(u["tail_s"] + (now - (u["ended"] + u["tail_s"])), 3),
            "own_rest_s": rest,
        }

    def _post(self, body: dict) -> None:
        req = urllib.request.Request(self.portal + "/breath",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            r.read()

    def _handle(self, mic: int, u: dict) -> None:
        now = time.time()
        m = measure(u, self.sr)
        tag = f"{time.strftime('%H:%M:%S')}  mic{mic} {m['duration_s']:4.1f}s rms {m['peak_rms']:.3f} gate {self.gates[mic].gate:.3f}"
        if m["why_not"]:
            if self.verbose:
                print(f"{tag}  — {m['why_not']}", flush=True)
            return
        if self.flooding(mic, now):
            print(f"{tag}  — the microphone is flooding — held back", flush=True)
            return
        body = self.payload(mic, u, m, now)
        self.turn[mic] = (body["voice_id"], u["ended"])
        self.accepted[mic].append(now)
        if self.verbose:
            print(f"{tag}  {m['note']} ~{m['hz']:.0f}Hz [{m['stability']}]  → {body['voice_id']}",
                  flush=True)
        if self.post:
            try:
                self._post(body)
                self.posted += 1
            except (urllib.error.URLError, OSError) as err:
                print(f"breath not delivered: {err}", file=sys.stderr, flush=True)

    def _worker(self) -> None:
        while not self.stop_flag.is_set():
            try:
                mic, u = self.work.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._handle(mic, u)
            except Exception as err:
                print(f"mic{mic}: {type(err).__name__}: {err}", file=sys.stderr, flush=True)

    def callback(self, indata, frames, _time, status) -> None:
        now = time.time()
        for mic, gate in self.gates.items():
            u = gate.feed(indata[:, mic - 1].copy(), now)
            if u is not None:
                self.work.put((mic, u))

    def start_worker(self) -> None:
        threading.Thread(target=self._worker, daemon=True, name="measure").start()

    def run(self) -> None:
        import sounddevice as sd

        self.start_worker()
        try:
            with sd.InputStream(device=self.device, channels=max(self.mics),
                                samplerate=self.sr, blocksize=int(self.sr * BLOCK_S),
                                dtype="float32", callback=self.callback):
                while not self.stop_flag.is_set():
                    self.stop_flag.wait(1.0)
        except Exception as err:
            print(f"the microphones could not be opened: {type(err).__name__}: {err}",
                  file=sys.stderr, flush=True)


# =============================================================================
# the command line
# =============================================================================

def load_config(path: str | None) -> dict:
    cfg = {}
    for p in ([Path(path)] if path else [HERE / "room.json"]):
        if p.exists():
            cfg = json.loads(p.read_text(encoding="utf-8"))
            break
    return cfg


def device_rate(device, kind: str) -> int:
    import sounddevice as sd
    try:
        return int(sd.query_devices(device, kind)["default_samplerate"])
    except Exception:
        return 48000


def cmd_devices(_a) -> None:
    import sounddevice as sd
    print(sd.query_devices())


def cmd_tone(a) -> None:
    import sounddevice as sd

    channels = a.channels
    n = max(channels.values())
    sr = device_rate(a.output_device, "output")
    mixer = Mixer(n)
    names = voice_names()
    todo = (sorted(channels.items(), key=lambda kv: kv[1]) if a.all
            else [(next((v for v, c in channels.items() if c == a.channel), "?"), a.channel)])
    with sd.OutputStream(device=a.output_device, channels=n, samplerate=sr,
                         dtype="float32", callback=mixer.callback):
        for vid, ch in todo:
            print(f"channel {ch}: {names.get(vid, vid)}  {a.hz:.0f} Hz, {a.seconds}s", flush=True)
            mixer.add(sine(a.hz, a.seconds, sr, peak=0.3), ch)
            time.sleep(a.seconds + 0.5)


def start_player(a) -> tuple[Player, object]:
    import sounddevice as sd

    n = max(a.channels.values())
    if a.humans_channel:
        n = max(n, a.humans_channel)
    sr = device_rate(a.output_device, "output")
    mixer = Mixer(n)
    stream = sd.OutputStream(device=a.output_device, channels=n, samplerate=sr,
                             dtype="float32", callback=mixer.callback)
    stream.start()
    player = Player(mixer, sr, a.channels, a.gain, a.portal, a.humans_channel, not a.quiet,
                    hold_s=a.hold)
    player.start()
    print(f"playing {len(a.channels)} voices on {n} channels of {a.output_device!r} at {sr} Hz",
          flush=True)
    return player, stream


def start_listener(a) -> Listener:
    sr = device_rate(a.input_device, "input")
    lst = Listener(a.input_device, a.mics, sr, a.portal, a.room, not a.quiet, post=not a.print)
    lst.start()
    print(f"listening on channels {a.mics} of {a.input_device!r} at {sr} Hz"
          + ("  (printing only, nothing posted)" if a.print else ""), flush=True)
    return lst


def cmd_play(a) -> None:
    start_player(a)
    wait_forever()


def cmd_listen(a) -> None:
    start_listener(a)
    wait_forever()


def cmd_run(a) -> None:
    """Both, on one full-duplex stream.

    26 October, Gray Area: with the player's output stream open on the
    UMC1820, a second, input-only stream on the same interface came up silent
    — `listen` alone heard every note, `run` heard none. One stream that reads
    the microphones and writes the speakers in the same callback is what
    CoreAudio actually wants from a single device, so that is what the room
    runs on. Separate devices (a different interface for the microphones)
    still get two streams.
    """
    import sounddevice as sd

    if str(a.input_device) != str(a.output_device):
        start_player(a)
        start_listener(a)
        wait_forever()
        return

    n_out = max(a.channels.values())
    if a.humans_channel:
        n_out = max(n_out, a.humans_channel)
    n_in = max(a.mics)
    sr = device_rate(a.output_device, "output")
    mixer = Mixer(n_out)
    listener = Listener(a.input_device, a.mics, sr, a.portal, a.room, not a.quiet,
                        post=not a.print)
    listener.start_worker()

    def both(indata, outdata, frames, t, status):
        listener.callback(indata, frames, t, status)
        mixer.callback(outdata, frames, t, status)

    stream = sd.Stream(device=(a.input_device, a.output_device), channels=(n_in, n_out),
                       samplerate=sr, blocksize=int(sr * BLOCK_S), dtype="float32",
                       callback=both)
    stream.start()
    player = Player(mixer, sr, a.channels, a.gain, a.portal, a.humans_channel, not a.quiet,
                    hold_s=a.hold)
    player.start()
    print(f"playing {len(a.channels)} voices on {n_out} channels and listening on {a.mics} "
          f"of {a.output_device!r} at {sr} Hz, one stream"
          + ("  (printing only, nothing posted)" if a.print else ""), flush=True)
    wait_forever()


def wait_forever() -> None:
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


def main(argv=None) -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)
    cfg = load_config(known.config)

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", help="room.json (default: next to this file)")
    p.add_argument("--portal", default=cfg.get("portal", PORTAL))
    p.add_argument("--output-device", default=cfg.get("output_device"))
    p.add_argument("--input-device", default=cfg.get("input_device"))
    p.add_argument("--channels", type=parse_channels,
                   default=cfg.get("channels") or default_channels(),
                   help="voice_00=1,voice_01=2,…  (1-based output channels)")
    p.add_argument("--gain", type=json.loads, default=cfg.get("gain") or {},
                   help='{"voice_00": 0.8}  software trim per voice')
    p.add_argument("--humans-channel", type=int, default=cfg.get("humans_channel"),
                   help="sound people singing from elsewhere (zoom, phones) on this channel")
    p.add_argument("--mics", type=lambda s: [int(x) for x in s.split(",")],
                   default=cfg.get("mics") or [1, 2, 3, 4, 5],
                   help="1-based input channels of the five microphones")
    p.add_argument("--room", default=cfg.get("room", "room"),
                   help="prefix of the people this room contributes: human_<room><mic>_xxxx")
    p.add_argument("--hold", type=float, default=cfg.get("hold_s", HOLD_S),
                   help=f"seconds behind the choir's clock each breath sounds (default {HOLD_S:g}; 0 = on arrival)")
    p.add_argument("--quiet", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices", help="list audio devices")
    t = sub.add_parser("tone", help="a sine on one output channel, or all in turn")
    t.add_argument("--channel", type=int, default=1)
    t.add_argument("--all", action="store_true")
    t.add_argument("--hz", type=float, default=220.0)
    t.add_argument("--seconds", type=float, default=2.0)
    sub.add_parser("play", help="the nine speakers")
    li = sub.add_parser("listen", help="the five microphones")
    li.add_argument("--print", action="store_true", help="measure and print, post nothing")
    r = sub.add_parser("run", help="both")
    r.add_argument("--print", action="store_true")
    a = p.parse_args(argv)
    if not hasattr(a, "print"):
        a.print = False
    {"devices": cmd_devices, "tone": cmd_tone, "play": cmd_play,
     "listen": cmd_listen, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
