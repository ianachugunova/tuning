#!/usr/bin/env python3
"""
live_v1/human.py — Human voice input via microphone + Whisper.

Each human performer:
  1. Sings into the microphone.
  2. A simple energy-based VAD detects when they start and stop.
  3. The recorded buffer is transcribed locally with Whisper.
  4. The transcript becomes a tone event in the shared field, with
     breath_chosen_s = actual duration of the utterance.

LLM agents read this event in their field_summary just like any other
voice — they can TUNE to the human's text or sing something NEW after.

USAGE — wire this into the orchestrator with --humans N:
  python3 -m live_v1.main --duration 180 --agents 10 --humans 1

REQUIREMENTS:
  pip install sounddevice openai-whisper numpy

  First run downloads a Whisper model (default: "base", ~140MB).
  Use --whisper-model tiny for fastest, --whisper-model small for quality.

IMPORTANT: USE HEADPHONES so the TTS playback of LLM voices does not feed
back into the microphone and get transcribed as another human utterance.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

# These imports are deferred so the rest of the project keeps working
# even if sounddevice/whisper are not installed.
#
# Two flags rather than one, because two different things are being asked for.
# Hearing a pitch is arithmetic and needs numpy alone; capturing a voice needs
# a sound card and a transcriber. Folding them together meant that anywhere
# without audio hardware — CI, above all — `_estimate_pitch` answered 0 Hz for
# every signal instead of refusing, and the fixtures that the browser is
# checked against were silently all zeros.
try:
    import numpy as np
    PITCH_DEPS_OK = True
    PITCH_DEPS_ERROR = None
except Exception as exc:
    PITCH_DEPS_OK = False
    PITCH_DEPS_ERROR = str(exc)

try:
    import sounddevice as sd
    import whisper
    HUMAN_DEPS_OK = PITCH_DEPS_OK
    HUMAN_DEPS_ERROR = None if PITCH_DEPS_OK else PITCH_DEPS_ERROR
except Exception as exc:
    HUMAN_DEPS_OK = False
    HUMAN_DEPS_ERROR = str(exc)


# ---------------------------------------------------------------------------
# Whisper is shared across all HumanVoice threads. A torch model is NOT
# thread-safe: concurrent .transcribe() calls from several mic threads corrupt
# its internal buffers and crash the process ("double free of object"). This
# module-level lock serialises inference so only one mic transcribes at a time.
# Transcription is short relative to a sung breath, so serialising 5 mics adds
# negligible latency and keeps the run stable.
# ---------------------------------------------------------------------------
_WHISPER_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# VAD / capture parameters
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000              # Whisper expects 16 kHz
BLOCK_DURATION_S = 0.1           # 100 ms blocks for streaming
VAD_RMS_THRESHOLD = 0.025        # voice above this RMS (raise if room is noisy)
SILENCE_TAIL_S = 0.5             # how much silence ends an utterance
                                 # (singers sustain tones rather than speak,
                                 #  so a short tail is fine — anything longer
                                 #  just adds delay before Whisper fires)
MIN_UTTERANCE_S = 0.6            # ignore clicks/breaths shorter than this
MAX_UTTERANCE_S = 30.0           # hard cap


# ---------------------------------------------------------------------------
# Pitch + voice quality (pure numpy, no extra deps)
# ---------------------------------------------------------------------------

# Sung voice range. Below 70Hz = inaudible/sub-bass; above 1100Hz = above
# typical sung range (the highest soprano top notes are ~C6 ≈ 1046 Hz).
PITCH_FMIN_HZ = 70.0
PITCH_FMAX_HZ = 1100.0

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _yin_frame(frame: "np.ndarray", sr: int, tau_min: int, tau_max: int,
               threshold: float = 0.15) -> float:
    """YIN pitch estimator for one frame (de Cheveigné & Kawahara, 2002).

    Avoids the octave / sub-harmonic errors of naive autocorrelation.
    Returns 0.0 for unvoiced frames.
    """
    frame = frame.astype(np.float32)
    n = len(frame)
    if n <= tau_max + 1:
        return 0.0
    # Difference function d(τ) = Σ (x[j] - x[j+τ])^2, vectorised.
    taus = np.arange(1, tau_max + 1)
    d = np.empty(tau_max + 1, dtype=np.float32)
    d[0] = 0.0
    for tau in taus:
        diff = frame[: n - tau] - frame[tau:]
        d[tau] = float(np.dot(diff, diff))
    # Cumulative mean normalised difference: d'(τ) = d(τ) · τ / Σ_{k≤τ} d(k).
    cum = np.cumsum(d[1:])
    d_prime = np.empty_like(d)
    d_prime[0] = 1.0
    d_prime[1:] = d[1:] * taus / np.maximum(cum, 1e-12)
    # First τ ≥ tau_min where d' dips below threshold; then descend to its
    # local minimum and parabolic-interpolate.
    tau = tau_min
    while tau <= tau_max:
        if d_prime[tau] < threshold:
            while tau + 1 <= tau_max and d_prime[tau + 1] < d_prime[tau]:
                tau += 1
            if 1 <= tau < tau_max:
                a, b, c = d_prime[tau - 1], d_prime[tau], d_prime[tau + 1]
                denom = 2.0 * (a - 2.0 * b + c)
                shift = (a - c) / denom if denom != 0.0 else 0.0
                tau_refined = tau + float(shift)
            else:
                tau_refined = float(tau)
            return sr / tau_refined if tau_refined > 0 else 0.0
        tau += 1
    return 0.0


def _estimate_pitch(
    audio: "np.ndarray",
    sample_rate: int = SAMPLE_RATE,
    fmin: float = PITCH_FMIN_HZ,
    fmax: float = PITCH_FMAX_HZ,
) -> tuple[float, float, float]:
    """YIN-based frame pitch tracker.

    Returns (median_f0_hz, std_f0_hz, voiced_fraction). Returns
    (0.0, 0.0, 0.0) if the utterance was effectively unvoiced (whisper,
    breath sound, consonant).
    """
    if not PITCH_DEPS_OK or audio.size == 0:
        return 0.0, 0.0, 0.0
    audio = audio.astype(np.float32)
    frame_n = int(sample_rate * 0.05)        # 50 ms
    hop_n = int(sample_rate * 0.025)         # 25 ms
    if len(audio) < frame_n:
        return 0.0, 0.0, 0.0
    tau_min = int(sample_rate / fmax)
    tau_max = int(sample_rate / fmin)
    if tau_max <= tau_min:
        return 0.0, 0.0, 0.0

    pitches: list = []
    n_frames = (len(audio) - frame_n) // hop_n + 1
    for i in range(n_frames):
        s = i * hop_n
        frame = audio[s : s + frame_n]
        if float(np.max(np.abs(frame))) < 0.01:
            continue  # silence inside the utterance
        f0 = _yin_frame(frame, sample_rate, tau_min, tau_max)
        if f0 > 0.0:
            pitches.append(f0)

    if not pitches:
        return 0.0, 0.0, 0.0
    arr = np.array(pitches, dtype=np.float32)
    voiced = len(arr) / max(1, n_frames)
    return float(np.median(arr)), float(np.std(arr)), float(voiced)


def _hz_to_note(hz: float) -> str:
    """Hz → nearest note name (A4 = 440 Hz)."""
    if not PITCH_DEPS_OK or hz <= 0:
        return ""
    semi = int(round(12 * np.log2(hz / 440.0)))
    idx = (semi + 9) % 12
    octave = 4 + (semi + 9) // 12
    return f"{_NOTE_NAMES[idx]}{octave}"


# ---------------------------------------------------------------------------
# Multi-channel microphone capture (e.g. MIDAS M32R via USB)
# ---------------------------------------------------------------------------


class MicMultiplexer:
    """Single InputStream on a multi-channel audio device, fanning out
    per-channel audio blocks to one queue per subscribed channel.

    Use this when you want N HumanVoice instances to share one physical
    device (e.g. MIDAS M32R presents 32 input channels through one USB
    cable; macOS will not let two streams open the same device). The
    multiplexer owns the only stream; each HumanVoice subscribes to one
    channel and reads from its private queue.
    """

    def __init__(
        self,
        device: Optional[object] = None,
        channels: int = 1,
        sample_rate: int = SAMPLE_RATE,
        block_duration_s: float = BLOCK_DURATION_S,
        verbose: bool = True,
    ) -> None:
        if not HUMAN_DEPS_OK:
            raise RuntimeError(
                f"MicMultiplexer requires sounddevice + numpy. "
                f"({HUMAN_DEPS_ERROR})"
            )
        self.device = device
        self.channels = channels
        self.sample_rate = sample_rate
        self.block_duration_s = block_duration_s
        self.verbose = verbose
        # channel index (0-based) → list of queues
        self._subscribers: dict = {}
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

    def subscribe(self, channel_1based: int) -> "queue.Queue":
        """Return a queue that receives audio blocks from the given
        channel (1-indexed, same numbering as the M32R touchscreen)."""
        idx = channel_1based - 1
        if not (0 <= idx < self.channels):
            raise ValueError(
                f"channel {channel_1based} out of range for device with "
                f"{self.channels} channel(s)"
            )
        q: "queue.Queue" = queue.Queue()
        with self._lock:
            self._subscribers.setdefault(idx, []).append(q)
        return q

    def _cb(self, indata, frames, time_info, status) -> None:
        if status and self.verbose:
            print(f"[mic-mux] audio status: {status}", flush=True)
        with self._lock:
            subs = dict(self._subscribers)
        for idx, queues in subs.items():
            try:
                ch_data = indata[:, idx].copy()
            except IndexError:
                continue
            for q in queues:
                q.put(ch_data)

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=int(self.sample_rate * self.block_duration_s),
            device=self.device,
            callback=self._cb,
        )
        self._stream.start()
        if self.verbose:
            print(
                f"[mic-mux] open: device={self.device!r}, "
                f"channels={self.channels}, sr={self.sample_rate}"
            )

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        self._stream = None


def _stability_label(std_hz: float, median_hz: float) -> str:
    """Verbal voice-quality label from pitch std relative to mean.

    Thresholds are tuned to sung human voice: a sustained note of ~10s
    typically drifts ~50–100 cents (half a semitone) just from breath
    pressure variation; that is "steady" for humans. "vibrato" is the
    deliberate ~±100 cent oscillation around a centre pitch. "sliding"
    means actual glissando / scoop, well beyond a semitone of range.
    """
    if median_hz <= 0:
        return ""
    cents = 1200.0 * np.log2((median_hz + std_hz) / median_hz) if std_hz > 0 else 0.0
    if cents < 30:
        return "very steady"
    if cents < 80:
        return "steady"
    if cents < 200:
        return "vibrato"
    return "sliding"


# ---------------------------------------------------------------------------
# Human voice listener
# ---------------------------------------------------------------------------


class HumanVoice:
    """Captures a single microphone, segments utterances by silence,
    transcribes via local Whisper, and appends events to the shared field."""

    def __init__(
        self,
        voice_id: str,
        field,
        whisper_model=None,
        whisper_model_name: str = "base",
        device: Optional[object] = None,
        audio_queue: Optional["queue.Queue"] = None,
        verbose: bool = True,
    ) -> None:
        """Create a human voice agent.

        Two modes:
          1. Standalone: pass `device` (or leave None for default mic).
             HumanVoice opens its own single-channel InputStream.
             Use this on a laptop with the built-in microphone.
          2. Shared (concert): pass `audio_queue` from a `MicMultiplexer`.
             HumanVoice reads from that queue and does not touch the
             audio device itself. Use this for 5 mics on an M32R.

        `whisper_model` lets multiple HumanVoice instances share a single
        loaded Whisper model (saves ~150MB per voice for the base model).
        """
        if not HUMAN_DEPS_OK:
            raise RuntimeError(
                f"Human voice requires sounddevice + whisper + numpy. "
                f"Install with:\n"
                f"  pip install sounddevice openai-whisper numpy\n"
                f"(import error: {HUMAN_DEPS_ERROR})"
            )

        self.voice_id = voice_id
        self.field = field
        self.device = device
        self.verbose = verbose

        if whisper_model is not None:
            self.model = whisper_model
        else:
            if verbose:
                print(f"[human] {self.voice_id}: loading Whisper '{whisper_model_name}'...")
            self.model = whisper.load_model(whisper_model_name)
            if verbose:
                print(f"[human] {self.voice_id}: Whisper ready.")

        if audio_queue is not None:
            self._raw_q = audio_queue
            self._owns_stream = False
        else:
            self._raw_q = queue.Queue()
            self._owns_stream = True
        self._running = False
        self._stream: Optional[sd.InputStream] = None
        self._capture_thread: Optional[threading.Thread] = None

    # --- audio callback (runs in PortAudio thread) ---------------------------

    def _audio_cb(self, indata, frames, time_info, status) -> None:
        if status and self.verbose:
            print(f"[human {self.voice_id}] audio status: {status}", flush=True)
        # Copy because PortAudio reuses the buffer
        self._raw_q.put(indata[:, 0].copy())

    # --- main capture loop (segments utterances) -----------------------------

    def _capture_loop(self) -> None:
        """Energy-based VAD: collect audio while RMS is above threshold,
        flush the buffer to Whisper after SILENCE_TAIL_S of quiet."""
        buf: list = []
        in_voice = False
        silence_blocks = 0
        silence_blocks_needed = int(SILENCE_TAIL_S / BLOCK_DURATION_S)
        max_blocks = int(MAX_UTTERANCE_S / BLOCK_DURATION_S)

        # Live level meter: even when nothing crosses the VAD threshold, print
        # the loudest RMS seen over the last ~2s so the operator can SEE whether
        # any signal is arriving and how far below the 0.025 threshold it sits.
        # Without this, a too-low preamp gain looks identical to a dead mic.
        meter_peak = 0.0
        meter_last = time.time()
        meter_interval_s = 2.0

        while self._running:
            try:
                chunk = self._raw_q.get(timeout=0.2)
            except queue.Empty:
                continue
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            is_voice = rms > VAD_RMS_THRESHOLD

            if self.verbose and not in_voice:
                meter_peak = max(meter_peak, rms)
                now = time.time()
                if now - meter_last >= meter_interval_s:
                    pct = min(1.0, meter_peak / VAD_RMS_THRESHOLD)
                    bar = "#" * int(pct * 20)
                    flag = "  <-- below threshold" if meter_peak <= VAD_RMS_THRESHOLD else ""
                    print(
                        f"[human {self.voice_id}] level peak={meter_peak:.4f} "
                        f"(thresh={VAD_RMS_THRESHOLD:.3f}) |{bar:<20s}|{flag}",
                        flush=True,
                    )
                    meter_peak = 0.0
                    meter_last = now

            if is_voice:
                if not in_voice and self.verbose:
                    print(f"[human {self.voice_id}] ▶ voice detected (rms={rms:.3f})")
                in_voice = True
                silence_blocks = 0
                buf.append(chunk)
            elif in_voice:
                buf.append(chunk)
                silence_blocks += 1
                if silence_blocks >= silence_blocks_needed or len(buf) >= max_blocks:
                    # flush
                    audio = np.concatenate(buf).astype(np.float32)
                    # strip the trailing silence we recorded
                    trim = int(silence_blocks * BLOCK_DURATION_S * SAMPLE_RATE)
                    if trim < len(audio):
                        audio = audio[:-trim]
                    duration_s = max(0.0, len(audio) / SAMPLE_RATE)
                    if duration_s >= MIN_UTTERANCE_S:
                        # transcribe in another thread so we keep capturing
                        threading.Thread(
                            target=self._transcribe_and_emit,
                            args=(audio, duration_s),
                            daemon=True,
                        ).start()
                    elif self.verbose:
                        print(f"[human {self.voice_id}] (too short: {duration_s:.2f}s — ignored)")
                    buf = []
                    in_voice = False
                    silence_blocks = 0

    # --- transcription + event emission --------------------------------------

    def _transcribe_and_emit(self, audio: "np.ndarray", duration_s: float) -> None:
        t0 = time.time()

        # Normalise + guard the audio buffer once for everything below.
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0

        # Audio characteristics (fast, no model needed).
        f0_med, f0_std, voiced = _estimate_pitch(audio, SAMPLE_RATE)
        note = _hz_to_note(f0_med)
        quality = _stability_label(f0_std, f0_med)

        # Whisper transcription (slow). Skip outright if the clip is too
        # quiet — Whisper otherwise can produce NaN logits and crash, since
        # it doesn't expect almost-silent input. Pad short clips up to one
        # second, which is the minimum where Whisper's internal mel pipeline
        # is numerically stable on Apple Silicon CPU.
        text = ""
        if rms >= 0.005:
            audio_for_whisper = audio
            if len(audio_for_whisper) < SAMPLE_RATE:
                audio_for_whisper = np.pad(
                    audio_for_whisper, (0, SAMPLE_RATE - len(audio_for_whisper))
                )
            try:
                # Serialise inference: the model is shared across all mic
                # threads and torch is not thread-safe (concurrent calls cause
                # a native double-free crash).
                with _WHISPER_LOCK:
                    result = self.model.transcribe(
                        audio_for_whisper,
                        fp16=False,           # safer on Apple Silicon
                        language=None,        # auto-detect
                        no_speech_threshold=0.4,
                    )
                text = (result.get("text") or "").strip()
            except Exception as exc:
                if self.verbose:
                    print(f"[human {self.voice_id}] whisper skipped ({type(exc).__name__})")
                text = ""
        latency_ms = int((time.time() - t0) * 1000)

        # Build content. LLMs see this verbatim in the field summary, and
        # sound.py's pitch detector will recognise note names + Hz numbers
        # inside it, so a replay can reproduce the human's actual pitch.
        parts: list[str] = []
        if note:
            parts.append(f"{note} ~{f0_med:.0f}Hz")
        if quality:
            parts.append(f"[{quality}]")
        if text:
            parts.append(text)
        content = " ".join(parts) if parts else ""

        if not content:
            if self.verbose:
                print(f"[human {self.voice_id}] (no voice / no speech)")
            return

        # Late import to avoid circular dep at module load. The package name
        # differs by launch mode (relative when run as `python3 live_v1/...`,
        # `live_v1.*` under pytest), so try each until one resolves.
        try:
            from .orchestrator import Event
        except Exception:
            try:
                from orchestrator import Event  # type: ignore
            except Exception:
                from live_v1.orchestrator import Event  # type: ignore

        ev = Event(
            timestamp=time.time(),
            voice_id=self.voice_id,
            model_short="human",
            kind="tone",
            action="HUMAN",         # so analyze/sound can treat it specially
            content=content,
            tuning_to=None,
            breath_chosen_s=duration_s,
            api_latency_ms=latency_ms,
        )
        self.field.append(ev)
        # Always print the full picture: pitch, note, quality, words.
        print(
            f"  ♫ {self.voice_id}  {note or '(unvoiced)':<5} "
            f"{f0_med:>6.1f}Hz  ±{f0_std:>4.1f}  [{quality or '-'}]  "
            f"rms={rms:.3f}  {duration_s:.1f}s  "
            f"voiced={voiced*100:.0f}%  "
            f"whisper={latency_ms}ms  \"{text}\""
        )

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        if self._owns_stream:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=int(SAMPLE_RATE * BLOCK_DURATION_S),
                device=self.device,
                callback=self._audio_cb,
            )
            self._stream.start()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        if self.verbose:
            mode = "own stream" if self._owns_stream else "shared multiplexer"
            print(f"[human {self.voice_id}] listening ({mode}).")

    def stop(self) -> None:
        self._running = False
        if self._owns_stream and self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)


def list_audio_devices() -> None:
    """Helper: list available microphones with their device indices."""
    if not HUMAN_DEPS_OK:
        print(f"sounddevice not available: {HUMAN_DEPS_ERROR}")
        return
    print(sd.query_devices())


if __name__ == "__main__":
    # When run directly: just list devices to help the user pick one
    list_audio_devices()
