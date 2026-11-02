// Pitch detection, ported from live_v1/human.py so the browser and the Python
// listener agree on every number. If a threshold changes here it must change
// there too, or a phone and a room microphone will disagree about the same
// sung note.
//
// Loaded both by pitch.html (as a plain script) and by test_pitch.mjs (in
// node), which is why it touches no DOM and exports through globalThis.

"use strict";

const TUNING = {
  TARGET_RATE: 16000,
  PITCH_FMIN_HZ: 70.0,      // below this is sub-bass, not a sung tone
  PITCH_FMAX_HZ: 1100.0,    // C6 ≈ 1046 Hz is about the top of sung range
  YIN_THRESHOLD: 0.15,
  FRAME_SILENCE_AMP: 0.01,  // frame-level gate inside an utterance
  VAD_RMS_THRESHOLD: 0.025,
  SILENCE_TAIL_S: 0.5,
  MIN_UTTERANCE_S: 0.6,
  MAX_UTTERANCE_S: 30.0,
  FRAME_S: 0.05,            // 50 ms analysis frame
  HOP_S: 0.025,             // 25 ms hop
};

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

/** YIN (de Cheveigné & Kawahara, 2002). Returns 0 for an unvoiced frame. */
function yinFrame(frame, sr, tauMin, tauMax, threshold = TUNING.YIN_THRESHOLD) {
  const n = frame.length;
  if (n <= tauMax + 1) return 0;

  // Difference function d(τ) = Σ (x[j] − x[j+τ])²
  const d = new Float32Array(tauMax + 1);
  for (let tau = 1; tau <= tauMax; tau++) {
    let sum = 0;
    const lim = n - tau;
    for (let j = 0; j < lim; j++) {
      const diff = frame[j] - frame[j + tau];
      sum += diff * diff;
    }
    d[tau] = sum;
  }

  // Cumulative mean normalised difference d'(τ) = d(τ)·τ / Σ_{k≤τ} d(k)
  const dp = new Float32Array(tauMax + 1);
  dp[0] = 1;
  let cum = 0;
  for (let tau = 1; tau <= tauMax; tau++) {
    cum += d[tau];
    dp[tau] = d[tau] * tau / Math.max(cum, 1e-12);
  }

  // First dip below threshold, descend to its local minimum, then
  // parabolic-interpolate for sub-sample accuracy.
  let tau = tauMin;
  while (tau <= tauMax) {
    if (dp[tau] < threshold) {
      while (tau + 1 <= tauMax && dp[tau + 1] < dp[tau]) tau++;
      let refined = tau;
      if (tau >= 1 && tau < tauMax) {
        const a = dp[tau - 1], b = dp[tau], c = dp[tau + 1];
        const denom = 2 * (a - 2 * b + c);
        if (denom !== 0) refined = tau + (a - c) / denom;
      }
      return refined > 0 ? sr / refined : 0;
    }
    tau++;
  }
  return 0;
}

/**
 * Frame-by-frame tracker over a whole utterance.
 * Returns { medianHz, stdHz, voicedFraction } — the same triple as
 * human.py's _estimate_pitch, and (0,0,0) for an unvoiced utterance.
 */
function estimatePitch(audio, sr, fmin = TUNING.PITCH_FMIN_HZ, fmax = TUNING.PITCH_FMAX_HZ) {
  const frameN = Math.floor(sr * TUNING.FRAME_S);
  const hopN = Math.floor(sr * TUNING.HOP_S);
  if (audio.length < frameN) return { medianHz: 0, stdHz: 0, voicedFraction: 0 };

  const tauMin = Math.floor(sr / fmax);
  const tauMax = Math.floor(sr / fmin);
  if (tauMax <= tauMin) return { medianHz: 0, stdHz: 0, voicedFraction: 0 };

  const pitches = [];
  const nFrames = Math.floor((audio.length - frameN) / hopN) + 1;
  for (let i = 0; i < nFrames; i++) {
    const frame = audio.subarray(i * hopN, i * hopN + frameN);
    let peak = 0;
    for (let j = 0; j < frame.length; j++) {
      const a = frame[j] < 0 ? -frame[j] : frame[j];
      if (a > peak) peak = a;
    }
    if (peak < TUNING.FRAME_SILENCE_AMP) continue;  // silence inside the utterance
    const f0 = yinFrame(frame, sr, tauMin, tauMax);
    if (f0 > 0) pitches.push(f0);
  }

  if (!pitches.length) return { medianHz: 0, stdHz: 0, voicedFraction: 0 };
  return {
    medianHz: median(pitches),
    stdHz: stddev(pitches),
    voicedFraction: pitches.length / Math.max(1, nFrames),
  };
}

/** Hz → nearest note name (A4 = 440 Hz). */
function hzToNote(hz) {
  if (hz <= 0) return "";
  const semi = Math.round(12 * Math.log2(hz / 440));
  const idx = (((semi + 9) % 12) + 12) % 12;
  const octave = 4 + Math.floor((semi + 9) / 12);
  return NOTE_NAMES[idx] + octave;
}

/** Cents off the nearest equal-tempered note: −50…+50. */
function centsOff(hz) {
  if (hz <= 0) return 0;
  const exact = 12 * Math.log2(hz / 440);
  return Math.round((exact - Math.round(exact)) * 100);
}

/**
 * Verbal voice-quality label. Thresholds are tuned to the sung voice: a
 * sustained note drifts 50–100 cents from breath pressure alone, which is
 * "steady" for a human; vibrato is the deliberate oscillation beyond that.
 */
function stabilityLabel(stdHz, medianHz) {
  if (medianHz <= 0) return "";
  const cents = stdHz > 0 ? 1200 * Math.log2((medianHz + stdHz) / medianHz) : 0;
  if (cents < 30) return "very steady";
  if (cents < 80) return "steady";
  if (cents < 200) return "vibrato";
  return "sliding";
}

// Roughly one value per this much of a held note, for the pitch trace below.
// Finer than any notation will use, and coarse enough that the longest
// utterance the page will emit stays inside the cap.
const CONTOUR_HOP_S = 0.25;
const CONTOUR_MAX_POINTS = 128;

/**
 * A held note's pitch trace, thinned to one median per block.
 *
 * `stabilityLabel` reduces a whole utterance to one of four words, which is the
 * right thing to show a singer and the wrong thing to leave in an archive:
 * "steady" covers everything from 30 to 80 cents of movement and nothing
 * recovers where inside that a voice actually was. The trace keeps the shape —
 * a slide, a wobble, a tone that goes somewhere and comes back — as a handful
 * of numbers, which is what a score needs and all it needs.
 *
 * Lives here rather than in pitch.html because it is pure arithmetic on the
 * pitches already measured, and because the score written from 18 September
 * onward cannot be corrected afterwards, so it should be under test.
 */
function contourOf(pitches, durationS) {
  if (!pitches.length || durationS <= 0) return [];
  const n = Math.max(1, Math.min(CONTOUR_MAX_POINTS,
                                 Math.round(durationS / CONTOUR_HOP_S)));
  const out = [];
  for (let i = 0; i < n; i++) {
    const a = Math.floor((i * pitches.length) / n);
    const b = Math.max(a + 1, Math.floor(((i + 1) * pitches.length) / n));
    out.push(Math.round(median(pitches.slice(a, b)) * 10) / 10);
  }
  return out;
}

// --- one open microphone in a shared hall ----------------------------------
//
// The gallery microphone is this same page, running on the room's machine
// inside the installation Gray Area builds (`modal/EXHIBITION.md`). What
// differs is not the measurement but what stands in front of it: nine other
// installations, a 1940 theatre floor, the choir's own circle of speakers,
// and nobody holding a device to their face.
//
// A phone can open its gate just above the room it is in, because on a phone
// everything above the room *is* the singer. An open microphone cannot: a gate
// that only follows the hall would faithfully notate the hall, and the score is
// the one thing this work leaves behind. So the room's microphone admits a
// breath only when four things are true at once — a person singing into a
// near-field mic satisfies all four without trying, and the hall satisfies
// none of them:
//
//   loud    an absolute level, and also well clear of the room's own running
//           floor, so a busy hour raises the bar instead of lowering it
//   long    a sung note rather than a door, a footstep or a cough
//   voiced  a pitch found in most of its frames, which noise does not give
//   held    not "sliding": conversation nearby yields a wandering estimate,
//           a held tone does not
//
// What fails is written nowhere. Logging the rejects "for the archive" would
// put the hall's noise floor into the score, which is precisely the thing that
// must not happen — see `modal/SCORE.md` §9.
const GALLERY = {
  GATE_FLOOR: 0.02,          // ≈ −34 dB rms; roughly 20 dB over a quiet hall
  GATE_CEIL: 0.10,           // the adaptive gate may climb this far, no further
  GATE_OVER_NOISE: 6.0,      // ≈ 16 dB above the room's own floor
  MIN_UTTERANCE_S: 1.0,      // a phone accepts 0.6; the room asks for a note
  MIN_VOICED_FRACTION: 0.6,
  // After this much silence the microphone belongs to somebody else. It is the
  // only thing an unattended open mic can honestly say about identity.
  SESSION_GAP_S: 90.0,
  // A ceiling on what one minute of room can add to the field. Nothing should
  // ever reach it; if something does — a speaker turned toward the mic, a
  // neighbour's soundcheck — the field gets twenty lines rather than a night.
  MAX_BREATHS_PER_MIN: 20,
};

/**
 * Is this measured utterance a person singing into the room's microphone?
 *
 * Returns "" when it is, and otherwise the reason in words, which the kiosk
 * prints while the piece is being installed — the gate has to be set against
 * the actual hall on 23 October, and a rejection that says why is the only way
 * to do that without a laptop full of recordings.
 */
function galleryAdmits(u) {
  const dur = Number(u.durationS) || 0;
  const med = Number(u.medianHz) || 0;
  const sd = Number(u.stdHz) || 0;
  const voiced = Number(u.voicedFraction) || 0;
  const peak = Number(u.peakRms) || 0;
  if (med <= 0) return "no pitch in it";
  if (dur < GALLERY.MIN_UTTERANCE_S) return "too short for a sung note";
  if (voiced < GALLERY.MIN_VOICED_FRACTION) return "too broken to be a voice";
  if (stabilityLabel(sd, med) === "sliding") return "wandering — the room, not a voice";
  if (peak < GALLERY.GATE_FLOOR) return "too quiet for an open microphone";
  return "";
}

/** The line human.py writes into the shared field, e.g. "D#4 ~305Hz [vibrato]". */
function fieldLine(medianHz, stdHz) {
  if (medianHz <= 0) return "";
  return `${hzToNote(medianHz)} ~${medianHz.toFixed(0)}Hz [${stabilityLabel(stdHz, medianHz)}]`;
}

function median(a) {
  const s = Float64Array.from(a).sort();
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Population standard deviation, to match numpy's default ddof=0. */
function stddev(a) {
  const mean = a.reduce((x, y) => x + y, 0) / a.length;
  return Math.sqrt(a.reduce((s, v) => s + (v - mean) ** 2, 0) / a.length);
}

const PitchCore = {
  TUNING, NOTE_NAMES, CONTOUR_HOP_S, CONTOUR_MAX_POINTS, GALLERY,
  yinFrame, estimatePitch, hzToNote, centsOff, stabilityLabel, fieldLine,
  contourOf, galleryAdmits, median, stddev,
};

if (typeof module !== "undefined" && module.exports) module.exports = PitchCore;
else globalThis.PitchCore = PitchCore;
