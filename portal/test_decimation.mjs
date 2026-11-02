// A phone will usually hand us 48 kHz, not the 16 kHz we ask for, so the page
// box-filters and decimates before running YIN. This checks that the detour
// does not move the pitch: same tones, generated at 48 kHz, decimated, and
// compared against the note they were built from.
//
//     node test_decimation.mjs

import { createRequire } from "node:module";
const { TUNING, yinFrame, hzToNote, stabilityLabel } =
  createRequire(import.meta.url)("./pitch-core.js");

const NATIVE = 48000;

/** Same box-filter decimation as pitch.html. */
function downsample(src, factor) {
  const n = Math.floor(src.length / factor);
  const dst = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    const base = i * factor;
    for (let k = 0; k < factor; k++) sum += src[base + k];
    dst[i] = sum / factor;
  }
  return dst;
}

/** A sung vowel: harmonic stack, not a bare sine. */
function voice(hz, sr, seconds = 0.3) {
  const n = Math.floor(sr * seconds);
  const x = new Float32Array(n);
  const weights = [1.0, 0.5, 0.33, 0.25, 0.16];
  let peak = 0;
  for (let i = 0; i < n; i++) {
    let v = 0;
    for (let k = 0; k < weights.length; k++) {
      v += weights[k] * Math.sin((2 * Math.PI * hz * (k + 1) * i) / sr);
    }
    x[i] = v;
    if (Math.abs(v) > peak) peak = Math.abs(v);
  }
  for (let i = 0; i < n; i++) x[i] = (x[i] / peak) * 0.3;
  return x;
}

const TONES = [
  ["A2", 110.0], ["D3", 146.83], ["G3", 196.0], ["A3", 220.0],
  ["D#4", 311.13], ["A4", 440.0], ["C5", 523.25], ["A5", 880.0],
];

const decimation = Math.round(NATIVE / TUNING.TARGET_RATE);
const workRate = NATIVE / decimation;
const frameLen = Math.floor(workRate * TUNING.FRAME_S);
const tauMin = Math.floor(workRate / TUNING.PITCH_FMAX_HZ);
const tauMax = Math.floor(workRate / TUNING.PITCH_FMIN_HZ);

console.log(`native ${NATIVE} Hz -> decimate by ${decimation} -> ${workRate} Hz`);
console.log(`frame ${frameLen} samples, tau ${tauMin}..${tauMax}\n`);
console.log("expected".padEnd(12) + "detected".padStart(10) + "cents".padStart(9) + "  note");

let failures = 0;
for (const [name, hz] of TONES) {
  const decimated = downsample(voice(hz, NATIVE), decimation);
  const frame = decimated.subarray(decimated.length - frameLen);
  const got = yinFrame(frame, workRate, tauMin, tauMax);

  const cents = got > 0 ? 1200 * Math.log2(got / hz) : NaN;
  const note = hzToNote(got);
  // A quarter-tone is 50 cents; anything inside 5 cents is inaudible drift.
  const ok = got > 0 && Math.abs(cents) < 5 && note === name;
  if (!ok) failures++;
  console.log(
    (ok ? "  " : "! ") + `${name} ${hz.toFixed(2)}`.padEnd(10) +
    got.toFixed(2).padStart(10) +
    (isNaN(cents) ? "—" : cents.toFixed(2)).padStart(9) +
    "  " + (note || "—") + (ok ? "" : "   <-- expected " + name)
  );
}

console.log(failures === 0
  ? `\nDecimation is transparent: all ${TONES.length} tones land on the right note.`
  : `\n${failures} of ${TONES.length} tones moved after decimation.`);
process.exit(failures === 0 ? 0 : 1);
