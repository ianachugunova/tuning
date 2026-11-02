// Check the browser pitch detector against live_v1/human.py on identical audio.
//
//     python3 make_fixtures.py && node test_pitch.mjs
//
// A phone and the room microphone have to agree about what was sung, so the
// tolerance here is tight: same note name, same stability word, and a
// frequency within a tenth of a cent.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const { createRequire } = await import("node:module");
const PitchCore = createRequire(import.meta.url)("./pitch-core.js");

const cases = JSON.parse(readFileSync(join(here, "fixtures.json"), "utf8"));

const CENTS_TOLERANCE = 0.1;
const centsBetween = (a, b) => (a > 0 && b > 0 ? Math.abs(1200 * Math.log2(a / b)) : 0);

let failures = 0;
console.log(
  "case".padEnd(18) + "python".padStart(11) + "javascript".padStart(13) +
  "cents".padStart(9) + "  note      stability"
);

for (const c of cases) {
  const audio = Float32Array.from(c.audio);
  const js = PitchCore.estimatePitch(audio, c.sample_rate);
  const py = c.python;

  const note = PitchCore.hzToNote(js.medianHz);
  const stability = PitchCore.stabilityLabel(js.stdHz, js.medianHz);
  const cents = centsBetween(js.medianHz, py.median_hz);

  const problems = [];
  if ((py.median_hz > 0) !== (js.medianHz > 0)) problems.push("voiced/unvoiced disagree");
  if (cents > CENTS_TOLERANCE) problems.push(`${cents.toFixed(2)} cents apart`);
  if (note !== py.note) problems.push(`note ${note} vs ${py.note}`);
  if (stability !== py.stability) problems.push(`stability "${stability}" vs "${py.stability}"`);
  if (Math.abs(js.voicedFraction - py.voiced_fraction) > 0.02) {
    problems.push(`voiced ${js.voicedFraction.toFixed(2)} vs ${py.voiced_fraction.toFixed(2)}`);
  }

  const ok = problems.length === 0;
  if (!ok) failures++;
  console.log(
    (ok ? "  " : "! ") + c.name.padEnd(16) +
    py.median_hz.toFixed(2).padStart(11) +
    js.medianHz.toFixed(2).padStart(13) +
    cents.toFixed(3).padStart(9) +
    "  " + (note || "—").padEnd(6) + "  " + (stability || "—") +
    (ok ? "" : "   <-- " + problems.join("; "))
  );
}

// The field line is what actually reaches the choir, so check the string too.
const sample = cases.find((c) => c.name === "voice_196");
if (sample) {
  const js = PitchCore.estimatePitch(Float32Array.from(sample.audio), sample.sample_rate);
  console.log(`\nfield line: "${PitchCore.fieldLine(js.medianHz, js.stdHz)}"`);
}

console.log(failures === 0
  ? `\nAll ${cases.length} cases agree with human.py.`
  : `\n${failures} of ${cases.length} cases disagree with human.py.`);
process.exit(failures === 0 ? 0 : 1);
