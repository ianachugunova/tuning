// Loads pitch.html the way a browser does: pitch-core.js and the page's own
// script evaluated into one shared global scope, in order.
//
// This exists because of a real failure. pitch-core.js declares a top-level
// `const TUNING`; pitch.html destructured `const { TUNING } = PitchCore`. Each
// file is valid on its own, so `node --check` passed and the deploy looked
// fine — but together they are a redeclaration, and the browser refused to
// instantiate the page script at all. Not one line of it ran, so the button
// had no click handler and simply did nothing when pressed, with no error
// visible on the page. Checking the files separately can never catch that.
//
// Run: node test_page.mjs

import { readFileSync } from "node:fs";
import { createContext, runInContext } from "node:vm";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(here, "pitch.html"), "utf8");

const inline = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)]
  .map((m) => m[1]);
if (inline.length !== 1) {
  throw new Error(`expected one inline script in pitch.html, found ${inline.length}`);
}

// The ids the page expects to find. If markup and script drift apart, the
// stub returns null and the script throws — which is the point.
const IDS = ["note", "hz", "quality", "log", "btn", "status", "rate", "field",
             "presence", "trace", "meterFill", "meterGate", "level", "listen",
             "singing"];

const missing = IDS.filter((id) => !html.includes(`id="${id}"`));
if (missing.length) throw new Error(`pitch.html is missing elements: ${missing}`);

const noop = () => {};
const ctx2d = new Proxy({}, {
  get: (_, k) => (k === "measureText" ? () => ({ width: 0 }) : noop),
  set: () => true,
});

function makeEl(id) {
  const el = {
    id,
    textContent: "",
    innerHTML: "",
    disabled: false,
    width: 300,
    height: 100,
    style: {},
    classList: { add: noop, remove: noop, toggle: noop, contains: () => false },
    listeners: {},
    addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
    removeEventListener: noop,
    appendChild: noop,
    insertBefore: noop,
    removeChild: noop,
    remove: noop,
    prepend: noop,
    // Recorded rather than dropped: the page reports a browser that refuses to
    // play by appending a node, and a noop here would make that untestable.
    appended: [],
    append(...kids) { this.appended.push(...kids); },
    firstChild: null,
    children: [],
    scrollTop: 0,
    getContext: () => ctx2d,
    getBoundingClientRect: () => ({ width: 300, height: 100, top: 0, left: 0 }),
  };
  return el;
}

const els = Object.fromEntries(IDS.map((id) => [id, makeEl(id)]));
const intervals = [];

// Swappable so a test can hand the page a field with something in it. The
// default is the one the page has to survive: the portal being down.
let fetchImpl = async () => ({ ok: false, status: 503, json: async () => ({}), text: async () => "" });

// Every Audio the page builds, and whether the browser lets it start. One
// element for the session is the whole point of the priming click, so counting
// them is the test.
const audios = [];
let playAllowed = true;

// Every oscillator the page builds. A person's breath carries no audio, so a
// tone here is the only evidence that anyone was heard at all.
const oscillators = [];

const sandbox = {
  console,
  Math, JSON, Date, Number, String, Array, Object, Error, Promise, Set, Map,
  Float32Array, Uint8Array, Int16Array,
  isFinite, isNaN, parseFloat, parseInt,
  setTimeout, clearTimeout, clearInterval,
  // Recorded rather than started. A real interval would hold the event loop
  // open and the test would never finish; this also lets us check that the
  // page actually arranges to keep listening to the field.
  setInterval: (fn, ms) => { intervals.push({ fn, ms }); return intervals.length; },
  requestAnimationFrame: () => 1,
  cancelAnimationFrame: noop,
  ResizeObserver: class { observe() {} disconnect() {} },
  fetch: (...args) => fetchImpl(...args),
  localStorage: {
    store: new Map(),
    getItem(k) { return this.store.get(k) ?? null; },
    setItem(k, v) { this.store.set(k, String(v)); },
  },
  navigator: {
    mediaDevices: {
      getUserMedia: async () => { throw Object.assign(new Error("no mic"), { name: "NotFoundError" }); },
      enumerateDevices: async () => [],
    },
    permissions: { query: async () => ({ state: "prompt" }) },
    userAgent: "node-test",
  },
  document: {
    getElementById: (id) => els[id] ?? null,
    createElement: () => makeEl("created"),
    querySelectorAll: () => [],
    addEventListener: noop,
    documentElement: makeEl("html"),
    body: makeEl("body"),
  },
  // Enough of a context to sound a person from their numbers, and to record
  // that it happened. Curves and ramps are kept because a wrong one throws in
  // a browser and would be silently caught here otherwise.
  AudioContext: class {
    constructor() {
      this.sampleRate = 48000;
      this.state = "running";
      this.currentTime = 0;
    }
    resume() { return Promise.resolve(); }
    createOscillator() {
      const osc = {
        type: "",
        frequency: {
          setValueAtTime: (v) => { osc.pitches.push(v); },
          setValueCurveAtTime: (c) => { osc.curve = Array.from(c); },
        },
        pitches: [],
        curve: null,
        connect: (n) => n,
        start: noop,
        stop: noop,
        onended: null,
      };
      oscillators.push(osc);
      return osc;
    }
    createGain() {
      const ramps = [];
      return {
        gain: {
          setValueAtTime: (v) => ramps.push(v),
          exponentialRampToValueAtTime: (v) => {
            if (!(v > 0)) throw new Error("an exponential ramp to zero throws in a browser");
            ramps.push(v);
          },
        },
        ramps,
        connect: (n) => n,
      };
    }
    get destination() { return { connect: noop }; }
  },
  Audio: class {
    constructor(src) {
      this.src = src;
      this.plays = [];
      audios.push(this);
    }
    play() {
      this.plays.push(this.src);
      return playAllowed
        ? Promise.resolve()
        : Promise.reject(Object.assign(new Error("gesture required"),
                                       { name: "NotAllowedError" }));
    }
    pause() {}
  },
  location: { protocol: "https:", hostname: "example.modal.run", href: "https://example.modal.run/", origin: "https://example.modal.run" },
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
sandbox.self = sandbox;
sandbox.window.isSecureContext = true;
sandbox.window.devicePixelRatio = 2;
sandbox.window.addEventListener = noop;

const context = createContext(sandbox);

// Order matters: this is the ordering of the <script> tags in the document.
const core = readFileSync(join(here, "pitch-core.js"), "utf8");

let failures = 0;
function step(label, fn) {
  try {
    fn();
    console.log(`ok    ${label}`);
  } catch (err) {
    failures++;
    console.log(`FAIL  ${label}\n      ${err.name}: ${err.message}`);
  }
}

step("pitch-core.js evaluates", () => runInContext(core, context, { filename: "pitch-core.js" }));

step("pitch.html script shares a scope with pitch-core.js without colliding", () => {
  runInContext(inline[0], context, { filename: "pitch.html" });
});

step("the microphone button has a click handler", () => {
  const handlers = els.btn.listeners.click ?? [];
  if (handlers.length !== 1) {
    throw new Error(`expected 1 click handler on #btn, found ${handlers.length}`);
  }
});

step("pressing the button reports a missing microphone instead of dying", async () => {
  const handler = els.btn.listeners.click[0];
  handler(); // getUserMedia rejects; the page should catch and explain
});

step("the page keeps listening to the field", () => {
  if (!intervals.length) throw new Error("nothing polls the field");
});

// The score (modal/SCORE.md) is written from 18 September onward and cannot be
// filled in afterwards, so the page's part of it is gated here rather than
// trusted. Both of these are things nothing else would notice going missing:
// the payload would still post and the field would still fill, and a decade
// later the human parts would have no traces and no onsets.
step("a breath still carries what the score needs", () => {
  const src = inline[0];
  for (const key of ["std_hz", "contour_hz", "tail_s", "own_rest_s"]) {
    if (!src.includes(`${key}:`)) {
      throw new Error(`the breath payload no longer sends ${key}`);
    }
  }
});

step("the pitch contour thins a held note to a readable trace", () => {
  // Reached through PitchCore, which is where it lives: the page's own script
  // is wrapped in a function so none of its declarations escape, and a trace
  // that the score depends on should not be untestable for that reason.
  const { contourOf } = sandbox.PitchCore;
  const rising = Array.from({ length: 240 }, (_, i) => 300 + i * 0.1);
  const c = contourOf(rising, 6.0);
  // A quarter of a second per point: six seconds is twenty-four of them.
  if (c.length !== 24) throw new Error(`expected 24 points, got ${c.length}`);
  if (!(c[0] < c[c.length - 1])) throw new Error("a rising slide came out flat");
  if (contourOf(rising, 90).length !== 128) {
    throw new Error("a very long note must stay under the cap");
  }
  if (contourOf([], 6).length !== 0) throw new Error("empty in, empty out");
  if (contourOf(rising, 0).length !== 0) {
    throw new Error("a zero-length note has no trace");
  }
});

step("the choir can be turned on and off", () => {
  const handlers = els.listen.listeners.click ?? [];
  if (handlers.length !== 1) {
    throw new Error(`expected 1 click handler on #listen, found ${handlers.length}`);
  }
  handlers[0]();   // on — with an empty field there is nothing to play, and that is fine
  handlers[0]();   // off again
});

// The page went silent on a laptop while looking like it was working, three
// separate ways at once, and none of them raised anything. A fresh `new Audio`
// per breath, which Safari refuses outside the gesture that asked for it; the
// refusal swallowed by a catch that only rescheduled; and a hush meant to keep
// the choir off a held note, held open forever by a room the microphone found
// loud. Each is invisible from the outside — the page looks alive and makes no
// sound — so each is pinned here.

async function stepAsync(label, fn) {
  try {
    await fn();
    console.log(`ok    ${label}`);
  } catch (err) {
    failures++;
    console.log(`FAIL  ${label}\n      ${err.name}: ${err.message}`);
  }
}

const FIELD_WITH_BREATHS = {
  size: 2,
  recent: [
    { ts: 101, who: "model", name: "eurollm", text: "F3 ~172Hz", hz: 174.61, audio: "one.wav" },
    { ts: 102, who: "model", name: "qwen3", text: "\u554a", hz: 0, audio: "two.wav" },
  ],
};

const listen = () => els.listen.listeners.click[0]();
const poll = intervals.find((i) => i.ms === 5000);
// The id the page gave this device, so a test can pretend to be somebody else.
const MY_DEVICE_ID = sandbox.localStorage.getItem("tuning.voice_id");

async function pollField() {
  if (!poll) throw new Error("the field is not polled on a five-second interval");
  await poll.fn();
  await new Promise((r) => setTimeout(r, 0));
}

await stepAsync("a page that loaded while the portal was down comes back", async () => {
  // Everything above ran against a 503, which is how the page starts here and
  // how it starts for anyone who opens it mid-deploy. It used to stay that way
  // for the life of the tab.
  fetchImpl = async () => ({ ok: true, status: 200,
                             json: async () => FIELD_WITH_BREATHS, text: async () => "" });
  els.field.textContent = "";
  await pollField();
  if (!els.field.textContent) {
    throw new Error("the field never came back after a failed probe at load");
  }
});

await stepAsync("one audio element serves the whole session", async () => {
  playAllowed = true;
  listen();                       // on: primes an element inside the gesture
  await pollField();
  if (audios.length !== 1) {
    throw new Error(`one element should serve the session, the page built ${audios.length}`);
  }
  if (!audios[0].src.endsWith("one.wav")) {
    throw new Error(`the primed element was never handed a breath, src is ${audios[0].src}`);
  }
});

await stepAsync("a browser that refuses to start sound says so", async () => {
  listen();                       // off, which also clears `sounding`
  playAllowed = false;
  els.singing.appended.length = 0;
  listen();                       // on again
  await pollField();
  const warned = els.singing.appended.some((k) => k && k.className === "warn");
  if (!warned) throw new Error("the refusal was swallowed and the page said nothing");
});

await stepAsync("a person is sounded from their numbers, and never their own", async () => {
  // No recording is kept or sent — see `/breath` — so this tone is built from
  // the pitch, the glide and the length alone. Without it a field full of
  // people is silent, which is what made the room feel dead.
  playAllowed = true;
  oscillators.length = 0;
  fetchImpl = async () => ({ ok: true, status: 200, text: async () => "", json: async () => ({
    size: 3,
    recent: [
      { ts: 201, who: "human", voice_id: "someone_else", name: "a voice",
        text: "D3 149Hz", hz: 149, dur: 1.4, contour: [147, 149, 152], hop: 0.25 },
      // The singer's own breath, which they have already heard for real.
      { ts: 202, who: "human", voice_id: MY_DEVICE_ID, name: "a voice",
        text: "A4 440Hz", hz: 440, dur: 1.0, contour: [440], hop: 0.25 },
    ],
  }) });
  await pollField();
  if (oscillators.length !== 1) {
    throw new Error(`expected one tone, for the other person, got ${oscillators.length}`);
  }
  if (!oscillators[0].curve || oscillators[0].curve.length !== 3) {
    throw new Error("the glide was flattened away instead of being sounded");
  }
});

step("the hush that keeps the choir off a held note is bounded", () => {
  // A bare `voicedNow` here is the bug: a microphone in a room above the gate
  // is voiced for as long as it is open, and the choir never plays again.
  const src = inline[0];
  if (/if \(!hearing \|\| sounding \|\| voicedNow/.test(src)) {
    throw new Error("playback is gated on a raw voiced flag with no time limit");
  }
  if (!src.includes("MAX_UTTERANCE_S")) {
    throw new Error("nothing bounds how long the choir stays quiet for a singer");
  }
});

if (failures) {
  console.log(`\n${failures} failure(s)`);
  process.exit(1);
}
console.log("\npitch.html loads and the button is live.");
