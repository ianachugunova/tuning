// Loads field.html the way a browser does: pitch-core.js, then listen.js, then
// the page's own script, all evaluated into one shared global scope, in order.
//
// This is test_page.mjs's twin, and it exists for the same reason. pitch-core.js
// declares a top-level `const TUNING`; a page that writes `const { TUNING } =
// PitchCore` at its own top level is a redeclaration, and the browser then
// refuses to instantiate the page script at all — not one line of it runs, so
// the button has no click handler and simply does nothing when pressed, with no
// error visible anywhere. Checking the files separately can never catch that,
// and the redesign adds a third script to the pile.
//
// What is gated here beyond loading is the part of the score the page is
// responsible for, and the three ways this page went silent on a laptop while
// looking like it was working.
//
// Run: node test_field_page.mjs

import { readFileSync } from "node:fs";
import { createContext, runInContext, Script } from "node:vm";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(here, "field.html"), "utf8");

const inline = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)]
  .map((m) => m[1]);
if (inline.length !== 1) {
  throw new Error(`expected one inline script in field.html, found ${inline.length}`);
}

// The ids the page expects to find. If markup and script drift apart, the stub
// returns null and the script throws — which is the point.
const IDS = ["veil", "veilStep1", "veilStep2", "begin", "enter", "pulse",
             "field", "nine",
             "tickName", "tickText", "tickGest", "note", "hz", "readout",
             "more", "diag", "mic", "choir", "ears", "status", "chorus",
             "school"];

const missing = IDS.filter((id) => !html.includes(`id="${id}"`));
if (missing.length) throw new Error(`field.html is missing elements: ${missing}`);

const noop = () => {};

// Recording rather than blind: the whole premise of this design is that a
// machine breath with no pitch is still drawn, under the axis, with its words.
// A proxy that swallows every call cannot tell that from a blank canvas.
let draws = [];
const ctx2d = new Proxy({}, {
  get: (_, k) => {
    if (k === "measureText") return () => ({ width: 10 });
    if (k === "setTransform" || k === "clearRect") return noop;
    return (...a) => { draws.push([String(k), ...a]); };
  },
  set: () => true,
});

function makeEl(id) {
  return {
    id,
    textContent: "",
    _html: "",
    get innerHTML() { return this._html; },
    set innerHTML(v) {
      this._html = v;
      if (v === "") this.children.length = 0;   // a clear, as in a browser
    },
    disabled: false,
    hidden: false,
    width: 300,
    height: 100,
    style: {},
    dataset: {},
    classList: {
      set: new Set(),
      add(c) { this.set.add(c); },
      remove(c) { this.set.delete(c); },
      contains(c) { return this.set.has(c); },
      toggle(c, on) {
        const want = on === undefined ? !this.set.has(c) : !!on;
        if (want) this.set.add(c); else this.set.delete(c);
        return want;
      },
    },
    listeners: {},
    addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
    removeEventListener: noop,
    // Kept rather than dropped: the rows under the ticker are how the page shows
    // that more than one voice is sounding, so a test has to be able to read
    // them. Setting innerHTML to "" is a clear, as it is in a browser.
    children: [],
    append(...kids) { this.children.push(...kids); },
    appendChild(kid) { this.children.push(kid); },
    removeChild: noop,
    remove: noop,
    prepend: noop,
    // Recorded rather than dropped: the page reports things by appending nodes,
    // and a noop here would make that untestable.
    appended: [],
    children: [],
    append(...kids) {
      this.appended.push(...kids);
      for (const k of kids) if (k && typeof k === "object") this.children.push(k);
    },
    firstChild: null,
    lastChild: null,
    scrollTop: 0,
    getContext: () => ctx2d,
    getBoundingClientRect: () => ({ width: 300, height: 100, top: 0, left: 0 }),
    querySelector: () => null,
  };
}

const els = Object.fromEntries(IDS.map((id) => [id, makeEl(id)]));
const intervals = [];

// Swappable so a test can hand the page a field with something in it. The
// default is the one the page has to survive: the portal being down.
let fetchImpl = async () => ({ ok: false, status: 503, json: async () => ({}), text: async () => "" });

const audios = [];
let playAllowed = true;
const oscillators = [];
// One per machine breath that was actually started, which is how simultaneity
// is checked: a queue of solos would only ever produce them one at a time.
const sources = [];
const gains = [];

// The page drives its own animation loop. Held rather than run, or draw() would
// recurse forever on its first call and the test would never finish.
const frames = [];

const sandbox = {
  console,
  Math, JSON, Date, Number, String, Array, Object, Error, Promise, Set, Map,
  Float32Array, Uint8Array, Int16Array, Infinity,
  isFinite, isNaN, parseFloat, parseInt,
  setTimeout, clearTimeout, clearInterval,
  setInterval: (fn, ms) => { intervals.push({ fn, ms }); return intervals.length; },
  requestAnimationFrame: (fn) => { frames.push(fn); return frames.length; },
  cancelAnimationFrame: noop,
  ResizeObserver: class { observe() {} disconnect() {} },
  // A breath is fetched as bytes and decoded, so several can sound at once;
  // everything else is the field itself and belongs to whichever test set it.
  fetch: (...args) => (/^audio\//.test(String(args[0]))
    ? Promise.resolve({ ok: true, status: 200,
                        arrayBuffer: async () => new ArrayBuffer(2048) })
    : fetchImpl(...args)),
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
    querySelector: () => null,
    addEventListener: noop,
    documentElement: makeEl("html"),
    body: makeEl("body"),
  },
  AudioContext: class {
    constructor() {
      this.sampleRate = 48000;
      this.state = "running";
      this.currentTime = 0;
    }
    resume() { return Promise.resolve(); }
    decodeAudioData(bytes) {
      return Promise.resolve({ duration: 5, byteLength: bytes.byteLength });
    }
    createBufferSource() {
      const src = {
        buffer: null,
        startedAt: null,
        connect: (n) => n,
        start(at) { src.startedAt = at ?? 0; },
        stop: noop,
        onended: null,
      };
      sources.push(src);
      return src;
    }
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
      const node = {
        gain: {
          value: 1,
          setValueAtTime: (v) => ramps.push(v),
          exponentialRampToValueAtTime: (v) => {
            if (!(v > 0)) throw new Error("an exponential ramp to zero throws in a browser");
            ramps.push(v);
          },
          linearRampToValueAtTime: (v) => { node.gain.value = v; ramps.push(v); },
          cancelScheduledValues: noop,
        },
        ramps,
        connect: (n) => n,
      };
      gains.push(node);
      return node;
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
  location: { protocol: "https:", hostname: "example.modal.run", search: "",
              href: "https://example.modal.run/next",
              origin: "https://example.modal.run" },
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
const listenSrc = readFileSync(join(here, "listen.js"), "utf8");

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
async function stepAsync(label, fn) {
  try {
    await fn();
    console.log(`ok    ${label}`);
  } catch (err) {
    failures++;
    console.log(`FAIL  ${label}\n      ${err.name}: ${err.message}`);
  }
}

step("pitch-core.js evaluates", () =>
  runInContext(core, context, { filename: "pitch-core.js" }));

step("listen.js shares a scope with pitch-core.js without colliding", () => {
  runInContext(listenSrc, context, { filename: "listen.js" });
  if (!sandbox.PitchListen) throw new Error("listen.js exported nothing");
});

step("field.html script shares a scope with both without colliding", () => {
  runInContext(inline[0], context, { filename: "field.html" });
});

step("the microphone button has a click handler", () => {
  const handlers = els.mic.listeners.click ?? [];
  if (handlers.length !== 1) {
    throw new Error(`expected 1 click handler on #mic, found ${handlers.length}`);
  }
});

step("pressing the button reports a missing microphone instead of dying", () => {
  els.mic.listeners.click[0]();   // getUserMedia rejects; the page must explain
});

step("the page keeps listening to the field", () => {
  if (!intervals.some((i) => i.ms === 5000)) {
    throw new Error("the field is not polled on a five-second interval");
  }
});

// The score (modal/SCORE.md) is written from 18 September onward and cannot be
// filled in afterwards. These are things nothing else would notice going
// missing: the payload would still post, the field would still fill, and a
// decade later the human parts would have no traces and no onsets. They now
// live in listen.js, which is the point of listen.js — one listener, one
// payload, whatever the page around it looks like.
step("a breath still carries what the score needs", () => {
  for (const key of ["std_hz", "contour_hz", "tail_s", "own_rest_s"]) {
    if (!listenSrc.includes(`${key}:`)) {
      throw new Error(`the breath payload no longer sends ${key}`);
    }
  }
});

step("the redesign measures through listen.js and not its own copy", () => {
  // The one rule this page is not allowed to break. A second implementation of
  // listening would be a second gate, a second onset convention and a second
  // idea of what a rest is, and the human halves of the score would stop being
  // comparable in the week they matter most. See modal/SCORE.md §9.
  const src = inline[0];
  if (!src.includes("PitchListen.create")) {
    throw new Error("the page does not use the shared listener");
  }
  for (const own of ["yinFrame(", "getUserMedia", "createAnalyser"]) {
    if (src.includes(own)) {
      throw new Error(`the page measures on its own: found ${own} in field.html`);
    }
  }
});

step("the hush that keeps the choir off a held note is bounded", () => {
  // A bare `voicedNow` here is the bug: a microphone in a room above the gate
  // is voiced for as long as it is open, and the choir never plays again.
  if (!listenSrc.includes("MAX_UTTERANCE_S")) {
    throw new Error("nothing bounds how long the choir stays quiet for a singer");
  }
  if (!inline[0].includes("holdingNote")) {
    throw new Error("playback is not gated on a held note at all");
  }
});

step("the threshold is two pages: the name of the work, then the score", () => {
  // The first page is the title and one sentence; the button turns it into
  // Oliveros's score, word for word. The score IS the instruction — she wrote
  // it as one, for people with no training — so if any sentence of it goes
  // missing, the page is giving people a partial instruction and calling it
  // the original.
  for (const line of [
    "letting it all the way out with air sound",
    "mind&rsquo;s ear for a tone",
    "using any vowel sound",
    "Select a voice distant from you",
    "singing a new tone that no one else is singing",
    "tuning to the tone of another voice alternately",
  ]) {
    if (!html.includes(line)) {
      throw new Error(`the score on the threshold lost a sentence: "${line}"`);
    }
  }
  const handlers = els.begin.listeners.click ?? [];
  if (handlers.length !== 1) {
    throw new Error(`expected 1 click handler on #begin, found ${handlers.length}`);
  }
  handlers[0]();
  if (!els.veilStep1.hidden || els.veilStep2.hidden) {
    throw new Error("the first button did not turn the name into the score");
  }
  if (!els.begin.hidden || els.enter.hidden) {
    throw new Error("the doors did not change hands: begin should give way to enter");
  }
});

step("the typeface is chosen, not offered", () => {
  // Three candidates were switchable on the live page for a while and the
  // grotesk won. A person entering the field has no reason to be offered
  // typography, so the switcher is gone — and it must stay gone.
  for (const leftover of ["typebar", "tuning.type", "data-type"]) {
    if (html.includes(leftover)) {
      throw new Error(`the type switcher is back: found "${leftover}"`);
    }
  }
  if (!/--voice:\s*"Helvetica Neue"/.test(html)) {
    throw new Error("the work's voice is no longer the grotesk");
  }
});

step("the threshold can be crossed", () => {
  const handlers = els.enter.listeners.click ?? [];
  if (handlers.length !== 1) {
    throw new Error(`expected 1 click handler on #enter, found ${handlers.length}`);
  }
  handlers[0]();
  if (!els.veil.classList.contains("gone")) {
    throw new Error("entering the field left the threshold on screen");
  }
});

step("crossing it is the only thing anybody has to press to hear the choir", () => {
  // A browser will not make a sound without a gesture, so the page gets exactly
  // one and must spend it on the way in. If it ever stops doing that, the choir
  // is silent for everybody who does not think to look in the footer for a
  // second button — and on an opening night nobody does.
  if (!els.choir.classList.contains("on") || els.choir.textContent !== "They are on") {
    throw new Error("the choir was not switched on by the gesture that entered the field");
  }
  if (!sandbox.window.AudioContext.prototype.resume) {
    throw new Error("the context is never resumed, so the gesture is wasted");
  }
});

const FIELD_WITH_BREATHS = {
  size: 2,
  choir_size: 2,
  sounding_since: 100,
  machines_now: 6,
  humans_now: 0,
  recent: [
    { ts: 101, who: "model", name: "eurollm", action: "NEW",
      text: "F3 ~172Hz", hz: 174.61, audio: "one.wav" },
    { ts: 102, who: "model", name: "qwen3", action: "TUNE",
      text: "\u554a", hz: 0, audio: "two.wav" },
  ],
};

const choir = () => els.choir.listeners.click[0]();
const poll = intervals.find((i) => i.ms === 5000);
const MY_DEVICE_ID = sandbox.localStorage.getItem("tuning.voice_id");

async function pollField() {
  // The page runs one request at a time, and it fired one the moment it
  // loaded. That one is still unsettled here — this file reaches its first
  // `await` only now — and a poll started on top of it is correctly refused.
  await new Promise((r) => setTimeout(r, 0));
  await poll.fn();
  await new Promise((r) => setTimeout(r, 0));
}

await stepAsync("a page that loaded while the portal was down comes back", async () => {
  // Everything above ran against a 503, which is how the page starts here and
  // how it starts for anyone who opens it mid-deploy. The old page stayed that
  // way for the life of the tab.
  //
  // Watched on the canvas rather than in the counter under it. This used to
  // assert that `pulse` filled up, which worked only because the page printed
  // the number of breaths there before the birth; that line is gone — the veil
  // is the quote and a door now, and the pulse stays empty until there is money
  // to report.
  //
  // So the recovery is watched where it is actually visible: the field repaints,
  // pitch axis and all, which it does when an answer arrives with something new
  // in it. The breath is stamped now and sung by a person, both deliberately — a
  // fixture from 1970 is correctly dropped as older than the window and would
  // paint nothing, and a human breath carries no audio, so proving the page is
  // alive does not also make a sound in the middle of the suite.
  const t = Date.now() / 1000;
  fetchImpl = async () => ({ ok: true, status: 200, text: async () => "", json: async () => ({
    size: 3, choir_size: 2, sounding_since: t - 3600,
    recent: [
      { ts: t - 2, who: "human", voice_id: "came_back", name: "a voice",
        text: "A4 440Hz", hz: 440, dur: 1.2,
        contour: [438.1, 441.7, 440.2, 439.5], hop: 0.3 },
    ],
  }) });
  await pollField();
  draws = [];
  const paint = frames.shift();
  if (!paint) throw new Error("the page stopped asking for animation frames");
  paint();
  const painted = draws.filter((d) => d[0] === "fillText").map((d) => String(d[1]));
  if (!painted.some((x) => /^[CA]\d$/.test(x))) {
    throw new Error("the field never came back after a failed probe at load; "
                    + `painted: ${painted}`);
  }
});

await stepAsync("the choir sounds as a choir and not as a queue of solos", async () => {
  // This was one shared `<audio>` element, which holds one file at a time, so
  // nine voices came out as consecutive solos with a pause bolted between
  // them — a tuning meditation played as a playlist. The simultaneity was in
  // the log the whole time: a breath lasts up to five seconds and the next one
  // starts two to eight seconds later.
  //
  // Two breaths four seconds apart must therefore become two sources four
  // seconds apart, both sounding, because the first has not finished. The gap
  // is checked as well as the count: playing a batch all at once would make a
  // chord the voices never took.
  //
  // The single element survives for one reason — Safari grants sound to an
  // element handed to it inside a gesture, and that silent element is what
  // tells us the browser will make sound at all.
  playAllowed = true;
  choir();                        // off (crossing the threshold turned it on)
  choir();                        // on again
  sources.length = 0;
  const t = Date.now() / 1000 - 6;
  fetchImpl = async () => ({ ok: true, status: 200, text: async () => "", json: async () => ({
    size: 9, choir_size: 9, sounding_since: 100,
    recent: [
      { ts: t, who: "model", voice_id: "voice_00", name: "gpt-2",
        text: "aaa", hz: 85.5, action: "NEW", audio: "one.wav" },
      { ts: t + 4, who: "model", voice_id: "voice_02", name: "phi-4-mini",
        text: "eee", hz: 342.1, action: "TUNE", tuning_to: "voice_00",
        audio: "two.wav" },
    ],
  }) });
  await pollField();
  if (sources.length !== 2) {
    throw new Error(`two breaths should sound as two voices, got ${sources.length}`);
  }
  const apart = sources[1].startedAt - sources[0].startedAt;
  if (Math.abs(apart - 4) > 0.2) {
    throw new Error(`the choir left four seconds between them, the page played ${apart}`);
  }
  if (audios.length !== 1) {
    throw new Error(`one element should serve the session, the page built ${audios.length}`);
  }
  // End them, or they stay on screen as voices that are still holding and the
  // next test starts with somebody else's chorus under its own.
  for (const s of sources) if (s.onended) s.onended();
});

await stepAsync("a breath sung before the listener arrived is not replayed", async () => {
  // The field is not a recording. Opening the page after a quiet hour used to
  // start by playing whatever was left in the queue, so the first thing a
  // stranger heard was six breaths from an hour ago, out of order with the
  // screen.
  sources.length = 0;
  fetchImpl = async () => ({ ok: true, status: 200, text: async () => "", json: async () => ({
    size: 9, choir_size: 9, sounding_since: 100,
    recent: [
      { ts: Date.now() / 1000 - 4000, who: "model", voice_id: "voice_03",
        name: "olmo-3", text: "old", hz: 199.8, action: "NEW",
        audio: "stale.wav" },
    ],
  }) });
  await pollField();
  if (sources.length !== 0) {
    throw new Error(`a breath from an hour ago sounded ${sources.length} time(s)`);
  }
});

await stepAsync("a browser that refuses to start sound says so", async () => {
  choir();                        // off, which also clears `sounding`
  playAllowed = false;
  els.status.innerHTML = "";
  choir();                        // on again
  await pollField();
  if (!/warn/.test(els.status.innerHTML)) {
    throw new Error("the refusal was swallowed and the page said nothing");
  }
});

await stepAsync("a person is sounded from their numbers, and never their own", async () => {
  // No recording is kept or sent — see `/breath` — so this tone is built from
  // the pitch, the glide and the length alone. Without it a field full of
  // people is silent, which is what made the room feel dead.
  playAllowed = true;
  oscillators.length = 0;
  fetchImpl = async () => ({ ok: true, status: 200, text: async () => "", json: async () => ({
    size: 3, choir_size: 2, sounding_since: 100,
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

await stepAsync("a person is named in the sounding column, like the machines", async () => {
  // With twenty people in the field the screen has to show people and machines
  // together, or the room reads as nine machines and an anonymous hum. A human
  // breath takes the same column a machine breath does, under the id the choir
  // itself addresses — a machine's "↺ tuning to human_02" must point at a name
  // that has been on this screen. Never your own from the server: you were
  // named when your breath was sent, and the echo would put you up twice.
  fetchImpl = async () => ({ ok: true, status: 200, text: async () => "", json: async () => ({
    size: 3, choir_size: 2, sounding_since: 100,
    recent: [
      { ts: 301, who: "human", voice_id: "human_02", name: "a voice",
        text: "G3 196Hz · steady", hz: 196, dur: 1.5, contour: [195, 196, 197], hop: 0.25 },
      { ts: 302, who: "human", voice_id: MY_DEVICE_ID, name: "a voice",
        text: "A4 440Hz", hz: 440, dur: 1.0, contour: [440], hop: 0.25 },
    ],
  }) });
  await pollField();
  if (els.tickName.textContent !== "human_02") {
    throw new Error(`the singer should hold the top line under their own id, found "${els.tickName.textContent}"`);
  }
  if (!els.tickName.classList.contains("human")) {
    throw new Error("a person on the top line should wear the singers' colour");
  }
  if (els.tickText.textContent !== "G3 196Hz · steady") {
    throw new Error(`the line should say what and how steadily they sang, found "${els.tickText.textContent}"`);
  }
});

await stepAsync("the room can watch its breaths counting toward the school bell", async () => {
  // The school always taught and always billed — truthfully, invisibly, in a
  // log file at night. During a session that is the wrong place for it: the
  // people singing are the ones turning the counter, and they should see it
  // turn. Quiet and grey while it counts; nothing rung, nothing warm.
  fetchImpl = async () => ({ ok: true, status: 200, text: async () => "", json: async () => ({
    size: 3, choir_size: 2, sounding_since: 100, machines_now: 9, recent: [],
    school: { toward_bell: 7, needs: 20, floor_h: 3.0, bell_ts: null, lessons: [] },
  }) });
  await pollField();
  if (els.school.textContent !== "7 of 20 breaths toward the school bell") {
    throw new Error(`the counter should read plainly, found "${els.school.textContent}"`);
  }
  if (els.school.classList.contains("rung")) {
    throw new Error("a counting school must not wear the bell's colour");
  }
});

await stepAsync("a rung bell names each voice as it returns changed, with its price", async () => {
  // The one dramatic moment the schedule was rearranged to land inside a
  // session — the bell every five minutes, the lightest voices first — is
  // worthless if the screen stays silent about it. From the ring until the
  // last voice is back, the line is warm and names the latest return and what
  // it paid from its own wallet: learning is the one expense that shortens a
  // life, and it belongs next to the singing that caused it.
  const t = Date.now() / 1000;
  fetchImpl = async () => ({ ok: true, status: 200, text: async () => "", json: async () => ({
    size: 3, choir_size: 2, sounding_since: 100, machines_now: 9, recent: [],
    school: {
      toward_bell: 0, needs: 20, floor_h: 3.0, bell_ts: t - 600,
      lessons: [
        { name: "tinystories", usd: 0.02, promoted: true, ts: t - 300 },
        { name: "gpt-2", usd: 0.05, promoted: false, ts: t - 60 },
      ],
    },
  }) });
  await pollField();
  const line = els.school.textContent;
  if (!line.includes("tinystories returned changed")) {
    throw new Error(`the latest return should be named, found "${line}"`);
  }
  if (!line.includes("$0.02")) {
    throw new Error(`the price of the lesson should be on screen, found "${line}"`);
  }
  if (!line.includes("1 of 9 back")) {
    throw new Error(`only promoted lessons count as returns, found "${line}"`);
  }
  if (!els.school.classList.contains("rung")) {
    throw new Error("a rung bell should be warm on screen");
  }
});

await stepAsync("a breath that arrives twice is drawn and sounded once", async () => {
  // The first request cannot say how far it has already heard, so it asks for
  // the last sixty breaths outright. On a slow connection the five-second timer
  // fires before that answer lands and asks for the same sixty again — and a
  // doubled breath is not cosmetic: the trace is drawn twice and the person is
  // sung back twice.
  playAllowed = true;
  oscillators.length = 0;
  const t = Date.now() / 1000;
  const body = {
    size: 9, choir_size: 4, sounding_since: t - 86400,
    recent: [
      { ts: t - 3, who: "human", voice_id: "someone_else", name: "a voice",
        text: "E4 330Hz", hz: 330, dur: 1.2, contour: [328, 330, 333], hop: 0.25 },
    ],
  };
  fetchImpl = () => new Promise((res) => setTimeout(() => res({
    ok: true, status: 200, text: async () => "", json: async () => body,
  }), 5));

  await Promise.all([poll.fn(), poll.fn()]);   // overlapping, on purpose
  await new Promise((r) => setTimeout(r, 20));

  if (oscillators.length !== 1) {
    throw new Error(`one breath, sounded ${oscillators.length} times`);
  }
});

await stepAsync("the field draws both halves, including the machines with no pitch", async () => {
  // The premise of this layout, and the thing that would silently regress into
  // an empty canvas: every machine breath in the real log comes back with
  // pitch_hz = 0, so they are drawn under the pitch axis with their words. If
  // that band ever stops being painted, the page shows a few human traces and
  // silently claims the choir is not answering.
  //
  // Timestamped against this device's own clock on purpose. Everything above
  // used a fixture from 1970, which the window correctly drops — a mark older
  // than the window is not drawn, and that is the behaviour, not a bug.
  const t = Date.now() / 1000;
  fetchImpl = async () => ({ ok: true, status: 200, text: async () => "", json: async () => ({
    size: 4, choir_size: 3, sounding_since: t - 86400,
    recent: [
      { ts: t - 8, who: "human", voice_id: "someone_else", name: "a voice",
        text: "G4 392Hz", hz: 392, dur: 1.5,
        contour: [399.9, 421.4, 415.0, 372.4, 369.6], hop: 0.25 },
      // The line the whole band exists for: it answered, and named no pitch.
      { ts: t - 4, who: "model", name: "qwen3", action: "TUNE",
        text: "\u554a", hz: 0, audio: null },
    ],
  }) });
  await pollField();

  draws = [];
  const draw = frames.shift();
  if (!draw) throw new Error("the page never asked for an animation frame");
  draw();
  const texts = draws.filter((d) => d[0] === "fillText").map((d) => String(d[1]));
  if (!texts.some((t) => t === "qwen3")) {
    throw new Error(`a pitchless machine breath was not drawn; painted: ${texts}`);
  }
  if (!texts.some((t) => t.includes("\u554a"))) {
    throw new Error("a machine's words were not drawn");
  }
  if (!draws.some((d) => d[0] === "arc")) {
    throw new Error("nothing round was drawn — no onsets and no machine marks");
  }
  if (!texts.some((t) => /^[CA]\d$/.test(t))) {
    throw new Error("the equal-tempered ladder was not labelled");
  }
});

// Last of the field tests on purpose: it leaves two fresh machine breaths in
// the window, and the drawing test above needs the room for its own labels.
await stepAsync("two voices sounding together are both named on screen", async () => {
  // The sound overlapped and the words did not: the ticker was one line, so a
  // listener heard a choir and read a solo. Anything still holding gets a row
  // under the newest, and gives the row back when it stops.
  //
  // Breath lengths are wildly unequal — a held vowel renders to 48 kB and a
  // paragraph of prose to a megabyte, which is twenty-one seconds — so the long
  // one is still sounding when the short one ends, and the screen follows that.
  sources.length = 0;
  // The singers from the tests above are still holding — a person stays in the
  // column for the length of their breath, on a real timer, and the longest
  // fixture breath is 1.5 s. Waiting them out both clears the column for the
  // machine assertions below and proves that people leave it when they stop.
  await new Promise((r) => setTimeout(r, 1600));
  if (els.chorus.children.length !== 0) {
    throw new Error("a person's breath ended and they were not let out of the column");
  }
  const t = Date.now() / 1000 - 3;
  fetchImpl = async () => ({ ok: true, status: 200, text: async () => "", json: async () => ({
    size: 9, choir_size: 9, sounding_since: 100,
    recent: [
      { ts: t, who: "model", voice_id: "voice_03", name: "olmo-3",
        text: "twenty-one seconds of prose", hz: 199.8, action: "NEW",
        audio: "long.wav" },
      { ts: t + 0.02, who: "model", voice_id: "voice_06", name: "qwen3",
        text: "eee", hz: 329.6, action: "TUNE", tuning_to: "voice_03",
        audio: "short.wav" },
    ],
  }) });
  await pollField();
  if (sources.length !== 2) {
    throw new Error(`expected two voices sounding, got ${sources.length}`);
  }
  if (els.tickName.textContent !== "qwen3") {
    throw new Error(`the newest voice should hold the top line, found "${els.tickName.textContent}"`);
  }
  const rows = els.chorus.children;
  if (rows.length !== 1) {
    throw new Error(`one voice is still holding, the page showed ${rows.length} row(s)`);
  }
  if (rows[0].children[0].textContent !== "olmo-3") {
    throw new Error(`the row under it should name olmo-3, found "${rows[0].children[0].textContent}"`);
  }
  // The short breath ends. The long one is still sounding, so it takes the top
  // line back rather than leaving the screen naming something silent.
  sources[1].onended();
  if (els.tickName.textContent !== "olmo-3") {
    throw new Error(`the voice still sounding should be named, found "${els.tickName.textContent}"`);
  }
  if (els.chorus.children.length !== 0) {
    throw new Error("nothing is under the newest any more, but rows remain");
  }
  sources[0].onended();
});

await stepAsync("a breath the plumbing delayed is still sung", async () => {
  // Age is not lateness. The choir's lines become visible on the volume a
  // commit late — sometimes a spell of minutes — and a tab parked behind Zoom
  // is throttled to a poll a minute. Both used to read as "stale" and were
  // dropped unplayed, so the page had holes the server never sang. A breath
  // sung while somebody was hearing belongs to them, however late it arrives;
  // only the gaps a held-up delivery stretched out are brought back to GAP_MAX.
  //
  // The page's script is a closure, so lateness is staged with the page's own
  // clock: the newest line in the log cannot be in the future, and a line
  // stamped sixty seconds ahead moves `skew` — and the clock does not move
  // back, which is why this is the last test that makes a sound.
  sources.length = 0;
  const t = Date.now() / 1000;
  fetchImpl = async () => ({ ok: true, status: 200, text: async () => "", json: async () => ({
    size: 9, choir_size: 9, sounding_since: 100,
    recent: [
      // Sixty, thirty and twenty-five seconds "ago" once the clock lands —
      // every one of them older than the old fifteen-second cutoff.
      { ts: t, who: "model", voice_id: "voice_04", name: "qwen3",
        text: "ooo", hz: 220.0, action: "NEW", audio: "late1.wav" },
      // Thirty seconds after the first: a gap to be capped, not kept.
      { ts: t + 30, who: "model", voice_id: "voice_05", name: "eurollm",
        text: "uuu", hz: 261.6, action: "TUNE", tuning_to: "voice_04",
        audio: "late2.wav" },
      // Five seconds after the second: a gap to be kept as it was.
      { ts: t + 35, who: "model", voice_id: "voice_06", name: "llm-jp",
        text: "iii", hz: 329.6, action: "TUNE", tuning_to: "voice_05",
        audio: "late3.wav" },
      // The line that sets the clock. No audio: it only has to be newest.
      { ts: t + 60, who: "model", voice_id: "voice_08", name: "tinystories",
        text: "a word with no pitch", hz: null, action: "NEW" },
    ],
  }) });
  await pollField();
  if (sources.length !== 3) {
    throw new Error(`three delayed breaths should sound, got ${sources.length}`);
  }
  const gap1 = sources[1].startedAt - sources[0].startedAt;
  const gap2 = sources[2].startedAt - sources[1].startedAt;
  if (Math.abs(gap1 - 8) > 0.2) {
    throw new Error(`a thirty-second gap should be capped at eight, got ${gap1}`);
  }
  if (Math.abs(gap2 - 5) > 0.2) {
    throw new Error(`a five-second gap should be kept, got ${gap2}`);
  }
  for (const s of sources) if (s.onended) s.onended();
});

step("score.html is syntactically whole", () => {
  // The score is the page that has to still work in December, when there is no
  // choir left to notice it is broken. It is not exercised here — it only reads
  // the field — but a syntax error in it would leave a blank document with no
  // error anywhere, which is the same failure this whole file exists for.
  const scoreHtml = readFileSync(join(here, "score.html"), "utf8");
  const scripts = [...scoreHtml.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)];
  if (scripts.length !== 1) {
    throw new Error(`expected one inline script in score.html, found ${scripts.length}`);
  }
  new Script(scripts[0][1], { filename: "score.html" });
  for (const mark of ["\u25cf", "\u25cb"]) {
    if (!scoreHtml.includes(mark)) {
      throw new Error(`score.html lost the ${mark} mark score_render.py uses`);
    }
  }
});

step("the nine are all present, alive or not", () => {
  // A voice is never removed and a dead one is never dark: `modal/EXHIBITION.md`
  // is explicit that a black row in a gallery reads as a flat battery rather
  // than a grave. Before 18 September nothing is born, and the column still has
  // to show nine names rather than an empty margin.
  if (els.nine.children.length !== 9) {
    throw new Error(`expected 9 voices in the margin, found ${els.nine.children.length}`);
  }
});

step("headphones, once declared, keep the field at full presence", () => {
  // The duck exists to keep the choir out of the microphone, and headphones do
  // that physically — so a person who declares them gets the machines at full
  // level under their own voice, and other people's tones stop being skipped.
  // It was found the hard way: on the first Zoom-protocol test the site's sound
  // "kept appearing and disappearing" — the page ducking under every breath of
  // a singer who could not feed back to begin with.
  //
  // No API reports where a device sends its sound, so it is declared, and both
  // gates must honour the declaration — the duck and the human-tone skip.
  const src = inline[0];
  const gates = src.match(/holdingNote\(\)\s*&&\s*!headphones/g) || [];
  if (gates.length < 2) {
    throw new Error(`both gates must honour the declaration; found ${gates.length}`);
  }
  if (!src.includes('"tuning.ears"')) {
    throw new Error("the declaration does not survive a reload");
  }
  if (!src.includes("ears=")) {
    throw new Error("a session link cannot arrive with headphones already on");
  }
  const handlers = els.ears.listeners.click ?? [];
  if (handlers.length !== 1) {
    throw new Error(`expected 1 click handler on #ears, found ${handlers.length}`);
  }
  handlers[0]();
  if (els.ears.textContent !== "headphones on" || !els.ears.classList.contains("on")) {
    throw new Error("declaring headphones did not light the button");
  }
  if (sandbox.localStorage.getItem("tuning.ears") !== "on") {
    throw new Error("the declaration was not remembered for the next visit");
  }
  handlers[0]();
  if (els.ears.textContent !== "headphones off" || els.ears.classList.contains("on")) {
    throw new Error("headphones cannot be undeclared");
  }
});

if (failures) {
  console.log(`\n${failures} failure(s)`);
  process.exit(1);
}
console.log("\nfield.html loads, measures through listen.js, and draws both halves.");
