// One microphone, one gate, one idea of what a breath is.
//
// This is the listening half of the portal, lifted out of pitch.html so that a
// second page can show the field differently without measuring it differently.
// That distinction is the whole reason this file exists. `pitch.html` says it
// in its own words (§"which microphone this is"): a second implementation of
// listening would be a second set of thresholds, a second onset convention and
// a second idea of what a rest is, and the human halves of the score would
// stop being comparable in the one week they matter most (`modal/SCORE.md` §9).
// A redesign is allowed to change what a singer sees. It is not allowed to
// change what the log says.
//
// So everything that ends up in `/breath` is here — the adaptive gate, the
// utterance boundaries, the rest before, the tail, the contour, the gallery's
// stricter admission, the refusal to send a breath the speakers were in — and
// nothing here draws anything. The caller supplies rendering and playback and
// gets numbers.
//
// `pitch.html` still carries its own inline copy of this logic, because it is
// what answers `/` from 18 September and it is not worth touching eight days
// out. When the new page takes over `/`, that copy is the thing to delete: two
// implementations existing at once is exactly what this file is meant to end.
//
// Loaded as a plain script after pitch-core.js, and by test_field_page.mjs in
// node, which is why it touches no DOM and exports through globalThis.

"use strict";

(() => {
  const Core = typeof PitchCore !== "undefined"
    ? PitchCore
    : (typeof require === "function" ? require("./pitch-core.js") : null);
  if (!Core) throw new Error("listen.js needs pitch-core.js");

  const { TUNING, GALLERY, yinFrame, hzToNote, centsOff, stabilityLabel,
          fieldLine, contourOf, galleryAdmits, median, stddev } = Core;

  // human.py's fixed 0.025 gate assumes a microphone with automatic gain
  // control. We switch AGC off on purpose — it is what preserves a held tone —
  // so the raw signal can sit far below that and no breath is ever registered.
  // Track the room's own noise floor instead and open the gate just above it.
  const PHONE = {
    GATE_FLOOR: 0.0035,
    GATE_CEIL: TUNING.VAD_RMS_THRESHOLD,
    GATE_OVER_NOISE: 3.5,
    // How far above the choir's own leakage a breath has to sit before it is
    // believed to be somebody singing. Roughly 10 dB.
    //
    // This replaces a flat refusal. "The choir was audible, so this breath is
    // not yours" is true of a laptop across a room and false of a phone held to
    // a mouth, and the piece needs the second case to work: a person tuning to
    // a machine has to be able to hear the pitch they are matching, so the
    // choir is playing *by design* while they sing. Refusing on audibility lost
    // real breaths; not refusing at all logged the choir's own pitches as a
    // person's, which is worse — the field log is the evidence this whole work
    // rests on. Loudness separates them, and it separates them by a wide margin:
    // leakage out of a phone's own speaker arrives tens of decibels under a
    // voice singing into that phone from a few centimetres away.
    BLEED_OVER: 3.0,
  };

  /**
   * A stable but anonymous name for this device, so the field can tell singers
   * apart without knowing who anyone is. Never leaves localStorage except as
   * this opaque string.
   *
   * `human_` and not `voice_`: the choir's own nine are `voice_00`…`voice_08`,
   * and a phone calling itself `voice_7f3a1b2c` collided with them. A stored
   * old name is carried over rather than replaced, so a singer who comes back
   * is the same part they were.
   */
  function deviceId() {
    const key = "tuning.voice_id";
    const fresh = () => "human_" + Math.random().toString(36).slice(2, 10);
    // Private browsing and blocked storage make localStorage throw, not return
    // null. Falling back to a per-visit name keeps the page alive.
    try {
      let id = localStorage.getItem(key);
      if (id && id.startsWith("voice_")) id = "human_" + id.slice(6);
      if (!id) id = fresh();
      localStorage.setItem(key, id);
      return id;
    } catch {
      return fresh();
    }
  }

  // Every one of these is off on purpose. Browser noise suppression is a
  // speech model: it hears a sustained vowel and the tone of a room as noise
  // and removes exactly the material this piece is made of. If a browser
  // refuses the raw ask we would still rather have the microphone than
  // nothing, so we walk down to plainer constraints and say so afterwards.
  const CONSTRAINTS = [
    { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    true,
  ];

  async function openMicrophone() {
    let lastErr = null;
    for (const audio of CONSTRAINTS) {
      try {
        return await navigator.mediaDevices.getUserMedia({ audio });
      } catch (err) {
        lastErr = err;
        // A refusal or a missing device will not be fixed by asking for less.
        if (err.name === "NotAllowedError" || err.name === "NotFoundError") throw err;
      }
    }
    throw lastErr;
  }

  /** Box-filter and decimate in one pass — crude anti-aliasing, cheap enough. */
  function downsample(src, dst, factor) {
    const n = Math.floor(src.length / factor);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      const base = i * factor;
      for (let k = 0; k < factor; k++) sum += src[base + k];
      dst[i] = sum / factor;
    }
    return n;
  }

  /**
   * Create a listener.
   *
   * @param {object} o
   * @param {boolean} [o.gallery]           the open microphone in the cube
   * @param {() => boolean} [o.speakerActive] is this device making sound right now
   * @param {(payload: object) => Promise<boolean>} [o.postBreath]
   * @param {(f: object) => void} [o.onFrame]      every animation frame
   * @param {(u: object) => void} [o.onUtterance]  one finished breath
   * @param {(s: object) => void} [o.onState]      running / stopped / settings
   * @param {() => void} [o.onVoicedEnd]           a held note has just stopped
   */
  function create(o = {}) {
    const gallery = !!o.gallery;
    const speakerActive = o.speakerActive || (() => false);
    const postBreath = o.postBreath || (async () => false);
    const onFrame = o.onFrame || (() => {});
    const onUtterance = o.onUtterance || (() => {});
    const onState = o.onState || (() => {});
    const onVoicedEnd = o.onVoicedEnd || (() => {});

    const GATE_FLOOR = gallery ? GALLERY.GATE_FLOOR : PHONE.GATE_FLOOR;
    const GATE_CEIL = gallery ? GALLERY.GATE_CEIL : PHONE.GATE_CEIL;
    const GATE_OVER_NOISE = gallery ? GALLERY.GATE_OVER_NOISE : PHONE.GATE_OVER_NOISE;

    const DEVICE_ID = deviceId();

    let noiseFloor = 0.01;
    let gate = GATE_CEIL;

    let ctx = null, stream = null, analyser = null, rawBuf = null, workBuf = null;
    let running = false, raf = 0;
    let workRate = TUNING.TARGET_RATE, decimation = 1;
    let tauMin = 0, tauMax = 0, frameLen = 0;

    // The utterance currently being sung.
    let utterPitches = [], utterFrames = 0, utterStart = 0, lastVoiced = 0;
    // The loudest frame in it. Only the room asks: a near-field microphone
    // with somebody singing into it sits far above a hall that merely has
    // people in it, and that difference is the one measurement that separates
    // the two.
    let utterPeak = 0;
    let inUtterance = false;
    let utterHeardChoir = false;   // the choir was audible while this was sung
    // How loud this device's own playback arrives back down its own microphone.
    // Learned from the microphone rather than assumed, because it depends on the
    // volume, on the phone, and on where the phone is being held, and none of
    // those are knowable from here.
    let bleedFloor = 0;
    // When this singer's previous breath stopped, on the audio clock. The rest
    // between two of somebody's own breaths is theirs and is measured here
    // rather than derived from arrival times on the server: the network is not
    // part of the music. Reset with the audio context, whose clock restarts.
    let lastUtterEnd = 0;

    let voicedNow = false, voicedSince = 0;

    // Who is at the room's microphone. There is no login, no camera and no way
    // to tell one visitor from the next, so "a person" is not a thing this data
    // can support. What the microphone can honestly report is a **turn at it**:
    // sound arriving with gaps shorter than a minute and a half is one
    // approach. See `modal/SCORE.md` §9.
    let roomTurn = "";
    let roomTurnEnd = 0;

    function singerAt(onset) {
      if (!gallery) return DEVICE_ID;
      if (!roomTurn || (roomTurnEnd && onset - roomTurnEnd > GALLERY.SESSION_GAP_S)) {
        roomTurn = "human_room_" + Math.random().toString(16).slice(2, 6);
        // A new turn has no rest before it. The gap belongs to the room, not
        // to the person who has just walked up to the microphone.
        lastUtterEnd = 0;
      }
      return roomTurn;
    }

    // Breath onsets already accepted, on the audio clock, for the one-minute
    // ceiling. A cheap last defence: if the room ever starts singing into its
    // own microphone, the field gains twenty lines and not a night's.
    const accepted = [];

    function flooding(now) {
      while (accepted.length && now - accepted[0] > 60) accepted.shift();
      return accepted.length >= GALLERY.MAX_BREATHS_PER_MIN
        ? "the microphone is flooding — held back" : "";
    }

    /**
     * Whether somebody is holding a note *now*, as opposed to sitting in a
     * room the microphone finds loud. A laptop in a café stays above the gate
     * indefinitely, and a plain `voicedNow` check would then suppress the
     * choir forever without ever saying so. No note outlasts MAX_UTTERANCE_S,
     * so neither does the hush.
     */
    function holdingNote() {
      if (!voicedNow) return false;
      return (ctx ? ctx.currentTime : 0) - voicedSince < TUNING.MAX_UTTERANCE_S;
    }

    /** Tell the singer whether the browser honoured the raw-audio ask. */
    function settings() {
      const s = stream.getAudioTracks()[0].getSettings?.() ?? {};
      const kept = [];
      if (s.noiseSuppression) kept.push("noise suppression");
      if (s.autoGainControl) kept.push("auto gain");
      if (s.echoCancellation) kept.push("echo cancellation");
      return kept;
    }

    async function start() {
      if (!window.isSecureContext) throw new Error("insecure-context");
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("no-getusermedia");

      // Safari ties microphone and audio permission to the click that started
      // this, and an await before constructing the AudioContext can lose that
      // gesture — so build it first and tear it down again if the mic fails.
      const Ctx = window.AudioContext || window.webkitAudioContext;
      try {
        ctx = new Ctx({ sampleRate: TUNING.TARGET_RATE });
      } catch {
        ctx = new Ctx();          // Safari may refuse a non-native rate
      }
      // A new context restarts the clock the rests are measured on, so the
      // first breath after the microphone is reopened has no measurable
      // before. Sending nothing is right: the gap really is unknown, and the
      // log should not guess.
      lastUtterEnd = 0;
      roomTurn = "";
      roomTurnEnd = 0;
      accepted.length = 0;

      try {
        stream = await openMicrophone();
      } catch (err) {
        ctx.close();
        ctx = null;
        throw err;
      }
      await ctx.resume();

      // Work near 16 kHz whatever the hardware gives us: YIN costs
      // O(tauMax x frame), which at 48 kHz is nine times the work for no
      // extra accuracy on a sung voice.
      decimation = Math.max(1, Math.round(ctx.sampleRate / TUNING.TARGET_RATE));
      workRate = ctx.sampleRate / decimation;

      analyser = ctx.createAnalyser();
      analyser.fftSize = 4096;     // >= 50 ms of audio even at 48 kHz
      analyser.smoothingTimeConstant = 0;
      ctx.createMediaStreamSource(stream).connect(analyser);

      rawBuf = new Float32Array(analyser.fftSize);
      workBuf = new Float32Array(Math.ceil(analyser.fftSize / decimation));
      frameLen = Math.floor(workRate * TUNING.FRAME_S);
      tauMin = Math.floor(workRate / TUNING.PITCH_FMAX_HZ);
      tauMax = Math.floor(workRate / TUNING.PITCH_FMIN_HZ);

      running = true;
      onState({
        running: true,
        hardwareRate: ctx.sampleRate,
        workRate,
        kept: settings(),
      });
      loop();
    }

    function stop() {
      running = false;
      if (typeof cancelAnimationFrame === "function") cancelAnimationFrame(raf);
      if (inUtterance) emitUtterance();
      if (stream) stream.getTracks().forEach((t) => t.stop());
      if (ctx) ctx.close();
      ctx = stream = analyser = null;
      voicedNow = false;
      onState({ running: false });
    }

    function loop() {
      if (!running) return;
      raf = requestAnimationFrame(loop);

      analyser.getFloatTimeDomainData(rawBuf);

      let work, n;
      if (decimation > 1) { n = downsample(rawBuf, workBuf, decimation); work = workBuf; }
      else { n = rawBuf.length; work = rawBuf; }
      if (n < frameLen) return;
      const frame = work.subarray(n - frameLen, n);

      let peak = 0, sumsq = 0;
      for (let i = 0; i < frame.length; i++) {
        const v = frame[i];
        const a = v < 0 ? -v : v;
        if (a > peak) peak = a;
        sumsq += v * v;
      }
      const rms = Math.sqrt(sumsq / frame.length);
      const now = ctx.currentTime;

      // Follow the room: fall to quiet quickly, rise slowly, so a sustained
      // note never gets mistaken for the noise floor it has to be measured
      // against.
      noiseFloor = rms < noiseFloor
        ? noiseFloor * 0.90 + rms * 0.10
        : noiseFloor * 0.9995 + rms * 0.0005;
      gate = Math.min(GATE_CEIL, Math.max(GATE_FLOOR, noiseFloor * GATE_OVER_NOISE));

      // The same shape as the noise floor above, measured off a different
      // source: while this device is making sound, most frames are the choir
      // arriving back down the microphone with nobody singing over it. Falling
      // fast and rising slowly is what makes that the number this converges to —
      // a person singing over the choir cannot drag it up to their own level.
      if (speakerActive()) {
        bleedFloor = !bleedFloor ? rms
          : rms < bleedFloor ? bleedFloor * 0.90 + rms * 0.10
          : bleedFloor * 0.995 + rms * 0.005;
      }

      let hz = 0;
      if (peak >= GATE_FLOOR) {
        hz = yinFrame(frame, workRate, tauMin, tauMax);
        if (hz < TUNING.PITCH_FMIN_HZ || hz > TUNING.PITCH_FMAX_HZ) hz = 0;
      }
      const voiced = rms >= gate;

      // Let a held note finish before the choir answers it.
      if (voiced && !voicedNow) voicedSince = now;
      if (voicedNow && !voiced) onVoicedEnd();
      voicedNow = voiced;

      if (voiced) {
        if (!inUtterance) {
          inUtterance = true; utterStart = now; utterPitches = []; utterFrames = 0;
          utterHeardChoir = false; utterPeak = 0;
        }
        // The choir, or another person sounded from their numbers: either way
        // it came out of these speakers and back down this microphone, and a
        // breath that contains it is not this singer's.
        if (speakerActive()) utterHeardChoir = true;
        lastVoiced = now;
        utterFrames++;
        if (rms > utterPeak) utterPeak = rms;
        if (hz > 0) utterPitches.push(hz);
        if (now - utterStart >= TUNING.MAX_UTTERANCE_S) emitUtterance();
      } else if (inUtterance && now - lastVoiced >= TUNING.SILENCE_TAIL_S) {
        emitUtterance();
      }

      onFrame({
        hz, rms, gate, voiced, now,
        note: hz > 0 && voiced ? hzToNote(hz) : "",
        cents: hz > 0 && voiced ? centsOff(hz) : 0,
        inUtterance,
      });
    }

    function emitUtterance() {
      const dur = lastVoiced - utterStart;
      const pitches = utterPitches;
      const frames = utterFrames;
      const peak = utterPeak;
      // How long ago the sound stopped. The page waits out half a second of
      // silence before it believes a note is over, so without this every onset
      // in the score would sit half a second late — a fixed error, in the one
      // dimension the piece is about, that nothing downstream could detect.
      const tail = Math.max(0, (ctx ? ctx.currentTime : lastVoiced) - lastVoiced);
      inUtterance = false;
      utterPitches = [];
      utterPeak = 0;
      // Said, not swallowed. This used to `return` in silence, so a stranger
      // who sang a short "a" saw nothing happen anywhere on the page and had no
      // way to learn that the field wanted a longer note — the difference
      // between getting a first breath out of somebody and losing them.
      if (dur < TUNING.MIN_UTTERANCE_S || pitches.length === 0) {
        onUtterance({
          state: "refused",
          reason: pitches.length === 0
            ? "no pitch in that — the field takes a sung note, not a word"
            : `too short — hold one note for about ${TUNING.MIN_UTTERANCE_S.toFixed(1)}s`,
          line: "", note: "", hz: 0, cents: 0, stability: "",
          dur: Math.max(0, dur), contour: [],
        });
        return;
      }

      const med = median(pitches);
      const sd = stddev(pitches);
      const voicedFrac = pitches.length / Math.max(1, frames);
      const contour = contourOf(pitches, dur);

      // This is exactly the line human.py puts into the shared field.
      const u = {
        line: fieldLine(med, sd),
        note: hzToNote(med),
        hz: med,
        cents: centsOff(med),
        stability: stabilityLabel(sd, med),
        dur,
        contour,
        state: "pending",
        reason: "",
      };

      // Heard through the speakers rather than sung: keep it out of the field.
      //
      // Decided by loudness and not by audibility. The choir plays while people
      // sing on purpose — that is how anybody tunes to it — so "the choir was
      // audible" cannot be the test. What separates the two cases is level: a
      // voice into the microphone it is singing into sits far above the same
      // device's own leakage. Below that margin, this is the choir listening to
      // itself, and the pitch would go into the log as a person's.
      if (utterHeardChoir && bleedFloor && peak < bleedFloor * PHONE.BLEED_OVER) {
        u.state = "refused";
        u.reason = "that was the choir coming back down the microphone — "
                 + "sing closer, or put headphones on";
        onUtterance(u);
        return;
      }

      // In the hall, everything that is not clearly a person singing into the
      // microphone stops here and is written nowhere. The reason is reported,
      // which is how the gate gets set against the real room during install.
      if (gallery) {
        const refused = galleryAdmits({
          durationS: dur, voicedFraction: voicedFrac,
          medianHz: med, stdHz: sd, peakRms: peak,
        }) || flooding(lastVoiced);
        if (refused) {
          u.state = "refused";
          u.reason = refused;
          onUtterance(u);
          return;
        }
      }

      // Who sang it. On a phone, this device; in the hall, whoever is
      // currently at the microphone — and asking may open a new turn, which
      // clears the rest before, so it is asked before that rest is measured.
      const singer = singerAt(utterStart);
      const restBefore = lastUtterEnd > 0 ? Math.max(0, utterStart - lastUtterEnd) : null;

      // This breath is now the one the next rest is measured from, whether or
      // not the field is online: a singer alone on a bad connection still has
      // a rhythm.
      lastUtterEnd = lastVoiced;
      roomTurnEnd = lastVoiced;
      accepted.push(lastVoiced);

      u.state = "sending";
      u.voiceId = singer;
      onUtterance(u);

      postBreath({
        voice_id: singer,
        source: gallery ? "gallery" : "portal",
        note: u.note,
        hz: med,
        cents: u.cents,
        stability: u.stability,
        duration_s: dur,
        voiced_fraction: voicedFrac,
        // The score's four additions. Measurements, all of them, and still no
        // audio: a spread, a trace of the pitch, and the two times that place
        // the note and this singer's own rest before it. See modal/SCORE.md.
        std_hz: sd,
        contour_hz: contour,
        tail_s: tail,
        own_rest_s: restBefore,
      }).then((ok) => {
        u.state = ok ? "sent" : "failed";
        onUtterance(u);
      });
    }

    return {
      start,
      stop,
      holdingNote,
      deviceId: () => DEVICE_ID,
      isRunning: () => running,
      isGallery: () => gallery,
    };
  }

  /** Say what went wrong in words that point at the fix. */
  function explain(err, where) {
    switch (err.message === "insecure-context" ? "insecure" :
            err.message === "no-getusermedia" ? "missing" : err.name) {
      case "insecure":
        return `<b>This page is not a secure context</b> (${where}). A microphone
                only opens over https, or on localhost.`;
      case "missing":
        return `<b>This browser exposes no getUserMedia</b> at ${where}.`;
      case "NotAllowedError":
        return `<b>Permission refused.</b> Two places to check: the padlock in
                the address bar for this site, and <b>System Settings → Privacy
                &amp; Security → Microphone</b>, where your browser itself needs
                the tick. macOS asks once and then silently refuses forever.`;
      case "NotFoundError":
        return `<b>No input device.</b> The browser sees no microphone at all.`;
      case "NotReadableError":
        return `<b>The microphone is there but cannot be read.</b> Usually
                another app is holding it.`;
      case "OverconstrainedError":
        return `<b>No device matches the request</b> (${err.constraint || "unknown"}).`;
      case "SecurityError":
        return `<b>Blocked by policy.</b> Microphone access is disabled for ${where}.`;
      default:
        return `<b>${err.name || "Error"}:</b> ${err.message}`;
    }
  }

  const PitchListen = { create, explain, PHONE, deviceId };

  if (typeof module !== "undefined" && module.exports) module.exports = PitchListen;
  else globalThis.PitchListen = PitchListen;
})();
