"""Can a voice hold a vowel at a named pitch? A probe, not a change.

    .venv-modal/bin/modal run portal/music_probe/sing_probe.py

Nothing here touches the choir. It renders the same breath three ways and
writes the wavs next to this file so they can be compared by ear:

    current_*   what the choir does today — Kokoro speaks, a sine holds the pitch
    sustain_*   the voice's own timbre held as one vowel at an exact frequency
    chant_*     the voice's actual sentence, sung on that one pitch

The mechanism is source-filter resynthesis (WORLD). Kokoro is asked for a
vowel, WORLD separates what the voice *sounds like* (spectral envelope,
aperiodicity) from what pitch it happened to speak at, and the pitch is then
replaced with the one the model asked for. The timbre is untouched, so the
nine fixed Kokoro identities in cast.py survive intact — this makes the voices
we already have sing, rather than importing nine strangers who can.

Pitch is exact by construction: F0 is written, not predicted. The probe
measures the result anyway and reports the error in cents, because a claim
about tuning in a piece about tuning should be checked.
"""

from __future__ import annotations

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

app = modal.App("tuning-music-probe")

SR = 24_000                 # Kokoro's sample rate
FRAME_MS = 5.0              # WORLD's frame period

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("espeak-ng", "libsndfile1")
    .pip_install(
        "kokoro>=0.9.4",
        "misaki[zh,ja]",
        "pyworld",
        "soundfile",
        "numpy<2",
        "torch",
    )
    # pyworld still does `import pkg_resources` at load, which Python 3.12 no
    # longer ships. Its own layer, after everything else, so nothing can
    # resolve setuptools back out again.
    .run_commands("pip install --no-cache-dir 'setuptools<81'")
)

# Four of the nine, chosen to be as unalike as the cast is: the elder's low
# male voice, an English female, the Italian, the Chinese. If four read as four
# beings, nine will.
VOICES = [
    # short_name,   kokoro,        vowel,  a real sentence in its own tongue
    ("gpt-2",       "am_onyx",     "ah",   "a long tone, warm and low"),
    ("olmo-3",      "af_heart",    "ah",   "I am holding D four, listening"),
    ("eurollm",     "if_sara",     "ah",   "tengo un tono lungo, ascolto"),
    ("qwen3",       "zf_xiaoxiao", "啊",    "我唱一个长音"),
]

# Two pitches a person could actually tune to, an octave apart in feel.
PITCHES = [("A3", 220.00), ("D4", 293.66)]


def _fade(wave, sr: int, ms: float = 60.0):
    import numpy as np

    n = min(int(sr * ms / 1000), len(wave) // 2)
    if n <= 0:
        return wave
    ramp = np.linspace(0.0, 1.0, n)
    wave[:n] *= ramp
    wave[-n:] *= ramp[::-1]
    return wave


def _sine(hz: float, seconds: float, sr: int, amp: float = 0.35):
    """The tone the choir uses today, copied from choir_app so the A/B is fair."""
    import numpy as np

    t = np.arange(int(sr * seconds)) / sr
    return _fade((amp * np.sin(2 * np.pi * hz * t)).astype("float64"), sr, 20.0)


@app.function(image=image, timeout=1800)
def probe() -> dict[str, bytes]:
    import io

    import numpy as np
    import pyworld as pw
    import soundfile as sf
    from kokoro import KPipeline

    pipelines: dict[str, object] = {}

    def speak(text: str, kokoro_voice: str):
        lang = kokoro_voice[0]
        if lang not in pipelines:
            pipelines[lang] = KPipeline(lang_code=lang)
        chunks = [a for _, _, a in pipelines[lang](text, voice=kokoro_voice)]
        if not chunks:
            return None
        return np.concatenate(chunks).astype("float64")

    def analyse(x):
        """What the voice sounds like, separated from what pitch it spoke at."""
        f0, t = pw.harvest(x, SR, frame_period=FRAME_MS)
        f0 = pw.stonemask(x, f0, t, SR)
        sp = pw.cheaptrick(x, f0, t, SR)
        ap = pw.d4c(x, f0, t, SR)
        return f0, sp, ap

    def longest_voiced(f0):
        """The steadiest stretch of vowel in the clip, as a slice."""
        voiced = f0 > 0
        best = cur = None
        for i, v in enumerate(voiced):
            if v and cur is None:
                cur = i
            elif not v and cur is not None:
                if best is None or i - cur > best[1] - best[0]:
                    best = (cur, i)
                cur = None
        if cur is not None and (best is None or len(f0) - cur > best[1] - best[0]):
            best = (cur, len(f0))
        return best

    def sustain(x, hz: float, seconds: float):
        """One vowel, this voice's timbre, held at exactly `hz`."""
        f0, sp, ap = analyse(x)
        span = longest_voiced(f0)
        if span is None:
            return None
        a, b = span
        # Trim the attack and release; the middle is where the vowel is itself.
        pad = max(1, (b - a) // 5)
        a, b = a + pad, max(a + pad + 2, b - pad)
        core_sp, core_ap = sp[a:b], ap[a:b]

        n = int(seconds * 1000 / FRAME_MS)
        # Ping-pong through the vowel rather than freezing one frame: a single
        # repeated spectrum sounds like an organ, the small drift of a real
        # held vowel is what makes it sound like someone holding it.
        idx = np.arange(n) % max(1, 2 * len(core_sp) - 2)
        idx = np.where(idx < len(core_sp), idx, 2 * len(core_sp) - 2 - idx)

        t = np.arange(n) * FRAME_MS / 1000.0
        # ±8 cents at 5 Hz. Enough to be a voice, far too little to be out of tune.
        vibrato = 1.0 + 0.0046 * np.sin(2 * np.pi * 5.0 * t)
        out_f0 = hz * vibrato

        y = pw.synthesize(out_f0, core_sp[idx].copy(order="C"),
                          core_ap[idx].copy(order="C"), SR, FRAME_MS)
        return _fade(y, SR, 120.0)

    def chant(x, hz: float):
        """The sentence as spoken, but all of it on one pitch."""
        f0, sp, ap = analyse(x)
        out_f0 = np.where(f0 > 0, hz, 0.0)
        y = pw.synthesize(out_f0, sp, ap, SR, FRAME_MS)
        return _fade(y, SR, 40.0)

    def measured_hz(y) -> float:
        f0, t = pw.harvest(np.ascontiguousarray(y), SR, frame_period=FRAME_MS)
        voiced = f0[f0 > 0]
        return float(np.median(voiced)) if len(voiced) else 0.0

    def to_wav(y) -> bytes:
        peak = float(np.max(np.abs(y))) or 1.0
        buf = io.BytesIO()
        sf.write(buf, (y * (0.89 / peak)).astype("float32"), SR,
                 format="WAV", subtype="PCM_16")
        return buf.getvalue()

    out: dict[str, bytes] = {}
    report: list[str] = []

    for short, kokoro, vowel, sentence in VOICES:
        vowel_audio = speak(vowel, kokoro)
        speech = speak(sentence, kokoro)
        if vowel_audio is None or speech is None:
            report.append(f"{short}: Kokoro rendered nothing")
            continue

        for note, hz in PITCHES:
            try:
                held = sustain(vowel_audio, hz, seconds=4.0)
            except Exception as err:
                report.append(f"{short} {note}: sustain failed — {err}")
                held = None
            if held is not None:
                out[f"sustain_{short}_{note}.wav"] = to_wav(held)
                got = measured_hz(held)
                cents = 1200 * np.log2(got / hz) if got else float("nan")
                report.append(f"{short:<10} {note:<3} held  asked {hz:7.2f} Hz, "
                              f"measured {got:7.2f} Hz  ({cents:+.1f} cents)")

        # One chant and one of today's rendering per voice, at D4 only —
        # enough to hear the difference without four times the files.
        note, hz = PITCHES[1]
        try:
            sung = chant(speech, hz)
            out[f"chant_{short}_{note}.wav"] = to_wav(sung)
            got = measured_hz(sung)
            cents = 1200 * np.log2(got / hz) if got else float("nan")
            report.append(f"{short:<10} {note:<3} chant asked {hz:7.2f} Hz, "
                          f"measured {got:7.2f} Hz  ({cents:+.1f} cents)")
        except Exception as err:
            report.append(f"{short} {note}: chant failed — {err}")

        # Today's rendering, for the A/B: speech laid over a sine.
        tone = _sine(hz, max(4.0, len(speech) / SR), SR)
        n = max(len(tone), len(speech))
        mix = np.zeros(n)
        mix[:len(tone)] += tone
        mix[:len(speech)] += speech * 0.9
        out[f"current_{short}_{note}.wav"] = to_wav(mix)

    print("\n" + "=" * 72)
    for line in report:
        print(line)
    print("=" * 72)
    return out


@app.local_entrypoint()
def main():
    written = probe.remote()
    for name, data in written.items():
        (HERE / name).write_bytes(data)
    print(f"\n{len(written)} files written to {HERE}")
    for name in sorted(written):
        print(f"  {name}")
