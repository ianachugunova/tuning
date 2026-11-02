"""Does the choir's own throat code work, and what does it cost? A probe.

    .venv-modal/bin/modal run portal/music_probe/throat_probe.py

The gate between `sing_probe.py`, which proved WORLD could make these voices
sing, and the choir actually doing it. Three things have to hold before an A100
is asked to run this:

  * **pyworld builds inside the choir's own image.** It ships no Linux wheel
    and compiles from an sdist whose setup still imports `pkg_resources`, next
    to torch and transformers and a numpy nobody pinned.
  * **The pitch is exact.** Measured back out of the audio, in cents, because
    a claim about tuning in a piece about tuning should be checked and not
    asserted.
  * **A breath does not get slower.** Every second here is charged to the
    voice that sang, and the wallets are calibrated on breath times. A body
    that costs a second a breath is a body that changes who dies first.

Runs on a CPU, imports the real functions rather than copies of them, and
writes its wavs next to this file so the result is also judged by ear.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# The point of this probe is that it runs the choir's real code in the choir's
# real image, so it imports the app itself rather than a copy — and `modal run`
# puts this file's directory on the path, not its parent's.
sys.path.insert(0, str(HERE.parent))

import choir_app  # noqa: E402

app = modal.App("tuning-throat-probe")

SR = choir_app.SR
D4 = 293.66

# The choir's image, unchanged, plus this file and the cast beside it. If it
# builds here it builds there, because it is the same image.
image = choir_app.image.add_local_file(
    Path(choir_app.__file__), remote_path="/root/choir_app.py"
)


@app.function(image=image, timeout=1800)
def probe() -> dict[str, bytes]:
    import io

    import numpy as np
    import soundfile as sf
    import pyworld as pw
    from kokoro import KPipeline

    import cast as cast_mod
    from choir_app import _chant, _sine, _sustain, _throat, _vowel_for

    print(f"numpy {np.__version__}, pyworld imported")

    pipelines: dict[str, KPipeline] = {}

    def speak(text: str, preset: str):
        lang = preset[0]
        if lang not in pipelines:
            pipelines[lang] = KPipeline(lang_code=lang)
        chunks = [a for _, _, a in pipelines[lang](text, voice=preset)]
        return np.concatenate(chunks).astype("float64") if chunks else None

    def measured(y) -> float:
        f0, _ = pw.harvest(np.ascontiguousarray(y, dtype="float64"), SR,
                           frame_period=choir_app.FRAME_MS)
        voiced = f0[f0 > 0]
        return float(np.median(voiced)) if voiced.size else 0.0

    def to_wav(y) -> bytes:
        peak = float(np.max(np.abs(y))) or 1.0
        buf = io.BytesIO()
        sf.write(buf, (y * (0.89 / peak)).astype("float32"), SR,
                 format="WAV", subtype="PCM_16")
        return buf.getvalue()

    out: dict[str, bytes] = {}
    rows: list[str] = []

    # What each voice would say if it were tuning, in the tongue it was raised
    # in — because the chant has to be tested on the consonants each voice
    # actually has, not on an English sentence for all of them.
    LINES = {
        "gpt-2": "a long tone, warm and low",
        "tinystories": "I sing a little note",
        "phi-4-mini": "holding D four, listening",
        "olmo-3": "I am holding D four, listening",
        "eurollm": "tengo un tono lungo, ascolto",
        "qwen3": "我唱一个长音",
    }

    for v in cast_mod.CHOIR:
        if not v.kokoro:
            continue
        vowel_text = _vowel_for(v.kokoro)

        t0 = time.time()
        vowel = speak(vowel_text, v.kokoro)
        if vowel is None:
            rows.append(f"{v.short_name:<14} Kokoro rendered no vowel")
            continue
        throat = _throat(vowel)
        measure_s = time.time() - t0
        if throat is None:
            rows.append(f"{v.short_name:<14} no voiced frame to measure")
            continue

        # A four-second breath, the way the choir will ask for one.
        t0 = time.time()
        held = _sustain(throat, D4, 4.0)
        hold_s = time.time() - t0
        got = measured(held)
        cents = 1200 * np.log2(got / D4) if got else float("nan")
        out[f"throat_{v.short_name}_sustain_D4.wav"] = to_wav(held)

        line = LINES.get(v.short_name, "a long tone")
        speech = speak(line, v.kokoro)
        t0 = time.time()
        sung = _chant(speech, D4) if speech is not None else None
        chant_s = time.time() - t0
        if sung is not None:
            out[f"throat_{v.short_name}_chant_D4.wav"] = to_wav(sung)
            # Today's rendering of the same line, for the A/B by ear.
            tone = _sine(D4, max(4.0, len(speech) / SR)).astype("float64")
            n = max(len(tone), len(speech))
            mix = np.zeros(n)
            mix[:len(tone)] += tone
            mix[:len(speech)] += speech * 0.9
            out[f"throat_{v.short_name}_today_D4.wav"] = to_wav(mix)

        rows.append(
            f"{v.short_name:<14} vowel {vowel_text!r:<6} "
            f"measure {measure_s:5.2f}s  hold {hold_s:5.2f}s  "
            f"chant {chant_s:5.2f}s   asked {D4:.2f} Hz, got {got:7.2f} Hz "
            f"({cents:+.1f} cents)"
        )

    print("\n" + "=" * 78)
    print("  measure — once per voice, ever. hold and chant — every breath.")
    for r in rows:
        print("  " + r)
    print("=" * 78)
    return out


@app.local_entrypoint()
def main():
    written = probe.remote()
    for name, data in written.items():
        (HERE / name).write_bytes(data)
    print(f"\n{len(written)} files written to {HERE}")
