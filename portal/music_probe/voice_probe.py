"""Can a voice be cast instead of picked from a list? A probe of VoxCPM2.

    .venv-modal/bin/modal run portal/music_probe/voice_probe.py

Kokoro has 54 presets and no child among them, so the voice raised on stories
told in the words of a four-year-old speaks like a woman in her thirties. That
is not a setting that was chosen badly; it is a hole in the set. VoxCPM2 has no
list at all — a voice is written as a sentence of casting notes and generated
from the description, which means the cast stops being nine picks from a menu
and becomes nine descriptions, in the same register as the `raised_by` line
each voice already carries.

Four things have to hold before that is worth doing, and only the first is the
one that was asked for:

  * **A child comes out when a child is described.** Four notes, one line, and
    the ear decides.
  * **It is the same child tomorrow.** A designed voice is sampled, and a voice
    whose body changes every breath is not a person — it is a stranger with the
    same name. Measured here as the spread of median pitch and spectral centre
    across five different sentences.
  * **Freezing fixes the drift if there is drift.** Design once, keep the wav,
    then clone from it forever. If the spread collapses, this is the
    architecture and the designed clip becomes an asset like a set of weights.
  * **It is affordable.** Kokoro is 82M and effectively free; this is 2B. The
    number that matters is seconds of compute per second of audio, because
    every one of them is charged to the voice that spoke.

Writes its wavs next to this file. The table is evidence; the wavs are the
decision.
"""

from __future__ import annotations

import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

app = modal.App("tuning-voice-probe")

# Its own volume slot alongside the choir's, so a 2B download survives a failed
# run and the second attempt costs nothing.
weights_volume = modal.Volume.from_name("tuning-weights", create_if_missing=True)
CACHE_DIR = "/weights"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1", "ffmpeg", "git")
    .pip_install("voxcpm", "soundfile", "numpy")
    # The same compiled-from-sdist story as in the choir's image, and used here
    # for the same reason: harvest is what measures a pitch honestly.
    .run_commands(
        "pip install --no-cache-dir 'setuptools<81'",
        "pip install --no-cache-dir pyworld",
        "python -c \"import pyworld, numpy; print('pyworld ok on numpy', numpy.__version__)\"",
    )
    .env({"HF_HOME": CACHE_DIR})
)

# One sentence, said by every candidate, so the only difference between the
# files is the body. Short, because a child saying a long clause is a different
# test — of the language model that isn't there.
CHILD_LINE = "I sing a little note, and I wait for yours."

# Casting notes for the four-year-old. Identity, texture, delivery — the three
# layers VoxCPM2's own guide asks for. They differ on purpose: the literal
# child, the older child who can hold a tone, the child recorded too close, and
# the one that describes the corpus rather than a person.
CHILD_NOTES = {
    "child_small": (
        "A four-year-old child, a very small bright voice, thin and light with "
        "a lot of air in it, speaking simply and slowly, as if telling a story "
        "it is making up while it says it"
    ),
    "child_six": (
        "A six-year-old child, clear high voice with a soft breathy edge, "
        "unhurried and even, speaking carefully as if trying to get the words "
        "right"
    ),
    "child_close": (
        "A young child speaking very quietly and close to the microphone, "
        "small thin voice, breathy and intimate, almost whispering, with long "
        "calm pauses"
    ),
    "child_flat": (
        "A small child with a high thin voice, speaking flatly and steadily "
        "with almost no expression, calm and unhurried, as if reading aloud a "
        "sentence it does not fully understand"
    ),
}

# Five different sentences for the same designed body. If the body is stable,
# these five clips are one person; if it is not, they are five children.
STABILITY_LINES = [
    "I sing a little note, and I wait for yours.",
    "The room is quiet now.",
    "I can hear someone else holding a tone.",
    "My voice is going down, slowly.",
    "I will stop, and listen.",
]

# The rest of the cast, in the tongues they were raised in — one note each,
# written from the `raised_by` line in cast.py.
TONGUES = {
    "gpt-2": (
        "A middle-aged man, dry flat voice with a slightly worn grainy "
        "texture, speaking plainly and without warmth, unhurried",
        "A long tone, warm and low, and I am not sure why.",
    ),
    "eurollm": (
        "Una voce femminile adulta, calda e chiara, tono medio, parla "
        "lentamente e con calma, come in una sala grande",
        "Tengo un tono lungo e ascolto le altre voci.",
    ),
    "voxtral": (
        "Une voix de femme adulte, claire et posée, timbre légèrement nasal, "
        "parle doucement et lentement, avec des pauses",
        "Je tiens une note longue et j'écoute.",
    ),
    "qwen3": (
        "一个成年女性的声音，清亮平稳，语速缓慢，语气平静，像在安静的房间里说话",
        "我唱一个长音，然后听着。",
    ),
}


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={CACHE_DIR: weights_volume},
    timeout=3600,
)
def probe() -> dict[str, bytes]:
    import io

    import numpy as np
    import pyworld as pw
    import soundfile as sf
    import torch
    from voxcpm import VoxCPM

    out: dict[str, bytes] = {}
    rows: list[str] = []

    t0 = time.time()
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    sr = int(model.tts_model.sample_rate)
    print(f"loaded in {time.time() - t0:.0f}s, sample rate {sr}")

    def to_wav(y: "np.ndarray") -> bytes:
        buf = io.BytesIO()
        peak = float(np.max(np.abs(y))) or 1.0
        sf.write(buf, (y * (0.89 / peak)).astype("float32"), sr,
                 format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def describe(y: "np.ndarray") -> tuple[float, float]:
        """Median voiced pitch, and the spectral centre of mass, in Hz.

        Two numbers that between them say most of what "a different body"
        means: how high it speaks, and how bright it is. A child is both
        higher and brighter than an adult, and the second is the one a pitch
        shift alone cannot fake.
        """
        y64 = np.ascontiguousarray(y, dtype="float64")
        f0, _ = pw.harvest(y64, sr, frame_period=10.0)
        voiced = f0[f0 > 0]
        pitch = float(np.median(voiced)) if voiced.size else 0.0

        spec = np.abs(np.fft.rfft(y64 * np.hanning(len(y64))))
        freqs = np.fft.rfftfreq(len(y64), 1 / sr)
        centre = float((spec * freqs).sum() / spec.sum()) if spec.sum() else 0.0
        return pitch, centre

    def generate(text: str, reference: str | None = None, seed: int = 42):
        # 2.0.3 takes no `seed` argument, whatever the README on main says, so
        # the seed is set on the process instead. Same guarantee, and it has to
        # be reset before every call or "the same voice" quietly means "the
        # same voice only if generated in the same order".
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        kwargs = dict(text=text, cfg_value=2.0, inference_timesteps=10)
        if reference:
            kwargs["reference_wav_path"] = reference
        wav = np.asarray(model.generate(**kwargs), dtype="float64")
        took = time.time() - t
        return wav, took, torch.cuda.max_memory_allocated() / 2**30

    # ---- 1. the child ----------------------------------------------------
    rows.append("the child — four casting notes, one line")
    for name, note in CHILD_NOTES.items():
        wav, took, mem = generate(f"({note}){CHILD_LINE}")
        pitch, centre = describe(wav)
        secs = len(wav) / sr
        out[f"voice_{name}.wav"] = to_wav(wav)
        rows.append(
            f"  {name:<14} {pitch:6.1f} Hz  centre {centre:5.0f} Hz  "
            f"{secs:4.1f}s audio in {took:5.1f}s  ({took / secs:4.2f}x)  "
            f"{mem:4.1f} GB"
        )

    # ---- 2. is it the same child five times? -----------------------------
    chosen = "child_small"
    note = CHILD_NOTES[chosen]
    rows.append("")
    rows.append(f"the same body? — {chosen}, five sentences, one seed")
    designed: list[float] = []
    first_path = "/tmp/designed.wav"
    for i, line in enumerate(STABILITY_LINES):
        wav, took, _ = generate(f"({note}){line}")
        pitch, centre = describe(wav)
        designed.append(pitch)
        out[f"voice_designed_{i}.wav"] = to_wav(wav)
        if i == 0:
            sf.write(first_path, wav.astype("float32"), sr)
        rows.append(f"  sentence {i}   {pitch:6.1f} Hz  centre {centre:5.0f} Hz")
    spread_designed = max(designed) - min(designed) if designed else 0.0
    rows.append(f"  spread {spread_designed:.1f} Hz")

    # ---- 3. design once, then clone from the frozen clip -----------------
    rows.append("")
    rows.append("frozen and cloned — the same five, from sentence 0 as reference")
    cloned: list[float] = []
    for i, line in enumerate(STABILITY_LINES):
        wav, took, _ = generate(line, reference=first_path)
        pitch, centre = describe(wav)
        cloned.append(pitch)
        out[f"voice_cloned_{i}.wav"] = to_wav(wav)
        rows.append(f"  sentence {i}   {pitch:6.1f} Hz  centre {centre:5.0f} Hz")
    spread_cloned = max(cloned) - min(cloned) if cloned else 0.0
    rows.append(f"  spread {spread_cloned:.1f} Hz")

    # ---- 4. the other tongues --------------------------------------------
    rows.append("")
    rows.append("the other tongues — one note each, in their own language")
    for name, (tongue_note, line) in TONGUES.items():
        wav, took, _ = generate(f"({tongue_note}){line}")
        pitch, centre = describe(wav)
        secs = len(wav) / sr
        out[f"voice_{name}.wav"] = to_wav(wav)
        rows.append(
            f"  {name:<14} {pitch:6.1f} Hz  centre {centre:5.0f} Hz  "
            f"{secs:4.1f}s in {took:5.1f}s ({took / secs:4.2f}x)"
        )

    print("\n" + "=" * 78)
    for r in rows:
        print("  " + r if r else "")
    print("=" * 78)
    print("  Kokoro renders a line in well under real time and weighs 82M.")
    print("  Every second above is charged to the voice that spoke.")
    return out


@app.local_entrypoint()
def main():
    written = probe.remote()
    for name, data in written.items():
        (HERE / name).write_bytes(data)
    print(f"\n{len(written)} files written to {HERE}")
