"""Is a designed voice the same voice tomorrow? A probe with a control.

    .venv-modal/bin/modal run portal/music_probe/voice_stability_probe.py

`voice_probe.py` got a child out of VoxCPM2 and then found the child's median
pitch moving 151 Hz across five sentences. Freezing the first clip and cloning
from it halved that to 80 Hz. Neither number means anything on its own, for two
reasons this probe exists to remove.

**Median pitch is not identity.** A person saying "the room is quiet now" and
"I can hear someone else holding a tone" has two different median pitches and
one body. What is wanted is a speaker embedding — the vector a verification
model uses to decide whether two recordings are the same person — and the
cosine between the takes.

**There is no scale without a control.** Kokoro is the incumbent, and its
consistency is not perfect either; it is simply the consistency the piece has
been living with. So the same five sentences are rendered by Kokoro's `af_sky`
and measured the same way. That number is the bar. Anything at or above it is
as much a person as what the choir already has.

Three ways of asking VoxCPM2 for the same body are compared:

  * **designed** — the casting note every time, which is what a naive port
    would do.
  * **cloned** — design once, keep the wav, pass it as `reference_wav_path`.
  * **continued** — the same clip passed as `prompt_wav_path` with its
    transcript, which the model's own documentation calls its highest-fidelity
    mode and which `voice_probe.py` never tried.
"""

from __future__ import annotations

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

app = modal.App("tuning-voice-stability")

weights_volume = modal.Volume.from_name("tuning-weights", create_if_missing=True)
CACHE_DIR = "/weights"

# Same image as voice_probe, plus the verification model's needs.
vox_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1", "ffmpeg", "git")
    .pip_install("voxcpm", "soundfile", "numpy", "torchaudio")
    .env({"HF_HOME": CACHE_DIR})
)

# Kokoro, exactly as the choir has it today. The control.
kokoro_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("espeak-ng", "libsndfile1")
    .pip_install("kokoro>=0.9.4", "misaki[zh,ja]", "soundfile", "numpy", "torch>=2.6")
    .run_commands(
        "python -m spacy download en_core_web_sm",
        "python -c \"import spacy; spacy.load('en_core_web_sm')\"",
    )
    .env({"HF_HOME": CACHE_DIR})
)

CHILD_NOTE = (
    "A four-year-old child, a very small bright voice, thin and light with "
    "a lot of air in it, speaking simply and slowly, as if telling a story "
    "it is making up while it says it"
)

# The control the first run of this probe was missing. WavLM's x-vectors are
# trained on adult speech, so a child scores loosely against itself for reasons
# that have nothing to do with the engine — and the incumbent it was being
# compared against, af_sky, is an adult woman. This note asks VoxCPM2 for
# roughly what Kokoro is already giving, so the remaining difference is the
# engine and not the age of the body.
ADULT_NOTE = (
    "An adult woman with a clear, even voice, medium pitch, neither bright "
    "nor dark, speaking calmly and unhurriedly at a steady pace"
)

LINES = [
    "I sing a little note, and I wait for yours.",
    "The room is quiet now.",
    "I can hear someone else holding a tone.",
    "My voice is going down, slowly.",
    "I will stop, and listen.",
]


@app.function(image=kokoro_image, volumes={CACHE_DIR: weights_volume}, timeout=1800)
def kokoro_takes() -> dict[str, bytes]:
    """The incumbent, saying the same five lines. The bar to clear."""
    import io

    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    pipe = KPipeline(lang_code="a")
    out: dict[str, bytes] = {}
    for i, line in enumerate(LINES):
        chunks = [a for _, _, a in pipe(line, voice="af_sky")]
        wav = np.concatenate(chunks).astype("float32")
        buf = io.BytesIO()
        sf.write(buf, wav, 24000, format="WAV", subtype="PCM_16")
        out[f"{i}"] = buf.getvalue()
    return out


@app.function(
    image=vox_image,
    gpu="A100-40GB",
    volumes={CACHE_DIR: weights_volume},
    timeout=3600,
)
def compare(kokoro_wavs: dict[str, bytes]) -> dict[str, bytes]:
    import io
    import itertools

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from transformers import AutoFeatureExtractor, WavLMForXVector
    from voxcpm import VoxCPM

    out: dict[str, bytes] = {}

    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    sr = int(model.tts_model.sample_rate)

    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    verifier = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()

    def embed(wav: "np.ndarray", rate: int):
        """The vector a verification model uses to say 'same person'."""
        y = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        if rate != 16000:
            y = torchaudio.functional.resample(y, rate, 16000)
        feats = extractor(y.squeeze(0).numpy(), sampling_rate=16000,
                          return_tensors="pt")
        with torch.no_grad():
            return verifier(**feats).embeddings[0]

    def cohesion(embeddings: list) -> tuple[float, float]:
        """Mean and worst cosine over every pair. One body scores near 1."""
        cos = torch.nn.functional.cosine_similarity
        pairs = [float(cos(a, b, dim=0)) for a, b in itertools.combinations(embeddings, 2)]
        return (sum(pairs) / len(pairs), min(pairs)) if pairs else (0.0, 0.0)

    def to_wav(y: "np.ndarray", rate: int) -> bytes:
        buf = io.BytesIO()
        peak = float(np.max(np.abs(y))) or 1.0
        sf.write(buf, (y * (0.89 / peak)).astype("float32"), rate,
                 format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def generate(text: str, seed: int = 42, **kw):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return np.asarray(
            model.generate(text=text, cfg_value=2.0, inference_timesteps=10, **kw),
            dtype="float64",
        )

    rows: list[str] = []

    # ---- the control ------------------------------------------------------
    kokoro_emb = []
    for i in sorted(kokoro_wavs):
        y, rate = sf.read(io.BytesIO(kokoro_wavs[i]), dtype="float64")
        kokoro_emb.append(embed(y, rate))
    k_mean, k_worst = cohesion(kokoro_emb)
    rows.append(f"kokoro af_sky   mean {k_mean:.3f}   worst pair {k_worst:.3f}   (the bar)")
    rows.append("")

    # ---- designed every time ---------------------------------------------
    designed = [generate(f"({CHILD_NOTE}){line}") for line in LINES]
    d_mean, d_worst = cohesion([embed(w, sr) for w in designed])
    rows.append(f"designed        mean {d_mean:.3f}   worst pair {d_worst:.3f}")

    # The frozen clip everything below clones from.
    frozen = "/tmp/frozen.wav"
    sf.write(frozen, designed[0].astype("float32"), sr)
    out["stability_frozen_reference.wav"] = to_wav(designed[0], sr)

    # ---- cloned from the frozen clip --------------------------------------
    cloned = [generate(line, reference_wav_path=frozen) for line in LINES]
    c_mean, c_worst = cohesion([embed(w, sr) for w in cloned])
    rows.append(f"cloned          mean {c_mean:.3f}   worst pair {c_worst:.3f}")

    # ---- continuation cloning, the mode the docs call highest fidelity ----
    continued = [
        generate(line, prompt_wav_path=frozen, prompt_text=LINES[0],
                 reference_wav_path=frozen)
        for line in LINES
    ]
    x_mean, x_worst = cohesion([embed(w, sr) for w in continued])
    rows.append(f"continued       mean {x_mean:.3f}   worst pair {x_worst:.3f}")

    # ---- the same three, on an adult body -------------------------------
    rows.append("")
    rows.append("an adult, so the measure is asked something it knows")
    adult_designed = [generate(f"({ADULT_NOTE}){line}") for line in LINES]
    a_mean, a_worst = cohesion([embed(w, sr) for w in adult_designed])
    rows.append(f"designed        mean {a_mean:.3f}   worst pair {a_worst:.3f}")

    adult_frozen = "/tmp/adult.wav"
    sf.write(adult_frozen, adult_designed[0].astype("float32"), sr)
    adult_continued = [
        generate(line, prompt_wav_path=adult_frozen, prompt_text=LINES[0],
                 reference_wav_path=adult_frozen)
        for line in LINES
    ]
    ac_mean, ac_worst = cohesion([embed(w, sr) for w in adult_continued])
    rows.append(f"continued       mean {ac_mean:.3f}   worst pair {ac_worst:.3f}")

    for label, takes in (("designed", designed), ("cloned", cloned),
                         ("continued", continued),
                         ("adult", adult_designed),
                         ("adult_continued", adult_continued)):
        for i, w in enumerate(takes):
            out[f"stability_{label}_{i}.wav"] = to_wav(w, sr)

    print("\n" + "=" * 78)
    print("  same five sentences. 1.000 would be one body; the bar is Kokoro.")
    for r in rows:
        print("  " + r if r else "")
    print("=" * 78)
    return out


@app.local_entrypoint()
def main():
    written = compare.remote(kokoro_takes.remote())
    for name, data in written.items():
        (HERE / name).write_bytes(data)
    print(f"\n{len(written)} files written to {HERE}")
