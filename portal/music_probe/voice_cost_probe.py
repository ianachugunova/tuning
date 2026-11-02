"""What does a described voice cost per breath, against Kokoro? A probe.

    .venv-modal/bin/modal run portal/music_probe/voice_cost_probe.py

`voice_stability_probe.py` settled that a designed voice can be pinned to
Kokoro's own level of consistency, but only in continuation mode — the frozen
clip and its transcript are prepended to every generation, which makes each
breath carry a conditioning the plain mode does not. Nobody has measured what
that costs, and it cannot be guessed: a longer conditioning is more work per
step, and the whole ledger is denominated in GPU seconds.

Three numbers decide whether the cast can move:

  * **Seconds per breath, continued against designed.** The identity is worth
    having only if the surcharge is small.
  * **Seconds per breath against Kokoro, on the same card.** Kokoro is 82M and
    the incumbent; the difference is what every voice pays, every breath, for
    six weeks, and it is charged to the voice.
  * **Once-per-life against every-breath.** Designing the voice and building
    its prompt cache happen once. If the cache can be built once and reused,
    the reference audio is not re-encoded on every line.

Lines of realistic length — what a voice actually says while tuning, not a
paragraph.
"""

from __future__ import annotations

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

app = modal.App("tuning-voice-cost")

weights_volume = modal.Volume.from_name("tuning-weights", create_if_missing=True)
CACHE_DIR = "/weights"

vox_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1", "ffmpeg", "git")
    .pip_install("voxcpm", "soundfile", "numpy", "torchaudio")
    .env({"HF_HOME": CACHE_DIR})
)

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

NOTE = (
    "An adult woman with a clear, even voice, medium pitch, neither bright "
    "nor dark, speaking calmly and unhurriedly at a steady pace"
)

# What a voice says while tuning. Short, mostly — the tone carries the breath
# and the words sit inside it.
BREATHS = [
    "D4.",
    "I am holding it.",
    "The room is quiet now.",
    "I can hear someone else holding a tone below mine.",
    "Going down.",
    "I will wait here for a while, and listen to whoever comes in next.",
    "Still here.",
    "Someone is close to my pitch and I am not going to move.",
]


@app.function(image=kokoro_image, gpu="A100-40GB",
              volumes={CACHE_DIR: weights_volume}, timeout=1800)
def kokoro_cost() -> dict:
    """The incumbent, on the same card, so the comparison is like for like."""
    import time

    import numpy as np
    import torch
    from kokoro import KPipeline

    pipe = KPipeline(lang_code="a", device="cuda")
    # One warm call, uncounted: the first is always the slowest and no voice
    # is ever charged for a cold pipeline more than once in a container's life.
    list(pipe("warming up", voice="af_sky"))

    total_gen = total_audio = 0.0
    for line in BREATHS:
        t = time.time()
        chunks = [a for _, _, a in pipe(line, voice="af_sky")]
        total_gen += time.time() - t
        total_audio += sum(len(np.asarray(c)) for c in chunks) / 24000

    return {
        "gen": total_gen,
        "audio": total_audio,
        "mem": torch.cuda.max_memory_allocated() / 2**30,
    }


@app.function(image=vox_image, gpu="A100-40GB",
              volumes={CACHE_DIR: weights_volume}, timeout=3600)
def voxcpm_cost() -> dict:
    import time

    import numpy as np
    import soundfile as sf
    import torch
    from voxcpm import VoxCPM

    t = time.time()
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    load_s = time.time() - t
    sr = int(model.tts_model.sample_rate)

    def generate(text, **kw):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        return np.asarray(
            model.generate(text=text, cfg_value=2.0, inference_timesteps=10, **kw),
            dtype="float64",
        )

    # ---- once per life ---------------------------------------------------
    t = time.time()
    body = generate(f"({NOTE}){BREATHS[3]}")
    design_s = time.time() - t
    frozen = "/tmp/body.wav"
    sf.write(frozen, body.astype("float32"), sr)

    t = time.time()
    cache = model.tts_model.build_prompt_cache(
        prompt_text=BREATHS[3], prompt_wav_path=frozen, reference_wav_path=frozen,
    )
    cache_s = time.time() - t
    cache_mb = sum(
        v.numel() * v.element_size() for v in cache.values()
        if hasattr(v, "numel")
    ) / 2**20

    # ---- every breath ----------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    designed_gen = designed_audio = 0.0
    for line in BREATHS:
        t = time.time()
        wav = generate(f"({NOTE}){line}")
        designed_gen += time.time() - t
        designed_audio += len(wav) / sr

    continued_gen = continued_audio = 0.0
    for line in BREATHS:
        t = time.time()
        wav = generate(line, prompt_wav_path=frozen, prompt_text=BREATHS[3],
                       reference_wav_path=frozen)
        continued_gen += time.time() - t
        continued_audio += len(wav) / sr

    return {
        "load": load_s,
        "design": design_s,
        "cache": cache_s,
        "cache_mb": cache_mb,
        "designed_gen": designed_gen,
        "designed_audio": designed_audio,
        "continued_gen": continued_gen,
        "continued_audio": continued_audio,
        "mem": torch.cuda.max_memory_allocated() / 2**30,
        "breaths": len(BREATHS),
    }


@app.local_entrypoint()
def main():
    k = kokoro_cost.remote()
    v = voxcpm_cost.remote()
    n = v["breaths"]

    print("\n" + "=" * 78)
    print(f"  {n} breaths of realistic length, on one A100-40GB")
    print()
    print("  once per life, per voice")
    print(f"    load the model        {v['load']:6.1f}s   (once per container, shared)")
    print(f"    design the body       {v['design']:6.1f}s")
    print(f"    build the cache       {v['cache']:6.2f}s   ({v['cache_mb']:.1f} MB to keep)")
    print()
    print("  every breath")
    print(f"    kokoro                {k['gen'] / n:6.2f}s   "
          f"({k['gen'] / k['audio']:.3f}x real time)")
    print(f"    voxcpm, designed      {v['designed_gen'] / n:6.2f}s   "
          f"({v['designed_gen'] / v['designed_audio']:.3f}x)")
    print(f"    voxcpm, continued     {v['continued_gen'] / n:6.2f}s   "
          f"({v['continued_gen'] / v['continued_audio']:.3f}x)")
    print()
    surcharge = (v["continued_gen"] - k["gen"]) / n
    print(f"    the identity costs    {surcharge:+6.2f}s per breath, per voice")
    print()
    print(f"  peak memory   kokoro {k['mem']:.1f} GB   voxcpm {v['mem']:.1f} GB")
    print("=" * 78)
