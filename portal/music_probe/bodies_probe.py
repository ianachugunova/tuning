"""Eight bodies, born from eight descriptions, in the choir's own image.

    .venv-modal/bin/modal run portal/music_probe/bodies_probe.py

The previous three probes each answered one question in a clean room. This one
asks whether the answers survive contact with the choir, and it is the only
probe of the four that could fail for a reason nothing else has tested:

  * **The two synthesisers have to share an image.** VoxCPM2 brings gradio,
    datasets and its own torch floor into a picture already holding
    transformers, kokoro, misaki, spacy, bitsandbytes and a pyworld compiled
    from source. Separate images proved nothing about one image.
  * **A body has to be born once and then be found.** `body_of` writes into
    `/weights/bodies` and commits, and the second container has to pick the
    same file up rather than generate a second person under the same name.
  * **Eight described voices have to be eight voices.** Same sentence, eight
    bodies, and the numbers should be as far apart as the descriptions are.

Runs the choir's real `speak` and `body_of` through a stub that holds only
what they touch, so nothing here is a copy of the logic it is checking.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import choir_app  # noqa: E402

app = modal.App("tuning-bodies-probe")

image = choir_app.image.add_local_file(
    Path(choir_app.__file__), remote_path="/root/choir_app.py"
)

# One line for everybody, so the only difference between the files is the body.
LINE = "I am holding a tone, and I am listening."


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={choir_app.CACHE_DIR: choir_app.weights_volume},
    timeout=3600,
)
def probe() -> dict[str, bytes]:
    import io

    import numpy as np
    import pyworld as pw
    import soundfile as sf
    import torch

    import cast as cast_mod
    import choir_app as ca

    print(f"torch {torch.__version__}, numpy {np.__version__}")

    class Stub:
        """Only the attributes `speak` and `body_of` actually reach for.

        A class body cannot say `torch = torch` — the name is being defined,
        so the right-hand side resolves to itself and not to the import.
        """

        synth = None
        bodies: dict = {}
        pipelines: dict = {}

        _synth = ca.Choir._synth
        body_of = ca.Choir.body_of
        speak = ca.Choir.speak

    Stub.torch = torch
    choir = Stub()
    out: dict[str, bytes] = {}
    rows: list[str] = []

    def describe(y, rate) -> tuple[float, float]:
        y64 = np.ascontiguousarray(y, dtype="float64")
        f0, _ = pw.harvest(y64, rate, frame_period=10.0)
        voiced = f0[f0 > 0]
        pitch = float(np.median(voiced)) if voiced.size else 0.0
        spec = np.abs(np.fft.rfft(y64 * np.hanning(len(y64))))
        freqs = np.fft.rfftfreq(len(y64), 1 / rate)
        centre = float((spec * freqs).sum() / spec.sum()) if spec.sum() else 0.0
        return pitch, centre

    def to_wav(y, rate) -> bytes:
        buf = io.BytesIO()
        peak = float(np.max(np.abs(y))) or 1.0
        sf.write(buf, (y * (0.89 / peak)).astype("float32"), rate,
                 format="WAV", subtype="PCM_16")
        return buf.getvalue()

    for v in cast_mod.CHOIR:
        if not v.casting:
            continue

        born = (Path(ca.BODIES_DIR) / f"{v.voice_id}.wav").exists()
        t0 = time.time()
        path = choir.body_of(v)
        birth_s = time.time() - t0
        if path is None:
            rows.append(f"{v.short_name:<15} was not born")
            continue

        # Its own tongue for the body, one shared English line for the compare.
        t0 = time.time()
        wav = choir.speak(v.first_words, v)
        say_s = time.time() - t0
        if wav is None:
            rows.append(f"{v.short_name:<15} has a body but no voice")
            continue

        pitch, centre = describe(wav, ca.SR)
        out[f"body_{v.short_name}_first_words.wav"] = to_wav(wav, ca.SR)

        shared = choir.speak(LINE, v)
        if shared is not None:
            out[f"body_{v.short_name}_same_line.wav"] = to_wav(shared, ca.SR)

        rows.append(
            f"{v.short_name:<15} {'found' if born else 'BORN':<5} "
            f"{birth_s:5.1f}s   say {say_s:5.2f}s   "
            f"{pitch:6.1f} Hz   centre {centre:5.0f} Hz   {len(wav) / ca.SR:4.1f}s"
        )

    print("\n" + "=" * 78)
    print("  born once, then found. every voice says the same line at the end.")
    for r in rows:
        print("  " + r)
    print(f"\n  peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    print("=" * 78)
    return out


@app.local_entrypoint()
def main():
    written = probe.remote()
    for name, data in written.items():
        (HERE / name).write_bytes(data)
    print(f"\n{len(written)} files written to {HERE}")
