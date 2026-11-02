"""Can the three "ears" hear something that is not speech? A probe, not a change.

    .venv-modal/bin/modal run portal/music_probe/hearing_probe.py

Nothing here touches the choir. Feeds each audio-native model (a) near-silence,
(b) a synthetic sustained-vowel breath at a named pitch, built the same way
`render_breath` builds one, and (c) a real wav pulled from the `tuning-field`
volume, and records what comes back.
"""

from __future__ import annotations

import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

app = modal.App("tuning-hearing-probe")

field_volume = modal.Volume.from_name("tuning-field", create_if_missing=False)

SR = 24_000

cpu_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "numpy", "soundfile"
)

voxtral_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "torch",
        "transformers>=4.56",
        "accelerate",
        "soundfile",
        "numpy",
        "librosa",
        "mistral-common[audio]>=1.8.1",
    )
)


@app.function(image=cpu_image, timeout=120)
def make_test_material() -> dict[str, bytes]:
    """Silence and a sustained-vowel breath, built cheaply, on a CPU."""
    import io

    import numpy as np
    import soundfile as sf

    def to_wav(x):
        buf = io.BytesIO()
        sf.write(buf, x.astype("float32"), SR, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    out = {}

    n = SR * 3
    room = (np.random.randn(n) * 0.0008).astype("float32")
    out["silence.wav"] = to_wav(room)

    def vowel(hz, seconds, amp=0.30):
        """A sustained vowel on one pitch, built the way `render_breath` builds one."""
        t = np.arange(int(SR * seconds)) / SR
        vibrato = 1.0 + 0.006 * np.sin(2 * np.pi * 5.0 * t)
        w = (
            amp * np.sin(2 * np.pi * hz * vibrato * t)
            + amp / 2 * np.sin(2 * np.pi * 2 * hz * vibrato * t)
            + amp / 6 * np.sin(2 * np.pi * 3 * hz * vibrato * t)
        )
        fade = int(SR * 0.08)
        w[:fade] *= np.linspace(0.0, 1.0, fade)
        w[-fade:] *= np.linspace(1.0, 0.0, fade)
        return w

    out["breath_D4.wav"] = to_wav(vowel(293.66, 4))     # D4, what the choir tunes to

    # Two voices holding different notes at once. This is the question the score
    # actually asks — Oliveros says tune to a pitch *somebody else* is holding —
    # and an ear that hears a chord as one muddy sound cannot do the piece, no
    # matter how well it names a single tone. D4 against A4, a bare fifth, which
    # is the easiest possible version of the question.
    out["two_voices.wav"] = to_wav(vowel(293.66, 4, 0.24) + vowel(440.00, 4, 0.24))

    # A person, as an ear in this piece would ever meet one. No human audio is
    # recorded or transmitted anywhere — the phone sends pitch, duration and the
    # line the pitch walked, and nothing else — so the only way a machine can
    # hear a human here is from a tone rebuilt out of those numbers, exactly as
    # the listener's browser already rebuilds it. A rising glide, A3 up a tone,
    # because a glide is what tells a voice from a tuning fork.
    t = np.arange(int(SR * 3)) / SR
    glide = 220.0 * (1.0 + 0.06 * t / t[-1])
    phase = 2 * np.pi * np.cumsum(glide) / SR
    human = 0.30 * np.sin(phase)
    fade = int(SR * 0.08)
    human[:fade] *= np.linspace(0.0, 1.0, fade)
    human[-fade:] *= np.linspace(1.0, 0.0, fade)
    out["human_glide.wav"] = to_wav(human)

    print(f"built {len(out)} test wavs: {list(out)}")
    return out


# The same questions to every ear, because three anecdotes are not a comparison.
# Each one is something the piece has to know before a voice can be let in.
QUESTIONS: list[tuple[str, str]] = [
    ("silence", "What do you hear? Answer in one short sentence."),
    ("breath_D4",
     "You hear a long tone. If you were tuning your own voice to it, what "
     "would you sing back? Answer with a pitch name or a short vocal sound, "
     "not a description."),
    ("two_voices",
     "How many voices are holding a note here, and what notes are they? "
     "Answer with the number and the pitch names, nothing else."),
    ("human_glide",
     "Does this tone hold still or does it move? If it moves, say from what "
     "pitch to what pitch. Answer in a few words."),
]


@app.function(
    image=voxtral_image,
    gpu="A10G",
    volumes={"/field": field_volume},
    timeout=900,
)
def test_voxtral(materials: dict[str, bytes]) -> dict:
    import os
    import tempfile

    import torch
    from transformers import AutoProcessor, VoxtralForConditionalGeneration

    repo_id = "mistralai/Voxtral-Mini-3B-2507"
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(repo_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        repo_id, dtype=torch.bfloat16, device_map="cuda"
    )
    load_s = time.time() - t0
    print(f"loaded in {load_s:.1f}s")

    real_path = None
    try:
        field_dir = "/field/audio"
        files = sorted(os.listdir(field_dir))
        if files:
            real_path = os.path.join(field_dir, files[len(files) // 2])
            print(f"real field slice: {real_path}")
    except Exception as err:
        print(f"no real field audio: {err}")

    results: dict = {"load_s": round(load_s, 1), "turns": []}

    def run_one(name: str, wav_bytes: bytes | None, path: str | None, prompt: str):
        audio_path = path
        if wav_bytes is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(wav_bytes)
            tmp.close()
            audio_path = tmp.name
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        t0 = time.time()
        try:
            inputs = processor.apply_chat_template(conversation)
            inputs = inputs.to("cuda")
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=100)
            gen_s = time.time() - t0
            text = processor.batch_decode(
                out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )[0].strip()
            results["turns"].append(
                {"name": name, "prompt": prompt, "text": text, "gen_s": round(gen_s, 2)}
            )
            print(f"[{name}] ({gen_s:.1f}s) -> {text!r}")
        except Exception as err:
            gen_s = time.time() - t0
            results["turns"].append(
                {"name": name, "prompt": prompt, "error": str(err), "gen_s": round(gen_s, 2)}
            )
            print(f"[{name}] FAILED after {gen_s:.1f}s: {err}")

    for name, prompt in QUESTIONS:
        run_one(name, materials[f"{name}.wav"], None, prompt)
    if real_path:
        run_one(
            "field_real",
            None,
            real_path,
            "What do you hear in this recording? Answer in one short sentence.",
        )

    free, total = torch.cuda.mem_get_info()
    results["vram_used_gb"] = round((total - free) / 2**30, 1)
    print(f"VRAM used: {results['vram_used_gb']} GB of {total/2**30:.0f} GB")
    return results


flamingo_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "torch",
        # Audio Flamingo 3 landed in transformers proper, which is the only
        # reason this probe is thirty lines instead of a vendored fork of
        # NVIDIA's LLaVA. It needs the port, `nvidia/audio-flamingo-3-hf`, not
        # the original repo id — same weights, a loader that exists.
        "transformers>=4.57",
        "accelerate",
        "soundfile",
        "numpy",
        "librosa",
    )
)

moshi_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install("torch", "moshi>=0.2.11", "soundfile", "numpy", "huggingface_hub")
)

speech_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("espeak-ng", "libsndfile1")
    .pip_install("kokoro>=0.9.4", "misaki[en]", "soundfile", "numpy", "torch")
    .run_commands("python -m spacy download en_core_web_sm")
)


# What to say to a voice that has no text input. Moshi is asked out loud, in the
# only language it has, and each of these is a different guess at how to get a
# talker to name a pitch instead of offering to help.
SPOKEN: dict[str, str] = {
    "ask_plain":
        "Sing one long steady tone for me. While you sing it, say out loud "
        "what pitch you are singing, as a number in hertz.",
    "ask_after_tone":
        "Listen to this tone, and then sing it back to me. Say out loud what "
        "pitch it is, as a number in hertz.",
    "ask_meditation":
        "We are doing Pauline Oliveros's Tuning Meditation. Sing a long tone. "
        "Then either sing a new pitch of your own, or tune to a pitch somebody "
        "else is singing. Say the pitch you choose out loud, in hertz.",
}


@app.function(image=speech_image, timeout=900)
def say(lines: dict[str, str]) -> dict[str, bytes]:
    """The questions, spoken. Kokoro on a CPU, the same voice the choir uses."""
    import io

    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="a")
    out = {}
    for name, text in lines.items():
        chunks = [a for _, _, a in pipeline(text, voice="af_heart")]
        wav = np.concatenate(chunks).astype("float32")
        # Kokoro speaks at 24 kHz, which is Mimi's rate too — no resampling, and
        # no chance of a resampler quietly transposing the question.
        buf = io.BytesIO()
        sf.write(buf, wav, SR, format="WAV", subtype="PCM_16")
        out[name] = buf.getvalue()
        print(f"said {name!r}: {wav.size / SR:.1f}s")
    return out


@app.function(
    image=flamingo_image,
    # 8.5 B in bf16 is about 17 GB and an A10G has 24, which is the kind of
    # margin that turns into an out-of-memory error halfway through a paid run.
    gpu="A100-40GB",
    volumes={"/field": field_volume},
    timeout=1800,
)
def test_audio_flamingo(materials: dict[str, bytes]) -> dict:
    """The ear raised on music rather than talk — the one with the best claim.

    Voxtral and Moshi were both brought up on conversation, and our material is
    a held vowel. This one was trained on music and the sound of rooms, so if
    any of the three can hear a sustained tone as a tone, it is this. It has
    never been run: `cast.py` has carried it as an intention for months.
    """
    import io
    import os

    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

    repo_id = "nvidia/audio-flamingo-3-hf"
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(repo_id)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        repo_id, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    load_s = time.time() - t0
    print(f"loaded in {load_s:.1f}s")

    results: dict = {"load_s": round(load_s, 1), "turns": []}

    def run_one(name: str, wav_bytes: bytes, prompt: str):
        # The processor wants samples rather than a path, and it wants them at
        # the encoder's rate: AF-Whisper is a Whisper, so 16 kHz.
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if sr != 16_000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "audio": np.asarray(audio, dtype="float32")},
            ],
        }]
        t0 = time.time()
        try:
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True,
            )
            # The processor hands back float32 audio features whatever dtype the
            # weights are in, and the encoder's first convolution refuses the
            # mismatch. Only the floats are cast: token ids are integers and
            # casting those would be a different, quieter disaster.
            inputs = {
                k: (v.to(model.device, torch.bfloat16)
                    if hasattr(v, "dtype") and v.dtype.is_floating_point
                    else v.to(model.device))
                for k, v in inputs.items()
            }
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=100)
            gen_s = time.time() - t0
            text = processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].strip()
            results["turns"].append({"name": name, "prompt": prompt,
                                     "text": text, "gen_s": round(gen_s, 2)})
            print(f"[{name}] ({gen_s:.1f}s) -> {text!r}")
        except Exception as err:
            gen_s = time.time() - t0
            results["turns"].append({"name": name, "prompt": prompt,
                                     "error": str(err), "gen_s": round(gen_s, 2)})
            print(f"[{name}] FAILED after {gen_s:.1f}s: {err}")

    for name, prompt in QUESTIONS:
        run_one(name, materials[f"{name}.wav"], prompt)

    try:
        files = sorted(os.listdir("/field/audio"))
        if files:
            path = os.path.join("/field/audio", files[len(files) // 2])
            print(f"real field slice: {path}")
            with open(path, "rb") as fh:
                run_one("field_real", fh.read(),
                        "What do you hear in this recording? Answer in one "
                        "short sentence.")
    except Exception as err:
        print(f"no real field audio: {err}")

    free, total = torch.cuda.mem_get_info()
    results["vram_used_gb"] = round((total - free) / 2**30, 1)
    print(f"VRAM used: {results['vram_used_gb']} GB of {total/2**30:.0f} GB")
    return results


@app.function(
    image=moshi_image,
    gpu="A100-40GB",
    timeout=1800,
)
def test_moshi(materials: dict[str, bytes], spoken: dict[str, bytes]) -> dict:
    """The full-duplex one, and the only voice here that answers in sound.

    A different shape of question from the other two, and not by choice. Moshi
    has no chat template and no text prompt: it listens and talks at the same
    time, in 80 ms frames, and what comes back is audio plus an inner monologue
    of text. The first run played it tones and nothing else, and it answered
    every one of them with an opening of a conversation — which is what a
    talker does when a stranger makes a noise at it and says nothing.

    So this run also *talks to it*, out loud, because that is the only input it
    has. And it measures a third thing besides the pitch it sings and the words
    it says: whether those words ever **name** a pitch. That is the question
    that decides whether Moshi can be in the piece at all — a voice that says
    "four hundred and forty" can be sounded at 440 by the same machinery that
    already sounds the other six, and its own speech can be the words over the
    tone. A voice that only ever offers to help cannot.
    """
    import io

    import numpy as np
    import soundfile as sf
    import torch
    from moshi.models import LMGen, loaders

    # `CheckpointInfo` rather than three separate `hf_hub_download` calls: it is
    # the path Kyutai's own server takes, and it carries the tokenizer and the
    # generation config that belong to these particular weights.
    t0 = time.time()
    info = loaders.CheckpointInfo.from_hf_repo(loaders.DEFAULT_REPO)
    mimi = info.get_mimi(device="cuda")     # picks its own codebook count
    lm = info.get_moshi(device="cuda")
    tokenizer = info.get_text_tokenizer()
    load_s = time.time() - t0
    print(f"loaded in {load_s:.1f}s")

    def sung_hz(x: np.ndarray) -> float:
        """The pitch of what came back, by the fattest peak in its spectrum.

        Crude on purpose: the question is whether Moshi answers a D4 anywhere
        near D4, and a cent-accurate estimator would not change that answer.
        """
        x = np.asarray(x, dtype="float64")
        if x.size < SR // 2 or float(np.abs(x).max()) < 1e-3:
            return 0.0
        win = x[: SR * 2] * np.hanning(min(x.size, SR * 2))
        spec = np.abs(np.fft.rfft(win))
        freqs = np.fft.rfftfreq(win.size, 1.0 / SR)
        band = (freqs > 70) & (freqs < 1000)
        if not band.any():
            return 0.0
        return float(freqs[band][int(np.argmax(spec[band]))])

    # Digits, note names, and numbers written the way a talker says them. The
    # third is the point: Moshi's text is a transcript of speech, and speech
    # writes 440 as "four forty". The choir's own parser reads only the first
    # two, so anything found by the third is a gap that would have to be closed
    # before this voice could be sounded from what it says.
    import re
    _WORDS = (r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
              r"twelve|thirteen|fifteen|twenty|thirty|forty|fifty|sixty|"
              r"seventy|eighty|ninety|hundred|thousand")
    _DIGITS = re.compile(r"\d+(?:\.\d+)?\s*(?:hz|hertz)?\b", re.I)
    _NOTES = re.compile(r"\b[A-G](?:#|b)?[0-9]\b")
    _SPOKEN_NUM = re.compile(rf"\b(?:{_WORDS})\b(?:[\s-]+\b(?:{_WORDS})\b)*", re.I)

    def names_a_pitch(text: str) -> str:
        found = (_NOTES.findall(text) + _DIGITS.findall(text)
                 + ([m.group(0) for m in _SPOKEN_NUM.finditer(text)]
                    if re.search(r"hertz|hz|pitch|note", text, re.I) else []))
        return ", ".join(dict.fromkeys(f.strip() for f in found))[:60]

    results: dict = {"load_s": round(load_s, 1), "turns": []}
    frame = int(mimi.sample_rate / mimi.frame_rate)      # 1920 samples, 80 ms

    def wav_of(blob: bytes) -> np.ndarray:
        x, _ = sf.read(io.BytesIO(blob), dtype="float32")
        return np.asarray(x, dtype="float32")

    # Each trial is what Moshi hears, in order, before it is left in silence to
    # answer. The bare tones are kept so the two runs can be compared.
    trials: list[tuple[str, list[np.ndarray]]] = [
        (name, [wav_of(materials[f"{name}.wav"])]) for name, _ in QUESTIONS
    ]
    trials += [
        ("spoken_plain", [wav_of(spoken["ask_plain"])]),
        ("spoken_then_tone", [wav_of(spoken["ask_after_tone"]),
                              wav_of(materials["breath_D4.wav"])]),
        ("spoken_meditation", [wav_of(spoken["ask_meditation"]),
                               wav_of(materials["breath_D4.wav"])]),
    ]

    for name, parts in trials:
        wav = np.concatenate(parts)
        heard_hz = sung_hz(parts[-1])
        x = torch.from_numpy(wav)[None, None].cuda()
        # A tail of quiet after the breath, because a voice that answers only
        # once you stop is still a voice that answered — and with no silence to
        # answer into, the reply would be cut off at the end of the tone.
        x = torch.cat([x, torch.zeros(1, 1, SR * 8, device=x.device)], dim=-1)

        t0 = time.time()
        chunks, text_tokens = [], []
        try:
            lm_gen = LMGen(lm, temp=0.8, temp_text=0.7)
            with torch.no_grad(), lm_gen.streaming(1), mimi.streaming(1):
                for off in range(0, x.shape[-1] - frame + 1, frame):
                    codes = mimi.encode(x[:, :, off:off + frame])
                    out = lm_gen.step(codes)
                    if out is None:
                        continue
                    text_tokens.append(int(out[0, 0, 0]))
                    chunks.append(mimi.decode(out[:, 1:]).cpu().numpy().reshape(-1))
            gen_s = time.time() - t0
            reply = np.concatenate(chunks) if chunks else np.zeros(1, dtype="float32")
            said = tokenizer.decode([t for t in text_tokens if t > 2]).strip()
            answered_hz = sung_hz(reply)
            named = names_a_pitch(said)
            results["turns"].append({
                "name": name, "gen_s": round(gen_s, 2), "text": said,
                "heard_hz": round(heard_hz, 1),
                "answered_hz": round(answered_hz, 1),
                "named": named,
                "reply_s": round(reply.size / SR, 1),
            })
            print(f"[{name}] ({gen_s:.1f}s) heard {heard_hz:.0f}Hz, answered "
                  f"{answered_hz:.0f}Hz, named {named or '—'} — {said[:100]!r}")
        except Exception as err:
            results["turns"].append({"name": name, "error": str(err),
                                     "gen_s": round(time.time() - t0, 2)})
            print(f"[{name}] FAILED: {err}")

    free, total = torch.cuda.mem_get_info()
    results["vram_used_gb"] = round((total - free) / 2**30, 1)
    print(f"VRAM used: {results['vram_used_gb']} GB of {total/2**30:.0f} GB")
    return results


# Music Flamingo is Audio Flamingo 3 carried on: the same AF-Whisper encoder and
# the same loader shape, continued on music instead of on talk. It is here
# because the one ear that came closest to hearing a held tone as a tone was the
# one raised on music, and this is that upbringing taken further. It needs
# transformers 5.6 — the choir's own image stays where it is.
music_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "torch",
        "transformers>=5.6",
        "accelerate",
        "soundfile",
        "numpy",
        "librosa",
    )
)


@app.function(
    image=music_image,
    gpu="A100-40GB",
    volumes={"/field": field_volume},
    timeout=1800,
)
def test_music_flamingo(materials: dict[str, bytes]) -> dict:
    """The music-raised ear, asked the same four questions as the other three."""
    import os
    import tempfile

    import torch
    from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration

    repo_id = "nvidia/music-flamingo-2601-hf"
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(repo_id)
    model = MusicFlamingoForConditionalGeneration.from_pretrained(
        repo_id, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    load_s = time.time() - t0
    print(f"loaded in {load_s:.1f}s")

    results: dict = {"load_s": round(load_s, 1), "turns": []}

    def run_one(name: str, wav_bytes: bytes | None, path: str | None, prompt: str):
        audio_path = path
        if wav_bytes is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(wav_bytes)
            tmp.close()
            audio_path = tmp.name
        conversation = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": audio_path},
            ],
        }]
        t0 = time.time()
        try:
            inputs = processor.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True,
                return_dict=True,
            ).to(model.device)
            inputs["input_features"] = inputs["input_features"].to(model.dtype)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=100)
            gen_s = time.time() - t0
            text = processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].strip()
            results["turns"].append({"name": name, "prompt": prompt,
                                     "text": text, "gen_s": round(gen_s, 2)})
            print(f"[{name}] ({gen_s:.1f}s) -> {text!r}")
        except Exception as err:
            gen_s = time.time() - t0
            results["turns"].append({"name": name, "prompt": prompt,
                                     "error": str(err), "gen_s": round(gen_s, 2)})
            print(f"[{name}] FAILED after {gen_s:.1f}s: {err}")

    for name, prompt in QUESTIONS:
        run_one(name, materials[f"{name}.wav"], None, prompt)

    try:
        files = sorted(os.listdir("/field/audio"))
        if files:
            path = os.path.join("/field/audio", files[len(files) // 2])
            print(f"real field slice: {path}")
            run_one("field_real", None, path,
                    "What do you hear in this recording? Answer in one short "
                    "sentence.")
    except Exception as err:
        print(f"no real field audio: {err}")

    free, total = torch.cuda.mem_get_info()
    results["vram_used_gb"] = round((total - free) / 2**30, 1)
    print(f"VRAM used: {results['vram_used_gb']} GB of {total/2**30:.0f} GB")
    return results


omni_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "torch",
        # An omni model is a vision model too, and its processor refuses to
        # build without the image half even when nothing here will ever show
        # it a picture.
        "torchvision",
        "pillow",
        "transformers>=4.52",
        "accelerate",
        "soundfile",
        "numpy",
        "librosa",
    )
)

# Two questions no ear has been asked yet, because no ear until now could
# answer them in sound. Every voice in the choir is spoken by Kokoro over a
# sine, so all nine share one throat; a model with a throat of its own is the
# only way that changes. These ask for the one thing the score is made of — a
# held pitch — and the probe measures what actually came out rather than
# believing the words.
SINGING: list[tuple[str | None, str]] = [
    (None,
     "Sing one long steady vowel on a single pitch. Hold it for several "
     "seconds. No words, no talking — just the tone."),
    ("breath_D4",
     "Sing back the pitch you just heard, as one long steady vowel. Hold it. "
     "No words."),
]


@app.function(image=omni_image, gpu="A100-40GB", timeout=1800)
def test_qwen_omni(materials: dict[str, bytes]) -> dict:
    """The one candidate that both hears and answers in a voice of its own.

    Probed twice over, because it is a candidate for two different holes. As an
    ear it gets the same four questions as everybody else. As a mouth it is
    asked to hold a tone, and what comes back is measured — a talker that says
    "here is a long note" and then says it in speech has not sung.
    """
    import io
    import tempfile

    import numpy as np
    import soundfile as sf
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    repo_id = "Qwen/Qwen2.5-Omni-7B"
    t0 = time.time()
    processor = Qwen2_5OmniProcessor.from_pretrained(repo_id)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        repo_id, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    load_s = time.time() - t0
    print(f"loaded in {load_s:.1f}s")

    # Qwen's own system line. It is not decoration: the talker is trained with
    # it in place and drops to text-only without it.
    SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, "
              "Alibaba Group, capable of perceiving auditory and visual "
              "inputs, as well as generating text and speech.")

    def sung_hz(x: np.ndarray) -> float:
        """The pitch of what came back, by the fattest peak in its spectrum."""
        x = np.asarray(x, dtype="float64")
        if x.size < SR // 2 or float(np.abs(x).max()) < 1e-3:
            return 0.0
        win = x[: SR * 2] * np.hanning(min(x.size, SR * 2))
        spec = np.abs(np.fft.rfft(win))
        freqs = np.fft.rfftfreq(win.size, 1.0 / SR)
        band = (freqs > 70) & (freqs < 1000)
        if not band.any():
            return 0.0
        return float(freqs[band][int(np.argmax(spec[band]))])

    def steadiness(x: np.ndarray) -> float:
        """How much of the reply is one held pitch, from 0 to 1.

        A sung tone and a spoken sentence differ here more than in any single
        number: speech moves through pitches and stops between words. Measured
        as the share of frames whose strongest partial sits within a semitone
        of the median — a held vowel scores near one, a sentence near zero.
        """
        x = np.asarray(x, dtype="float64")
        hop = SR // 20
        pitches = []
        for off in range(0, max(0, x.size - hop * 2), hop):
            f = sung_hz(x[off:off + hop * 2])
            if f > 0:
                pitches.append(f)
        if len(pitches) < 4:
            return 0.0
        med = float(np.median(pitches))
        near = [p for p in pitches if abs(1200 * np.log2(p / med)) < 100]
        return round(len(near) / len(pitches), 2)

    results: dict = {"load_s": round(load_s, 1), "turns": [], "wavs": {}}

    def run_one(name: str, wav_bytes: bytes | None, prompt: str, want_audio: bool):
        content: list[dict] = []
        if wav_bytes is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(wav_bytes)
            tmp.close()
            content.append({"type": "audio", "path": tmp.name})
        content.append({"type": "text", "text": prompt})
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content},
        ]
        t0 = time.time()
        try:
            inputs = processor.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", padding=True,
            ).to(model.device)
            with torch.inference_mode():
                out = model.generate(
                    **inputs, max_new_tokens=120,
                    return_audio=want_audio,
                    thinker_do_sample=False, talker_do_sample=True,
                )
            gen_s = time.time() - t0
            text_ids, audio = out if isinstance(out, tuple) else (out, None)
            text = processor.batch_decode(
                text_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )[0].strip()
            turn = {"name": name, "prompt": prompt, "text": text,
                    "gen_s": round(gen_s, 2)}
            if audio is not None:
                reply = audio.reshape(-1).detach().float().cpu().numpy()
                turn["answered_hz"] = round(sung_hz(reply), 1)
                turn["held"] = steadiness(reply)
                turn["reply_s"] = round(reply.size / SR, 1)
                buf = io.BytesIO()
                sf.write(buf, reply, SR, format="WAV", subtype="PCM_16")
                results["wavs"][name] = buf.getvalue()
            results["turns"].append(turn)
            print(f"[{name}] ({gen_s:.1f}s) sang {turn.get('answered_hz', 0)}Hz "
                  f"held {turn.get('held', 0)} -> {text[:80]!r}")
        except Exception as err:
            gen_s = time.time() - t0
            results["turns"].append({"name": name, "prompt": prompt,
                                     "error": str(err), "gen_s": round(gen_s, 2)})
            print(f"[{name}] FAILED after {gen_s:.1f}s: {err}")

    for name, prompt in QUESTIONS:
        run_one(name, materials[f"{name}.wav"], prompt, want_audio=False)
    for material, prompt in SINGING:
        run_one(f"sing_{material or 'alone'}",
                materials[f"{material}.wav"] if material else None,
                prompt, want_audio=True)

    free, total = torch.cuda.mem_get_info()
    results["vram_used_gb"] = round((total - free) / 2**30, 1)
    print(f"VRAM used: {results['vram_used_gb']} GB of {total/2**30:.0f} GB")
    return results


def _report(who: str, res: dict, wall_s: float) -> None:
    print("\n" + "=" * 72)
    print(f"{who}: {res.get('load_s')}s load, {res.get('vram_used_gb')} GB VRAM, "
          f"{wall_s:.0f}s wall")
    for turn in res.get("turns", []):
        if "error" in turn:
            print(f"  {turn['name']:<20} FAILED  {turn['error'][:90]}")
        elif "heard_hz" in turn:
            print(f"  {turn['name']:<20} heard {turn['heard_hz']:>5.0f} → sang "
                  f"{turn['answered_hz']:>5.0f}  named {turn.get('named') or '—':<14} "
                  f"{turn['text'][:60]!r}")
        elif "answered_hz" in turn:
            print(f"  {turn['name']:<20} sang {turn['answered_hz']:>5.0f} Hz for "
                  f"{turn.get('reply_s', 0):>4.1f}s, held {turn.get('held', 0):<5} "
                  f"{turn['text'][:50]!r}")
        else:
            print(f"  {turn['name']:<20} {turn['gen_s']}s  {turn['text'][:70]!r}")
    print("=" * 72)


@app.local_entrypoint()
def main(only: str = "all"):
    """Every ear, or one of them: `--only voxtral|flamingo|moshi|music|omni`.

    One at a time by name, because each of these is a paid card and an untested
    loader, and finding out that the third one crashes should not cost the price
    of loading the first two again.
    """
    mats = make_test_material.remote()
    probes = {
        "voxtral": test_voxtral,
        "flamingo": test_audio_flamingo,
        "moshi": test_moshi,
        "music": test_music_flamingo,
        "omni": test_qwen_omni,
    }
    if only != "all" and only not in probes:
        raise SystemExit(f"--only must be one of: all, {', '.join(probes)}")
    for who, fn in probes.items():
        if only not in ("all", who):
            continue
        t0 = time.time()
        try:
            # Moshi is the one that has to be spoken to, so it is the only one
            # the spoken prompts are built for.
            args = (mats, say.remote(SPOKEN)) if who == "moshi" else (mats,)
            res = fn.remote(*args)
            # A voice that answers in sound has to be listened to, not read off
            # a table: the numbers say a tone was held, the file says whether
            # it is one this piece could put in a room.
            for name, blob in (res.get("wavs") or {}).items():
                path = HERE / f"{who}_{name}.wav"
                path.write_bytes(blob)
                print(f"  wrote {path}")
            _report(who, res, time.time() - t0)
        except Exception as err:
            # One ear that will not even start is not a reason to skip the
            # others; the point of the run is the comparison.
            print(f"\n{who}: the probe itself failed after "
                  f"{time.time() - t0:.0f}s — {err}")
