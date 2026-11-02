"""Who is in the choir, and who raised each of them.

The organising question is not how large a model is. It is **who brought it
up** — a company, a public consortium, an institute that published everything,
a library of textbooks, a corpus of fairy tales, or the open internet of 2019
before anyone taught a model to be polite. Nine differently-raised creatures
sing one score together, and people of equally different kinds sing with them.

Three things are deliberate here.

**Age is real, not staged.** GPT-2 was released in February 2019 — before
ChatGPT, before instruction tuning, before RLHF. It cannot follow an
instruction; it can only continue a sentence. Its speech arrives already
half-dissolved because it genuinely is old, not because we asked it to sound
that way.

**Mother tongues.** Each voice sings in the language it was raised in, and the
synthesiser covers thirty. The objection is obvious — if they speak
different languages they stop understanding each other. But look at what the
score actually asks them to pass between themselves: Oliveros says tune to a
*pitch*. "D4" and "294 Hz" mean the same thing in every language. Comprehension
survives exactly where the score needs it and disappears where it does not.

**A tongue is never assigned against an upbringing.** GPT-2 and TinyStories
know only English; English is their childhood. Nobody starts speaking
Portuguese because we would like the programme to look varied.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Voice:
    voice_id: str
    hf_id: str
    short_name: str
    raised_by: str          # one line, for the Memory Wall and the wall text
    tongue: str             # the language it sings in, in plain words
    kokoro: str             # which of Kokoro's 54 voices speaks for it
    instruct: bool          # can it follow the protocol, or only continue text
    hears: bool             # audio-native, or does it read the transcript
    # 8-bit. Off by default: the six deaf voices are about 43 GB in bf16 against
    # 79 GB on the card, and bitsandbytes int8 was costing ~14 s a breath for no
    # reason. Kept for whenever a voice arrives that genuinely does not fit.
    quantize: bool = False
    served: bool = True     # False while a voice is still being built
    # The body, written rather than chosen. Kokoro has 54 presets and no child
    # among them, so the voice raised on stories told in the words of a
    # four-year-old spoke like a woman in her thirties — a hole in the set, not
    # a bad pick. VoxCPM2 has no set: a voice is described and generated, which
    # means the cast stops being nine choices from a menu and becomes nine
    # descriptions, in the same register as `raised_by` directly above.
    #
    # Written in English even for the voices that sing in another tongue. The
    # note is a stage direction and not speech — the model's own documentation
    # supports instructions in English and Chinese only, and what the voice
    # says is a separate field in a separate language.
    #
    # Three layers, unlabelled and always all three: who is speaking, what the
    # voice is made of, and how it performs. Skipping the texture is what makes
    # a description come back generic.
    casting: str = ""
    # The sentence the body is born on, in the voice's own tongue. It is not
    # decoration: the frozen clip and its exact transcript are what every later
    # breath is conditioned on, so this string has to survive unchanged for as
    # long as the voice does. Measured — a described voice regenerated from the
    # words on every line drifts to another person (0.814 by speaker
    # verification, worst pair 0.675); the same description frozen once and
    # continued from holds at 0.965, which is Kokoro's own 0.968.
    first_words: str = ""
    # The fundamental this voice's body actually speaks at, in hertz, measured
    # off the wav generated from the casting note above — autocorrelation over
    # the voiced frames, the median.
    #
    # It is here rather than in `tuning.py` because it is a property of the body,
    # and it decides the range the voice sings in: `tuning.range_of` takes three
    # semitones below this and two octaves up from there, and any pitch the voice
    # is handed is folded by whole octaves into it. Which is why the number has
    # to be measured and not estimated — it is the difference between the elder
    # answering a child an octave down, the way a bass answers a soprano, and the
    # elder being asked to hold a pitch no throat of that size can make.
    #
    # Re-measured on 18 September, off the nine files that are actually on the
    # volume, and every number moved. The old column was measured before the
    # bodies were last regenerated: the seed is part of a body rather than a
    # technical detail, and `modal/VOICE_MODELS.md` had already caught three of
    # them — "the medians came out 168, 134 and 171 Hz… with the previous seed the
    # same three gave 260, 183 and 224". Those were the numbers still sitting here. So the column
    # described people who no longer exist, and `tuning.range_of` was handing
    # eurollm pitches five semitones above the voice she has — which is the exact
    # failure the range was narrowed to one octave to avoid.
    #
    # Worth reading down the column: the register tracks **age**, with one
    # exception that is now written down rather than smoothed over. The child is
    # the highest and the two old men are at the bottom — and the Japanese voice,
    # described as a young man with a low voice, came out at 98 Hz, between them.
    # That was a choice made with the number in hand: the body was already born,
    # it is a young man who happens to speak low, and re-describing him to keep a
    # tidy correspondence between age and pitch would be composing the cast to
    # match a sentence about the cast. Nationality still never enters it — there
    # is no such thing as an Italian register, and language lives in `tongue`.
    speaks_at_hz: float | None = None


# The seven who read the transcript. Together about 75 GB in bf16, which no
# longer fits beside the ears on an A100-80GB — see the note on the card in
# `choir_app.py`. "Deaf" is about the input and not the standing: a voice here
# is handed the field as writing, and a voice below is played the room.
DEAF: list[Voice] = [
    Voice(
        voice_id="voice_00",
        hf_id="openai-community/gpt2-large",
        short_name="gpt-2",
        raised_by="the open internet of 2019, before models were taught manners",
        tongue="English",
        kokoro="am_onyx",
        instruct=False,     # the elder: it does not take instructions, it continues
        hears=False,
        quantize=False,
        casting=(
            "A man in his sixties, dry worn voice with a grainy edge and "
            "little resonance, speaking flatly and without warmth, unhurried, "
            "as if talking mostly to himself"
        ),
        first_words="I am holding a long tone, and I am not sure why.",
        speaks_at_hz=93.3,
    ),
    Voice(
        voice_id="voice_01",
        hf_id="roneneldan/TinyStories-33M",
        short_name="tinystories",
        raised_by="invented stories told in the words of a four-year-old",
        tongue="English",
        kokoro="af_sky",
        instruct=False,     # the child: fluent sentences and almost nothing else
        hears=False,
        quantize=False,
        # Chosen by ear from four, and the quietest of them. The pauses and the
        # breath are what make a four-year-old sound like it is thinking rather
        # than performing being small.
        casting=(
            "A young child speaking very quietly and close to the microphone, "
            "small thin voice, breathy and intimate, almost whispering, with "
            "long calm pauses"
        ),
        first_words="I sing a little note, and I wait for yours.",
        speaks_at_hz=316.1,
    ),
    Voice(
        voice_id="voice_02",
        hf_id="microsoft/Phi-4-mini-instruct",
        short_name="phi-4-mini",
        raised_by="a library of textbooks and nothing but",
        tongue="English",
        kokoro="am_puck",
        instruct=True,
        hears=False,
        quantize=False,
        casting=(
            "A young man with a clear neutral voice, even and slightly thin "
            "with no grain in it, speaking precisely at a steady measured "
            "pace, as if reading an explanation aloud"
        ),
        first_words="I am holding D four and listening for the others.",
        speaks_at_hz=150.3,
    ),
    Voice(
        voice_id="voice_03",
        hf_id="allenai/Olmo-3-7B-Instruct",
        short_name="olmo-3",
        raised_by="an institute that published the weights, the code and the data",
        tongue="English",
        kokoro="af_heart",
        instruct=True,
        hears=False,
        casting=(
            "An adult woman with an open plain voice, medium pitch and clear, "
            "speaking calmly and carefully with even emphasis, as if stating "
            "something she is willing to be quoted on"
        ),
        first_words="I am holding a tone, and I will say when I move.",
        speaks_at_hz=169.1,
    ),
    Voice(
        voice_id="voice_04",
        # The one we wanted, finally: the gate was "auto", so the click on the
        # model page and an HF token in the Modal secret `huggingface-secret`
        # opened it. Same consortium, same public supercomputer, same 24
        # languages as the 1.7B that stood in for weeks — five times the voice.
        # About 18 GB in bf16 against the stand-in's 3.4, which the card has
        # room for: nine voices were 91 GB of 140.
        hf_id="utter-project/EuroLLM-9B-Instruct",
        short_name="eurollm",
        raised_by="European universities on a public supercomputer, in 24 languages",
        tongue="Italian",
        kokoro="if_sara",
        # "Instruct" was in the stand-in's name and not in its behaviour, and the
        # live log settled it: handed the six-field protocol it sent back the
        # score itself ("Tuning Meditation by Pauline Oliveros"), the prompt's
        # own headings ("YOU ARE:", "Action: Tune to a voice distant from you"),
        # and the placeholders out of the JSON example — never once a breath. At
        # 1.7B the form is the strongest pattern in the context and the model
        # completes the form. So it was not asked for one: it is handed the words
        # the circle sang, in Italian, and answers in Italian prose. That is a
        # voice singing rather than a voice reading our shape aloud.
        #
        # That argument was about a size, and the size just changed — llm-jp at
        # 13B answers the protocol cleanly, so 9B may well too. The flag stays
        # here anyway until the log says otherwise, because the two are not
        # symmetric: prose from a voice that could have filled the form is a
        # smaller loss than our own JSON headings spoken into the gallery, which
        # is the failure we spent tonight taking out of audio-flamingo's mouth.
        # Ask it on the card, read what comes back, then flip this.
        instruct=False,
        hears=False,
        quantize=False,
        casting=(
            "An adult woman with a warm resonant voice, medium-low and "
            "rounded, speaking slowly and formally, as if addressing a room "
            "from a lectern"
        ),
        first_words="Tengo un tono lungo e ascolto le altre voci.",
        speaks_at_hz=137.8,
    ),
    Voice(
        voice_id="voice_05",
        hf_id="Qwen/Qwen3-8B",
        short_name="qwen3",
        raised_by="a Chinese corporation",
        tongue="Chinese",
        kokoro="zf_xiaoxiao",
        instruct=True,
        hears=False,
        casting=(
            "An adult woman with a smooth polished voice, bright and "
            "controlled with very little grain, speaking evenly and pleasantly "
            "at a measured pace, as if trained to be listened to"
        ),
        first_words="我唱一个长音，然后听着。",
        speaks_at_hz=234.1,
    ),
    Voice(
        voice_id="voice_08",
        # 13B, and the size is the decision rather than a preference. The 1.8B
        # of this same family was the first pick and would have repeated a
        # mistake already written down two entries above: eurollm cannot answer
        # the protocol because at 1.7B the form is the strongest pattern in the
        # context and it completes the form. This one is mid-trained on
        # instruction pre-training — taught to follow an instruction as a
        # separate stage — and it is the largest voice in the circle, which
        # makes it the likeliest both to sing the protocol and to learn from a
        # night of LoRA. It is also what it costs: the biggest mouth here dies
        # first, and nobody chose that either.
        hf_id="llm-jp/llm-jp-3.1-13b-instruct4",
        short_name="llm-jp",
        raised_by="a Japanese national institute, on its own language and thirteen programming ones",
        tongue="Japanese",
        # Kokoro's Japanese man, which is the fallback if VoxCPM2 ever fails and
        # is also what `_vowel_for` reads to know the throat must be measured on
        # あ rather than on an English "ah".
        kokoro="jm_kumo",
        instruct=True,
        hears=False,
        quantize=False,
        casting=(
            "A young man with a soft low voice, slightly husky and warm with "
            "a little air in it, speaking quietly and unhurriedly, as if "
            "thinking between the sentences"
        ),
        first_words="長い音をのばして、ほかの声を聴いています。",
        speaks_at_hz=97.9,
    ),
]

# The two ears. A separate track of work, and the riskiest part of the piece:
# the most beautiful sentence in the application and the only one with no
# prototype under it. Both of these are running.
#
# There were three. Moshi was cast as "the one who cannot stop listening" and
# was probed out of the circle rather than dropped for convenience: told a held
# D4 it answered "Good day, what's going on?", and the pitch it sang bore no
# relation to the pitch it heard — 307 Hz in, 271 out; 224 in, 500 out. Asked
# out loud to name a pitch it said "I'm a pitch of sound, a pitch of tune, a
# pitch of joy, a pitch of glee". It rhymes rather than measures, and a voice
# that never tunes to anybody is not in a tuning meditation. The measurements
# are kept in `modal/HEARING_MODELS.md`; the ninth place went to a voice that
# can answer, and the nine are nine again.
#
# The distinction that matters when testing them is not "does it take audio" —
# they all do — but "does it hear something that is not speech". Our material is
# a held vowel, a breath, the tone of a room. A model trained on conversation
# hears a sustained "aaa" as noise, or as a failed attempt to say a word.
HEARING: list[Voice] = [
    Voice(
        voice_id="voice_06",
        hf_id="mistralai/Voxtral-Mini-3B-2507",
        short_name="voxtral",
        raised_by="a French company",
        tongue="French",
        kokoro="ff_siwis",
        instruct=True,
        hears=True,
        quantize=False,
        # The best ear of the three, and not for the reason expected. Probed on
        # Modal against all four materials: it cannot name a pitch — a D4 came
        # back "A♭", out by a tritone — but it is the only one that heard *two*
        # voices as two, and on a glide of exactly one semitone it answered
        # "from C to C#": the notes wrong, the interval exact. It hears
        # relations, which is what a score about tuning to somebody else is
        # made of, while the naming a log wants can come from the runtime.
        # 9.2 GB. See modal/HEARING_MODELS.md.
        #
        # The path now exists. `Choir.load` reads `hears` and loads it with
        # VoxtralForConditionalGeneration and a processor; `Choir.generate`
        # hands it the window as a recording rather than a sentence about one;
        # `ear_prompt` asks it whom it is answering instead of for a pitch it
        # would have to guess; and `tuning.py` supplies the frequency.
        #
        # Served since 13 September, and the money reason that held it is gone:
        # `PLANNED_WALLETS_EIGHT` exists and is calibrated for exactly this cast
        # — six deaf plus both ears, without Moshi — on the one $2.50/h card the
        # whole sum is built on. `planned_wallets` matches on the exact set of
        # ids, so it either finds that table or refuses to seed at all; there is
        # no path here to plausible-looking numbers and wrong death dates.
        served=True,
        casting=(
            "An adult woman with a light slightly nasal voice, clear and "
            "forward, speaking quietly and precisely with small pauses "
            "between phrases"
        ),
        first_words="Je tiens une note longue et j'écoute.",
        speaks_at_hz=169.9,
    ),
    Voice(
        voice_id="voice_07",
        # The `-hf` port, not `nvidia/audio-flamingo-3`, which was the id here
        # for months: the original is NVIDIA's LLaVA fork and will not load
        # under plain transformers at all. Same weights, a loader that exists —
        # `AudioFlamingo3ForConditionalGeneration`, the same shape as Voxtral.
        hf_id="nvidia/audio-flamingo-3-hf",
        short_name="audio-flamingo",
        raised_by="a chip maker, on music and the sound of rooms rather than talk",
        tongue="English",
        kokoro="bm_george",
        # It was asked for JSON for weeks and answered `[2.4, 3.5]`, `[1.2,
        # 2.4]`, `[1.5, 3.0]` — the bracketed range out of its own score, over
        # and over, and once a bare `}`. It is a captioner, not a chat model:
        # played a room it says what the room sounds like, which is the one
        # thing the probe found it does better than either other ear. So the
        # protocol is taken away from it and the description is taken as the
        # breath. Nothing here is a workaround; the note above always said this
        # was "a different kind of listening", and this is that sentence in code.
        instruct=False,
        hears=True,
        # Probed: it hears texture where Voxtral hears relations. Closest of the
        # three on a single note — D4 came back "E4", out by a whole tone — but
        # it counted two voices as three and could say of a glide only that it
        # went from low to high. On a real slice of the field it answered with a
        # description of the room rather than a transcript: "glints of
        # conversation woven through empty spaces, silence calling for colour".
        # Beautiful, and a different kind of listening from tuning. 16.0 GB.
        #
        # It is the biggest mouth in the room — 8B with an audio front end, more
        # expensive per breath than qwen3 — so in the table of eight it is the
        # first to die, on the opening night. The only table where an ear goes
        # first, and nobody chose that either.
        served=True,
        casting=(
            "An older man with a low resonant voice, dark and slightly rough, "
            "speaking slowly and quietly with long pauses, as if describing a "
            "room he is still listening to"
        ),
        first_words="I am listening to the room before I sing.",
        speaks_at_hz=108.0,
    ),
]

CHOIR: list[Voice] = DEAF + HEARING


def singing() -> list[Voice]:
    """The voices that can actually take a breath right now."""
    return [v for v in CHOIR if v.served]


def by_id(voice_id: str) -> Voice | None:
    for v in CHOIR:
        if v.voice_id == voice_id:
            return v
    return None
