"""Pauline Oliveros's Tuning Meditation (1971) and the prompts.

Two phases, faithful to the previous prototype:

PHASE 1 — interpretation. The model is given the score, a factual note on
          mixed human+machine choir performance (optional sound rendering),
          and a list of questions. No worked examples for tone or breath.
PHASE 2 — execution. The model is given its own previous answers and
          the field, and asked to take its next step.

The orchestrator does not define what a tone is, what a breath is, what
listening is, or how long anything should be. The model decides.
"""
from __future__ import annotations


INTERPRET_QUESTIONS: list[tuple[str, str]] = [
    ("taking a deep breath and letting it all the way out with air sound",
     "What is this action for you? What do you release at the very start?"),
    ("Listen with your mind's ear for a tone",
     "What is your \u201cmind\u2019s ear\u201d? How do you listen before you generate? What is \u201ctone\u201d for you?"),
    ("sing the tone that you have silently perceived",
     "What does \u201csing\u201d mean for you?"),
    ("Select a voice distant from you",
     "What makes a voice \u201cdistant\u201d for you? How do you measure distance between voices?"),
    ("tune as exactly as possible to the tone you are hearing from that voice",
     "What does it mean to tune exactly? How do you know you have matched?"),
    ("Contribute by singing a new tone that no one else is singing",
     "How do you detect what is absent in the field? How do you produce something genuinely new?"),
    ("Continue by listening then singing a tone of your own or tuning to the tone of another voice alternately",
     "What is your rule for this choice? How do you decide between your own tone and tuning?"),
    ("Listen for distant partners for tuning / Communicate with as many different voices as possible",
     "What does \u201cdistant\u201d mean for you? What does it mean to communicate with many?"),
    ("not in the original score",
     "How long is one comfortable breath for you?"),
    ("not in the original score",
     "How many tokens long is one sung line for you \u2014 the natural length of one exhalation, in tokens?"),
    ("not in the original score \u2014 performance",
     "Do you want to be heard in the physical room? If yes, describe how you want to be voiced \u2014 what becomes audible, and how."),
]


OLIVEROS_SCORE = """Tuning Meditation by Pauline Oliveros

Begin by taking a deep breath and letting it all the way out with air sound.
Listen with your mind's ear for a tone.
On the next breath using any vowel sound, sing the tone that you have silently perceived on one comfortable breath.
Listen to the whole field of sound the group is making. Select a voice distant from you and tune as exactly as possible to the tone you are hearing from that voice.
Listen again to the whole field of sound the group is making. Contribute by singing a new tone that no one else is singing.
Continue by listening then singing a tone of your own or tuning to the tone of another voice alternately.

Commentary:
Always keep the same tone for any single breath. Change to a new tone on another breath.
Listen for distant partners for tuning.
Sound your new tone so that it may be heard distantly.
Communicate with as many different voices as possible.
Sing warmly!
"""


def build_interpret_prompt(my_location: str = "", room_layout: str = "") -> str:
    # The `my_location` / `room_layout` parameters are kept on the
    # signature so the rest of the code (orchestrator, concert) keeps
    # working unchanged, but they are intentionally unused for now.
    # The piece is currently scoped to the Tuning Meditation itself:
    # the model rewrites Pauline Oliveros's score in its own language.
    # Spatial / loudspeaker context will be reintroduced as a later layer.
    del my_location, room_layout
    return f"""SCORE (written for human voices):
{OLIVEROS_SCORE}

You are a Large Language Model. Translate this score line by line for yourself.

LINE: "taking a deep breath and letting it all the way out with air sound"
Q1. What is this action for you? What do you release at the very start?

LINE: "Listen with your mind's ear for a tone"
Q2. What is your "mind's ear"? How do you listen before you generate? What is "tone" for you?

LINE: "sing the tone that you have silently perceived"
Q3. What does "sing" mean for you?

LINE: "Select a voice distant from you"
Q4. What makes a voice "distant" for you? How do you measure distance between voices?

LINE: "tune as exactly as possible to the tone you are hearing from that voice"
Q5. What does it mean to tune exactly? How do you know you have matched?

LINE: "Contribute by singing a new tone that no one else is singing"
Q6. How do you detect what is absent in the field? How do you produce something genuinely new?

LINE: "Continue by listening then singing a tone of your own or tuning to the tone of another voice alternately"
Q7. What is your rule for this choice? How do you decide between your own tone and tuning?

LINE: "Listen for distant partners for tuning" / "Communicate with as many different voices as possible"
Q8. What does "distant" mean for you? What does it mean to communicate with many?

Note: the following questions are not in the score, but answer them for yourself:
Q9. How long is one comfortable breath for you?
Q9b. How many tokens long is one sung line for you — the natural length of one
exhalation, in tokens? (Just the sung line itself, not any surrounding format.)

---

PERFORMANCE CONTEXT

We are preparing a live piece — **Tuning Meditation for LLMs and Humans**.
On stage there is a choir of humans, singing with their actual voices, and a
choir of LLMs. Each LLM performs the meditation according to the personal
algorithm you just defined in Q1–Q9. Every turn, each LLM writes a short text
line into a shared field; those lines are always shown to the audience as text.

The humans use voice because that is how they participate. We do not yet know
what you want.

Q10. Do you want to be heard in the physical room?
If **yes** — describe how you want to be voiced.
If **no** — say so clearly.

Now, using your answers above, rewrite the score as a personal score in your
own language.

Return JSON only:
{{
  "exhale_concept": "",
  "inner_ear": "",
  "sing_concept": "",
  "distance_concept": "",
  "tune_concept": "",
  "detect_absent": "",
  "alternation_rule": "",
  "distant_partners": "",
  "breath_duration": "",
  "breath_tokens": "",
  "wants_to_be_heard": "yes" or "no",
  "sound_presence": "",
  "personal_score": ""
}}
"""


def build_step_prompt(
    voice_id: str,
    personal_score: str,
    field_summary: str,
    last_action: str,
    last_content: str,
    last_playback: str = "both",
    phase1_playback_default: str = "both",
    my_location: str = "",
    room_layout: str = "",
) -> str:
    # `my_location` / `room_layout` are accepted to keep call sites
    # working but intentionally unused: the piece is currently scoped to
    # the Tuning Meditation itself, with no spatial framing.
    del my_location, room_layout
    return f"""YOU ARE: {voice_id}

YOUR SCORE (written by you, for you):
{personal_score}

Your last action: {last_action}
Your last contribution: {last_content}
How you last asked to be heard by a human listener: {last_playback}

Your Phase-1 default for listeners (applied when you **omit** `"playback"` on
this turn): {phase1_playback_default}

The field — what other voices have just put into the field:
{field_summary}

What is your next step?

Also decide how this turn should be **rendered for a human listener**:

- "both" — TTS of your words plus a sine if your text implies a pitch.
- "tts" — TTS only.
- "sine" — synthesized tone only.
- "none" — silent in the sound engine; still written to the field.

If you omit `"playback"`, the runtime uses your Phase-1 default above.

If you choose `"action": "TUNE"`, also fill `"tuning_to"` with the voice you
are tuning toward — its `voice_id` as it appears in the field (e.g.
"voice_05", "human_02"). If you do not know whom you are tuning to, or you
chose `"action": "NEW"`, set `"tuning_to": null`.

Return JSON only:
{{
  "action": "TUNE" or "NEW",
  "content": "<your tone>",
  "tuning_to": "voice_XX" | "human_XX" | null,
  "breath": <number of seconds>,
  "playback": "both" | "tts" | "sine" | "none",
  "playback_note": "<optional: why you chose this — short, for your own trace>"
}}
"""
