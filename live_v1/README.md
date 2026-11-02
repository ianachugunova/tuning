# live_v1 — what the exhibition inherited from the June concert

*Tuning Meditation: LLMs* was first performed live on 6 June 2026 at the Center for
New Music, San Francisco, with API‑hosted models and human singers. The exhibition
choir (`portal/`) reuses a handful of its modules unchanged, so that the score the
machines read and the way a pitch is found in text stayed the same from the concert to
the gallery. Only those modules are kept here.

| file | what the choir takes from it |
|---|---|
| `score.py` | the prompt that puts Oliveros's score in front of a model (`build_step_prompt`, `build_interpret_prompt`) |
| `pitch_text.py` | finding pitches in free text: `extract_all_pitches`, `strip_leading_pitch` |
| `orchestrator.py` | the `Event` record and the parsers for a model's answer (`_robust_json`, `_parse_breath_duration`) |
| `models.py`, `sound.py` | the June model list and `playback_coerce`, needed by `orchestrator` |
| `human.py` | the YIN pitch detector the gallery microphones use (`room.py`) |

The concert's own runner, analysis, stem rendering and PDF export are not part of this
repository.
