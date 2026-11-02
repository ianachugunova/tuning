# Tuning Meditation: LLMs — live_v1

Live (real-time) version of *Tuning Meditation: LLMs* for the upcoming
"Tuning Meditation for LLMs and Humans" performance.

This is **not** a re-render of a previously recorded session (that lives in
`../sound_v1/`). This is the live orchestrator: 10 different LLMs, all
running concurrently through OpenRouter, breathing on a shared clock,
listening to the same shared event field, and producing one tone per
breath.

## Day 1 scope (today)

Text-only. No audio, no microphone yet. The point is to confirm that:

- 10 different models can be called in parallel through OpenRouter
- They can run continuously for 3+ minutes without falling over
- Each agent can read the recent shared field and respond to it
- Latency and cost match the budget we estimated

## How to run

From the repo root:

```bash
python3 -m Tuning_Meditation_LLMs.live_v1.main \
    --duration 180 \
    --agents 10
```

Each session creates a self-contained folder under `out/`:

```
out/
  runs_index.csv                 — one row per run, cross-run overview
  run_2026-05-09_01-32-29/
    meta.json                    — full run config + summary stats
    events.jsonl                 — every event in raw form (one JSON per line, streamed live)
    events.csv                   — only "tone" events, tabular
    interpretations.json         — Phase 1 self-definitions (raw)
    interpretations.csv          — Phase 1 self-definitions, tabular
    agents.csv                   — per-agent stats: breaths, TUNE/NEW counts, avg breath, avg latency
```

The folder name is an ISO-style UTC timestamp so runs sort chronologically
in the file system. Open any of the CSVs in Numbers, Excel, or Cursor to
compare across runs — each model interprets the score differently every
time, and `interpretations.csv` is where that variation is most visible.

You can `tail -f out/run_<id>/events.jsonl` during a session to watch
events arrive in real time (this is also the source for the live video
panels in `panels_v1/`, when those exist).

## What is intentionally not in Day 1

- No audio output.
- No microphone input.
- No human voices.
- No MIDI.
- No Darwinian "self-interpret the score" phase from the older prototype.
  Each agent gets a fixed, well-tested system prompt directly.

These come on Days 2–4.

## Architecture (Day 1)

```
                ┌──────────────────────────────┐
                │   Conductor (asyncio loop)   │
                │   shared clock + event field │
                └───────┬───────────────┬──────┘
                        │               │
       ┌────────────────┘               └────────────────┐
       ▼                                                  ▼
  10 × Agent (one per LLM)                   Field (in-memory + JSONL)
  - own breath duration                      - append-only events
  - own model + provider                     - listening window query
  - hard timeout per call
  - retry on 429 once
```

Each agent is an independent async task. They all share one `Field` object
which is the only synchronization point. No locks: the `Field` is single-writer
per agent and append-only.
