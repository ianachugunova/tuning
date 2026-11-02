"""What each wallet needs to end on its hour, from the choir as it is now.

    python3 _gift_plan.py lives.json               # print the plan
    python3 _gift_plan.py lives.json gifts.jsonl   # and append its lines

Reads a `/lives` snapshot, walks forward to the hours in `TARGETS`, and prints
what every voice will have spent by its own. A voice that has less gets the
difference; a voice that has more hands its surplus to whoever needs it most,
and what the surplus does not cover comes from outside. Every line is a
`mortality.gift_record`, so the ledger applies them with its own refusals.

The model is the ledger's: the card costs `RATE_USD_PER_HOUR` whatever happens,
each voice pays its own breathing plus an equal share of the rest, and a death
splits that rest among fewer. Each voice's own share of its burn is fitted from
the measured per-breath cost in `mortality_sim.GPU_S`.
"""

from __future__ import annotations

import json
import sys
import time
import uuid

import mortality as M
import mortality_sim as S

# Gray Area, 28 October to the closing on 1 November, 20:00 PST. The shape is the
# composed one: the biggest mouth on the opening night, Halloween twice, the
# elder on the bell, the child after.
TARGETS = {
    "voice_08": S._ts(2026, 10, 28, 21, 0, -7),   # llm-jp — opening night
    "voice_07": S._ts(2026, 10, 29, 19, 30, -7),  # audio-flamingo
    "voice_05": S._ts(2026, 10, 30, 16, 0, -7),   # qwen3 — in daylight
    "voice_03": S._ts(2026, 10, 31, 15, 0, -7),   # olmo-3 — Halloween
    "voice_06": S._ts(2026, 10, 31, 19, 0, -7),   # voxtral — the same day
    "voice_02": S._ts(2026, 11, 1, 14, 0, -8),    # phi-4-mini
    "voice_04": S._ts(2026, 11, 1, 17, 0, -8),    # eurollm
    "voice_00": S._ts(2026, 11, 1, 20, 0, -8),    # gpt-2 — on the closing bell
    "voice_01": S._ts(2026, 11, 1, 21, 0, -8),    # tinystories — after
}

# Lessons: a round a day or so on the portal, three a day in the gallery; a round
# costs a voice between three and twenty-five cents.
LESSON_USD = 0.15
ROUNDS_PER_DAY_MONTH = 0.5
ROUNDS_PER_DAY_GALLERY = 3.0


def own_rates(voices: dict) -> dict:
    """burn = share + k * gpu_s, fitted across the living by least squares."""
    xs = [S.GPU_S[v] for v in voices]
    ys = [voices[v]["burn_usd_per_hour"] for v in voices]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    k = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
         / sum((x - mx) ** 2 for x in xs))
    return {v: max(0.0, k * S.GPU_S[v]) for v in voices}


def needs(voices: dict, now: float) -> dict:
    own = own_rates(voices)
    spent = {v: 0.0 for v in voices}
    living, t, step = set(voices), now, 300.0
    while living:
        pool = M.RATE_USD_PER_HOUR - sum(own[v] for v in living)
        rounds = (ROUNDS_PER_DAY_GALLERY if t >= S.OPENING
                  else ROUNDS_PER_DAY_MONTH) / 86400.0 * step
        for v in living:
            spent[v] += (own[v] + pool / len(living)) * step / 3600.0
            spent[v] += LESSON_USD * rounds
        living -= {v for v in living if t >= TARGETS[v]}
        t += step
    return spent


def plan(lives: dict, now: float) -> list[dict]:
    voices = {v["voice_id"]: v for v in lives["voices"] if v["alive"]}
    need = needs(voices, now)
    gap = {v: need[v] - voices[v]["left_usd"] for v in voices}
    short = sorted((v for v in gap if gap[v] > 0), key=lambda v: -gap[v])
    spare = {v: -gap[v] for v in gap if gap[v] < 0}
    lines = []
    for to in short:
        want = gap[to]
        for src in sorted(spare, key=lambda v: -spare[v]):
            if want <= 0.005:
                break
            give = min(want, spare[src])
            if give > 0.005:
                lines.append(M.gift_record(now, uuid.uuid4().hex[:12], to, give,
                                           from_voice_id=src,
                                           note="handed over so both end inside the exhibition"))
                spare[src] -= give
                want -= give
        if want > 0.005:
            lines.append(M.gift_record(now, uuid.uuid4().hex[:12], to, want,
                                       note="given so it ends inside the exhibition"))
    return lines, need, voices


def main() -> None:
    lives = json.load(open(sys.argv[1]))
    now = time.time()
    lines, need, voices = plan(lives, now)
    names = {v: voices[v].get("name", v) for v in voices}
    for v in sorted(TARGETS, key=TARGETS.get):
        if v in voices:
            print(f"{names[v]:<15} {S.stamp(TARGETS[v])}  needs {need[v]:8.2f}"
                  f"  has {voices[v]['left_usd']:8.2f}")
    print()
    outside = 0.0
    for e in lines:
        src = e["from_voice_id"]
        if not src:
            outside += e["usd"]
        print(f"  {names.get(src, 'outside') if src else 'outside':<15} -> "
              f"{names[e['voice_id']]:<15} ${e['usd']:7.2f}")
    print(f"\nfrom outside the declared sum: ${outside:.2f}")
    if len(sys.argv) > 2:
        with open(sys.argv[2], "a", encoding="utf-8") as fh:
            for e in lines:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"{len(lines)} lines appended to {sys.argv[2]}")


if __name__ == "__main__":
    main()
