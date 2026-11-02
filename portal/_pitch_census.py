"""What the choir actually sings, voice by voice — for the speaker decision.

The equipment request promised Gray Area an answer by 15 October: two 8-inch
cabinets for the two lowest voices, or nine small ones — decided by whether
the voices, as they train, start naming pitches well below 100 Hz. That is a
countable fact in choir.jsonl, and this counts it.

Run: ../.venv-modal/bin/modal run _pitch_census.py::census
"""

import modal

app = modal.App("tuning-pitch-census")
vol = modal.Volume.from_name("tuning-field")


@app.function(volumes={"/field": vol}, timeout=300)
def census():
    import json
    from collections import defaultdict

    sung = defaultdict(list)      # short name -> [hz, ...] everything sounded
    named = defaultdict(list)     # -> [hz, ...] only pitches the voice chose
    kinds = defaultdict(lambda: defaultdict(int))

    with open("/field/choir.jsonl", encoding="utf-8") as fh:
        for ln in fh:
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("kind") != "tone":
                continue
            hz = r.get("pitch_hz")
            if not hz:
                continue
            who = r.get("model_short") or r.get("voice_id") or "?"
            hz = float(hz)
            sung[who].append(hz)
            kind = str(r.get("pitch_kind") or "?")
            kinds[who][kind] += 1
            if "given" not in kind and "fold" not in kind:
                named[who].append(hz)

    def pct(xs, p):
        return sorted(xs)[int(p * (len(xs) - 1))] if xs else 0.0

    print(f"{'voice':>16}  {'tones':>6}  {'min':>6}  {'p10':>6}  "
          f"{'median':>6}  {'p90':>6}  {'<100Hz':>7}  {'<80Hz':>6}")
    total = []
    for who in sorted(sung, key=lambda w: pct(sung[w], .5)):
        xs = sung[who]
        total += xs
        low = sum(1 for x in xs if x < 100.0)
        vlow = sum(1 for x in xs if x < 80.0)
        print(f"{who:>16}  {len(xs):>6}  {min(xs):>6.1f}  {pct(xs, .1):>6.1f}  "
              f"{pct(xs, .5):>6.1f}  {pct(xs, .9):>6.1f}  "
              f"{low / len(xs) * 100:>6.1f}%  {vlow / len(xs) * 100:>5.1f}%")
    low = sum(1 for x in total if x < 100.0)
    vlow = sum(1 for x in total if x < 80.0)
    print(f"{'all':>16}  {len(total):>6}  {min(total):>6.1f}  "
          f"{pct(total, .1):>6.1f}  {pct(total, .5):>6.1f}  {pct(total, .9):>6.1f}  "
          f"{low / len(total) * 100:>6.1f}%  {vlow / len(total) * 100:>5.1f}%")

    print("\npitch_kind per voice (who names their own):")
    for who in sorted(kinds):
        parts = ", ".join(f"{k}: {n}" for k, n in
                          sorted(kinds[who].items(), key=lambda kv: -kv[1]))
        print(f"{who:>16}  {parts}")

    print("\nself-named pitches only (the request's 15 October question):")
    for who in sorted(named, key=lambda w: pct(named[w], .5)):
        xs = named[who]
        if not xs:
            continue
        low = sum(1 for x in xs if x < 100.0)
        print(f"{who:>16}  {len(xs):>6}  min {min(xs):>6.1f}  "
              f"median {pct(xs, .5):>6.1f}  <100Hz {low / len(xs) * 100:.1f}%")
