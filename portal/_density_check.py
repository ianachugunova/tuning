"""How continuous is the field, actually? Coverage and gaps, from both logs.

Run: ../.venv-modal/bin/modal run _density_check.py::check --hours 3
"""

import modal

app = modal.App("tuning-density-check")
vol = modal.Volume.from_name("tuning-field")


@app.function(volumes={"/field": vol}, timeout=120)
def check(hours: float = 3.0):
    import json
    import time

    def tail(path, size=6_000_000):
        try:
            with open(path, "rb") as fh:
                fh.seek(0, 2)
                end = fh.tell()
                fh.seek(max(0, end - size))
                return fh.read().decode("utf-8", "replace").splitlines()[1:]
        except FileNotFoundError:
            return []

    now = time.time()
    since = now - hours * 3600

    # Sounding intervals, machines and people together — the field as a room
    # hears it, not as either log alone tells it.
    spans = []          # (t0, t1, who)
    humans = []
    for ln in tail("/field/choir.jsonl"):
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if r.get("kind") != "tone":
            continue
        ts = float(r.get("timestamp") or 0)
        if ts < since:
            continue
        dur = float(r.get("sound_s") or r.get("breath_s") or 0)
        t0 = float(r.get("started_ts") or ts)
        spans.append((t0, t0 + max(dur, 0.1), "m"))
    for ln in tail("/field/field.jsonl"):
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        ts = float(r.get("ts") or 0)
        if ts < since:
            continue
        t0 = float(r.get("started_ts") or ts)
        t1 = float(r.get("ended_ts") or (t0 + float(r.get("duration_s") or 0)))
        spans.append((t0, max(t1, t0 + 0.1), "h"))
        humans.append(ts)

    spans.sort()
    m = sum(1 for s in spans if s[2] == "m")
    print(f"window: last {hours:.1f} h   machine breaths: {m}   human breaths: {len(spans) - m}")
    if not spans:
        print("nothing sounded at all")
        return

    # Union coverage and the gaps between sound.
    covered = 0.0
    gaps = []
    cur0, cur1 = spans[0][0], spans[0][1]
    for t0, t1, _ in spans[1:]:
        if t0 <= cur1:
            cur1 = max(cur1, t1)
        else:
            covered += cur1 - cur0
            gaps.append(t0 - cur1)
            cur0, cur1 = t0, t1
    covered += cur1 - cur0
    window = now - max(since, spans[0][0])

    tones = [t1 - t0 for t0, t1, w in spans if w == "m"]
    starts = sorted(t0 for t0, _, w in spans if w == "m")
    waits = [b - a for a, b in zip(starts, starts[1:])]

    def pct(xs, p):
        return sorted(xs)[int(p * (len(xs) - 1))] if xs else 0.0

    print(f"sound covered {covered / window * 100:.0f}% of the window "
          f"({covered / 60:.1f} of {window / 60:.1f} min)")
    print(f"silences: n={len(gaps)}  median {pct(gaps, .5):.1f}s  "
          f"p90 {pct(gaps, .9):.1f}s  longest {max(gaps) if gaps else 0:.1f}s")
    print(f"machine tone length: median {pct(tones, .5):.1f}s  "
          f"p90 {pct(tones, .9):.1f}s")
    print(f"machine start-to-start: median {pct(waits, .5):.1f}s  "
          f"p90 {pct(waits, .9):.1f}s")

    # The same numbers for the stretch when somebody was actually there,
    # because ALONE_STRETCH makes the empty field eleven times slower on
    # purpose and the two regimes must not be averaged into one verdict.
    if humans:
        h0, h1 = min(humans) - 60, max(humans) + 300
        inside = [(a, b) for a, b, _ in spans if a >= h0 and a <= h1]
        if len(inside) > 3:
            cov = 0.0
            g2 = []
            c0, c1 = inside[0]
            for a, b in inside[1:]:
                if a <= c1:
                    c1 = max(c1, b)
                else:
                    cov += c1 - c0
                    g2.append(a - c1)
                    c0, c1 = a, b
            cov += c1 - c0
            span = h1 - h0
            print(f"\nwhile people were in the field ({(h1 - h0) / 60:.0f} min): "
                  f"covered {cov / span * 100:.0f}%  "
                  f"median silence {pct(g2, .5):.1f}s  longest {max(g2) if g2 else 0:.1f}s")
    else:
        print("\n(no people in this window — the choir was resting at the "
              "empty-field stretch)")
