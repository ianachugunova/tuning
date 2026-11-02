"""Does the field survive a festival? Measured, not argued.

    modal run load_probe.py::race          # many containers, one file
    modal run load_probe.py::mixed_race    # the choir's file and the humans' file, together
    modal run load_probe.py::read_cost     # what /field costs as the log grows
    modal run load_probe.py::cleanup       # remove everything this wrote

Endzeit on 18 September is a festival, and the opening could put a hundred or
more phones onto `/breath` and `/field` inside the same minute. Nothing here has
ever been tried with more than a handful. Two things could go wrong and they are
not equally bad:

**The log could be corrupted.** `field.jsonl` is not a record of the work any
more, it *is* the work — the score is the artefact (`modal/SCORE.md`) — so a
torn or interleaved line is unrecoverable damage, not a glitch. The portal
defends against this by refusing to scale: `max_containers=1`, one writer,
forever. This file exists to find out what that defence is actually worth, by
doing the forbidden thing on purpose — several containers appending to one file
on the same Modal volume — and reading the result back line by line.

**The log could be slow.** A commit to a Modal volume is a network round trip,
and the portal does one per breath from inside an async endpoint, which means
every other request in that container waits for it. That is a throughput
ceiling and it can be measured in seconds per hundred breaths.

**The two files could interfere even though nothing ever writes both.** By
design `choir_app.py` only ever appends to `choir.jsonl` and `portal_app.py`
only ever appends to `field.jsonl` — "one writer per file" (`modal/SCORE.md`
§1). But for the whole of the six weeks both are true simultaneously: the
choir is appending to its file roughly every twelve seconds while the portal
is appending to its own from however many phones are singing, and both do it
by calling `.commit()` on the *same* `tuning-field` volume object. Nothing in
Modal's documentation says a commit from one container is scoped to only the
file it touched, so the untested question is not "does one file survive many
writers" (`race`, below) but "does committing volume A while volume A is
simultaneously being written and committed by a completely different app ever
lose or corrupt either file." That is `mixed_race`, and until it has run this
had never been tried — every previous run of this probe, and the portal's own
festival rehearsals, only ever exercised humans writing alone.

Everything here runs on CPU and writes only under `PROBE_DIR`, a scratch
directory on the volume with nothing else in it. **Nothing in this file opens
`field.jsonl`.** That is deliberate and it is the reason the probe is a separate
app: the field is append-only, so a test breath written into it could never be
taken out again.
"""

from __future__ import annotations

import modal

app = modal.App("tuning-load-probe")

# The same volume the portal and the choir share, because the question is about
# this volume and no other. A different directory on it, because the answer must
# not cost anything.
field_volume = modal.Volume.from_name("tuning-field", create_if_missing=True)

FIELD_DIR = "/field"
PROBE_DIR = f"{FIELD_DIR}/loadprobe"

image = modal.Image.debian_slim(python_version="3.12")

# One line of the field as it is actually written now — a human breath with its
# notation, contour and all. Measuring the cost of reading the log with toy
# lines would measure nothing: a tone line is about a kilobyte, and that is the
# number that decides what `/field` costs in November.
SHAPE = {
    "ts": 1789000000.0, "voice_id": "human_7f3a1b2c", "note": "D4",
    "hz": 293.66, "cents": 0, "stability": "steady", "duration_s": 4.2,
    "voiced_fraction": 0.97, "source": "portal", "kind": "tone", "who": "human",
    "started_ts": 1789000000.0, "ended_ts": 1789000004.2, "tail_s": 0.5,
    "sound_s": 4.2, "own_rest_s": 12.5, "std_hz": 2.4,
    "contour_hz": [293.1, 293.6, 294.0, 293.8] * 4,
    "contour_hop_s": 0.25, "content": "D4 ~294Hz [steady]",
    "pitch_hz": 293.66, "pitch_note": "D4", "pitch_cents": 0,
    "pitch_label": "", "pitch_kind": "sung", "pitch_all": [],
}


@app.function(image=image, volumes={FIELD_DIR: field_volume}, timeout=600)
def append_many(writer: int, breaths: int, path: str) -> dict:
    """One container appending `breaths` lines, exactly the way the portal does.

    Open, write one line, close, commit. The sequence number and the writer id
    go into the line so that afterwards it can be said not only that the right
    number of lines arrived but that no writer's line was lost while another
    writer's was kept — which is what "one writer per file" is protecting.
    """
    import json
    import os
    import socket
    import time

    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_ms, commit_ms = [], []
    host = socket.gethostname()
    for i in range(breaths):
        row = dict(SHAPE, writer=writer, seq=i, host=host, ts=time.time())
        line = json.dumps(row, ensure_ascii=False) + "\n"
        t0 = time.time()
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
        t1 = time.time()
        field_volume.commit()
        t2 = time.time()
        write_ms.append((t1 - t0) * 1000)
        commit_ms.append((t2 - t1) * 1000)

    def stats(xs: list[float]) -> dict:
        xs = sorted(xs)
        return {"n": len(xs), "median": round(xs[len(xs) // 2], 1),
                "p95": round(xs[int(len(xs) * 0.95)], 1),
                "max": round(xs[-1], 1)}

    return {"writer": writer, "host": host, "bytes_per_line": len(line),
            "write_ms": stats(write_ms), "commit_ms": stats(commit_ms)}


@app.function(image=image, volumes={FIELD_DIR: field_volume}, timeout=600)
def verify(path: str, writers: int, breaths: int) -> dict:
    """Read the file back and say whether it is still a log.

    Three separate questions. Is every line parseable JSON — i.e. did two
    appends ever land inside each other? Is every line that was written still
    there — i.e. did a commit from one container overwrite another's? And did
    any writer's lines arrive out of their own order, which would mean the file
    cannot be read as a chronology even if nothing was lost.
    """
    import json

    field_volume.reload()
    torn, rows = [], []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as err:
                torn.append({"line": lineno, "why": str(err),
                             "text": line[:120]})

    by_writer: dict[int, list[int]] = {}
    for r in rows:
        by_writer.setdefault(int(r.get("writer", -1)), []).append(int(r.get("seq", -1)))
    lost = {w: sorted(set(range(breaths)) - set(seqs))[:8]
            for w, seqs in by_writer.items()
            if len(set(seqs)) != breaths}
    unordered = [w for w, seqs in by_writer.items() if seqs != sorted(seqs)]

    return {
        "expected_lines": writers * breaths,
        "lines_read": len(rows),
        "torn_lines": torn[:5],
        "torn_count": len(torn),
        "writers_seen": sorted(by_writer),
        "containers_seen": sorted({str(r.get("host")) for r in rows}),
        "writers_with_missing_lines": lost,
        "writers_out_of_own_order": unordered,
        "intact": (len(rows) == writers * breaths and not torn and not lost),
    }


@app.local_entrypoint()
def race(writers: int = 6, breaths: int = 40):
    """Several containers append to one file at once. The forbidden thing.

    This is the experiment the portal's `max_containers=1` exists to make
    unnecessary, and the only way to know what that setting is worth is to
    remove it here and look at the file afterwards. If this comes back intact,
    the field's safety does not rest on a single container being alive; if it
    comes back torn, then it does, and every deploy during the festival is a
    moment when two containers exist at once.
    """
    import json
    import time

    path = f"{PROBE_DIR}/race.jsonl"
    print(f"{writers} containers x {breaths} breaths -> {path}\n")
    t0 = time.time()
    results = list(append_many.starmap(
        [(w, breaths, path) for w in range(writers)]
    ))
    wall = time.time() - t0
    for r in results:
        print(f"  writer {r['writer']} on {r['host'][:12]:<12} "
              f"line={r['bytes_per_line']}B  "
              f"write {r['write_ms']['median']}/{r['write_ms']['p95']}ms  "
              f"commit {r['commit_ms']['median']}/{r['commit_ms']['p95']}ms")
    total = writers * breaths
    print(f"\n{total} breaths in {wall:.1f}s  = {total / wall:.1f} breaths/s "
          f"across {writers} containers")
    print(json.dumps(verify.remote(path, writers, breaths), indent=1))


@app.local_entrypoint()
def mixed_race(humans: int = 6, human_breaths: int = 60,
               choir_writers: int = 1, choir_breaths: int = 60):
    """Two different files, two different roles, the same volume, at once.

    One writer shaped like the choir appends to a `choir.jsonl`-shaped file
    while several writers shaped like humans through the portal append to a
    separate `field.jsonl`-shaped file, all against `tuning-field` at the
    same time — `.spawn()` rather than `.starmap()` so the two groups overlap
    in wall-clock time instead of running one after the other. `choir_writers`
    defaults to 1 because that is the real design (`max_containers=1` on the
    choir); it can be raised to ask the same question `race()` asks, but of
    the choir's own file, for free.

    Verified independently per file, because that is what "one writer per
    file" promises — a lost line in the humans' file while the choir's stays
    intact would still be a launch blocker, and averaging the two together
    would hide exactly that.
    """
    import json
    import time

    choir_path = f"{PROBE_DIR}/mixed-choir.jsonl"
    field_path = f"{PROBE_DIR}/mixed-field.jsonl"
    print(f"{choir_writers} choir-shaped writer(s) x {choir_breaths} breaths -> {choir_path}")
    print(f"{humans} human-shaped writers x {human_breaths} breaths -> {field_path}\n")

    t0 = time.time()
    choir_calls = [append_many.spawn(w, choir_breaths, choir_path)
                   for w in range(choir_writers)]
    human_calls = [append_many.spawn(100 + w, human_breaths, field_path)
                   for w in range(humans)]
    choir_results = [c.get() for c in choir_calls]
    human_results = [c.get() for c in human_calls]
    wall = time.time() - t0

    print("choir-shaped:")
    for r in choir_results:
        print(f"  writer {r['writer']:>3} on {r['host'][:12]:<12} "
              f"write {r['write_ms']['median']}/{r['write_ms']['p95']}ms  "
              f"commit {r['commit_ms']['median']}/{r['commit_ms']['p95']}ms")
    print("human-shaped:")
    for r in human_results:
        print(f"  writer {r['writer']:>3} on {r['host'][:12]:<12} "
              f"write {r['write_ms']['median']}/{r['write_ms']['p95']}ms  "
              f"commit {r['commit_ms']['median']}/{r['commit_ms']['p95']}ms")

    total = (choir_writers * choir_breaths) + (humans * human_breaths)
    print(f"\n{total} breaths across both files in {wall:.1f}s wall "
          f"({total / wall:.1f} breaths/s combined)\n")

    choir_verdict = verify.remote(choir_path, choir_writers, choir_breaths)
    field_verdict = verify.remote(field_path, humans, human_breaths)
    print("choir.jsonl-shaped file:")
    print(json.dumps(choir_verdict, indent=1))
    print("field.jsonl-shaped file:")
    print(json.dumps(field_verdict, indent=1))

    both_intact = choir_verdict["intact"] and field_verdict["intact"]
    print(f"\nBOTH FILES INTACT: {both_intact}")
    if not both_intact:
        print("!! the volume dropped or tore a write while a second file on "
              "the same volume was being written at the same time. This is "
              "the concurrency the choir and the portal will actually run "
              "under for six weeks, and it is not safe to deploy on top of "
              "this result.")


@app.function(image=image, volumes={FIELD_DIR: field_volume}, timeout=900)
def grow_and_time(lines: int) -> dict:
    """What one `/field` request costs when the log is `lines` long.

    The portal answers `/field` by reading and parsing the whole of
    `field.jsonl` every time — that is how it knows the size, the first breath
    and the distinct voices. A hundred phones polling every five seconds is
    twenty of these a second, and six weeks of a festival is a log of a few
    hundred thousand lines. The two multiply.
    """
    import json
    import os
    import time

    os.makedirs(PROBE_DIR, exist_ok=True)
    path = f"{PROBE_DIR}/grow-{lines}.jsonl"
    line = json.dumps(SHAPE, ensure_ascii=False) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(line for _ in range(lines))
    field_volume.commit()

    def timed(fn):
        t0 = time.time()
        out = fn()
        return round((time.time() - t0) * 1000, 1), out

    # Exactly what the endpoint does today.
    def whole_file():
        with open(path, encoding="utf-8") as fh:
            return [json.loads(x) for x in fh.readlines()]

    def count_lines():
        with open(path, "rb") as fh:
            return sum(1 for _ in fh)

    def tail_only():
        # What it could do instead: the last 400 kB, parsed, which is what the
        # page can actually display.
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            fh.seek(max(0, size - 400_000))
            blob = fh.read().decode("utf-8", "replace")
        return [json.loads(x) for x in blob.split("\n")[1:] if x.strip()]

    parse_ms, rows = timed(whole_file)
    count_ms, _ = timed(count_lines)
    tail_ms, tail_rows = timed(tail_only)
    reload_ms, _ = timed(field_volume.reload)
    return {
        "lines": lines,
        "megabytes": round(os.path.getsize(path) / 1e6, 2),
        "read_and_parse_all_ms": parse_ms,
        "count_lines_ms": count_ms,
        "tail_400kb_ms": tail_ms,
        "tail_rows": len(tail_rows),
        "volume_reload_ms": reload_ms,
        "rows": len(rows),
    }


@app.local_entrypoint()
def read_cost():
    """The cost of one `/field`, at four sizes the field will actually reach."""
    import json

    # 4 000 is where it is now-ish; 40 000 is the first fortnight; 250 000 is
    # six weeks of nine voices breathing every twelve seconds plus the people.
    for n in (4_000, 40_000, 120_000, 250_000):
        print(json.dumps(grow_and_time.remote(n)))


@app.function(image=image, volumes={FIELD_DIR: field_volume}, timeout=300)
def _remove() -> list[str]:
    import os
    import shutil

    if not os.path.isdir(PROBE_DIR):
        return []
    gone = sorted(os.listdir(PROBE_DIR))
    shutil.rmtree(PROBE_DIR)
    field_volume.commit()
    return gone


@app.local_entrypoint()
def cleanup():
    """Delete everything the probe wrote. Nothing of the field is under it."""
    print("removed:", _remove.remote() or "(nothing was there)")
