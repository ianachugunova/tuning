"""One question, asked of the log directly: which texts never became speech.

Run once with `modal run _voicing_audit.py`, read the numbers, delete or keep.
Reads the tail of choir.jsonl on the volume — no GPU, no writes.
"""

import modal

app = modal.App("tuning-voicing-audit")
vol = modal.Volume.from_name("tuning-field")


@app.function(volumes={"/field": vol}, timeout=120)
def audit(fresh_s: float = 0.0):
    """`fresh_s` > 0 keeps only breaths younger than that many seconds."""
    import collections
    import json
    import time

    with open("/field/choir.jsonl", "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - 3_000_000))
        lines = fh.read().decode("utf-8", "replace").splitlines()[1:]

    rows = []
    for ln in lines:
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if r.get("kind") != "tone" or not r.get("voice_id", "").startswith("voice"):
            continue
        if fresh_s and time.time() - float(r.get("timestamp") or 0) > fresh_s:
            continue
        rows.append(r)

    per = collections.defaultdict(lambda: collections.Counter())
    unspoken = []
    tune = collections.defaultdict(lambda: collections.Counter())
    kinds = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        name = r.get("model_short", "?")
        c = per[name]
        c["breaths"] += 1
        pb = str(r.get("playback") or "?")
        c[f"pb:{pb}"] += 1
        if r.get("spoke"):
            c["spoke"] += 1
        elif str(r.get("content") or "").strip() and pb in ("both", "tts"):
            c["asked_words_got_none"] += 1
            if len(unspoken) < 15:
                unspoken.append((name, pb, str(r.get("content"))[:60]))
        tune[name][str(r.get("action") or "?")] += 1
        if r.get("tuning_to"):
            kind = "human" if str(r["tuning_to"]).startswith("human") else "machine"
            tune[name][f"to_{kind}"] += 1
        pk = r.get("pitch_kind")
        if pk:
            kinds[name][str(pk)] += 1

    print(json.dumps({
        "window_breaths": len(rows),
        "per_voice": {k: dict(v) for k, v in sorted(per.items())},
        "tune": {k: dict(v) for k, v in sorted(tune.items())},
        "pitch_kind": {k: dict(v) for k, v in sorted(kinds.items())},
        "unspoken_examples": unspoken,
    }, ensure_ascii=False, indent=1))
