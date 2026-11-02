"""Did the bell ring? Read both logs and say what the bell itself would see."""

import modal

app = modal.App("tuning-school-check")
vol = modal.Volume.from_name("tuning-field")


@app.function(volumes={"/field": vol}, timeout=120)
def check():
    import json
    import time

    def tail(path, size=2_000_000):
        try:
            with open(path, "rb") as fh:
                fh.seek(0, 2)
                end = fh.tell()
                fh.seek(max(0, end - size))
                return fh.read().decode("utf-8", "replace").splitlines()[1:]
        except FileNotFoundError:
            return []

    now = time.time()
    day_ago = now - 24 * 3600

    breaths = []
    for ln in tail("/field/field.jsonl"):
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        ts = float(r.get("ts") or 0)
        if ts > day_ago:
            breaths.append(ts)
    print(f"human breaths in the last 24 h: {len(breaths)}")
    if breaths:
        for ts in breaths[-5:]:
            print("  breath at", time.strftime("%H:%M:%S UTC", time.gmtime(ts)))

    print("\nschool.jsonl, last 24 h:")
    lessons = 0
    for ln in tail("/field/school.jsonl"):
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        ts = float(r.get("timestamp") or 0)
        if ts <= day_ago:
            continue
        lessons += 1
        when = time.strftime("%H:%M UTC", time.gmtime(ts))
        if r.get("kind") == "train":
            print(f"  {when}  {r.get('model_short','?'):15s} "
                  f"${r.get('usd',0):.2f}  {r.get('status','?')}"
                  f"  promoted={r.get('promoted')}")
        else:
            print(f"  {when}  {r.get('kind','?')}: "
                  f"{json.dumps({k: v for k, v in r.items() if k not in ('kind', 'timestamp')}, ensure_ascii=False)[:120]}")
    if not lessons:
        print("  (nothing)")

    print("\nadapter each voice sang its last breath through:")
    latest = {}
    for ln in tail("/field/choir.jsonl"):
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if r.get("kind") == "tone" and str(r.get("voice_id", "")).startswith("voice"):
            latest[r.get("model_short", "?")] = (
                r.get("adapter_version"), float(r.get("timestamp") or 0))
    for name, (ver, ts) in sorted(latest.items()):
        when = time.strftime("%H:%M UTC", time.gmtime(ts))
        print(f"  {name:16s} {ver or '(base, no adapter)'}  (last breath {when})")
