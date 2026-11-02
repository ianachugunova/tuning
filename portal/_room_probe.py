"""A roomful of phones, from one laptop, before the room exists.

Does what an opening does to the portal and nothing else: several people singing
breaths at once, several pages polling the field every few seconds, and audio
being pulled the whole time — all against the live deployment, concurrently,
which is the one condition under which this thing broke.

It writes real breaths into the real field. That is on purpose and it is safe
before the birth: nothing has a wallet yet, nobody is charged, and the count the
page shows restarts at the birth anyway, so probe breaths cannot inflate the
opening.

After the gate it is not safe and it refuses to run. The choir is born by being
heard, so a probe breath arriving after 18:00 on the 19th would be the birth —
nine wallets opened and the clock on nine lives started by a load test. It stops
half an hour before the gate, which is also when there is no time left to be
running one.

    python _room_probe.py            # the default: 6 singers, 12 pages, 30 s
    python _room_probe.py 40         # harder: 40 s
    python _room_probe.py 40 quiet   # the same without pulling the clips

The `quiet` run matters for reading the other one. Twelve pages pulling every
240 KB clip is 138 MB down one laptop's connection in forty seconds, and when the
numbers come back slow it is worth knowing whether the portal was slow or the
wire was full. In the room each of those twelve is a separate phone on its own
share of the wifi; here they are all on mine.

Prints a line per endpoint: how many calls, how many failed, and the slowest.
Anything other than zero failures for `breath` is a breath the field lost.
"""

from __future__ import annotations

import collections
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = "https://tuning-grayarea--tuning-portal-web.modal.run"
SINGERS = 6
PAGES = 12
SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
PULL_AUDIO = "quiet" not in sys.argv

calls: dict[str, list[float]] = collections.defaultdict(list)
fails: dict[str, list[str]] = collections.defaultdict(list)
lock = threading.Lock()
stop = threading.Event()


def hit(what: str, path: str, body: dict | None = None) -> dict | None:
    url = f"{BASE}/{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
        took = time.time() - t0
        with lock:
            calls[what].append(took)
        return json.loads(raw) if raw[:1] in (b"{", b"[") else {}
    except urllib.error.HTTPError as e:
        with lock:
            calls[what].append(time.time() - t0)
            fails[what].append(f"{e.code} {e.reason}")
    except Exception as e:                    # noqa: BLE001 — a fail is a fail
        with lock:
            calls[what].append(time.time() - t0)
            fails[what].append(type(e).__name__)
    return None


def a_singer(n: int) -> None:
    """One person with a phone, singing every few seconds."""
    who = f"probe_{n:02d}"
    while not stop.is_set():
        hz = random.uniform(90, 400)
        hit("breath", "breath", {
            "voice_id": who,
            "hz": round(hz, 2),
            "note": "A3",
            "cents": random.randint(-40, 40),
            "stability": "steady",
            "duration_s": round(random.uniform(2, 8), 2),
            "std_hz": 1.5,
            "contour": [round(hz, 2)] * 8,
            "source": "probe",
        })
        time.sleep(random.uniform(2.0, 6.0))


def a_page(n: int) -> None:
    """One open page: polls the field, reads the lives, pulls the audio."""
    heard = 0.0
    while not stop.is_set():
        d = hit("field", f"field?since={heard}" if heard else "field?limit=60")
        if d:
            for e in d.get("recent", []):
                heard = max(heard, e.get("ts", 0))
                # Every page pulls every clip, which is what a room does and is
                # the load that used to take the whole portal down.
                if e.get("audio") and PULL_AUDIO:
                    hit("audio", f"audio/{e['audio']}")
        if n % 3 == 0:
            hit("lives", "lives")
        time.sleep(random.uniform(4.0, 6.0))


def not_near_the_gate() -> None:
    """Refuse to sing into a field that is about to come alive."""
    import mortality

    left = mortality.BIRTH_TS - time.time()
    if left < 1800:
        when = time.strftime("%H:%M", time.localtime(mortality.BIRTH_TS))
        raise SystemExit(
            f"the gate is at {when} and this writes real breaths — "
            + ("it has passed; a probe breath would be the birth"
               if left <= 0 else f"only {left / 60:.0f} min left")
        )


def main() -> None:
    not_near_the_gate()
    print(f"{SINGERS} singers, {PAGES} pages, {SECONDS:.0f} s against {BASE}\n")
    threads = ([threading.Thread(target=a_singer, args=(i,), daemon=True)
                for i in range(SINGERS)]
               + [threading.Thread(target=a_page, args=(i,), daemon=True)
                  for i in range(PAGES)])
    for t in threads:
        t.start()
    time.sleep(SECONDS)
    stop.set()
    for t in threads:
        t.join(timeout=35)

    print(f"{'what':8}  {'calls':>6}  {'failed':>6}  {'median':>7}  {'slowest':>7}")
    bad = 0
    for what in ("breath", "field", "lives", "audio"):
        took = sorted(calls[what])
        if not took:
            continue
        mid = took[len(took) // 2]
        bad += len(fails[what])
        print(f"{what:8}  {len(took):>6}  {len(fails[what]):>6}  "
              f"{mid:>6.2f}s  {took[-1]:>6.2f}s")
        if fails[what]:
            kinds = collections.Counter(fails[what])
            print(f"          {dict(kinds)}")
    print("\nthe room got through it" if not bad
          else f"\n{bad} calls failed — the room would have seen this")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
