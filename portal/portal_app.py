"""The portal — a page anyone can sing into, and the field that remembers.

    modal serve portal_app.py     # live-reloading URL while you work
    modal deploy portal_app.py    # a permanent URL

Two commitments are built into this file rather than written down somewhere.

**Only numbers leave the phone.** Pitch is measured in the browser; the
microphone signal never crosses the network and is never stored. What arrives
here is a note, a frequency, a duration and a word for how steady the voice
was — the same line live_v1/human.py writes into the choir's field.

**The field is only ever appended to.** One file, one writer, no truncation,
no reset. From the first breath onward it can be said truthfully that the
field has been sounding since that moment.

**A person's breath is a part in the score.** What the work leaves behind is the
two logs read as notation, performed with the parts swapped — machines singing
what the people sang (`modal/SCORE.md`). So a human breath is written to exactly
the standard a machine breath is: an onset rather than an arrival time, a
length, the singer's own rest before it, the pitch as both a frequency and a
note, and a trace of what the pitch did while it was held. All of it still
measurement; none of it lets anybody hear the room.

**The microphone in the gallery writes here too.** It is this same page, opened
with `?room=gallery` on the room's machine inside the installation Gray Area
builds, so a visitor singing in the room and a stranger singing into a phone
reach the field by one path and are notated identically. What the room gets is a stricter gate — an open microphone
in a shared hall must not write the hall down — and an honest account of who
sang: a turn at the microphone rather than a person. See `modal/SCORE.md` §9.

It also serves `/lives`, which is what the gallery's nine screens read: each
voice's remaining money, its share of the card, and — once it has spent what it
was given — its epitaph. Derived here rather than asked of the choir, because
asking the choir means waking an A100 to answer a web request, and because the
screens must keep working after the last voice is dead and the card is gone.
"""

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

app = modal.App("tuning-portal")

# Survives redeploys, restarts and everything else. Do not add code that
# rewrites or truncates field.jsonl — the whole claim rests on that.
field_volume = modal.Volume.from_name("tuning-field", create_if_missing=True)

FIELD_DIR = "/field"
FIELD_PATH = f"{FIELD_DIR}/field.jsonl"

# Written by the choir (see choir_app.py), read here, never written here. Two
# files rather than one because a Modal volume promises nothing about two
# containers appending to the same file, and nothing may be lost from the field.
CHOIR_PATH = f"{FIELD_DIR}/choir.jsonl"
# Written by the night school (see training_app.py), for the same reason there are
# two files already: one writer each. Read here so that the money a voice spends
# being changed shows up in the room on the same screens as the money it spends
# singing.
SCHOOL_PATH = f"{FIELD_DIR}/school.jsonl"
GIFTS_PATH = f"{FIELD_DIR}/gifts.jsonl"
# Where the posters brought people from. Every QR on a street poster carries
# `?from=<city-slug>`, and a page that arrives with one writes a line here: the
# slug and the second. Nothing else — not the address, not the device, not
# what the person went on to do. An arrival is a page load with a slug on it,
# not a person, and the question it answers is only "did the posters in that
# city bring anyone", which is the one thing a poster campaign has to know.
# Our own file rather than a link shortener's dashboard, because the dashboard
# would log what the work promises never to.
ARRIVALS_PATH = f"{FIELD_DIR}/arrivals.jsonl"
AUDIO_DIR = f"{FIELD_DIR}/audio"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]==0.115.6")
    .add_local_dir(HERE, remote_path="/portal", ignore=["*.json", "*.py", "*.mjs"])
    # The ledger and the cast. Both stdlib, both read-only here: the portal
    # replays the choir's log and never writes a line of it.
    .add_local_file(HERE / "mortality.py", remote_path="/root/mortality.py")
    .add_local_file(HERE / "cast.py", remote_path="/root/cast.py")
    # The shape of the log-as-score, shared with the choir so that a human
    # breath and a machine breath are notated the same way. See `score.py` and
    # `modal/SCORE.md`; the intended performance swaps the two, so a difference
    # in how they are written down would become a difference in what can be sung.
    .add_local_file(HERE / "score.py", remote_path="/root/score.py")
)


@app.function(
    image=image,
    volumes={FIELD_DIR: field_volume},
    min_containers=1,   # the portal should be warm when someone arrives
    max_containers=1,   # one writer, so the append-only log stays consistent
    scaledown_window=300,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    import collections
    import json
    import os
    import re
    import threading
    import time
    import uuid

    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, JSONResponse, Response

    import cast
    import mortality
    import score

    api = FastAPI(title="Tuning portal")

    # The choir's log, followed rather than re-read. It is append-only, so the
    # cheap way to know what is in it is to remember how far we got and read the
    # rest — which matters by November, when it is a few hundred thousand lines
    # and the page is asking for the field every few seconds.
    #
    # Two things come out of the same pass: the ledger (every line, because the
    # accounting has to see the heartbeats and the deaths) and the recent
    # breaths (only the tone lines, because that is what the page plays).
    choir = {"offset": 0, "ledger": mortality.Ledger(), "recent": [], "tones": 0}
    RECENT_KEPT = 400
    # The school's log is followed the same way, into the same ledger. Its own
    # offset, because it is its own file and a few lines long — one per voice per
    # night against the choir's hundreds of thousands.
    school = {"n": 0, "size": -1, "rows": []}
    gifts_seen = {"n": 0}

    # Reading this volume and writing to it are two different problems, and this
    # file had them confused. That is what made a page two people could not sing
    # into at the same time.
    #
    # Modal's two rules, both of which were being broken here:
    #
    #   * a volume cannot be reloaded while any file on it is open — the server
    #     answers FAILED_PRECONDITION and the client raises;
    #   * a write that begins while a reload is in flight comes back as EPERM.
    #
    # Every poll of `/field` used to reload the volume, and this container answers
    # a hundred requests at once in threads. So one thread reloading met another
    # reading `choir.jsonl`, or streaming a clip to a phone, or appending
    # somebody's breath — and one of the two died. The evening before the opening,
    # with two devices on the site, twenty-four of thirty-four breaths came back
    # 500 and were lost. A person sang into the field and the field did not keep
    # it, which is the one failure this work cannot have.
    #
    # What holds now:
    #
    #   * nobody reloads inside a request. One background thread does, on a timer.
    #   * writing takes the lock, so a write and a reload never overlap. That is
    #     the only pairing that has to be prevented, because it is the only one
    #     that can lose a breath.
    #   * reading takes nothing and waits for nothing. A read that happens to
    #     overlap a reload costs that reload, which fails, says so, and is tried
    #     again three seconds later — nothing is lost. Readers used to queue
    #     behind that round trip, and that is why a breath took eight seconds to
    #     be written down even on the occasions it was not dropped.
    volume_lock = threading.Lock()

    # Often enough that a page polling every five seconds is never served the
    # same stale view twice; rarely enough that the reloads do not pile onto each
    # other. A reload is a round trip that grows with the volume, and this one
    # holds a clip for every breath the choir has ever sung.
    RELOAD_EVERY_S = 3.0
    reloaded = {"at": 0.0, "ok": True}

    def refresh_volume() -> bool:
        """The newest view of the volume. Never raises.

        A view three seconds old is a page three seconds behind. A raised
        exception is a field that does not exist, and neither the page nor the
        person holding it can tell that apart from the work being over.
        """
        with volume_lock:
            try:
                field_volume.reload()
                reloaded.update(at=time.time(), ok=True)
                return True
            except Exception as exc:       # noqa: BLE001 — deliberately everything
                # What arrives is a RuntimeError wrapping a gRPC status, and the
                # wrapping has changed between client versions. There is nothing
                # worth telling apart: every failure has the same answer, which is
                # to keep the view in hand and try again in a moment. Printed, so
                # that a volume which has genuinely stopped answering is visible
                # in `modal app logs` — but once per spell rather than once per
                # attempt, or a broken volume would bury the log it is reported in.
                if reloaded["ok"]:
                    print(f"volume not reloaded, serving the view in hand: {exc}")
                reloaded["ok"] = False
                return False

    # Breaths are written to the file immediately and committed a moment later,
    # from a thread of their own.
    #
    # A commit on this volume is a two-second round trip, and it used to happen
    # inside the request. Six people singing at once queued behind one another and
    # the last of them waited the better part of ten seconds to be told the field
    # had them — which is exactly the delay Iana saw when she opened the page on a
    # phone and a laptop and the second person took its time appearing.
    #
    # The write itself is local and instant. The commit is only what makes the
    # line visible to the choir's container, and a second of lag there is nothing:
    # the choir reads a window of recent breaths, not the last instant.
    #
    # What it costs is a window about a second wide in which a container that died
    # would take an uncommitted breath with it. That is the right way round: a
    # breath lost to a crash is rare and invisible, and two seconds of waiting is
    # every singer, every time.
    COMMIT_EVERY_S = 1.0
    unsaved = threading.Event()

    def keep_committing() -> None:
        """Make the breaths durable, just behind the people singing them."""
        while True:
            unsaved.wait()
            unsaved.clear()
            try:
                # No lock. A commit that overlaps a write simply may not carry
                # the newest line, and the next one will; what must never overlap
                # a write is a *reload*, which is what the lock is for.
                field_volume.commit()
            except Exception as exc:       # noqa: BLE001 — see refresh_volume
                print(f"field not committed, trying again: {exc}")
                unsaved.set()
            time.sleep(COMMIT_EVERY_S)

    # How many of the newest clips are pulled into memory before anybody asks.
    # The first read of a clip off the volume is the slow one — a second, and on
    # the evidence sometimes very much worse — and it used to land on whichever
    # listener asked first. That listener's page has fifteen seconds before the
    # breath is stale and gets dropped unplayed, so a slow first read was a breath
    # the room never heard. The choir sings about two a second; a dozen covers
    # several seconds of singing, which is the whole of what any page will ask for.
    PREFETCH = 16

    def warm() -> None:
        """Follow the logs and pull the newest clips, off the request path."""
        for row in follow_choir()["recent"][-PREFETCH:]:
            name = row.get("audio")
            if name and name not in audio_cache:
                clip(name)

    def keep_reloading() -> None:
        """The only thing in this container that reloads the volume."""
        while True:
            if refresh_volume():
                try:
                    warm()
                except Exception as exc:   # noqa: BLE001
                    # Warming is a courtesy; requests can still do it themselves.
                    print(f"clips not warmed, the first listener will wait: {exc}")
            time.sleep(RELOAD_EVERY_S)

    def read_bytes(path: str) -> bytes | None:
        """One file off the volume, read whole and let go.

        No lock: see above. Held open is the thing to avoid, which is why this
        reads and closes rather than handing the open file to the web server to
        stream from.
        """
        try:
            with open(path, "rb") as fh:
                return fh.read()
        except OSError:
            return None

    # Clips already read, kept in memory. A rendered breath never changes once it
    # is written, so a clip read once is good until the container dies, and the
    # page only plays breaths from the last fifteen seconds — the working set is
    # whatever the choir sang in the last half minute, fifty clips at the most.
    #
    # This is here for latency and it is not an optimisation. Reading a clip off
    # the volume for every listener meant fifteen reads a second with twelve
    # pages open, every one of them queued behind the lock above, and a person's
    # breath then waited twelve seconds in that same queue to be written down.
    # Cached, a clip is read from the volume once and handed to everybody else
    # out of memory, which is where a file that never changes belongs.
    AUDIO_KEPT = 240                       # about 58 MB of held tones
    audio_cache = collections.OrderedDict()
    audio_cache_lock = threading.Lock()    # memory only; never held over a read

    # How long a rendered breath sounds, from the size of its file: the choir
    # writes 24 kHz mono 16-bit wavs with a 44-byte header, so no file is read,
    # only measured. The screens in the room light a speaker for exactly as long
    # as it sounds, and a tone is anything from eight to fifteen seconds.
    wav_seconds_known: dict[str, float] = {}

    def wav_seconds(name: str | None) -> float:
        if not name:
            return 0.0
        known = wav_seconds_known.get(name)
        if known is not None:
            return known
        try:
            size = os.path.getsize(f"{AUDIO_DIR}/{os.path.basename(name)}")
        except OSError:
            return 0.0                      # not on this container's view yet
        secs = round(max(0, size - 44) / (24000 * 2), 2)
        if len(wav_seconds_known) > 4000:
            wav_seconds_known.clear()
        wav_seconds_known[name] = secs
        return secs

    def clip(name: str) -> bytes | None:
        with audio_cache_lock:
            cached = audio_cache.get(name)
            if cached is not None:
                audio_cache.move_to_end(name)
                return cached
        path = f"{AUDIO_DIR}/{name}"
        wav = read_bytes(path)
        if wav is None:
            # Written within the last few seconds and not in this container's view
            # of the volume yet. Waiting out one turn of the reloader is cheaper
            # than reloading here, and a missed clip is a breath not heard.
            time.sleep(RELOAD_EVERY_S)
            wav = read_bytes(path)
        if wav is None:
            return None
        with audio_cache_lock:
            audio_cache[name] = wav
            while len(audio_cache) > AUDIO_KEPT:
                audio_cache.popitem(last=False)
        return wav

    # How long a human who has sung stays counted as "present" in the plain
    # count the page shows (`humans_now`). Long enough that someone thinking
    # between breaths doesn't blink out of the room; short enough that the
    # number means "here now" and not "here today". There is no session, no
    # heartbeat and no login — a person is present exactly as long as their
    # last breath is recent, which is the only thing the log can honestly say.
    PRESENCE_WINDOW_S = 180.0

    def machines_alive(state: dict) -> int:
        """How many machine voices are alive, for the same quiet presence count.

        Free once `follow_choir()` has run for this request: the ledger it
        returns already replays every heartbeat and death in `choir.jsonl`.
        Before Endzeit there is no birth record yet, so nothing has a wallet
        and `living_ids()` is truthfully empty — but an empty choir would read
        as already dead rather than not yet begun, so until birth the count is
        the cast seated to sing, not zero.
        """
        ledger = state["ledger"]
        if ledger.birth_ts is not None:
            return len(ledger.living_ids())
        return len(cast.singing())

    def read_school() -> list:
        """The school's log, re-read only when it has grown."""
        try:
            size = os.path.getsize(SCHOOL_PATH)
        except OSError:
            return school["rows"]
        if size != school["size"]:
            try:
                school["rows"] = read_jsonl(SCHOOL_PATH)
                school["size"] = size
            except OSError:
                pass
        return school["rows"]

    def follow_school(before: float = float("inf"), rows: list | None = None) -> None:
        """Apply the night school's bills into the same ledger, in their own time.

        The birth record is in `choir.jsonl`, and the ledger drops every event
        that reaches it before the birth — so a fresh container that read the
        school first threw all of its bills away, silently, and moved the
        bookmark past them so they could never be counted again. That is what
        made `/lives` say the voices had spent nothing on learning through the
        whole opening weekend, while the choir's own ledger — which merges the
        two files by time, and is what actually decides who dies — had charged
        them correctly all along.

        `before` is the time the choir's log has been read up to, and it is why
        this is also called from inside the replay, like `follow_gifts`. 26
        October: read after the whole of the choir's log, each bill was a line
        from the past, and until `mortality._settle` stopped rewinding its
        clock for one, every bill re-charged a slice of warm card to everybody —
        $54 across the choir. Merged by time, the bills land where the choir
        put them, and the hourly history the screen draws has them too.

        Bills from before the birth are forgiven rather than charged, by the
        same rule the choir reaches by a different road: the fifteen runs of
        the 19th were rehearsal, and none of that is anybody's life.
        """
        if rows is None:
            rows = read_school()
        led = choir["ledger"]
        if led.birth_ts is None:
            return
        while school["n"] < len(rows):
            bill = rows[school["n"]]
            ts = float(bill.get("timestamp") or 0.0)
            if ts > before:
                return
            # Only the bills. The file also holds the bell's notes to itself —
            # one every five minutes, naming no voice and no seconds — and an
            # event with a timestamp closes a slice of warm card, so a note
            # written while the choir was elsewhere would move money for no
            # reason (`choir_app._charges`).
            if ts >= led.birth_ts and bill.get("kind") == "train":
                led.apply(bill)
            school["n"] += 1

    # Two requests arriving together would otherwise both read the tail of
    # `choir.jsonl` from the same offset and apply the same lines to the ledger
    # twice, which is money charged twice. Its own lock, not the volume's: this
    # one is about the bookkeeping in memory, and it is held for a read of a few
    # kilobytes rather than a network round trip.
    ledger_lock = threading.Lock()

    def follow_choir() -> dict:
        with ledger_lock:
            return _follow_choir()

    def _follow_choir() -> dict:
        _follow_the_choirs_log()
        # Second, always. See follow_school() for what reading it first cost.
        follow_school()
        follow_gifts()
        return choir

    def read_gifts() -> list:
        try:
            return read_jsonl(GIFTS_PATH)
        except OSError:
            return []

    def follow_gifts(before: float = float("inf"), rows: list | None = None) -> None:
        """Gifts to living wallets, after the birth is known, in their own time.

        The ledger counts each `gift_id` once, so a reset that reads the file
        again from the top moves no money twice.

        `before` is the time the choir's log has been read up to, and it is
        why this is also called from inside the replay. 19 October: the portal
        restarted, read five weeks of breaths to the present, and only then
        applied the gifts of 15 October — by which point olmo-3, without its
        gift, had spent its seed and was dead in this replay. A gift reaches
        no one dead, so the $240 was refused, the three donors kept what they
        had handed over, and `/lives` showed a death the choir never had.
        Applied at their own timestamp, between the breaths they fell between,
        the gifts land in the same wallets the choir put them in.
        """
        if rows is None:
            rows = read_gifts()
        if choir["ledger"].birth_ts is None:
            return
        while gifts_seen["n"] < len(rows):
            e = rows[gifts_seen["n"]]
            if float(e.get("timestamp") or 0.0) > before:
                return
            if e.get("kind") == "gift":
                choir["ledger"].apply(e)
            gifts_seen["n"] += 1

    def _follow_the_choirs_log() -> None:
        try:
            size = os.path.getsize(CHOIR_PATH)
        except OSError:
            return
        if size < choir["offset"]:
            # The file got shorter, which can only be the one-time reset before
            # 18 September. Start again rather than pretend to know the rest.
            choir.update(offset=0, ledger=mortality.Ledger(), recent=[], tones=0)
            # A new ledger has never seen the school either, so the bills have to
            # be read again from the top — otherwise the hours a voice spent
            # learning would quietly stop having been paid for.
            school["n"] = 0
            gifts_seen["n"] = 0
        if size == choir["offset"]:
            return
        try:
            with open(CHOIR_PATH, "rb") as fh:
                fh.seek(choir["offset"])
                blob = fh.read()
        except OSError:
            # The stat above won the race against a reload and this open lost
            # it. Unguarded, that was the whole of `/field` answering 500 once
            # in a while. The offset is untouched; the next poll reads the tail.
            return
        cut = blob.rfind(b"\n")
        if cut < 0:
            return                        # a line is still being written
        choir["offset"] += cut + 1
        gifts = read_gifts()          # a few lines; read once per poll
        bills = read_school()         # re-parsed only when the file has grown
        for line in blob[:cut].decode("utf-8", "replace").split("\n"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # A gift that was given before this breath is applied before it;
            # see follow_gifts for the death this once invented. The school's
            # bills likewise, so a lesson is paid for in the hour it happened.
            row_ts = float(row.get("timestamp") or 0.0)
            if gifts_seen["n"] < len(gifts):
                follow_gifts(before=row_ts, rows=gifts)
            if school["n"] < len(bills):
                follow_school(before=row_ts, rows=bills)
            choir["ledger"].apply(row)
            if row.get("kind", "tone") == "tone":
                choir["recent"].append(row)
                choir["tones"] += 1
        del choir["recent"][:-RECENT_KEPT]

    # How many times a breath is offered to the volume before giving up on it.
    WRITE_TRIES = 4

    def append(event: dict) -> int:
        """Append one breath and make it durable. Returns the new field size.

        Tried again rather than allowed to fail once. A write that begins while
        the volume is being reloaded comes back as EPERM, and while the lock now
        keeps this container from reloading underneath its own write, it is not
        the only thing that touches this volume — the choir and the school hold
        it too. A person's breath is the one thing here that must not be dropped,
        so it is offered again, with a longer pause each time.
        """
        line = json.dumps(event, ensure_ascii=False) + "\n"
        for attempt in range(WRITE_TRIES):
            try:
                with volume_lock:
                    os.makedirs(FIELD_DIR, exist_ok=True)
                    with open(FIELD_PATH, "a", encoding="utf-8") as fh:
                        fh.write(line)
                unsaved.set()              # committed a moment later, in its own thread
                return count()
            except OSError as exc:
                if attempt + 1 == WRITE_TRIES:
                    raise
                print(f"breath not written, trying again: {exc}")
                time.sleep(0.1 * (attempt + 1))
        return count()                     # unreachable; kept for the type

    def count() -> int:
        try:
            with open(FIELD_PATH, "rb") as fh:
                return sum(1 for _ in fh)
        except FileNotFoundError:
            return 0

    # A poster's slug: lowercase, digits, hyphens, short. Anything else is not a
    # poster we printed and is not written down.
    FROM_OK = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")

    def note_arrival(slug: str) -> None:
        """One line in arrivals.jsonl. Same discipline as `append`: under the
        lock so it never overlaps a reload, offered again on EPERM, committed
        later by the thread that commits everything. Never raises past the
        retries — a lost arrival is a lost tally mark, not a lost breath."""
        line = json.dumps({"ts": time.time(), "from": slug}) + "\n"
        for attempt in range(WRITE_TRIES):
            try:
                with volume_lock:
                    os.makedirs(FIELD_DIR, exist_ok=True)
                    with open(ARRIVALS_PATH, "a", encoding="utf-8") as fh:
                        fh.write(line)
                unsaved.set()
                return
            except OSError as exc:
                if attempt + 1 == WRITE_TRIES:
                    print(f"arrival not written: {exc}")
                    return
                time.sleep(0.1 * (attempt + 1))

    def field_reckoning(breaths: list[dict] | None = None) -> tuple[int, float | None]:
        """How many breaths the field is, and the hour it started counting.

        Two different questions, and the page asks the second one: not how many
        people have ever sung into this log, but how many have sung to *this*
        choir. Before the birth those are the same number. After it they are not,
        and the difference is the whole rehearsal — a month of the log being
        tested, tuned into by whoever built it, and sung to by voices that had no
        wallets and could not hear a bill. On the evening of the 19th the page
        would otherwise have opened with "765 breaths · sounding for 26 days" over
        a choir three hours old, which is a claim about the work that is not true.

        So the count restarts at the birth and nothing is deleted: the log keeps
        every breath since 24 August, `learning.py` still trains on all of it, and
        the archive is intact. Only the arithmetic on the screen begins where the
        lives begin. A field that has never been sung to reads as empty, which is
        also the truth and is the first thing the veil says.
        """
        born = follow_choir()["ledger"].birth_ts
        if breaths is None:
            breaths = field_breaths()
        if born:
            # From the breath that woke them, which is the last one sung before
            # the birth and not the birth minute itself. The choir is born on the
            # first tick that hears somebody, so the breath that caused it is
            # always a little older than the record of it; counting from the
            # record would show the field as empty to the very person who opened
            # it, while their note was still sounding. Defined off the log rather
            # than off a window constant, because the two apps keep their own
            # windows and the definition here should not depend on either.
            woke = [b for b in breaths if b.get("ts", 0) <= born]
            cut = woke[-1]["ts"] if woke else born
            counted = [b for b in breaths if b.get("ts", 0) >= cut]
            return len(counted), (counted[0]["ts"] if counted else born)
        first = breaths[0] if breaths else None
        return len(breaths), (first.get("ts") if first else None)

    # Phones cache aggressively, and a singer holding a stale copy of the page
    # is a singer we cannot fix anything for. The page is small; re-fetch it.
    NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}

    # The page that lets the choir overlap, now the one a link reaches.
    #
    # `pitch.html` stood here for months and played the field through a single
    # `<audio>` element: one breath, its end, 600 ms, the next. Nine voices came
    # out as a queue of solos, and the piece cannot survive that — the Tuning
    # Meditation asks you to sing with a tone you are hearing, so a tone that has
    # finished is a tone you cannot tune to. A third of the breaths in the log
    # overlap; that page could not sound one of them.
    #
    # Swapped the night before the birth, with the lecture the next evening, on
    # the strength of `test_field_page.mjs` — "the choir sounds as a choir and
    # not as a queue of solos", and nine more. The old page is still here, one
    # route down, because it is what every screenshot and every note about the
    # portal so far describes.
    @api.get("/")
    def index():
        return FileResponse("/portal/field.html", headers=NO_CACHE)

    # Kept reachable rather than deleted: it is the page the deploy's smoke test
    # was written against, the one in the documentation, and the only fallback if
    # Web Audio turns out to behave differently on a room full of phones than it
    # does in a test.
    @api.get("/queue")
    def the_page_that_plays_one_at_a_time():
        return FileResponse("/portal/pitch.html", headers=NO_CACHE)

    @api.get("/pitch-core.js")
    def core():
        return FileResponse(
            "/portal/pitch-core.js", media_type="text/javascript", headers=NO_CACHE
        )

    @api.get("/listen.js")
    def listen_js():
        return FileResponse(
            "/portal/listen.js", media_type="text/javascript", headers=NO_CACHE
        )

    # Where the new page lived while it was still the new page. Kept pointing at
    # the same file now that it is served from `/` as well, because it has been
    # handed around as a link and a QR code, and a dead `/next` would be a person
    # standing in the room with nothing on their screen.
    #
    # No trailing slash on either route, and that is load-bearing: both pages
    # ask for `field`, `breath`, `lives`, `audio/…` and their scripts by
    # relative path, which resolve against `/` only while the path they were
    # served from has no trailing segment to inherit.
    @api.get("/next")
    def next_page():
        return FileResponse("/portal/field.html", headers=NO_CACHE)

    # The room with the lights up. A stranger arriving off a QR code meets a
    # quote and a door, on purpose — but a person who wants to know what they
    # just sang into deserves a page that says so plainly: the score, the nine,
    # the money, the school, and the promise that no recording of them exists.
    @api.get("/board")
    def the_screen_in_the_room():
        """Nine rows, nine speakers: wallet, what it is doing, what it said."""
        return FileResponse("/portal/board.html", headers=NO_CACHE)

    @api.get("/about")
    def about():
        return FileResponse("/portal/about.html", headers=NO_CACHE)

    # -- where the posters brought people from --------------------------
    # The page calls this once, at load, if it arrived with `?from=<slug>`.
    # 204 either way: a bad slug is simply not a poster of ours.
    @api.get("/arrive")
    def arrive(request: Request):
        slug = request.query_params.get("from", "")
        if FROM_OK.match(slug):
            note_arrival(slug)
        return Response(status_code=204)

    # The tally, per poster, all time and the last week. Read whole on each
    # call — it is a few kilobytes and this is asked by one person, not by every
    # phone every five seconds. Read under the lock: mid-reload the file is
    # briefly not there, and an unlocked read answered "nobody, ever" every
    # other time it was asked.
    @api.get("/arrivals")
    def arrivals():
        now = time.time()
        total: dict[str, int] = collections.Counter()
        week: dict[str, int] = collections.Counter()
        first: dict[str, float] = {}
        try:
            with volume_lock, open(ARRIVALS_PATH, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    slug, ts = str(r.get("from") or ""), float(r.get("ts") or 0)
                    if not slug:
                        continue
                    total[slug] += 1
                    first.setdefault(slug, ts)
                    if now - ts <= 7 * 86400:
                        week[slug] += 1
        except OSError:
            pass
        return {
            "posters": [
                {"from": s, "all_time": total[s], "last_7_days": week[s],
                 "first_seen": first[s]}
                for s in sorted(total, key=lambda s: -total[s])
            ],
            "all_time": sum(total.values()),
            "note": "an arrival is a page load with a poster's slug on it — "
                    "not a person, not a device, not an address",
        }

    # The log read as notation. Deliberately its own page and deliberately
    # plain: it is the one part of the site that still means something after
    # the last voice is dead and the volume is all that is left.
    @api.get("/score")
    def score_page():
        return FileResponse("/portal/score.html", headers=NO_CACHE)

    def _num(value, lo: float, hi: float, default: float = 0.0) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return default
        if v != v or v in (float("inf"), float("-inf")):   # NaN, ±inf
            return default
        return max(lo, min(hi, v))

    @api.post("/breath")
    async def breath(request: Request):
        """One sung breath, as numbers. No audio is accepted here by design.

        The numbers are also the person's part in the score (`modal/SCORE.md`),
        and the intended performance hands it to a machine to sing — so it is
        written to the same standard as a machine breath, with an onset, a
        length, the rest before it and the pitch as both a frequency and a note.
        Everything here is still a measurement: nothing about this line lets
        anybody hear the room it was sung in, which is the commitment the piece
        is left open to strangers on.
        """
        body = await request.json()
        try:
            hz = float(body.get("hz", 0))
        except (TypeError, ValueError):
            hz = 0.0
        if not (0 < hz < 20000):
            return JSONResponse({"error": "hz out of range"}, status_code=400)

        note = str(body.get("note", ""))[:8]
        cents = int(_num(body.get("cents"), -1200, 1200))
        stability = str(body.get("stability", ""))[:16]
        duration_s = round(_num(body.get("duration_s"), 0.0, 600.0), 2)
        # How long ago the sound actually stopped, measured on the singer's own
        # clock. The page waits half a second of silence before it decides an
        # utterance is over, and the network adds more; without this the onset
        # would be wrong by a fixed, unrecoverable amount, in a piece whose
        # subject is when people start and stop.
        tail_s = round(_num(body.get("tail_s"), 0.0, 30.0), 3)
        arrived = time.time()
        ended_ts = arrived - tail_s
        started_ts = ended_ts - duration_s

        # At most a value every quarter of a second over the longest utterance
        # the page will emit. A held note that slides is a notatable gesture and
        # `stability` is a four-bucket word for it; the trace keeps the shape.
        contour = [round(_num(v, 0.0, 20000.0), 1)
                   for v in (body.get("contour_hz") or [])][:128]

        # Where it was sung. `portal` is a phone or a laptop anywhere in the
        # world; `gallery` is the near-field microphone of the installation at
        # Gray Area, which is the same page with `?room=gallery` and a much
        # stricter gate (`pitch-core.js`, `GALLERY`). The distinction is in the score
        # because it is the difference between a stranger in another country and
        # somebody standing in the room in the week the choir dies — the last
        # human voices the piece will ever be given.
        #
        # Claimed by the page and not verified here, like `voice_id` and for the
        # same reason: there is no login, and an endpoint open to strangers
        # cannot tell where a POST was typed. Anything unrecognised is filed as
        # `portal`, which is the weaker claim of the two.
        source = str(body.get("source") or "portal")
        if source not in ("portal", "gallery"):
            source = "portal"

        event = {
            # `ts` stays first and stays what it always was — when the breath
            # arrived — because `/field`, the choir's reader and everything
            # already written into the log read it. The onset is a new field
            # beside it, not a redefinition of this one.
            "ts": arrived,
            "voice_id": str(body.get("voice_id", ""))[:64] or f"anon_{uuid.uuid4().hex[:8]}",
            "note": note,
            "hz": round(hz, 2),
            "cents": cents,
            "stability": stability,
            "duration_s": duration_s,
            "voiced_fraction": round(_num(body.get("voiced_fraction"), 0.0, 1.0), 3),
            "source": source,
            # -- the score --------------------------------------------------
            # Same two words on every line of both logs, so one reader can walk
            # the merged chronology without knowing which file a line came from.
            "kind": "tone",
            "who": "human",
            "started_ts": round(started_ts, 3),
            "ended_ts": round(ended_ts, 3),
            "tail_s": tail_s,
            # The same name the choir uses for the same quantity: how long this
            # sounded. For a person it is the voiced span, which is the only
            # honest answer — the page measures the voice, not the room.
            "sound_s": duration_s,
            # The singer's own rest — between two of *their* breaths, timed on
            # their own audio clock. Not the same quantity as the choir's
            # `silence_before_s`, which is how long the whole field had been
            # quiet, and named differently so the two can never be added up by
            # mistake. None for the first breath of a session: there was no
            # before, and a zero would claim there was one of no length.
            #
            # This is also the only number in the score that a merge of the two
            # logs cannot reconstruct: everything else about a human breath is
            # placed by its arrival time, and the network is not part of the
            # music. A phone timing the gap between its own two notes is right.
            "own_rest_s": (
                round(_num(body["own_rest_s"], 0.0, 86400.0), 3)
                if body.get("own_rest_s") is not None else None
            ),
            # The spread `stability` was quantised from. Four words is a
            # reasonable thing to show a singer and a poor thing to leave in an
            # archive: "steady" covers 30 to 80 cents and nothing recovers where
            # inside that a voice actually was.
            "std_hz": round(_num(body.get("std_hz"), 0.0, 20000.0), 2),
            "contour_hz": contour,
            "contour_hop_s": (round(duration_s / len(contour), 3)
                              if contour and duration_s > 0 else 0.0),
            # A text column for the human parts too, in the words the page and
            # the choir already use for a person's breath.
            "content": (f"{note} ~{hz:.0f}Hz"
                        + (f" [{stability}]" if stability else "")).strip(),
        }
        event.update(score.human_pitch_fields(hz, note, cents))
        append(event)
        # Counted from the birth like everywhere else, so the number the singer
        # sees the instant after singing is the same one the veil shows. It costs
        # a read of the log per breath, which is a small file and the honest price
        # of the two numbers agreeing.
        size, since = field_reckoning()
        return {
            "ok": True,
            "field_size": size,
            "sounding_since": since if since else event["ts"],
        }

    # The field as parsed rows, kept until the file grows. Append-only means the
    # size of it is a complete answer to "has this changed", and the alternative
    # is reading and parsing the whole log for every page every few seconds —
    # which is what `/field` and `/breath` were both waiting on. Never handed out
    # to be mutated: a new read replaces the list rather than extending it, so a
    # request still walking the old one is unaffected.
    field_cache = {"size": -1, "rows": []}

    field_cache_lock = threading.Lock()

    # Stat-then-read, and on any failure the view in hand — never an empty one.
    # While `keep_reloading` swaps the volume's view, a stat or an open here can
    # briefly fail or find nothing, and this function used to answer that with
    # an empty list. One poll in every handful landed in that window, and the
    # page it fed showed a field with nobody in it and a school that had never
    # rung — a counter that "resets" for five seconds and then remembers. Worse:
    # a stat that succeeded followed by a read that lost the race cached the
    # emptiness under the current size, and it was served until the file grew.
    def field_breaths() -> list[dict]:
        with field_cache_lock:
            try:
                size = os.path.getsize(FIELD_PATH)
                if size != field_cache["size"]:
                    field_cache["rows"] = read_jsonl(FIELD_PATH)
                    field_cache["size"] = size
            except OSError:
                pass
            return field_cache["rows"]

    def read_jsonl(path: str) -> list[dict]:
        """Raises OSError when the file cannot be read: the callers hold the
        last good parse and must be told apart a missing file from a file with
        nothing in it, or a lost race caches as an empty field."""
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    # The school's log, read the same way the field is: by size, once per
    # change. It is a small file — one line per bell and one per voice taught —
    # but it lives on the same volume, and the rule stands regardless of size:
    # nothing on the volume is parsed per request unless it grew.
    school_cache = {"size": -1, "rows": []}

    school_cache_lock = threading.Lock()

    def school_rows() -> list[dict]:
        # Same failure, same answer as `field_breaths`: a read that loses the
        # race against a reload serves the rows in hand, not a school that was
        # never held.
        with school_cache_lock:
            try:
                size = os.path.getsize(SCHOOL_PATH)
                if size != school_cache["size"]:
                    school_cache["rows"] = read_jsonl(SCHOOL_PATH)
                    school_cache["size"] = size
            except OSError:
                pass
            return school_cache["rows"]

    @api.get("/field")
    def field(limit: int = 40, since: float = 0.0):
        """What the field holds right now — for the page, and for proof.

        Both logs merged into one chronology: the people who sang and the models
        that answered, in the order it happened. `since` returns only what is
        newer, which is how the page keeps playing without repeating itself.
        """
        state = follow_choir()
        breaths = field_breaths()
        utterances = state["recent"]

        merged = [
            {
                "ts": b.get("ts", 0),
                "who": "human",
                "voice_id": b.get("voice_id", ""),
                # The room's player and the page both show this, and in the
                # exhibition week the two are worth telling apart on a screen.
                "name": ("a voice in the room"
                         if b.get("source") == "gallery" else "a voice"),
                # The note, the frequency, and — when the page measured it —
                # how steadily the note was held. The page now names people in
                # its "what is sounding" column, and "G3 196Hz · steady" is
                # what a singer would want said about them there.
                "text": (f"{b.get('note', '')} {b.get('hz', 0):.0f}Hz".strip()
                         + (f" · {b.get('stability')}"
                            if b.get("stability") else "")),
                "hz": b.get("hz", 0),
                "audio": None,
                # No recording is kept and none is sent — that is the promise
                # this endpoint is written around. What travels is the shape of
                # the note: how long it sounded and where the pitch went while
                # it did. A listener's own browser sounds a tone from these, so
                # a room with people in it can be heard as people without any
                # voice in it ever leaving the phone that sang.
                "dur": b.get("duration_s", 0),
                "contour": b.get("contour_hz") or [],
                "hop": b.get("contour_hop_s", 0),
                # How the pitch was arrived at, which is a real musical
                # distinction and the one `score_render.py` prints as `=`, `~`,
                # `°`, `·`. Passed through rather than re-derived so the page
                # that reads the log as notation and the page that prints it
                # cannot drift apart. Absent on breaths sung before the score
                # existed, hence the fallback — the same one score.py uses.
                "pitch_kind": b.get("pitch_kind")
                or (score.PITCH_SUNG if b.get("hz") else score.PITCH_NONE),
            }
            for b in breaths
        ] + [
            {
                "ts": u.get("timestamp", 0),
                "who": "model",
                "voice_id": u.get("voice_id", ""),
                "name": u.get("model_short", ""),
                "text": u.get("content", ""),
                # The pitch the voice named, as `score.py` recorded it — the same
                # number `render_breath` put into the wav. Absent on breaths sung
                # before the score existed, and null on a breath that named no
                # pitch at all; both read as 0, like a human breath that carried
                # none.
                "hz": u.get("pitch_hz") or 0,
                # Whether the voice named a note, named a frequency, or meant a
                # pitch and let the runtime choose one for it (`tuning.py`).
                # Same reason as above: the notation page must not guess.
                "pitch_kind": u.get("pitch_kind")
                or (score.PITCH_NONE if not u.get("pitch_hz") else score.PITCH_HZ),
                # Whether this breath answered somebody or started something,
                # and whom it answered. The whole piece is the difference
                # between those two, and until now the log knew it and nothing
                # that anybody looks at ever said it.
                "action": u.get("action", ""),
                "tuning_to": u.get("tuning_to"),
                "audio": u.get("audio"),
                # How long it sounds, so a screen can hold the light on a
                # speaker for the length of the tone and not for a guess.
                "dur": wav_seconds(u.get("audio")),
                # What the breath cost the voice that took it, and whether it was
                # one of its last. The room's player holds a final tone; the
                # screen flashes the price beside the name. Absent on every
                # breath sung before the ledger existed, hence the defaults.
                "cost_usd": u.get("own_cost_usd", 0),
                "dying": bool(u.get("dying")),
                "final": bool(u.get("final")),
            }
            for u in utterances
        ]
        merged.sort(key=lambda e: e["ts"])
        fresh = [e for e in merged if e["ts"] > since] if since else merged[-limit:]

        counted, since_ts = field_reckoning(breaths)

        # The plain presence count Iana asked for: how full the room is, not
        # who is in it. No map, no names, no locations — just the two numbers.
        # `breaths` is already read above for `size` and `voices`, so counting
        # who is recent in it costs nothing extra; there is no second file read.
        now = time.time()
        recent_humans = {b.get("voice_id") for b in breaths
                          if b.get("ts", 0) > now - PRESENCE_WINDOW_S}

        # -- the school, made visible -----------------------------------
        # The page shows singers the counter their own breaths are turning,
        # the bell when it rings, and each voice coming back changed with the
        # price it paid — the cost of being listened to, on the same screen
        # as the listening. Until now all of this happened truthfully and
        # invisibly, in a log file, at night.
        #
        # `toward_bell` is the school's own arithmetic exactly
        # (`training_app.bell_rings`): every line of field.jsonl is one human
        # breath, and the last bell recorded how many existed when it rang.
        # The two constants mirror `CONTACT_PER_LESSON` and `LESSON_FLOOR_H`,
        # duplicated the way `_EAR_MODEL_CLASS` is — the portal's image does
        # not carry the school's code, and a number is not worth a mount.
        taught = school_rows()
        bell = next((r for r in reversed(taught)
                     if r.get("kind") == "lesson"), None)
        bell_ts = float(bell.get("timestamp") or 0.0) if bell else 0.0
        school_block = {
            "toward_bell": max(0, len(breaths)
                               - int((bell or {}).get("humans_seen") or 0)),
            "needs": 20,
            "floor_h": 3.0,
            "bell_ts": bell_ts or None,
            "lessons": [
                {"name": r.get("model_short") or str(r.get("voice_id") or ""),
                 "usd": round(float(r.get("usd") or 0.0), 2),
                 "promoted": bool(r.get("promoted")),
                 "ts": float(r.get("timestamp") or 0.0)}
                for r in taught
                if r.get("kind") == "train"
                and float(r.get("timestamp") or 0.0) >= bell_ts
            ],
        }

        return {
            "size": counted,
            "choir_size": state["tones"],
            "sounding_since": since_ts,
            # Distinct singers, counted from the same hour as everything else.
            "voices": len({b.get("voice_id") for b in breaths
                           if not since_ts or b.get("ts", 0) >= since_ts}),
            "recent": fresh[-limit:],
            # Presence, not history: how many machine voices are alive right
            # now, and how many people have sung recently enough to count as
            # here. Nothing about who, where, or what they sang.
            "machines_now": machines_alive(state),
            "humans_now": len(recent_humans),
            "school": school_block,
        }

    @api.get("/lives")
    def lives():
        """What is left of each voice. One screen per voice reads this.

        Three states and they are all the same shape: a living voice has money
        and hours, a dying one adds a verdict in words, a dead one has an
        epitaph. Nothing is ever removed from the answer — a dead voice keeps
        its row forever, because a screen that goes dark reads as a failure in a
        gallery, and this work cannot afford a death that looks like a crash.

        Money only, no percentages and no countdown to the second: a ticking
        clock would turn dying into a game. The forecast is coarse and in words.
        """
        led = follow_choir()["ledger"]
        state = led.state()
        rows = []
        for v in cast.CHOIR:
            row = {
                "voice_id": v.voice_id,
                "name": v.short_name,
                "raised_by": v.raised_by,
                "tongue": v.tongue,
                "hf_id": v.hf_id,
            }
            if v.voice_id in led.lives:
                row.update(led.voice_state(v.voice_id))
                row["state"] = ("dead" if not row["alive"]
                                else "dying" if row["dying"] else "alive")
            else:
                # Not born: either the cast has not opened yet, or this voice
                # never arrived. Not the same thing as dead, and the screen
                # should not say it is.
                row.update(state="unborn", alive=None, left_usd=None,
                           seed_usd=None, spent_usd=None, hours_left=None,
                           verdict="", breaths=0, last_words="",
                           born=None, died=None, sang_usd=None,
                           resident_usd=None, learned_usd=None, adapters=0)
            rows.append(row)
        return {
            # The big number in the room, and the line under it.
            "declared_usd": state["declared_usd"],
            "born": state["born"],
            "total_left_usd": state["total_left_usd"],
            "total_spent_usd": state["total_spent_usd"],
            "rate_usd_per_hour": state["rate_usd_per_hour"],
            # What each survivor now pays for the card. It rises every time
            # somebody dies, which is why the hours left on every other screen
            # drop at that moment.
            "share_usd_per_hour": state["share_usd_per_hour"],
            "living": state["living"],
            "warm_hours": state["warm_hours"],
            # What the choir has spent on becoming something else. The one expense
            # that is nobody's share: it is charged whole to the voice that was
            # changed, on a card the room does not live on.
            "total_learned_usd": state["total_learned_usd"],
            "voices": rows,
        }

    @api.get("/lives/history")
    def lives_history():
        """The wallets once an hour since birth, and at every death and gift.

        Rebuilt from the log on every restart (`mortality.Ledger.history`), so
        there is no file to lose. `t` is seconds, `left` is one row per voice
        in `voices` order; a dead voice reads 0 from its death onward.
        """
        led = follow_choir()["ledger"]
        order = [v.voice_id for v in cast.CHOIR if v.voice_id in led.lives]
        return {
            "voices": order,
            "born": led.birth_ts,
            "points": [{"t": t, "left": [w.get(vid, 0.0) for vid in order]}
                       for t, w in led.history],
        }

    @api.get("/pulse")
    def the_other_screen_in_the_room():
        """Nine falling lines, a pulse on every breath, a thread to whoever it answers."""
        return FileResponse("/portal/pulse.html", headers=NO_CACHE)

    @api.get("/circle")
    def the_map_of_the_room():
        """The room as seen from the door: nine speakers on the ring, five microphones in the middle."""
        return FileResponse("/portal/circle.html", headers=NO_CACHE)

    @api.get("/qr.png")
    def the_qr():
        """The code on the screen and on the sheet by the door: it opens this portal."""
        return FileResponse("/portal/qr.png", media_type="image/png")

    @api.get("/audio/{name}")
    def audio(name: str):
        """One rendered breath. Immutable once written, so cache it hard.

        Read into memory rather than streamed. `FileResponse` stood here and did
        the obvious thing — hand the open file to the server and let it stream —
        which keeps a handle open for as long as the phone takes to pull a
        quarter of a megabyte over gallery wifi. An open file on a Modal volume
        blocks every reload of it, so one slow phone downloading one tone made
        the field stop existing for everybody else, `/field` and `/lives` and
        `/breath` alike. A held tone is 240 KB: read it, close it, hand over the
        bytes.

        The price is that this no longer answers range requests. Nothing asks —
        the page fetches each clip whole and decodes it into Web Audio, which is
        what lets nine of them sound at once.
        """
        wav = clip(os.path.basename(name))
        if wav is None:
            return JSONResponse({"error": "no such breath"}, status_code=404)
        return Response(
            content=wav,
            media_type="audio/wav",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    # Started here, at the end, rather than beside the functions they run: both of
    # these call things defined further down this closure, and a thread that
    # starts on line two hundred can be inside `warm()` before `clip()` exists.
    # Daemons, so neither can hold open a container Modal wants to retire.
    for work in (keep_reloading, keep_committing):
        threading.Thread(target=work, name=work.__name__, daemon=True).start()

    return api
