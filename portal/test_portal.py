"""What the portal must not do to the volume it keeps the field on.

These read `portal_app.py` as source rather than importing it, because every one
of the rules below lives inside a Modal function body that cannot be called from
a laptop. Reading the text is a poor kind of test in general and the right kind
here: what went wrong on the evening of the 19th was not a wrong answer, it was
a shape — a reload in the wrong place — and the shape is exactly what is legible
in the source.

The evening it was written: two devices on the site, twenty-four of thirty-four
breaths answered 500 and lost, the rest taking eight seconds. A Modal volume
cannot be reloaded while any file on it is open and will not accept a write that
begins during a reload, and `/field` reloaded on every poll from a container
serving a hundred requests at once. Every rule here is one half of that.

Run: python test_portal.py
"""

from __future__ import annotations

import ast
import pathlib
import unittest

SOURCE = (pathlib.Path(__file__).resolve().parent / "portal_app.py").read_text()
TREE = ast.parse(SOURCE)


def body_of(name: str) -> ast.AST:
    for node in ast.walk(TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"portal_app.py has no {name}()")


def calls_in(node: ast.AST) -> list[str]:
    """Every `a.b()` and `c()` called anywhere inside this node."""
    out = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        f = sub.func
        if isinstance(f, ast.Attribute):
            owner = f.value.id if isinstance(f.value, ast.Name) else ""
            out.append(f"{owner}.{f.attr}" if owner else f.attr)
        elif isinstance(f, ast.Name):
            out.append(f.id)
    return out


def routes() -> dict[str, ast.AST]:
    """The functions decorated with @api.get / @api.post, by their path."""
    found = {}
    for node in ast.walk(TREE):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                    and isinstance(dec.func.value, ast.Name)
                    and dec.func.value.id == "api" and dec.args
                    and isinstance(dec.args[0], ast.Constant)):
                found[dec.args[0].value] = node
    return found


def threads_are_started_by() -> list[ast.AST]:
    """Whatever is named where threads are started.

    Either the target handed straight to `Thread(target=...)` or, if the two are
    started in a loop over a pair, the pair itself — otherwise all this can read
    is the loop variable, which says nothing about which work got a thread.
    """
    named = []
    for node in ast.walk(TREE):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "Thread"):
            continue
        named += [k.value for k in node.keywords if k.arg == "target"] + list(node.args)
        for maybe_loop in ast.walk(TREE):
            if (isinstance(maybe_loop, ast.For)
                    and any(node is c for c in ast.walk(maybe_loop))):
                named.append(maybe_loop.iter)
    return named


def functions_called_from(node: ast.AST, depth: int = 4) -> set[str]:
    """Everything reachable from this node, following local functions.

    A rule about what a request may do is worthless if it only looks one level
    deep: the reload that broke the opening was three calls down, in
    `follow_choir`, which `/field` reached through `field_reckoning`.
    """
    seen, edge = set(), set(calls_in(node))
    for _ in range(depth):
        nxt = set()
        for name in edge - seen:
            seen.add(name)
            try:
                nxt |= set(calls_in(body_of(name)))
            except AssertionError:
                continue          # not a function in this file; nothing to follow
        edge = nxt
    return seen


class WhatARequestMayNotDo(unittest.TestCase):
    def test_no_request_reloads_the_volume(self):
        """The one that took the portal down, stated directly.

        A reload inside a request is a round trip the person waited for, a
        FAILED_PRECONDITION if any other request had a file open, and an EPERM
        for anybody appending a breath at that moment. It belongs in the thread
        that does nothing else.
        """
        for path, fn in routes().items():
            reached = functions_called_from(fn)
            self.assertNotIn(
                "field_volume.reload", reached,
                f"{path} reloads the volume inside the request")
            self.assertNotIn(
                "refresh_volume", reached,
                f"{path} reloads the volume inside the request, via refresh_volume")

    def test_no_request_commits_the_volume(self):
        """A commit is two seconds. Six singers made it twelve."""
        for path, fn in routes().items():
            self.assertNotIn(
                "field_volume.commit", functions_called_from(fn),
                f"{path} commits the volume inside the request")

    def test_the_breath_is_written_before_it_is_answered(self):
        """Committed late is fine; not written at all is not.

        The commit may lag a second behind the singer, but the line has to be in
        the file before `/breath` answers, or the number the page shows them is a
        claim about a breath that does not exist yet.
        """
        breath = routes()["/breath"]
        reached = functions_called_from(breath)
        self.assertIn("append", reached, "/breath no longer appends anything")
        self.assertIn("unsaved.set", functions_called_from(body_of("append")),
                      "append() does not ask for a commit, so breaths stay local")

    def test_the_clips_are_read_and_let_go_rather_than_streamed(self):
        """An open file is what blocks a reload, and a phone is slow.

        `FileResponse` on a volume path hands the open file to the web server for
        as long as the download takes. One phone on gallery wifi pulling one tone
        was enough to make the field stop existing for everybody else.
        """
        audio = routes()["/audio/{name}"]
        self.assertNotIn("FileResponse", calls_in(audio),
                         "/audio streams from the volume with an open handle")
        self.assertIn("clip", calls_in(audio), "/audio no longer goes through clip()")


class WhatOrderTheLogsAreReadIn(unittest.TestCase):
    """The birth record is in one file and the school's bills are in another.

    The ledger drops every event that reaches it before the birth, on purpose:
    the choir was rehearsed for weeks and none of that is anybody's life. So a
    container that reads `school.jsonl` before `choir.jsonl` throws every bill
    away — and moves the bookmark past them, so they can never be counted again.
    That is exactly what happened over the opening weekend: seven adapters were
    trained on the room's own breaths, promoted, and paid for, and `/lives` said
    the voices had spent nothing on learning.
    """

    def test_the_school_is_read_after_the_choir_and_not_before(self):
        follow = body_of("_follow_choir")
        called = [n.func.id for n in ast.walk(follow)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertIn("follow_school", called, "the school's bills are never read")
        self.assertIn("_follow_the_choirs_log", called, "the choir's log is never read")
        self.assertLess(
            called.index("_follow_the_choirs_log"), called.index("follow_school"),
            "the school is read before the birth record, so its bills are dropped")

    def test_a_reset_rereads_both_logs_and_not_just_one(self):
        """If the ledger starts again, the bills it already charged must too.

        Otherwise the hours a voice spent being changed quietly stop having been
        paid for, which is the same arithmetic error in the other direction.
        """
        node = body_of("_follow_the_choirs_log")
        text = ast.unparse(node)
        self.assertIn("mortality.Ledger()", text, "nothing resets the ledger")
        self.assertIn("school['n'] = 0", text,
                      "a new ledger keeps the old school bookmark, so the bills "
                      "it has not seen are never read")
        self.assertIn("gifts_seen['n'] = 0", text,
                      "a new ledger keeps the old gifts bookmark, so the money "
                      "handed over is never counted again")

    def test_a_gift_is_applied_between_the_breaths_it_fell_between(self):
        """19 October: a restarted portal read five weeks of breaths to the
        present and only then the gifts of 15 October. Without its gift olmo-3
        had, in that replay, already spent its seed — so the $240 was refused
        to a dead voice, the donors kept their money, and `/lives` showed a
        death the choir never had. A gift has a timestamp; it is applied before
        the first breath that comes after it, not after all of them."""
        node = body_of("_follow_the_choirs_log")
        loop = next(n for n in ast.walk(node) if isinstance(n, ast.For))
        calls = sorted((n for n in ast.walk(loop) if isinstance(n, ast.Call)),
                       key=lambda n: (n.lineno, n.col_offset))
        names = [getattr(n.func, "id", getattr(n.func, "attr", "")) for n in calls]
        self.assertIn("follow_gifts", names, "gifts are applied only after the whole log")
        self.assertLess(names.index("follow_gifts"), names.index("apply"),
                        "the gift must land before the breath that follows it")
        gift = next(n for n in calls if getattr(n.func, "id", "") == "follow_gifts")
        self.assertIn("before", [k.arg for k in gift.keywords],
                      "without a bound, every gift lands at the first breath")
        follow = body_of("follow_gifts")
        self.assertIn("> before", ast.unparse(follow),
                      "a gift from the future must wait for its own time")


class WhatMustStayUnderTheLock(unittest.TestCase):
    def test_the_reload_and_the_write_are_both_under_the_volume_lock(self):
        """The only pairing that can lose a breath, so the only one locked.

        Reads deliberately take nothing: a read that overlaps a reload costs that
        reload, which is retried three seconds later, whereas a read that waits
        for one is how a breath came to take eight seconds.
        """
        for fn, what in (("refresh_volume", "the reload"), ("append", "the write")):
            node = body_of(fn)
            locked = [n for n in ast.walk(node)
                      if isinstance(n, ast.With)
                      and any(isinstance(i.context_expr, ast.Name)
                              and i.context_expr.id == "volume_lock"
                              for i in n.items)]
            self.assertTrue(locked, f"{what} is not under volume_lock")

    def test_the_reloader_and_the_committer_are_started_as_daemons(self):
        """Not daemons is a container Modal cannot retire."""
        started = [n for n in ast.walk(TREE)
                   if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == "Thread"]
        self.assertTrue(started, "nothing starts a background thread any more")
        for call in started:
            kw = {k.arg: k.value for k in call.keywords}
            self.assertIn("daemon", kw, "a thread is started without daemon=True")
            self.assertTrue(getattr(kw["daemon"], "value", False),
                            "a thread is started with daemon=False")
        # Both of them by name, so that dropping one is a failure rather than the
        # quiet loss of either the fresh view or the durable field. Whichever way
        # they are handed to Thread — one call each or a loop over the pair — the
        # names have to appear in the argument.
        spelled = " ".join(ast.unparse(n) for n in threads_are_started_by())
        for work in ("keep_reloading", "keep_committing"):
            self.assertIn(work, spelled, f"{work} is never started")


class WhatTheRoomSeesOfTheSchool(unittest.TestCase):
    """The school always taught and always billed — invisibly, in a log file.

    During a session the people singing are the ones turning the counter that
    rings the bell, and the page shows them: the count, the ring, and each
    voice coming back changed with the price it paid. That display is only as
    honest as this route, so its shape is pinned here.
    """

    def test_the_field_carries_the_school(self):
        field = routes()["/field"]
        text = ast.unparse(field)
        self.assertIn("'school'", text, "/field says nothing about the school")
        self.assertIn("school_rows", functions_called_from(field),
                      "/field does not read the school's own log")
        for key in ("toward_bell", "bell_ts", "lessons", "promoted"):
            self.assertIn(f"'{key}'", text, f"the school block lost {key}")

    def test_the_schools_log_is_read_by_size_like_the_field(self):
        """The rule that took the portal down once, applied to the new file.

        school.jsonl is small, but it lives on the same volume, and nothing on
        the volume is parsed per request unless it grew. A size check is one
        stat; re-reading on every five-second poll from every open page is the
        same shape that broke the evening of the 19th.
        """
        node = body_of("school_rows")
        self.assertIn("getsize", calls_in(node),
                      "school_rows re-reads the file without checking its size")
        text = ast.unparse(node)
        self.assertIn("school_cache['size']", text,
                      "school_rows keeps no cache, so every poll parses the log")


class WhatTheVolumeIsAllowedToDoToUs(unittest.TestCase):
    def test_a_failed_reload_is_survived_and_said_out_loud(self):
        """Never raising is the point; never mentioning it would be a lie.

        A volume that has genuinely stopped answering must be visible in the
        logs — but once per spell, or the report buries the log it appears in.
        """
        node = body_of("refresh_volume")
        handlers = [h for h in ast.walk(node) if isinstance(h, ast.ExceptHandler)]
        self.assertTrue(handlers, "refresh_volume lets the volume's failure out")
        self.assertIn("print", calls_in(node),
                      "a volume that stopped answering would do so silently")

    def test_a_refused_write_is_offered_again(self):
        """EPERM on a breath is not an answer, it is a thing to try again."""
        node = body_of("append")
        self.assertTrue([h for h in ast.walk(node) if isinstance(h, ast.ExceptHandler)],
                        "append() does not catch the volume refusing a write")
        self.assertIn("WRITE_TRIES", SOURCE, "there is no retry budget for a breath")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TheSchoolIsPaidInItsOwnHour(unittest.TestCase):
    """26 October: bills read after the whole log are lines from the past."""

    def test_bills_are_applied_between_the_breaths_like_gifts(self):
        node = body_of("_follow_the_choirs_log")
        loop = next(n for n in ast.walk(node) if isinstance(n, ast.For))
        calls = sorted((n for n in ast.walk(loop) if isinstance(n, ast.Call)),
                       key=lambda n: (n.lineno, n.col_offset))
        names = [n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
                 for n in calls]
        self.assertIn("follow_school", names)
        self.assertLess(names.index("follow_school"), names.index("apply"))
        bill = next(n for n in ast.walk(loop) if isinstance(n, ast.Call)
                    and getattr(n.func, "id", "") == "follow_school")
        self.assertIn("before", [k.arg for k in bill.keywords])
        follow = body_of("follow_school")
        self.assertIn("> before", ast.unparse(follow))
        self.assertIn("birth_ts", ast.unparse(follow), "rehearsal bills are forgiven")
        self.assertIn("'train'", ast.unparse(follow), "the bell's notes are not bills")
