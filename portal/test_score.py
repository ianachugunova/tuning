"""Tests for the other thing in this repository that cannot be fixed later.

    python3 test_score.py

`test_mortality.py` guards the arithmetic that has to be right before 18
September. This guards the notation, for exactly the same reason: the log is
append-only, the weights are deleted in November, and whatever is not written
into a line while the choir is alive is not in the piece. A missing duration
cannot be added in December — there is nothing left to measure it from.

So the properties are tested rather than the output. That every breath carries a
duration and that the duration is the sound and not the intention. That every
silence in a merged chronology is recoverable. That a pitch survives as both a
name and a number, and that how it was named survives too. That lines written
before today still read. And that adding all of this did not change what the
ledger makes of the same log, because the accounting replays the same file.

Stdlib only, like the modules it tests.
"""

import os
import sys
import unittest

import mortality as M
import score as S
import score_render as R

# The one seam that neither module can test alone: `live_v1.pitch_text` decides
# which part of a model's line is a pitch, and `score.py` decides how that is
# notated. They must agree, and they live in different trees. Imported softly
# because `score.py` is stdlib-only by design and must stay importable in the
# portal's image, which has no `live_v1` in it at all.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from live_v1.pitch_text import extract_all_pitches
except Exception:                       # noqa: BLE001 — any import failure
    extract_all_pitches = None


def machine_row(ts, vid="voice_02", content="D4, a hum", pitches=(("D4", 293.66),),
                breath=6.0, sound=6.4, tuning=None, prev_end=None, **extra):
    """A tone line in the shape `choir_app.take_a_breath` now writes."""
    row = {
        "timestamp": ts, "voice_id": vid, "model_short": "phi-4-mini",
        "kind": "tone", "action": "TUNE" if tuning else "NEW",
        "content": content, "tuning_to": tuning,
        "breath_chosen_s": breath, "api_latency_ms": 900,
        "playback": "both", "playback_note": "", "gpu_s": 2.0,
        "sound_s": sound, "toned": True, "spoke": True,
        "silence_before_s": S.silence_before(prev_end, ts),
        "heard": [], "tuning_to_id": tuning, "tongue": "English",
    }
    row.update(S.machine_pitch_fields([(hz, label) for label, hz in pitches]))
    row.update(extra)
    return row


def human_row(started, dur=4.6, hz=305.2, vid="human_7f3a", rest_before=None,
              tail=0.5, **extra):
    """A breath in the shape `portal_app./breath` now writes."""
    row = {
        "ts": started + dur + tail, "voice_id": vid, "kind": "tone",
        "who": "human", "source": "portal",
        "note": S.note_name(hz), "hz": hz, "cents": S.cents_off(hz),
        "stability": "steady", "duration_s": dur, "voiced_fraction": 0.94,
        "started_ts": started, "ended_ts": started + dur, "tail_s": tail,
        "sound_s": dur, "own_rest_s": rest_before,
        "std_hz": 2.4, "contour_hz": [303.1, 305.0, 306.2, 305.4],
        "contour_hop_s": round(dur / 4, 3),
        "content": f"{S.note_name(hz)} ~{hz:.0f}Hz [steady]",
    }
    row.update(S.human_pitch_fields(hz, S.note_name(hz), S.cents_off(hz)))
    row.update(extra)
    return row


class Pitch(unittest.TestCase):
    """A pitch has to survive as a number, as a name, and as a way of naming."""

    def test_a_named_note_is_notated_as_that_note(self):
        f = S.machine_pitch_fields([(293.66, "D4")])
        self.assertEqual(f["pitch_kind"], S.PITCH_NOTE)
        self.assertEqual(f["pitch_note"], "D4")
        self.assertEqual(f["pitch_label"], "D4")
        self.assertAlmostEqual(f["pitch_hz"], 293.66, places=2)

    def test_a_named_frequency_keeps_the_frequency_and_gains_a_note(self):
        """The swap needs this: a person cannot sing "294 Hz" off a page."""
        f = S.machine_pitch_fields([(294.0, "294Hz")])
        self.assertEqual(f["pitch_kind"], S.PITCH_HZ)
        self.assertEqual(f["pitch_note"], "D4")
        self.assertEqual(f["pitch_cents"], 2)

    def test_a_sung_pitch_is_marked_as_named_by_nobody(self):
        """And the cents are kept, because 33 cents under D#4 is the piece."""
        f = S.human_pitch_fields(305.2, "D#4", -33)
        self.assertEqual(f["pitch_kind"], S.PITCH_SUNG)
        self.assertEqual(f["pitch_note"], "D#4")
        self.assertEqual(f["pitch_cents"], -33)

    def test_a_line_with_no_pitch_says_so_rather_than_guessing(self):
        f = S.machine_pitch_fields([])
        self.assertEqual(f["pitch_kind"], S.PITCH_NONE)
        self.assertIsNone(f["pitch_hz"])

    def test_note_names_match_the_browser(self):
        """`pitch-core.js` names the human pitches; these must be the same names.

        A human breath is named in the browser and a machine breath is named
        here. If the two disagreed, the same frequency would appear as two
        different note heads depending on who sang it.
        """
        for hz, name in [(440.0, "A4"), (261.63, "C4"), (293.66, "D4"),
                         (196.0, "G3"), (369.99, "F#4"), (1046.5, "C6")]:
            self.assertEqual(S.note_name(hz), name, f"{hz} Hz")
        self.assertEqual(S.cents_off(440.0), 0)
        self.assertEqual(S.note_name(0), "")

    def test_a_voice_that_names_two_tones_keeps_both(self):
        f = S.machine_pitch_fields([(293.66, "D4"), (349.23, "F4")])
        self.assertEqual(len(f["pitch_all"]), 2)
        self.assertAlmostEqual(f["pitch_hz"], 293.66, places=2)


@unittest.skipIf(extract_all_pitches is None, "live_v1 is not importable here")
class WhatTheVoicesActuallyWrite(unittest.TestCase):
    """The seam: `live_v1` finds the pitch, `score.py` notates it.

    Run against lines the cast has actually produced, or plausibly will. The
    numbers here are not invented — they come from the same function
    `render_breath` calls, so a disagreement between this and the wav is a
    disagreement this test would catch.
    """

    def kind_of(self, text):
        return S.machine_pitch_fields(extract_all_pitches(text))

    def test_the_three_ways_a_voice_names_a_pitch(self):
        self.assertEqual(self.kind_of("D4, a low steady hum")["pitch_kind"],
                         S.PITCH_NOTE)
        self.assertEqual(self.kind_of("294Hz \u2014 \u6211\u8ddf\u7740\u4f60")["pitch_kind"],
                         S.PITCH_HZ)
        self.assertEqual(self.kind_of("a long tone that does not end")["pitch_kind"],
                         S.PITCH_NONE)

    def test_a_bare_number_is_a_frequency_and_a_duration_is_not(self):
        """The elder writes prose. "440" is a pitch; "3 s" is not."""
        self.assertEqual(self.kind_of("440")["pitch_note"], "A4")
        self.assertIsNone(self.kind_of("silence for 3 s")["pitch_hz"])

    def test_a_refused_duration_is_not_then_accepted_in_pieces(self):
        """A decimal duration used to be read twice: refused whole, kept in part.

        "12.5s" is a duration and is turned away — and the parser then tried
        again on the "12" in front of the dot, which nothing forbade, and wrote
        down 12 Hz. Every second a voice mentions was a tone, and so was every
        timestamp in the field summary the voices are given to read. gpt-2
        echoed its own prompt back and the log recorded it singing 31 Hz.
        """
        for line in ("held for 12.5s",
                     "a rest of 4.5s",
                     "[-31.2s] human_ynmq9xv8 (human):",
                     "[-6.4s] voice_02 (phi-4-mini)"):
            self.assertIsNone(self.kind_of(line)["pitch_hz"], msg=line)
        # And the tone still survives a duration standing next to it.
        self.assertEqual(self.kind_of("a rest of 4.5s then D4")["pitch_note"], "D4")
        self.assertEqual(self.kind_of("220.5")["pitch_hz"], 220.5)

    def test_a_pitch_nobody_can_sing_is_folded_and_still_notated(self):
        """"0.5 kHz" is a real answer, and it has a note name."""
        f = self.kind_of("0.5 kHz, a shimmer")
        self.assertEqual(f["pitch_note"], "B4")
        self.assertEqual(f["pitch_kind"], S.PITCH_HZ)

    def test_a_gesture_of_two_tones_keeps_both_and_sounds_the_first(self):
        f = self.kind_of("G#3 then F4")
        self.assertEqual(f["pitch_note"], "G#3")
        self.assertEqual(len(f["pitch_all"]), 2)

    def test_the_pitch_in_the_log_is_the_pitch_in_the_wav(self):
        """`render_breath` sounds `extract_all_pitches(text)[0][0]`. So does this.

        Stated as a test because it is the reason `machine_pitch_fields` takes
        the list rather than the text: a second implementation of "which pitch
        did it mean" would drift, and the log would stop describing the sound.
        """
        for text in ("D4, a hum", "294Hz", "440", "G#3 then F4", "0.5kHz"):
            sounded = extract_all_pitches(text)[0][0]
            self.assertAlmostEqual(self.kind_of(text)["pitch_hz"], sounded,
                                   places=2, msg=text)


class Durations(unittest.TestCase):
    """The one thing a score cannot be performed without."""

    def test_the_length_notated_is_the_sound_and_not_the_intention(self):
        """A spoken breath is stretched to cover its words. That was invisible.

        `breath_chosen_s` is what the voice asked for; the tone is then extended
        to the length of the speech mixed over it. Every breath that carried
        words sounded for longer than the log used to say, by an amount nothing
        recorded.
        """
        row = machine_row(1000.0, breath=6.0, sound=9.4)
        note = S.read([row], [])[0]
        self.assertAlmostEqual(note.sound_s, 9.4)
        self.assertNotAlmostEqual(note.sound_s, row["breath_chosen_s"])

    def test_a_chosen_silence_has_a_duration_of_zero_and_still_a_pitch(self):
        """A voice may name a tone and refuse to sound it. That is notatable."""
        row = machine_row(1000.0, sound=0.0, playback="none", toned=False,
                          spoke=False)
        note = S.read([row], [])[0]
        self.assertEqual(note.sound_s, 0.0)
        self.assertEqual(note.pitch_note, "D4")

    def test_a_human_breath_has_an_onset_that_is_not_its_arrival(self):
        """The page waits out half a second of silence before it decides.

        Plus the network. If the onset were `ts` every human note in the score
        would sit late by its own length plus a tail, and nothing downstream
        could detect it.
        """
        row = human_row(1000.0, dur=4.6, tail=0.5)
        note = S.read([], [row])[0]
        self.assertAlmostEqual(note.t, 1000.0)
        self.assertAlmostEqual(row["ts"] - note.t, 5.1)
        self.assertAlmostEqual(note.end, 1004.6)

    def test_every_breath_in_a_rendered_slice_has_a_duration(self):
        choir, human = R.demo_logs()
        notes = [n for n in S.read(choir, human) if n.kind == "tone"]
        self.assertTrue(notes)
        for n in notes:
            self.assertIsNotNone(n.sound_s, n.text)
            self.assertGreaterEqual(n.sound_s, 0.0)
        self.assertTrue(any(n.sound_s > 0 for n in notes))


class Silence(unittest.TestCase):
    """A tuning meditation is mostly rests. They have to be recoverable."""

    def test_the_rest_is_written_down_by_the_voice_that_heard_it(self):
        rows = [machine_row(1000.0, sound=6.0),
                machine_row(1031.0, sound=5.0, prev_end=1006.0)]
        notes = S.read(rows, [])
        self.assertIsNone(notes[0].rest_s)          # nothing before the first
        self.assertAlmostEqual(notes[1].rest_s, 25.0)

    def test_the_rest_is_also_derivable_when_it_was_not_written(self):
        """Which is what makes the lines from before today still performable."""
        rows = [{"timestamp": 1000.0, "voice_id": "voice_00", "kind": "tone",
                 "model_short": "gpt-2", "content": "a hum",
                 "breath_chosen_s": 6.0},
                {"timestamp": 1031.0, "voice_id": "voice_02", "kind": "tone",
                 "model_short": "phi-4-mini", "content": "D4",
                 "breath_chosen_s": 5.0}]
        notes = S.read(rows, [])
        self.assertAlmostEqual(notes[1].rest_s, 25.0)

    def test_the_score_tiles_the_time_it_covers(self):
        """Sound plus silence is the whole span, with nothing unaccounted for.

        This is the property that says the six weeks are reconstructible. Walk
        the chronology: every breath either begins after the field went quiet,
        in which case its rest is exactly the length of that quiet, or it begins
        while somebody else is still sounding, in which case its rest is zero
        and the overlap is carried by the absolute onset. There is no third
        case, and no interval that is neither sound nor a notated silence.
        """
        choir, human = R.demo_logs()
        notes = [n for n in S.read(choir, human) if n.kind == "tone"]
        end = notes[0].end
        for n in notes[1:]:
            self.assertIsNotNone(n.rest_s, n.text)
            if n.rest_s > 0:
                self.assertAlmostEqual(n.t, end + n.rest_s, places=2, msg=n.text)
            else:
                # A zero rest is an entry into a field that is still sounding.
                self.assertLessEqual(n.t, end + 1e-6, msg=n.text)
            end = max(end, n.end)
        self.assertAlmostEqual(end, max(n.end for n in notes), places=2)

    def test_silence_across_the_two_logs_is_one_silence(self):
        """A machine resting while a person sings is not resting."""
        notes = S.read([machine_row(1000.0, sound=6.0)],
                       [human_row(1010.0, dur=4.0)])
        self.assertAlmostEqual(notes[1].rest_s, 4.0)

    def test_the_voices_own_account_of_the_rest_is_kept_beside_the_derived_one(self):
        """Two different facts, and the log has room for both.

        The derived rest is the archive's: it is exact and it is what a
        conductor reads. `heard_rest_s` is what the voice itself measured
        against the field it had been given, which is what it was answering —
        and the only rest that survives if the other log is ever lost.
        """
        rows = [machine_row(1000.0, sound=6.0),
                machine_row(1031.0, sound=5.0, prev_end=1006.0)]
        # A person sang in between, so the field was quiet for less than the
        # machine thought. Both numbers are true; they answer different things.
        notes = S.read(rows, [human_row(1010.0, dur=4.0)])
        machine = [n for n in notes if n.who == "machine"][1]
        self.assertAlmostEqual(machine.rest_s, 17.0)        # since the field
        self.assertAlmostEqual(machine.heard_rest_s, 25.0)  # since it listened

    def test_a_singers_own_rest_is_kept_as_measured_not_reconstructed(self):
        """A phone timing its own two notes beats two arrival times."""
        notes = S.read([], [human_row(1000.0, dur=4.0),
                            human_row(1100.0, dur=3.0, rest_before=95.5)])
        self.assertIsNone(notes[0].own_rest_s)
        self.assertAlmostEqual(notes[1].own_rest_s, 95.5)

    def test_a_machines_own_rest_is_derived_from_its_own_part(self):
        """What the performer of that one part reads off their own page."""
        notes = S.read([machine_row(1000.0, vid="voice_02", sound=6.0),
                        machine_row(1050.0, vid="voice_05", sound=4.0),
                        machine_row(1100.0, vid="voice_02", sound=5.0)], [])
        self.assertAlmostEqual(notes[2].rest_s, 46.0)        # since anybody
        self.assertAlmostEqual(notes[2].own_rest_s, 94.0)    # since itself

    def test_no_before_is_none_and_never_zero(self):
        """A zero would claim a rest was measured. None says it was not."""
        self.assertIsNone(S.silence_before(None, 1000.0))
        self.assertEqual(S.silence_before(994.0, 1000.0), 6.0)
        # Voices overlap; a negative gap is an overlap, not a negative rest.
        self.assertEqual(S.silence_before(1006.0, 1000.0), 0.0)


class WhoAndWhereTo(unittest.TestCase):
    """Parts, and the arrows between them."""

    def test_the_two_logs_merge_into_one_chronology(self):
        notes = S.read([machine_row(1000.0), machine_row(1100.0)],
                       [human_row(1050.0)])
        self.assertEqual([n.who for n in notes],
                         ["machine", "human", "machine"])

    def test_a_part_is_a_voice_and_both_kinds_of_voice_have_one(self):
        """The intended performance swaps them, so both must stand alone."""
        choir, human = R.demo_logs()
        by_part = S.parts(S.read(choir, human))
        self.assertIn("voice_02", by_part)
        self.assertIn("human_7f3a", by_part)
        for part in by_part.values():
            self.assertTrue(all(n.sound_s is not None for n in part))

    def test_tuning_to_resolves_to_a_part_when_the_prose_names_one(self):
        heard = ["voice_00", "voice_05", "human_7f3a"]
        self.assertEqual(S.resolve_tuning_to("voice_05", heard), "voice_05")
        self.assertEqual(S.resolve_tuning_to("the voice 7f3a was singing",
                                             heard), "human_7f3a")
        self.assertIsNone(S.resolve_tuning_to("the low warm hum", heard))
        self.assertIsNone(S.resolve_tuning_to(None, heard))

    def test_unresolvable_prose_is_kept_rather_than_dropped(self):
        row = machine_row(1000.0, tuning="the low warm hum", tuning_to_id=None)
        note = S.read([row], [])[0]
        self.assertEqual(note.tuning_to, "the low warm hum")
        self.assertEqual(note.tuning_to_id, "")


class Compatibility(unittest.TestCase):
    """The log is extended, never replaced. Everything else must still work."""

    def test_the_ledger_reads_the_new_shape_exactly_as_it_read_the_old(self):
        """The accounting replays the same file. It must not have noticed.

        `mortality.py` is the one thing in the repository that cannot be wrong,
        and it walks every line of `choir.jsonl` including the ones this change
        widened. Same money, same deaths, same warm hours.
        """
        old = {"timestamp": 0.0, "kind": "tone", "voice_id": "a", "gpu_s": 3.0,
               "content": "D4"}
        for shape in ("old", "new"):
            led = M.Ledger()
            led.apply(M.birth_record(M.BIRTH_TS, {"a": 50.0, "b": 50.0},
                                     [{"voice_id": "a"}, {"voice_id": "b"}]))
            t = M.BIRTH_TS
            led.apply(M.heartbeat_record(t, ["a", "b"]))
            for i in range(1, 200):
                e = dict(old, timestamp=t + i * 30 - 3)
                if shape == "new":
                    e = machine_row(t + i * 30 - 3, vid="a", sound=9.4,
                                    prev_end=t + i * 30 - 20)
                    e["gpu_s"] = 3.0
                led.apply(e)
                led.apply(M.heartbeat_record(t + i * 30, ["a", "b"]))
            if shape == "old":
                before = (round(led.charged_usd, 9), round(led.warm_s, 6),
                          round(led.lives["a"].spent_usd, 9),
                          led.lives["a"].breaths)
            else:
                after = (round(led.charged_usd, 9), round(led.warm_s, 6),
                         round(led.lives["a"].spent_usd, 9),
                         led.lives["a"].breaths)
        self.assertEqual(before, after)

    def test_the_invariant_still_holds_over_the_new_shape(self):
        led = M.Ledger()
        led.apply(M.birth_record(M.BIRTH_TS, {"a": 50.0}, [{"voice_id": "a"}]))
        t = M.BIRTH_TS
        led.apply(M.heartbeat_record(t, ["a"]))
        prev = None
        for i in range(1, 300):
            row = machine_row(t + i * 30 - 3, vid="a", prev_end=prev)
            row["gpu_s"] = 3.0
            prev = t + i * 30 - 3 + row["sound_s"]
            led.apply(row)
            led.apply(M.heartbeat_record(t + i * 30, ["a"]))
        self.assertAlmostEqual(led.charged_usd,
                               M.RATE_USD_PER_S * (led.warm_s + led._owed),
                               places=9)

    def test_a_line_from_before_the_score_existed_still_reads(self):
        """Nothing in the log so far has any of these fields."""
        rows = [{"timestamp": 1000.0, "kind": "tone", "voice_id": "voice_00",
                 "model_short": "gpt-2", "action": "NEW",
                 "content": "a long low tone", "tuning_to": None,
                 "breath_chosen_s": 7.0, "api_latency_ms": 800,
                 "playback": "both"}]
        humans = [{"ts": 1010.0, "voice_id": "anon_1", "note": "D#4",
                   "hz": 305.2, "cents": -33, "stability": "steady",
                   "duration_s": 4.6, "voiced_fraction": 0.9,
                   "source": "portal"}]
        notes = S.read(rows, humans)
        self.assertEqual(len(notes), 2)
        self.assertAlmostEqual(notes[0].sound_s, 7.0)   # the floor, not the truth
        self.assertAlmostEqual(notes[1].t, 1005.4)      # onset, approximated
        self.assertEqual(notes[1].pitch_note, "D#4")
        self.assertEqual(notes[1].pitch_kind, S.PITCH_SUNG)

    def test_the_ledger_still_sees_a_breath_that_answered_a_person(self):
        """`mortality` matches `tuning_to` against "human". Still true."""
        led = M.Ledger()
        led.apply(M.birth_record(M.BIRTH_TS, {"a": 50.0}, [{"voice_id": "a"}]))
        led.apply(M.heartbeat_record(M.BIRTH_TS, ["a"]))
        row = machine_row(M.BIRTH_TS + 30, vid="a", tuning="human_7f3a")
        row["gpu_s"] = 4.0
        led.apply(row)
        self.assertGreater(led.lives["a"].answered_human_usd, 0.0)


class Rendering(unittest.TestCase):
    """A demonstration beats an argument. It also has to keep working."""

    def test_a_slice_renders_and_shows_what_a_performer_needs(self):
        choir, human = R.demo_logs()
        text = R.render(choir, human, tail=0, by_part=True)
        for needed in ("ПАРТИТУРА", "ПАРТИЯ", "СМЕРТЬ", "пауза", "D4", "D#4",
                       "поёт человек", "поёт машина"):
            self.assertIn(needed, text)
        # Every sounding breath prints a duration in seconds.
        self.assertGreaterEqual(text.count("s "), 8)

    def test_an_empty_log_says_so_instead_of_crashing(self):
        self.assertIn("пуст", R.render([], []))

    def test_a_slide_and_a_held_tone_are_told_apart(self):
        rising = S.Note(t=0, kind="tone", who="human",
                        contour_hz=[300.0, 310.0, 320.0, 330.0])
        held = S.Note(t=0, kind="tone", who="human",
                      contour_hz=[300.0, 300.4, 300.1, 300.2])
        self.assertEqual(R.gesture(rising), "\u2197")
        self.assertEqual(R.gesture(held), "\u2014")
        self.assertEqual(R.gesture(S.Note(t=0, kind="tone", who="machine")), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
