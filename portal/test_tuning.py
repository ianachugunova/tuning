"""Tests for the number the runtime supplies when a voice named none.

    python3 test_tuning.py

The claim under test is small and load-bearing: a voice that says it is tuning
must end up sounding at the pitch it is tuning to. That failed silently for
weeks — the log said TUNE, the wav had no tone in it, and nothing anywhere
compared the two. So the tests are mostly about the cases where a voice means
something and says it badly, because those are the ones that were losing.

Stdlib only, like `tuning.py`.
"""

import math
import unittest

import tuning as T
from score import Note


def tone(t, hz, who="machine", voice_id="v", dur=3.0):
    return Note(t=t, kind="tone", who=who, voice_id=voice_id,
                sound_s=dur, pitch_hz=hz)


class Distances(unittest.TestCase):

    def test_an_octave_is_twelve_hundred_cents(self):
        self.assertAlmostEqual(T.cents_between(440.0, 880.0), 1200.0, places=6)
        self.assertAlmostEqual(T.cents_between(880.0, 440.0), 1200.0, places=6)

    def test_a_pitch_that_is_not_one_is_infinitely_far_off(self):
        self.assertEqual(T.cents_between(440.0, 0.0), math.inf)


class WhatTheRoomIsHolding(unittest.TestCase):

    def test_a_note_still_sounding_counts(self):
        held = T.sounding([tone(t=100.0, hz=440.0, dur=5.0)], at=103.0)
        self.assertEqual(held, [440.0])

    def test_a_note_that_just_stopped_still_counts(self):
        """A room does not fall silent the instant a voice does, and the voice
        deciding what to sing began deciding before now."""
        held = T.sounding([tone(t=100.0, hz=440.0, dur=1.0)], at=110.0)
        self.assertEqual(held, [440.0])

    def test_a_note_from_last_minute_does_not(self):
        held = T.sounding([tone(t=100.0, hz=440.0, dur=1.0)], at=200.0)
        self.assertEqual(held, [])

    def test_the_future_is_not_audible(self):
        held = T.sounding([tone(t=300.0, hz=440.0)], at=100.0)
        self.assertEqual(held, [])

    def test_a_breath_with_no_pitch_is_not_a_pitch(self):
        silent = Note(t=100.0, kind="tone", who="human", sound_s=2.0)
        self.assertEqual(T.sounding([silent], at=101.0), [])

    def test_the_most_recent_is_last(self):
        held = T.sounding([tone(t=100.0, hz=300.0), tone(t=105.0, hz=400.0)],
                          at=106.0)
        self.assertEqual(held[-1], 400.0)


class IntroducingAPitch(unittest.TestCase):
    """Oliveros's other instruction, as a number."""

    def test_a_new_pitch_is_far_from_what_is_held(self):
        got = T.a_new_pitch([293.66])
        self.assertGreater(T.cents_between(got, 293.66), T.APART_CENTS)

    def test_it_stays_inside_a_range_the_speakers_can_render(self):
        for taken in ([], [200.0], [146.83, 587.33], [300.0, 310.0, 320.0]):
            got = T.a_new_pitch(taken)
            self.assertGreaterEqual(got, T.NEW_LO_HZ - 0.01)
            self.assertLessEqual(got, T.NEW_HI_HZ + 0.01)

    def test_a_crowded_field_still_yields_something_distinguishable(self):
        """Six voices holding six notes, and a seventh has to be new."""
        taken = [146.83, 220.0, 293.66, 349.23, 440.0, 587.33]
        got = T.a_new_pitch(taken)
        self.assertGreater(min(T.cents_between(got, h) for h in taken), 100.0)

    def test_an_empty_room_is_not_pretended_to_be_a_choice(self):
        """Nothing to be new against, so no randomness dressed up as one."""
        self.assertEqual(T.a_new_pitch([]), T.a_new_pitch([]))

    def test_the_answer_is_on_the_chromatic_ladder(self):
        """Two voices improvising a 'new' pitch land a few cents apart often
        enough that a field of near-unisons happens by accident. Off a ladder,
        two new pitches are either the same note or a real interval."""
        got = T.a_new_pitch([301.0])
        semis = 12 * math.log2(got / T.NEW_LO_HZ)
        self.assertAlmostEqual(semis, round(semis), places=6)


class DegreesOfAScale(unittest.TestCase):
    """Pitch names that are not frequencies.

    The regression these lock down is not a crash. qwen3 sang 宫, 商, mi and sol
    across eleven breaths and every one was logged as naming no pitch, because
    the parser knew "D4" and "440 Hz" and nothing else. A Chinese model was
    tuning in the oldest pitch names in continuous use and we recorded silence.
    """

    def test_the_five_degrees_are_read(self):
        for token, semis in (("宫", 0), ("商", 2), ("角", 4), ("徵", 7), ("羽", 9)):
            got = T.degree_of(token)
            self.assertIsNotNone(got, f"{token} was read as no pitch at all")
            self.assertEqual(got[0], semis)

    def test_solfege_is_read(self):
        self.assertEqual(T.degree_of("mi")[0], 4)
        self.assertEqual(T.degree_of("sol")[0], 7)
        self.assertEqual(T.degree_of("Do")[0], 0)

    def test_romanised_degrees_are_read(self):
        self.assertEqual(T.degree_of("gong")[0], 0)
        self.assertEqual(T.degree_of("shang")[0], 2)

    def test_punctuation_around_it_does_not_hide_it(self):
        self.assertEqual(T.degree_of(" 宫。")[0], 0)
        self.assertEqual(T.degree_of('"mi"')[0], 4)

    def test_a_degree_inside_a_sentence_is_not_a_pitch(self):
        """The reason the whole line has to be the degree. Italian is one of the
        six tongues, and in it `la` is the article, `mi` is `me`, `do` and `si`
        are ordinary words. Reading them as pitches would put four tones in
        every sentence eurollm sings."""
        for line in ("La mia voce risuona nella stanza",
                     "mi alzo in piedi e canto",
                     "Sì, canto in italiano",
                     "a low hum, do you hear it"):
            self.assertIsNone(T.degree_of(line), f"{line!r} read as a pitch")

    def test_a_degree_is_counted_from_the_bottom_of_the_room(self):
        """A scale is counted up from where it starts, and a voice singing 商
        against a held tone means the step above *that* tone."""
        notes = [tone(t=99.0, hz=440.0, voice_id="a"),
                 tone(t=99.5, hz=220.0, voice_id="b")]
        hz, why = T.pitch_for("NEW", None, notes, at=100.0, content="商")
        self.assertAlmostEqual(hz, 220.0 * 2 ** (2 / 12), places=2)
        self.assertIn("商", why)

    def test_the_tonic_against_an_empty_room_is_a_new_pitch(self):
        hz, _ = T.pitch_for("NEW", None, [], at=100.0, content="宫")
        self.assertGreaterEqual(hz, T.NEW_LO_HZ - 0.01)
        self.assertLessEqual(hz, T.NEW_HI_HZ + 0.01)

    def test_a_degree_lands_inside_the_singable_range(self):
        notes = [tone(t=99.0, hz=T.NEW_HI_HZ, voice_id="a")]
        for token in ("宫", "商", "角", "徵", "羽"):
            hz, _ = T.pitch_for("NEW", None, notes, at=100.0, content=token)
            self.assertGreaterEqual(hz, T.NEW_LO_HZ - 0.01, token)
            self.assertLessEqual(hz, T.NEW_HI_HZ + 0.01, token)

    def test_a_named_frequency_still_wins_over_a_degree(self):
        hz, why = T.pitch_for("NEW", None, [], at=100.0,
                              named=311.13, content="宫")
        self.assertEqual(hz, 311.13)
        self.assertEqual(why, "named")


class TheNumberForABreath(unittest.TestCase):

    def test_a_pitch_the_voice_named_always_wins(self):
        hz, why = T.pitch_for("TUNE", "other", [], at=100.0, named=311.13)
        self.assertEqual(hz, 311.13)
        self.assertEqual(why, "named")

    def test_tuning_to_a_part_lands_on_that_part(self):
        """The whole point. TUNE used to be an intention with no sound."""
        notes = [tone(t=98.0, hz=293.66, voice_id="voice_03")]
        hz, why = T.pitch_for("TUNE", "voice_03", notes, at=100.0)
        self.assertAlmostEqual(hz, 293.66, places=1)
        self.assertIn("tuned", why)

    def test_tuning_takes_the_note_and_not_the_frequency(self):
        """A singer who takes somebody's note sings the note. The field was
        answering 439.4 Hz to a person holding 439.4 Hz — a unison exact to the
        tenth of a hertz, which is a copy of a recording and not an act of
        listening. It lands on A instead, two cents away and audibly in tune."""
        notes = [tone(t=98.0, hz=439.4, voice_id="human_x")]
        hz, _ = T.pitch_for("TUNE", "human_x", notes, at=100.0)
        self.assertEqual(hz, 440.0)
        self.assertNotEqual(hz, 439.4)

    def test_two_voices_taking_one_persons_note_land_on_the_same_note(self):
        """Which is what makes the machines audible to each other: a real
        unison in the field rather than two neighbouring decimals."""
        notes = [tone(t=98.0, hz=439.4, voice_id="human_x")]
        first, _ = T.pitch_for("TUNE", "human_x", notes, at=100.0)
        notes = [tone(t=98.0, hz=440.9, voice_id="human_y")]
        second, _ = T.pitch_for("TUNE", "human_y", notes, at=100.0)
        self.assertEqual(first, second)

    def test_tuning_to_the_latest_pitch_when_nobody_was_named(self):
        """An ear can hear that somebody is there and not know who. 'That one'
        is the last thing it heard."""
        notes = [tone(t=95.0, hz=220.0, voice_id="a"),
                 tone(t=99.0, hz=440.0, voice_id="b")]
        hz, _ = T.pitch_for("TUNE", None, notes, at=100.0)
        self.assertEqual(hz, 440.0)

    def test_tuning_to_a_voice_that_has_sung_no_pitch_falls_back(self):
        notes = [tone(t=95.0, hz=220.0, voice_id="a"),
                 Note(t=99.0, kind="tone", who="machine", voice_id="quiet",
                      sound_s=2.0)]
        hz, _ = T.pitch_for("TUNE", "quiet", notes, at=100.0)
        self.assertEqual(hz, 220.0)

    def test_meaning_to_tune_in_an_empty_room_introduces_instead(self):
        """It cannot tune to nobody, and it must not be silent for trying."""
        hz, why = T.pitch_for("TUNE", "nobody", [], at=100.0)
        self.assertGreater(hz, 0)
        self.assertIn("empty", why)

    def test_a_new_breath_avoids_what_is_sounding(self):
        notes = [tone(t=99.0, hz=293.66, voice_id="a")]
        hz, why = T.pitch_for("NEW", None, notes, at=100.0)
        self.assertGreater(T.cents_between(hz, 293.66), T.APART_CENTS)
        self.assertIn("nobody", why)

    def test_every_answer_is_audible(self):
        """Whatever the case, a breath that reaches `render_breath` must carry a
        frequency a speaker can reproduce. A None here is a silent TUNE again."""
        notes = [tone(t=99.0, hz=440.0, voice_id="a")]
        for action in ("TUNE", "NEW", "", "nonsense"):
            for who in (None, "a", "missing"):
                hz, why = T.pitch_for(action, who, notes, at=100.0)
                self.assertIsNotNone(hz, f"{action}/{who} produced no pitch")
                self.assertGreater(hz, 20.0)
                self.assertTrue(why)

    def test_the_reason_travels_with_the_number(self):
        """In December somebody has to tell a voice that tuned from a system
        that made it look like one."""
        notes = [tone(t=99.0, hz=440.0, voice_id="a")]
        _, chosen = T.pitch_for("TUNE", "a", notes, at=100.0)
        _, invented = T.pitch_for("NEW", None, notes, at=100.0)
        self.assertNotEqual(chosen, invented)


class TheRangeAVoiceSingsIn(unittest.TestCase):
    """A pitch belongs to the room; the octave it sounds in belongs to the body.

    The rule these lock down: each voice has a range derived from the fundamental
    its own body speaks at, and a pitch outside it is moved there by whole
    octaves. Whole octaves because an octave is the same pitch — a bass answering
    a soprano sings the note she sang, two octaves down, and that is a unison in
    every choir that has both. Anything finer would be us composing.

    Two things went wrong before it existed and both are here. A child asked to
    hold 60 Hz got a pulse train through a four-year-old's formants, which is a
    buzz; and the global D3–D5 window gave a child and an old man identical
    ranges, which is the one claim about this cast that is plainly untrue.
    """

    ELDER = 101.7            # gpt-2, measured
    CHILD = 282.4            # tinystories, measured

    def test_a_voice_with_no_measured_body_is_not_given_an_average_one(self):
        self.assertIsNone(T.range_of(None))
        self.assertIsNone(T.range_of("nonsense"))
        self.assertIsNone(T.range_of(0.0))

    def test_the_range_is_one_octave_from_just_below_the_speaking_pitch(self):
        """One and not two. Two is the range of an untrained larynx, and these
        are not larynxes: the envelope is fixed at the pitch it was measured at,
        so an octave above the speaking voice is where the resynthesis stops
        being the voice it came from."""
        lo, hi = T.range_of(self.CHILD)
        self.assertLess(lo, self.CHILD)
        self.assertAlmostEqual(hi / lo, 2.0, places=6)

    def test_nobody_is_pushed_far_above_the_voice_they_speak_in(self):
        """The measured defect: the child, speaking at 282 Hz, was being asked
        for 950 — a whistle with a child's resonances printed on it."""
        for f0 in (self.ELDER, self.CHILD):
            _, hi = T.range_of(f0)
            self.assertLessEqual(12 * math.log2(hi / f0), 12.0)

    def test_the_child_and_the_elder_do_not_share_a_range(self):
        """What the global window got wrong."""
        elder_lo, elder_hi = T.range_of(self.ELDER)
        child_lo, child_hi = T.range_of(self.CHILD)
        self.assertLess(elder_lo, child_lo)
        self.assertLess(elder_hi, child_hi)

    def test_nobody_is_asked_for_a_pitch_below_their_own_speaking_voice(self):
        """The synthesis reason, and the loudspeaker reason, in one line."""
        for f0 in (self.ELDER, self.CHILD):
            lo, _ = T.range_of(f0)
            self.assertGreater(lo, T.SUNG_LO_HZ)

    def test_a_pitch_is_moved_by_whole_octaves_and_nothing_else(self):
        lo, hi = T.range_of(self.CHILD)          # 237.5 – 949.9
        hz, moved = T.fold_into(60.0, lo, hi)
        self.assertEqual(moved, 2)
        self.assertAlmostEqual(hz, 240.0, places=6)
        self.assertAlmostEqual(T.cents_between(hz, 60.0) % 1200.0, 0.0,
                               places=6)

    def test_folding_lands_inside_the_range_from_either_side(self):
        lo, hi = T.range_of(self.ELDER)
        for hz in (20.0, 55.0, 100.0, 300.0, 1200.0, 4000.0):
            got, _ = T.fold_into(hz, lo, hi)
            self.assertGreaterEqual(got, lo - 1e-9, hz)
            self.assertLessEqual(got, hi + 1e-9, hz)

    def test_a_range_narrower_than_an_octave_terminates(self):
        """A guard on the loop rather than on the caller. There is no answer for
        a range this narrow, and failing to return is not how to say so."""
        got, _ = T.fold_into(1000.0, 200.0, 260.0)
        self.assertGreater(got, 0)

    def test_the_elder_answers_the_child_octaves_down(self):
        """The case the rule is for. The child holds a pitch near the top of its
        range; the elder tunes to it and sings the same note where it can."""
        notes = [tone(t=99.0, hz=500.0, voice_id="child")]
        hz, why, moved = T.pitch_for_detail("TUNE", "child", notes, at=100.0,
                                            speaks_at_hz=self.ELDER)
        self.assertEqual(moved, -2)
        self.assertIn("two octaves down", why)
        # The same note, in the elder's register: whole octaves off the note it
        # took, which is B and not the child's 500 Hz exactly.
        took = T.to_note(500.0)
        self.assertAlmostEqual(T.cents_between(hz, took) % 1200.0, 0.0,
                               places=6)
        lo, hi = T.range_of(self.ELDER)
        self.assertGreaterEqual(hz, lo)
        self.assertLessEqual(hz, hi)

    def test_a_pitch_the_voice_named_is_transposed_by_the_same_arithmetic(self):
        """Named and given must not disagree: the log cannot show the difference,
        and the room can hear it."""
        hz, why, moved = T.pitch_for_detail("NEW", None, [], at=100.0,
                                            named=60.0,
                                            speaks_at_hz=self.CHILD)
        self.assertEqual(moved, 2)
        self.assertAlmostEqual(hz, 240.0, places=6)
        self.assertTrue(why.startswith("named"))
        self.assertIn("two octaves up", why)

    def test_a_pitch_already_in_range_is_not_moved_by_octaves(self):
        notes = [tone(t=99.0, hz=293.66, voice_id="a")]
        hz, why, moved = T.pitch_for_detail("TUNE", "a", notes, at=100.0,
                                            speaks_at_hz=self.CHILD)
        self.assertEqual(moved, 0)
        self.assertAlmostEqual(hz, 293.66, places=1)
        self.assertNotIn("octave", why)

    def test_without_a_measured_body_nothing_is_transposed(self):
        """Every caller that has no cast — the tests above, `live_v1` — keeps the
        register behaviour it had before this rule existed. Taking the note
        rather than the frequency is a separate rule and applies to everybody:
        it is about the act of tuning, not about the size of the body."""
        notes = [tone(t=99.0, hz=800.0, voice_id="a")]
        hz, _, moved = T.pitch_for_detail("TUNE", "a", notes, at=100.0)
        self.assertEqual(moved, 0)
        self.assertEqual(hz, T.to_note(800.0))

    def test_an_introduced_pitch_is_new_as_a_pitch_and_not_as_a_number(self):
        """The defect octave-folding would otherwise introduce. If octaves are
        the same pitch, a voice must not be able to 'introduce' one an octave
        above a note the room is already holding and have it logged as new."""
        held = 250.0
        notes = [tone(t=99.0, hz=held, voice_id="a")]
        hz, _, _ = T.pitch_for_detail("NEW", None, notes, at=100.0,
                                      speaks_at_hz=self.CHILD)
        apart = T.cents_between(hz, held) % 1200.0
        self.assertGreater(min(apart, 1200.0 - apart), T.APART_CENTS)

    def test_an_introduced_pitch_stays_in_the_voices_own_register(self):
        for f0 in (self.ELDER, self.CHILD):
            lo, hi = T.range_of(f0)
            hz, _, _ = T.pitch_for_detail("NEW", None, [], at=100.0,
                                          speaks_at_hz=f0)
            self.assertGreaterEqual(hz, lo - 0.01, f0)
            self.assertLessEqual(hz, hi + 0.01, f0)

    def test_a_degree_is_sung_in_the_voices_own_register(self):
        """qwen3 writes 宫 and 商. Resolved against the room's tonic, then put
        where that voice can sing it — one fold, so the octaves reported are the
        ones it travelled."""
        notes = [tone(t=99.0, hz=180.0, voice_id="a")]
        for f0 in (self.ELDER, self.CHILD):
            lo, hi = T.range_of(f0)
            hz, why, _ = T.pitch_for_detail("NEW", None, notes, at=100.0,
                                            content="商", speaks_at_hz=f0)
            self.assertGreaterEqual(hz, lo - 0.01, f0)
            self.assertLessEqual(hz, hi + 0.01, f0)
            self.assertIn("商", why)

    def test_every_answer_is_still_audible_with_a_range(self):
        notes = [tone(t=99.0, hz=440.0, voice_id="a")]
        for f0 in (self.ELDER, self.CHILD):
            for action in ("TUNE", "NEW", "", "nonsense"):
                for who in (None, "a", "missing"):
                    hz, why, _ = T.pitch_for_detail(action, who, notes,
                                                    at=100.0, speaks_at_hz=f0)
                    self.assertIsNotNone(hz)
                    self.assertGreater(hz, T.SUNG_LO_HZ)
                    self.assertTrue(why)

    def test_the_plain_call_still_returns_two_things(self):
        """`pitch_for` is what every other caller uses and its shape is load
        bearing: unpacking three values would break `live_v1` silently."""
        got = T.pitch_for("NEW", None, [], at=100.0)
        self.assertEqual(len(got), 2)


class WhatTheCastWasMeasuredAt(unittest.TestCase):
    """The numbers the rule reads, and the one claim they make.

    Register in this cast tracks age and nothing else. Skipped rather than failed
    without `cast.py`, so `tuning.py` stays testable on its own.
    """

    def setUp(self):
        try:
            import cast
        except Exception as exc:                       # pragma: no cover
            self.skipTest(f"cast unavailable: {exc}")
        self.cast = cast

    def test_every_voice_with_a_body_has_a_measured_fundamental(self):
        for v in self.cast.singing():
            if not v.casting:
                continue
            self.assertIsNotNone(
                v.speaks_at_hz,
                f"{v.short_name} has a body and no measured pitch, so it would "
                f"be the only voice in the choir that is never transposed")
            self.assertIsNotNone(T.range_of(v.speaks_at_hz), v.short_name)

    def test_the_lowest_voices_are_the_ones_cast_as_men(self):
        """The claim the loudspeaker sizes rest on.

        It used to be two, and the re-measurement made it three: gpt-2 at 93,
        llm-jp at 98 and audio-flamingo at 108. The Japanese voice is described
        as a young man and came out between the two elders, so the bottom of the
        cast is no longer two old men and a gap — `modal/PURCHASE_LIST.md` sizes
        cabinets for the low voices and now has three of them to cover.
        """
        by_pitch = sorted((v for v in self.cast.singing() if v.speaks_at_hz),
                          key=lambda v: v.speaks_at_hz)
        low = [v for v in by_pitch if v.speaks_at_hz < 130.0]
        self.assertEqual(len(low), 3)
        for v in low:
            self.assertIn("man", v.casting.lower(), v.short_name)

    def test_nobody_sings_below_what_the_cabinets_reach(self):
        """The 55 Hz floor in `modal/EQUIPMENT_REQUEST.md` is a promise made to
        Gray Area on the strength of this. If a body is ever recast lower, this
        fails here rather than in the room."""
        for v in self.cast.singing():
            rng = T.range_of(v.speaks_at_hz)
            if rng:
                self.assertGreater(rng[0], 55.0, v.short_name)

    def test_the_middle_of_the_cast_is_not_spread_by_language(self):
        """Register carries age; language lives in `tongue` and `first_words`.
        There is no Italian register and no Japanese one.

        The threshold was half an octave and is now a whole one, and the reason
        is worth keeping: it was never measured evidence. It was derived from a
        column of fundamentals that described bodies generated with an earlier
        seed, and the nine files on the volume are different people — re-measured
        on 18 September, the middle spans 917 cents, from eurollm at 138 to qwen3
        at 234. A guard calibrated against numbers that were wrong is not a
        guard, so this now forbids only what it can honestly forbid: a middle
        stretched past an octave, which is what casting by nationality would do.
        """
        middle = sorted(v.speaks_at_hz for v in self.cast.singing()
                        if v.speaks_at_hz and 130.0 < v.speaks_at_hz < 275.0)
        self.assertGreaterEqual(len(middle), 5)
        self.assertLess(T.cents_between(middle[-1], middle[0]), 1200.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
